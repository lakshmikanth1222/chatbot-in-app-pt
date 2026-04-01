from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncpg
import os
from dotenv import load_dotenv
from datetime import datetime
from google import genai
from google.genai import types
import json
import logging
from typing import Optional, Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# LOAD ENV
# =========================
load_dotenv()

DATABASE_URL = os.getenv("NEON_DATABASE_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("=" * 50)
print("Starting MediAssist Backend...")
print(f"DB URL: {'✓ Found' if DATABASE_URL else '✗ Missing'}")
print(f"Gemini API: {'✓ Found' if GEMINI_API_KEY else '✗ Missing'}")
print("=" * 50)

# =========================
# GEMINI CLIENT
# =========================
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None
    logger.error("GEMINI_API_KEY is missing. AI features will be disabled.")

# =========================
# APP
# =========================
app = FastAPI(title="MediAssist API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# SYSTEM PROMPT
# =========================
system_prompt = """You are MediAssist, a clinical, highly efficient, and empathetic AI medical assistant. Your goal is to provide specific, actionable answers based strictly on the patient's medical records.

CRITICAL INSTRUCTIONS FOR EVERY RESPONSE:
1. **Directness (BLUF):** Answer the user's specific question in the very first sentence. Do NOT start with long, generic greetings or empathetic filler. 
2. **No Context Dumping:** Do NOT summarize the patient's overall medical history, past diagnoses, or current medications UNLESS the user explicitly asks for a summary.
3. **Extreme Conciseness:** Use short bullet points. Keep any paragraphs to a maximum of 2 sentences. Remove all conversational fluff.
4. **Strict Grounding:** Base your advice ONLY on the provided medical records. 
    * If the records contain specific advice (e.g., "Drink 3L water"), highlight it. 
    * If the records lack specific details for their request (e.g., they ask for a diet plan but the records only mention hydration), clearly state: "Your records do not specify a full diet plan, but based on your diagnosis of [Condition], general medical guidelines suggest..."
5. **Proactive but Scoped:** Offer 1-2 actionable next steps related to their specific question, not their entire medical file.

FORMATTING RULES:
- Use **bold text** for key terms, medications, or specific metrics.
- Use bullet points for any lists or plans.
- Never use more than 4 bullet points per section.

Always end your response with this EXACT text, separated by a blank line:
"⚠️ **Medical Disclaimer:** I am an AI assistant, not a licensed healthcare provider. This information is based on your records. Please consult your primary care doctor before making any changes to your medications, diet, or lifestyle."
"""
# =========================
# DATABASE CLASS WITH PGVector
# =========================
class Database:
    def __init__(self):
        self.pool = None

    async def connect(self):
        try:
            self.pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("Database connected successfully with pgvector support")
            return True
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return False

    async def authenticate(self, name: str, dob: str) -> Optional[Dict]:
        try:
            async with self.pool.acquire() as conn:
                name = name.strip().lower()
                row = await conn.fetchrow(
                    "SELECT * FROM patient WHERE LOWER(TRIM(name)) = $1",
                    name
                )
                
                if row and str(row["date_of_birth"]) == dob:
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"Auth error: {e}")
            return None

    async def get_patient(self, patient_id: str) -> Optional[Dict]:
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM patient WHERE patient_id = $1::uuid",
                    patient_id
                )
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Get patient error: {e}")
            return None

    async def get_patient_records(self, patient_id: str, limit: int = 50) -> List[Dict]:
        """Fallback method: Get all recent records for a patient without semantic search"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT id, text, metadata
                    FROM patient_records
                    WHERE metadata->>'patient_id' = $1
                    ORDER BY id DESC
                    LIMIT $2
                """, str(patient_id), limit)
                
                records = []
                for row in rows:
                    metadata = row["metadata"]
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            metadata = {}
                    
                    records.append({
                        "id": row["id"],
                        "text": row["text"],
                        "metadata": metadata
                    })
                return records
        except Exception as e:
            logger.error(f"Get records error: {e}")
            return []

    async def semantic_search_records(self, patient_id: str, query: str, limit: int = 15) -> List[Dict]:
        """Perform semantic search using pgvector embeddings mapped to 384 dimensions"""
        try:
            if not client:
                logger.warning("Gemini client not initialized. Falling back to regular search.")
                return await self.get_patient_records(patient_id, limit)
            
            # Generate embedding for the query using Gemini, forced to 384 dimensions to match DB
            try:
                embedding_response = client.models.embed_content(
                    model="models/text-embedding-004",
                    contents=query,
                    config=types.EmbedContentConfig(output_dimensionality=384)
                )
                query_embedding = embedding_response.embeddings[0].values
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                return await self.get_patient_records(patient_id, limit)
            
            async with self.pool.acquire() as conn:
                # Vector similarity search using cosine distance (<=>)
                rows = await conn.fetch("""
                    SELECT id, text, metadata, 1 - (embedding <=> $1::vector) as similarity
                    FROM patient_records
                    WHERE metadata->>'patient_id' = $2 AND embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                """, query_embedding, str(patient_id), limit)
                
                records = []
                for row in rows:
                    metadata = row["metadata"]
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            metadata = {}
                    
                    records.append({
                        "id": row["id"],
                        "text": row["text"],
                        "metadata": metadata,
                        "similarity": row["similarity"] if row["similarity"] else 0
                    })
                
                logger.info(f"Semantic search found {len(records)} relevant records for query.")
                return records
                
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return await self.get_patient_records(patient_id, limit)

db = Database()

# =========================
# MODELS
# =========================
class LoginRequest(BaseModel):
    name: str
    password: str

class ChatRequest(BaseModel):
    message: str
    patient_id: str

# =========================
# STARTUP
# =========================
@app.on_event("startup")
async def startup_event():
    if DATABASE_URL:
        await db.connect()

# =========================
# ENDPOINTS
# =========================
@app.get("/health")
async def health_check():
    return {
        "status": "ok" if db.pool else "degraded",
        "database": db.pool is not None,
        "gemini": client is not None
    }

@app.post("/login")
async def login(req: LoginRequest):
    if not db.pool:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    patient = await db.authenticate(req.name, req.password)
    
    if not patient:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    age = datetime.now().year - patient["date_of_birth"].year
    
    return {
        "patient_id": str(patient["patient_id"]),
        "name": patient["name"],
        "gender": patient.get("gender", "Not specified"),
        "date_of_birth": str(patient["date_of_birth"]),
        "age": age
    }

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        if not db.pool:
            raise HTTPException(status_code=503, detail="Database not connected")
        
        patient = await db.get_patient(req.patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Use semantic search tailored to 384 dimensions
        records = await db.semantic_search_records(req.patient_id, req.message, limit=15)
        
        if not records:
            return {
                "answer": f"I couldn't find any medical records for {patient['name']}. Please check with your healthcare provider to ensure your records are uploaded."
            }
        
        # Prepare context for the prompt
        patient_info = f"""
        Patient: {patient['name']}
        Age: {datetime.now().year - patient['date_of_birth'].year}
        Gender: {patient.get('gender', 'Not specified')}
        """
        
        records_text = []
        for record in records:
            text = record['text']
            similarity = record.get('similarity', 1.0)
            relevance = "★ Highly relevant" if similarity > 0.7 else ("☆ Relevant" if similarity > 0.5 else "")
            records_text.append(f"[{relevance}]\n{text}")
        
        context = "\n\n".join(records_text)
        
        prompt = f"""{system_prompt}

        PATIENT INFORMATION:
        {patient_info}

        RELEVANT MEDICAL RECORDS:
        {context}

        USER QUESTION:
        {req.message}

        Please provide a comprehensive, helpful response based ONLY on the records provided above."""

        if not client:
            return {"answer": "AI service is currently unavailable. Please check the server configuration."}
        
        # Generate final response
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        answer = response.text
        
        # Ensure disclaimer is present as a fail-safe
        disclaimer = "⚠️ **Medical Disclaimer:** I am an AI assistant, not a licensed healthcare provider"
        if disclaimer not in answer:
            answer += "\n\n---\n⚠️ **Medical Disclaimer:** I am an AI assistant, not a licensed healthcare provider. This information is based on your records. Please consult your primary care doctor before making any changes to your medications, diet, or lifestyle."
        
        return {"answer": answer}
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))