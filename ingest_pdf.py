import os
import logging
from urllib.parse import urlparse
from pypdf import PdfReader

# LlamaIndex Core & Database
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.postgres import PGVectorStore

# Local Embeddings
from llama_index.embeddings.fastembed import FastEmbedEmbedding

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 1. CONFIGURATION
# ==========================================
# No API keys needed for ingestion anymore! It is 100% local.
NEON_DATABASE_URI = "postgresql://neondb_owner:npg_Wt1zuaco6Vsv@ep-proud-sun-a16kwegi-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require" 
DATA_FOLDER = "dummy_data"

# ==========================================
# 2. SETUP LOCAL EMBEDDINGS & DATABASE
# ==========================================
logging.info("Initializing local FastEmbed engine (Zero API limits)...")
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

def get_vector_store():
    logging.info("Connecting to Neon PostgreSQL...")
    parsed = urlparse(NEON_DATABASE_URI)
    
    return PGVectorStore.from_params(
        host=parsed.hostname,
        port=parsed.port or 5432,
        user=parsed.username,
        password=parsed.password,
        database=parsed.path.lstrip("/"),
        table_name="patient_records_deepseek", # New table name
        embed_dim=384,
    )

# ==========================================
# 3. LOCAL PDF EXTRACTION
# ==========================================
def extract_text_locally(filepath):
    filename = os.path.basename(filepath)
    logging.info(f"Reading {filename} locally...")
    
    try:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        
        if not text.strip():
            logging.warning(f"No text found in {filename}. If it's a scanned image, local OCR is required.")
            return None
            
        logging.info(f"Extracted {len(text)} characters from {filename}.")
        return text.strip()

    except Exception as e:
        logging.error(f"Error reading {filename}: {e}")
        return None

# ==========================================
# 4. RUN INGESTION
# ==========================================
def main():
    if not os.path.exists(DATA_FOLDER):
        logging.error(f"Please create a '{DATA_FOLDER}' folder and put PDFs inside.")
        return

    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    documents_to_embed = []

    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".pdf")]
    
    for filename in pdf_files:
        filepath = os.path.join(DATA_FOLDER, filename)
        raw_text = extract_text_locally(filepath)
        
        if raw_text:
            doc = Document(
                text=f"Source File: {filename}\n\nContent:\n{raw_text}",
                metadata={"file_name": filename}
            )
            documents_to_embed.append(doc)

    if documents_to_embed:
        logging.info(f"Vectorizing and uploading {len(documents_to_embed)} documents to Neon Database...")
        # Since this is local, there are no rate limits. It will process all at once.
        VectorStoreIndex.from_documents(documents_to_embed, storage_context=storage_context)
        logging.info("SUCCESS! All data is securely stored in your Postgres Database.")
    else:
        logging.warning("No PDF data was processed successfully.")

if __name__ == "__main__":
    main()