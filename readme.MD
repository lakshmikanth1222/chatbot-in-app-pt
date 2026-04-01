🩺 MediAssist: AI-Powered Medical Record Assistant
📖 Overview
MediAssist is a secure, end-to-end Retrieval-Augmented Generation (RAG) system designed to help patients interact with and understand their personal medical documents.

The application architecture separates data ingestion, backend orchestration, and frontend user experience. It leverages local embedding models for secure document parsing, a cloud-based vector database for semantic search, and Google's Gemini 2.5 Flash LLM to act as a proactive, empathetic "temporary doctor" with strict clinical guardrails.

🏗️ System Architecture
The project is structured into three main workflows:

Offline Data Ingestion (ingest_pdf.py)

Extracts text from raw medical PDFs using PyPDF.

Generates 384-dimensional vector embeddings locally using FastEmbed (BAAI/bge-small-en-v1.5) to ensure privacy and eliminate API costs during ingestion.

Upserts text chunks, metadata, and embeddings into a Neon Serverless PostgreSQL database.

FastAPI Backend (backend.py)

Acts as the secure middle layer handling authentication and LLM orchestration.

Embeds incoming user queries using Google's text-embedding-004 (explicitly truncated to 384 dimensions to match the database schema).

Performs Cosine Similarity vector searches via PostgreSQL's pgvector extension.

Constructs a highly structured prompt enforcing strict guidelines (Bottom-Line-Up-Front directness, no context dumping, and medical disclaimers) and queries the Gemini 2.5 Flash model.

Streamlit Frontend (frontend.py)

Provides a responsive, Dark Mode UI for patients.

Features a secure mock-login portal (validating Name and Date of Birth against the database).

Offers a chat interface alongside "Quick Insight" buttons for one-click access to health summaries, medication interactions, and diet plans.

💻 Tech Stack
Frontend Interface: Streamlit, Requests

Backend Framework: FastAPI, Uvicorn, Pydantic

Database: Neon Serverless PostgreSQL, asyncpg, pgvector

Large Language Model (LLM): Google Gemini 2.5 Flash (via google-genai SDK)

Embedding Models: * Ingestion: FastEmbed (BAAI/bge-small-en-v1.5) via LlamaIndex.

Querying: Gemini text-embedding-004 (Configured for 384 Output Dimensionality).

Document Parsing: LlamaIndex Core, PyPDF

📁 Project Structure
Plaintext
mediassist/
│
├── ingest_pdf.py       # Offline data pipeline for chunking and embedding PDFs
├── backend.py          # FastAPI application for DB connections and AI orchestration
├── frontend.py         # Streamlit chat interface and secure portal UI
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (API Keys and DB URI)
└── dummy_data/         # Directory containing sample medical PDFs for ingestion
🚀 Setup & Installation
1. Prerequisites
Python 3.10+

A Neon Database account with the pgvector extension enabled.

A Google Gemini API Key.

2. Install Dependencies
Clone the repository and install the required packages:

Bash
pip install -r requirements.txt
pip install google-genai  # Ensure the latest Google GenAI SDK is installed
3. Environment Configuration
Create a .env file in the root directory and add your credentials:

Code snippet
GEMINI_API_KEY=your_google_gemini_api_key_here
NEON_DATABASE_URI=postgresql://user:password@host/dbname?sslmode=require
⚙️ Usage Instructions
Step 1: Ingest Medical Records
Place your medical PDFs (e.g., ct_scan_results.pdf, prescriptions.pdf) into a folder named dummy_data in the root directory.
Run the ingestion script to parse, embed, and upload the data to your database:

Bash
python ingest_pdf.py
Note: The ingestion process runs locally using FastEmbed, meaning it does not consume Gemini API credits.

Step 2: Start the Backend Server
Open a terminal and start the FastAPI server:

Bash
uvicorn backend:app --reload
The backend will run on http://127.0.0.1:8000. You can view the API documentation at http://127.0.0.1:8000/docs.

Step 3: Launch the Frontend App
Open a second terminal window and start the Streamlit UI:

Bash
streamlit run frontend.py
A browser window will automatically open at http://localhost:8501. Log in using patient credentials that exist in your patient database table (e.g., Name: "Arjun Mehta", DOB: "1982-11-30").

🔌 API Endpoints (Backend)
GET /health: Health check endpoint verifying Database and Gemini API connectivity.

POST /login: Authenticates a user based on exact matches of name and password (mapped to Date of Birth) in the database. Returns patient metadata.

POST /chat: The core RAG endpoint. Accepts a patient_id and a message. Performs vector retrieval against patient records, synthesizes context, and returns a formatted medical insight from the LLM.

🛡️ Security & Privacy Guardrails
Data Isolation: The semantic search query (backend.py) uses a strict WHERE metadata->>'patient_id' = $2 clause. A patient can only retrieve their own embedded records.

AI System Prompting: The LLM is heavily constrained by system instructions to prevent hallucination. It is instructed to only answer based on the retrieved context, use direct BLUF (Bottom-Line-Up-Front) formatting, and avoid context dumping.

Mandatory Disclaimer: Every response programmatically includes a medical disclaimer emphasizing that MediAssist is an AI prototype and not a licensed healthcare provider.
