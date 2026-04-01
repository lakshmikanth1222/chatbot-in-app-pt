# 🩺 MediAssist: AI-Powered Medical Record Assistant

## 📖 Overview
MediAssist is a secure, end-to-end Retrieval-Augmented Generation (RAG) prototype designed to help patients interact with and understand their personal medical documents (such as CT scans, prescriptions, and lab results). 

The application architecture separates data ingestion, backend orchestration, and frontend user experience[cite: 14]. [cite_start]It leverages local embedding models for secure document parsing, a cloud-based vector database for semantic search, and Google's Gemini 2.5 Flash LLM to act as a proactive, empathetic "temporary doctor" with strict clinical guardrails.

---

## 🏗️ System Architecture

The project is structured into two primary workflows:

1.  **Offline Data Ingestion (`ingest_pdf.py`) **
    * Reads raw medical PDF files from the local directory.
    * Chunks the text and generates 384-dimensional vector embeddings locally using `FastEmbed` (`BAAI/bge-small-en-v1.5`) to ensure privacy and eliminate API costs during ingestion.
    * Upserts the text chunks, vectors, and metadata into a Neon Serverless PostgreSQL database.

2.  **Online Chat Interface (`app.py` / `frontend.py` & `backend.py`) **
    * **Frontend:** Captures user queries via a sleek Dark Mode Streamlit interface. Features a secure mock-login portal and offers a chat interface alongside "Quick Insight" buttons.
    * **Backend:** Acts as the secure middle layer handling authentication and LLM orchestration. Converts the query into a vector and performs a semantic similarity search against the Neon database to retrieve the top relevant medical document snippets.
    * **LLM Integration:** Constructs a strict system prompt combining the user query, retrieved context, and medical safety guardrails, then streams the generated, structured response from Gemini back to the user.

---

## 💻 Tech Stack

* **Frontend UI:** Streamlit 
* **AI Orchestration:** LlamaIndex 
* **Large Language Model (LLM):** Google Gemini 2.5 Flash 
* **Embedding Model:** FastEmbed (BAAI/bge-small-en-v1.5) - Runs locally for zero API costs and enhanced privacy.
* **Vector Database:** Neon Serverless PostgreSQL (utilizing the pgvector extension) 
* **Document Parsing:** PyPDF 

---

## 📁 Project Structure

```text
mediassist/
│
├── ingest_pdf.py       # Offline data pipeline for chunking and embedding PDFs 
├── backend.py          # FastAPI application for DB connections and AI orchestration
├── frontend.py         # Streamlit chat interface and secure portal UI (formerly app.py) 
├── requirements.txt    # The complete list of Python dependencies required to run the application
├── .env                # Environment variables (API Keys and DB URI)
└── dummy_data/         # Directory containing sample medical PDFs for ingestion (*.pdf)
