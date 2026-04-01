# 🩺 MediAssist: AI-Powered Medical Record Assistant

## 📖 Overview
[cite_start]MediAssist is a secure, end-to-end Retrieval-Augmented Generation (RAG) prototype designed to help patients interact with and understand their personal medical documents (such as CT scans, prescriptions, and lab results)[cite: 4, 5]. 

[cite_start]The application architecture separates data ingestion, backend orchestration, and frontend user experience[cite: 14]. [cite_start]It leverages local embedding models for secure document parsing, a cloud-based vector database for semantic search, and Google's Gemini 2.5 Flash LLM to act as a proactive, empathetic "temporary doctor" with strict clinical guardrails[cite: 5, 9, 10, 11].

---

## 🏗️ System Architecture

[cite_start]The project is structured into two primary workflows[cite: 14]:

1.  [cite_start]**Offline Data Ingestion (`ingest_pdf.py`) [cite: 15]**
    * Reads raw medical PDF files from the local directory[cite: 16].
    * [cite_start]Chunks the text and generates 384-dimensional vector embeddings locally using `FastEmbed` (`BAAI/bge-small-en-v1.5`) to ensure privacy and eliminate API costs during ingestion[cite: 10, 17].
    * [cite_start]Upserts the text chunks, vectors, and metadata into a Neon Serverless PostgreSQL database[cite: 18].

2.  [cite_start]**Online Chat Interface (`app.py` / `frontend.py` & `backend.py`) [cite: 19]**
    * [cite_start]**Frontend:** Captures user queries via a sleek Dark Mode Streamlit interface[cite: 20]. Features a secure mock-login portal and offers a chat interface alongside "Quick Insight" buttons.
    * **Backend:** Acts as the secure middle layer handling authentication and LLM orchestration. Converts the query into a vector and performs a semantic similarity search against the Neon database to retrieve the top relevant medical document snippets[cite: 21, 22].
    * [cite_start]**LLM Integration:** Constructs a strict system prompt combining the user query, retrieved context, and medical safety guardrails, then streams the generated, structured response from Gemini back to the user[cite: 23, 24].

---

## 💻 Tech Stack

* [cite_start]**Frontend UI:** Streamlit [cite: 7]
* **AI Orchestration:** LlamaIndex [cite: 8]
* [cite_start]**Large Language Model (LLM):** Google Gemini 2.5 Flash [cite: 9]
* [cite_start]**Embedding Model:** FastEmbed (BAAI/bge-small-en-v1.5) - Runs locally for zero API costs and enhanced privacy[cite: 10].
* **Vector Database:** Neon Serverless PostgreSQL (utilizing the pgvector extension) [cite: 11]
* [cite_start]**Document Parsing:** PyPDF [cite: 12]

---

## 📁 Project Structure

```text
mediassist/
│
[cite_start]├── ingest_pdf.py       # Offline data pipeline for chunking and embedding PDFs [cite: 52]
├── backend.py          # FastAPI application for DB connections and AI orchestration
[cite_start]├── frontend.py         # Streamlit chat interface and secure portal UI (formerly app.py) [cite: 51]
[cite_start]├── requirements.txt    # The complete list of Python dependencies required to run the application [cite: 53]
├── .env                # Environment variables (API Keys and DB URI)
[cite_start]└── dummy_data/         # Directory containing sample medical PDFs for ingestion (*.pdf) [cite: 54]
