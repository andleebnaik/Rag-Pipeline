# RAG-Pipeline

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using FastAPI. The pipeline processes PDF documents, generates vector embeddings, stores them in Qdrant, and provides endpoints for querying answers and predicting ratings.

---

## 🔧 Project Structure

RAG-PIPELINE/
├── app/
│ ├── routers/
│ │ └── user.py # API endpoints (FastAPI routers)
│ ├── services/
│ │ ├── process_file.py # PDF processing and text extraction
│ │ ├── response_generation.py # Generates responses using LLM
│ │ └── vector_store.py # Handles vector DB operations (Qdrant)
├── prompt_studio.json # Prompt configuration for LLM
├── .env # Environment variables (API keys, DB configs)
├── app.log # Log file for server/debug logs
├── main.py # Main FastAPI app runner
├── requirements.txt # Python package dependencies
└── README.md # Project documentation


---

##  Features

-  Upload and process PDF files
-  Generate answers using an LLM (e.g., Gemini Pro)
-  Store and retrieve document embeddings using Qdrant
-  Prompt management via `prompt_studio.json`
-  REST API endpoints for interaction

---

##  Setup

### 1. Clone the repository

git clone https://github.com/andleebnaik/Rag-Pipeline.git
cd rag-pipeline

### 2. Create virtual env and install requirements
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

### 3. Run the Application

uvicorn main:app --reload

API will be available at: http://127.0.0.1:8000

### Technologies Used

FastAPI – Web framework

Qdrant – Vector database

gpt-4o – LLM

PyMuPDF / pdfplumber / unstructured – PDF parsing

Pydantic – Data validation

Faiss - vector search