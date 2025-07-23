# app/routers/user.py - FastAPI Routes
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import time
from pathlib import Path
from app.services.process_file import FileUploadHandler
from app.services.vector_store import QuadrantManager
from app.services.vector_store import FAISSSearchEngine
from app.services.response_generation import LLMService
import uuid

#from services.llm_service import LLMService

                                                            # Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

                                                            # Initialize FastAPI and APIRouter

router = APIRouter()

# Pydantic models
class QueryRequest(BaseModel):
    query: str



@router.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information"""
    logging.info("Root endpoint accessed")
    return {
        "message": "Philippine History RAG Pipeline API",
        "version": "1.0.0",
        "description": "Upload PDFs and query Philippine History content using advanced RAG",
        "endpoints": {
            "upload": "POST /upload - Upload a PDF file for processing",
            "query": "POST /query - Query the loaded document",
            "status": "GET /status - Check system status",
            "demo-queries": "GET /demo-queries - Get sample queries",
            "reset": "DELETE /reset - Reset the system"
        },
        "features": [
            "PDF text extraction and chunking",
            "FAISS vector similarity search", 
            "OpenAI GPT integration",
            "Background processing"
        ]
    }

@router.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    try:
        handler = FileUploadHandler(file)
        response = await handler.upload()
        return response,
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/parse-file/{file_id}")

def parse_files(file_id: str):
    try:
        handler = FileUploadHandler()
        file_attributes = handler.get_file_path(file_id)
        logging.info(f"Parsing file at path: {file_attributes['file_path']}")
        
        parsed_chunks = handler.parse_file(file_id) #Embedded chunks
        if not parsed_chunks:
            raise HTTPException(status_code=400, detail="Parsed content is empty.")

        manager = QuadrantManager()
        client, response = manager.connect_qdrant()

        doc_id = str(uuid.uuid4())  # Optional: keep doc ID consistent for all chunks

        if response["status_code"] == 200:
            logging.info(f"parsed text keys: {parsed_chunks.keys()}")
            for chunk_id, meta_data in parsed_chunks.items():
                logging.info(f"----> Upserting chunk with text:{ type(parsed_chunks)}\n {chunk_id} \n {type(meta_data)} \n {client} \n {doc_id} \n {file_id}")  
                manager.upsert_data(client, meta_data, file_id, doc_id)

            return {
                "status_code": 200,
                "message": "Parsing completed and data saved in Qdrant!"
            }
        else:
            logging.error(f"Failed to connect to Qdrant: {response['message']}")
            raise HTTPException(status_code=response["status_code"], detail=response["message"])
    except Exception as e:
        logging.error(f"Parsing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")

    

@router.post("/user-query", tags=["RAG Pipeline"])
async def user_query(query_object: QueryRequest):
    """Process user query against the uploaded document"""
    start_time = time.time()
    try:
        # Initialize file handler and vector store manager
        handler = FileUploadHandler()
        manager = QuadrantManager()

        # Step 1: Generate embedding
        query_embedding = handler.generate_embeddings(query_object.query)
        logging.info(f"User Query: {query_object.query}")

        # Step 2: Fetch all vectors and payloads from Qdrant
        all_embeddings, all_payloads = manager.fetch_all_vectors_and_payloads()

        # Step 3: Initialize FAISS search engine
        search_engine = FAISSSearchEngine(all_embeddings, all_payloads)

        # Step 4: Perform similarity search
        search_results = search_engine.search(query_embedding, top_k=5)

        logging.info(f"Top results from FAISS: {search_results}\n execution_time : {round(time.time() - start_time, 2)}")

        # Step 5: Generate Response  
        llm_service = LLMService()
        references =  [item['payload']['text'] for item in search_results]
        response_obj = {
            "query": query_object.query,
            "references": references
        }
        query_response = llm_service.generate_response(response_obj )  
        return {
            "query": query_object.query,
            "response": query_response
        }
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
