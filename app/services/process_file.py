import os
import uuid
import shutil
import logging
import tempfile
from typing import Optional
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse
from unstructured.partition.pdf import partition_pdf
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


openai_api_key = os.getenv('TEAMIFIED_OPENAI_API_KEY')
embedding_model = os.getenv('EMBEDDING_MODEL')
client = OpenAI(api_key=openai_api_key)

class FileUploadHandler:
    UPLOAD_TEMP_DIR = tempfile.mkdtemp()  # Shared temp directory for all instances

    def __init__(self, file: Optional[UploadFile] = None):
        self.file = file
        self.upload_dir = FileUploadHandler.UPLOAD_TEMP_DIR
        self.file_metadata = {}  # to store file_id to original filename mapping
        
    async def upload(self) -> dict:
        if not self.file:
            raise ValueError("No file provided for upload")

        try:
            file_id = str(uuid.uuid4())
            file_path = os.path.join(self.upload_dir, f"{file_id}.pdf")

            with open(file_path, "wb") as f:
                shutil.copyfileobj(self.file.file, f)

            original_filename = getattr(self.file, 'filename', 'unknown')

            logging.info(f"Uploaded file '{original_filename}' saved at: {file_path}")

            return {
                "file_id": file_id,
                "file_name": original_filename
            }

        except Exception as e:
            logging.error(f"Error uploading file {getattr(self.file, 'filename', 'unknown')}: {str(e)}")
            raise e

        

    def get_file_path(self, file_id: str) -> dict:
        file_path = os.path.join(self.upload_dir, f"{file_id}.pdf")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found for file_id: {file_id}")

        # Assuming you stored the original filename in a dict: self.file_metadata[file_id] = filename
        original_filename = self.file_metadata.get(file_id, "unknown")  # fallback to 'unknown' if not found

        return {
            "file_path": file_path,
            "file_name": original_filename
        }

    def generate_embeddings(self, text):
        try:
            query_embedding = client.embeddings.create(
                input=text,
                model=embedding_model
            ).data[0].embedding

            return query_embedding
        except Exception as exc:
            logging.error(f"Processing failed with exception: {exc}")
            raise HTTPException(status_code=500, detail=f"Processing failed with exception: {exc}")

    def parse_file(self, file_id: str, chunk_size: int = 1500) -> dict:
        file_attr = self.get_file_path(file_id)
        try:
            elements = partition_pdf(
                filename=file_attr["file_path"],
                strategy="fast",
                extract_images_in_pdf=False
            )

            chunks = {}
            current_chunk = ""
            chunk_counter = 1

            for el in elements:
                el_text = str(el).strip()
                if not el_text:
                    continue

                # If adding this element exceeds the chunk size, save the current chunk and start a new one
                if len(current_chunk) + len(el_text) + 1 > chunk_size:
                    chunk_id = f"chunk_{chunk_counter}"
                    chunks[chunk_id] = {
                        "_id": chunk_id,
                        "file_id": file_id,
                        "file_name": file_attr["file_name"],
                        "text": current_chunk.strip(),
                        "total_chars": len(current_chunk.strip()),
                        "embedding": self.generate_embeddings(current_chunk.strip())
                    }
                    chunk_counter += 1
                    current_chunk = ""

                current_chunk += el_text + "\n"

            # Add the last remaining chunk if any
            if current_chunk.strip():
                chunk_id = f"chunk_{chunk_counter}"
                chunks[chunk_id] = {
                    "_id": chunk_id,
                    "file_id": file_id,
                    "file_name": file_attr["file_name"],
                    "text": current_chunk.strip(),
                    "total_chars": len(current_chunk.strip()),
                    "embedding": self.generate_embeddings(current_chunk.strip())
                }

            logging.info(f"Successfully created {len(chunks)} chunks for file.")
            #logging.info(f"----> Chunks: {chunks}")
            return chunks

        except Exception as e:
            logging.error(f"Error parsing file {file_id}: {str(e)}")
            raise e
