import logging
import traceback
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
load_dotenv()
import os,logging,uuid
import faiss
import numpy as np

class FAISSSearchEngine:
    def __init__(self, embeddings, payloads):
        self.dimension = len(embeddings[0])  # e.g. 1536
        self.index = faiss.IndexFlatL2(self.dimension)  # or IndexFlatIP for cosine-like
        self.embeddings = np.array(embeddings).astype('float32')
        self.payloads = payloads  # List of text chunks or dicts
        
        self.index.add(self.embeddings)
        logging.info(f"FAISS index built with {self.index.ntotal} vectors.")

    def search(self, query_embedding, top_k=5):
        """
        Search for similar vectors using FAISS
        
        Args:
            query_embedding: The query vector (list or numpy array)
            top_k: Number of top results to return
            
        Returns:
            List of tuples: [(distance, payload), ...]
        """
        try:
            # Ensure query_embedding is the right format
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding).astype('float32')
            
            # Reshape to 2D array if needed (FAISS expects 2D)
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Perform the search
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Format results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx != -1:  # Valid result
                    payload = self.payloads[idx]
                    results.append({
                        'distance': float(distance),
                        'payload': payload,
                        'rank': i + 1
                    })
            
            logging.info(f"FAISS search returned {len(results)} results")
            return results
            
        except Exception as e:
            logging.error(f"FAISS search failed: {str(e)}")
            return []

    def search_similar_chunks(self, embedding, top_k=5):
        """
        Alias for search method to maintain backward compatibility
        """
        return self.search(embedding, top_k)


class QuadrantManager:
    """Manager for Qdrant vector database operations"""
    
    def __init__(self, collection_name: str = "philippine_history"):

        qdrant_url = os.getenv('QDRANT_URL')
        self.qdrant_client = QdrantClient(url=qdrant_url, timeout= 30)
        self.collection_name = collection_name
        self.qdrant_client, self.status = self.connect_qdrant(collection_name)

    def connect_qdrant(self, collection_name="philippine_history"):
        logging.info("Connecting Qdrant")
        try:
            # Check if the collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            logging.info(f"Qdrant Collections fetched: {collection_names}")

            if collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
                logging.info(f"Collection '{collection_name}' created successfully.")
            else:
                logging.info(f"Collection '{collection_name}' already exists.")

            return self.qdrant_client, {"status_code": 200, "message": "Connection Established!"}

        except Exception as e:
            logging.error(f"Couldn't connect to vector database. \n ERROR:: {traceback.format_exc()}")
            return None, {"status_code": 401, "message": "Couldn't connect to vector database."}

    def upsert_data(self,client,response_obj, pdf_file_id, doc_id ):
        try:
            embedding_vector = response_obj["embedding"]  # fixed key name
            payload = {
                "doc_id": doc_id,
                "text": response_obj["text"],
                "file_id": pdf_file_id
            }

            client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding_vector,
                        payload=payload
                    )
                ]
            )

            logging.info(f"Data inserted successfully.")
        
        except Exception as e:
            logging.error(f"Error inserting data into Qdrant: {e}")
            raise e
    
    def fetch_all_vectors_and_payloads(self):
            """Fetch all vectors and their payloads from the Qdrant collection"""
            scroll_result = self.self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # adjust limit based on expected data size
                with_payload=True,
                with_vectors=True
            )
            vectors = [point.vector for point in scroll_result[0]]
            payloads = [point.payload for point in scroll_result[0]]
            return vectors, payloads