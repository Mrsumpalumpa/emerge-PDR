import os
from qdrant_client import QdrantClient, models
from typing import List, Dict, Any, Optional
import httpx # Import httpx here for direct use in this file

# Constants for Qdrant (can be moved to config or env vars)
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT_GRPC = os.getenv("QDRANT_PORT_GRPC", "6334") # gRPC port for client
QDRANT_PORT_REST = os.getenv("QDRANT_PORT_REST","6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None) # If Qdrant requires an API key

# Constants for Ollama (for embedding generation)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text") # Or another embedding model

class QdrantManager():
    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT_REST, api_key=QDRANT_API_KEY)
        self.ollama_base_url = OLLAMA_BASE_URL

    async def _get_embedding(self, text: str) -> List[float]:
        """Generates embedding for a given text using Ollama."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={"model": EMBEDDING_MODEL_NAME, "prompt": text},
                    timeout=60.0 # Increased timeout for embedding generation
                )
                response.raise_for_status() # Raise an exception for HTTP errors
                data = response.json()
                return data["embedding"]
        except Exception as e:
            print(f"Error getting embedding from Ollama: {e}")
            # Fallback or re-raise based on your error handling strategy
            raise

    async def ensure_collection_exists(self, collection_name: str, vector_size: int = 768):
        """Ensures the Qdrant collection exists or creates it."""
        # Use a try-except block to handle potential issues with collection existence check
        try:
            collections = await self.client.get_collections()
            if collection_name not in [c.name for c in collections.collections]:
                print(f"Collection '{collection_name}' not found. Creating...")
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                )
                print(f"Collection '{collection_name}' created.")
            else:
                print(f"Collection '{collection_name}' already exists.")
        except Exception as e:
            print(f"Error checking/creating collection '{collection_name}': {e}")
            # Depending on severity, you might want to re-raise or try to recover

    async def add_system_instruction(self, collection_name: str, instruction_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Adds a system instruction to a Qdrant collection.
        Generates embedding for the content.
        """
        try:
            embedding = await self._get_embedding(content)
            points = [
                models.PointStruct(
                    id=instruction_id,
                    vector=embedding,
                    payload={"content": content, **(metadata if metadata else {})}
                )
            ]
            response = await self.client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points
            )
            print(f"Added instruction '{instruction_id}' to '{collection_name}'. Status: {response.status}")
            return response.status == models.UpdateStatus.COMPLETED
        except Exception as e:
            print(f"Failed to add instruction to Qdrant: {e}")
            return False

    async def get_system_instruction(self, collection_name: str, instruction_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a system instruction by its ID."""
        try:
            point = await self.client.retrieve(
                collection_name=collection_name,
                ids=[instruction_id],
                with_payload=True,
                with_vectors=False
            )
            return point[0].payload if point else None
        except Exception as e:
            print(f"Failed to retrieve instruction from Qdrant: {e}")
            return None

    async def search_system_instructions(self, collection_name: str, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for system instructions semantically similar to the query.
        """
        try:
            query_embedding = await self._get_embedding(query_text)
            search_result = await self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True
            )
            return [hit.payload for hit in search_result]
        except Exception as e:
            print(f"Failed to search instructions in Qdrant: {e}")
            return []

    async def delete_system_instruction(self, collection_name: str, instruction_id: str) -> bool:
        """Deletes a system instruction by its ID."""
        try:
            response = await self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=[instruction_id])
            )
            print(f"Deleted instruction '{instruction_id}' from '{collection_name}'. Status: {response.status}")
            return response.status == models.UpdateStatus.COMPLETED
        except Exception as e:
            print(f"Failed to delete instruction from Qdrant: {e}")
            return False

    async def list_collections(self) -> List[str]:
        """Lists all collections in Qdrant."""
        try:
            collections = await self.client.get_collections()
            return [c.name for c in collections.collections]
        except Exception as e:
            print(f"Failed to list collections from Qdrant: {e}")
            return []

    async def list_instructions_in_collection(self, collection_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Lists all instructions (payloads) in a given collection."""
        try:
            scroll_result, _ = await self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                offset=0 # Start from the beginning
            )
            return [point.payload for point in scroll_result]
        except Exception as e:
            print(f"Failed to list instructions from collection '{collection_name}': {e}")
            return []