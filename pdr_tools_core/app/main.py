from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware # <--- ADD THIS LINE
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from qdrant_manager import QdrantManager
from models import (
    AddInstructionPayload, GetInstructionPayload, SearchInstructionsPayload,
    DeleteInstructionPayload, ListInstructionsPayload, ListCollectionsPayload,
    StatusResponse, SystemInstructionResponse
)
import os
from typing import List, Dict, Any, Optional
# from fastapi_mcp import FastApiMCP


app = FastAPI(title="MCP Server (Management Control Point) - RESTful", version="0.1.0")
# mcp = FastApiMCP(app)
# mcp.mount()
qdrant_manager = QdrantManager()
# --- ADD CORS MIDDLEWARE ---
origins = [
    "http://localhost",
    "http://localhost:3000", # The origin where Open WebUI frontend is running
    "http://127.0.0.1:3000", # Common alternative for localhost
    # You might also need to add the internal Docker network address for debugging,
    # but the browser CORS error is about the *client's* origin.
    # For production, you would list your actual Open WebUI domain here.
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"], # Allows all headers
)
# Define a default collection name or get from env
DEFAULT_COLLECTION = os.getenv("DEFAULT_QDRANT_COLLECTION", "system_instructions")

@app.on_event("startup")
async def startup_event():
    """Ensures default Qdrant collection exists on startup."""
    print(f"Starting up RESTful MCP Server. Ensuring default collection '{DEFAULT_COLLECTION}' exists...")
    # For 'nomic-embed-text', vector size is 768. Adjust if using a different embedding model.
    await qdrant_manager.ensure_collection_exists(DEFAULT_COLLECTION, vector_size=768)
    print("RESTful MCP Server startup complete.")

# --- RESTful Endpoints ---

@app.post("/instructions",
 response_model=StatusResponse, 
 status_code=status.HTTP_201_CREATED, 
 tags=["Instructions"],
 operation_id="create_instruction")
async def add_instruction(payload: AddInstructionPayload):
    """Adds a new system instruction to a Qdrant collection."""
    try:
        success = await qdrant_manager.add_system_instruction(
            payload.collection_name,
            payload.instruction_id,
            payload.content,
            payload.metadata
        )
        if success:
            return StatusResponse(status="success", message=f"Instruction {payload.instruction_id} added successfully.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to add instruction.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error adding instruction: {e}")

@app.get("/instructions/{collection_name}/{instruction_id}", response_model=SystemInstructionResponse, tags=["Instructions"])
async def get_instruction(collection_name: str, instruction_id: str):
    """Retrieves a system instruction by its ID."""
    try:
        instruction = await qdrant_manager.get_system_instruction(
            collection_name, instruction_id
        )
        if instruction:
            # Ensure the instruction_id is always present in the response
            return SystemInstructionResponse(
                instruction_id=instruction_id,
                content=instruction.get("content", ""),
                metadata={k: v for k, v in instruction.items() if k != "content"} # Exclude content from direct metadata
            )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Instruction not found.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error retrieving instruction: {e}")


@app.post("/instructions/search", response_model=List[SystemInstructionResponse], tags=["Instructions"])
async def search_instructions(payload: SearchInstructionsPayload):
    """Searches for system instructions semantically similar to the query."""
    try:
        results = await qdrant_manager.search_system_instructions(
            payload.collection_name, payload.query, payload.limit
        )
        return [
            SystemInstructionResponse(
                instruction_id=r.get('id', 'N/A'),
                content=r.get('content', ''),
                metadata={k: v for k, v in r.items() if k not in ["id", "content"]}
            ) for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error searching instructions: {e}")

@app.delete("/instructions/{collection_name}/{instruction_id}", response_model=StatusResponse, tags=["Instructions"])
async def delete_instruction(collection_name: str, instruction_id: str):
    """Deletes a system instruction by its ID."""
    try:
        success = await qdrant_manager.delete_system_instruction(
            collection_name, instruction_id
        )
        if success:
            return StatusResponse(status="success", message=f"Instruction {instruction_id} deleted successfully.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete instruction.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error deleting instruction: {e}")

@app.get("/collections", response_model=List[str], tags=["Collections"], operation_id="get_collections_list")
async def list_collections():
    """Lists all collections in Qdrant."""
    try:
        collections = await qdrant_manager.list_collections()
        print("COLLECTIONS:",str(collections))
        return collections
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error listing collections: {e}")
    
@app.get("/mojones", response_model=List[str], tags=["mojones"], operation_id="get_mojones_list")
async def list_mojones():
    """Lists all mojones"""
    try:
        mojones = [{"pepe":"2"},{"manolo":"5"},{"perico":"4"},{"snake":"10"}]
        json_item_data = jsonable_encoder(mojones)
        print("COLLECTIONS:",json_item_data)
        return JSONResponse(content=json_item_data)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error listing collections: {e}")

@app.post("/collections/instructions/list", 
    response_model=List[SystemInstructionResponse], 
    tags=["Collections"],
    operation_id="get_instructions_list"
)
async def list_collection_instructions(payload: ListInstructionsPayload):
    """Lists all instructions (payloads) in a given collection."""
    try:
        instructions = await qdrant_manager.list_instructions_in_collection(
            payload.collection_name, payload.limit
        )
        return [
            SystemInstructionResponse(
                instruction_id=i.get('id', 'N/A'),
                content=i.get('content', ''),
                metadata={k: v for k, v in i.items() if k not in ["id", "content"]}
            ) for i in instructions
        ]
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error listing instructions in collection: {e}")

# You can still have regular FastAPI endpoints, e.g., a health check
@app.get("/", tags=["Health"])
async def read_root():
    return {"message": "MCP Server (FastAPI RESTful) is running!"}