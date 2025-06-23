from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Literal, List

# --- Modelos específicos para fastapi_mcp (Mensajes/Comandos) ---

class Message(BaseModel):
    """Base model for all messages processed by MCP."""
    id: str = Field(..., description="Unique ID for the message/command")
    type: str = Field(..., description="Type of the message/command (e.g., 'add_instruction', 'get_instruction')")
    payload: Dict[str, Any] = Field({}, description="Data payload for the message")

class AddInstructionPayload(BaseModel):
    collection_name: str
    instruction_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class GetInstructionPayload(BaseModel):
    collection_name: str
    instruction_id: str

class SearchInstructionsPayload(BaseModel):
    collection_name: str
    query: str
    limit: int = 5

class DeleteInstructionPayload(BaseModel):
    collection_name: str
    instruction_id: str

class ListInstructionsPayload(BaseModel):
    collection_name: str
    limit: int = 100

class ListCollectionsPayload(BaseModel):
    pass # No specific payload needed

# --- Modelos de respuesta ---

class SystemInstructionResponse(BaseModel):
    instruction_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class StatusResponse(BaseModel):
    status: Literal["success", "failure"]
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None # Generic data field for responses


