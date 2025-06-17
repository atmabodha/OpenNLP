from pydantic import BaseModel
from typing import List, Optional

class Reference(BaseModel):
    source: str
    link: str

class ChatCompletionRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class ChatCompletionResponse(BaseModel):
    response: str
    references: List[Reference]
    confidence: float

class CompletionResponse(BaseModel):
    response: str
