from pydantic import BaseModel
from typing import List
from api.helper.chat_completion import Reference

class RegenerateRequest(BaseModel):
    query: str
    previous_response: str
    feedback: str

class RegenerateResponse(BaseModel):
    new_response: str
    references: List[Reference]
