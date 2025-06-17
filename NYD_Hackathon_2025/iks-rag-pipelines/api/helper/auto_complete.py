from pydantic import BaseModel
from typing import List

# Request model
class AutocompleteRequest(BaseModel):
    partial_query: str

# Response model
class AutocompleteResponse(BaseModel):
    suggestions: List[str]

