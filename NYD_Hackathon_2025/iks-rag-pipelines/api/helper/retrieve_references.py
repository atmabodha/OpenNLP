from pydantic import BaseModel
from typing import List, Optional

class ReferenceDetail(BaseModel):
    text: str
    source: str
    link: str

class RetrieveReferencesRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class RetrieveReferencesResponse(BaseModel):
    references: List[ReferenceDetail]
