from pydantic import BaseModel
from typing import List

class QueryPredictionRequest(BaseModel):
    query: str

class QueryPredictionResponse(BaseModel):
    predicted_query: str
    confidence: float