from fastapi import FastAPI
from dotenv import load_dotenv
import sys
sys.path.append('..')
"""from app.inference.pipeline import pipeline_rag"""
from app.inference.cache_pipeline import pipeline_rag_cache
from app.inference.recommendation import get_recommended_questions
from pydantic import BaseModel
from typing import List

load_dotenv()
from api.helper.auto_complete import (
    AutocompleteRequest,
    AutocompleteResponse
)
from api.helper.query_prediction import (
    QueryPredictionRequest,
    QueryPredictionResponse
)
from api.helper.chat_completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionResponse,
    Reference
)
from api.helper.retrieve_references import (
    RetrieveReferencesRequest,
    RetrieveReferencesResponse,
    ReferenceDetail
)
from api.helper.regenerate_response import (
    RegenerateRequest,
    RegenerateResponse
)
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.yogavivek.org",
        "https://yogavivekchatbot-ui.onrender.com",
        "https://books-67622.web.app",
        "https://yogavivek-4haf.onrender.com",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AutocompleteRequest(BaseModel):
    text: str

class AutocompleteResponse(BaseModel):
    suggestions: List[str]

@app.post("/v1/autocomplete", response_model=AutocompleteResponse)
async def autocomplete(request: AutocompleteRequest):
    """
    Endpoint to provide query autocompletion suggestions.
    """
    print(f"Received query: {request.text}")
    suggestions = get_recommended_questions(request.text)
    print(f"Generated response: {suggestions}")
    
    # Make sure we're returning a list of strings
    if isinstance(suggestions, dict) and 'suggestions' in suggestions:
        suggestions_list = suggestions['suggestions']
    elif isinstance(suggestions, list):
        suggestions_list = suggestions
    else:
        suggestions_list = []
    
    return AutocompleteResponse(suggestions=suggestions_list)

@app.post("/v1/predict-query", response_model=QueryPredictionResponse)
async def predict_query(request: QueryPredictionRequest):
    """
    Endpoint to predict and complete user queries.
    """
    # This is a placeholder implementation
    
    # predicted = f"{request.query} Bhagavad Gita?"

    predicted = get_recommended_questions(request.query)
    print(f"Generated response: {predicted}")
    
    return QueryPredictionResponse(
        predicted_query=predicted['suggestions'][0],
        confidence=0.92
    )

@app.post("/v1/chat", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    """
    Endpoint to generate responses using RAG.
    """
    # This is a placeholder implementation
    dummy_response = "The Bhagavad Gita emphasizes self-discipline as essential for spiritual growth."
    dummy_references = [
        Reference(
            source="Bhagavad Gita, Chapter 6, Verse 5",
            link="https://example.com/gita/ch6v5"
        ),
        Reference(
            source="Bhagavad Gita, Chapter 2, Verse 47",
            link="https://example.com/gita/ch2v47"
        )
    ]
    
    return ChatCompletionResponse(
        response=dummy_response,
        references=dummy_references,
        confidence=0.97
    )

@app.post("/v1/retrieve-references", response_model=RetrieveReferencesResponse)
async def retrieve_references(request: RetrieveReferencesRequest):
    """
    Endpoint to retrieve relevant references based on a query.
    """
    # This is a placeholder implementation
    dummy_references = [
        ReferenceDetail(
            text="Karma yoga is the path of selfless action, without attachment to results.",
            source="Patanjali Yoga Sutras, Verse 2.1",
            link="https://example.com/yoga-sutras/v2-1"
        ),
        ReferenceDetail(
            text="The practice of karma yoga purifies the mind and leads to spiritual wisdom.",
            source="Patanjali Yoga Sutras, Verse 3.5",
            link="https://example.com/yoga-sutras/v3-5"
        )
    ]
    
    return RetrieveReferencesResponse(references=dummy_references)

@app.post("/v1/regenerate", response_model=RegenerateResponse)

async def regenerate_response(request: RegenerateRequest):
    """
    Endpoint to regenerate a response based on feedback.
    """
    # This is a placeholder implementation
    dummy_response = (
        "Detachment in Bhagavad Gita is about acting without attachment to results. "
        "It is discussed in Chapter 2, where Krishna advises Arjuna to focus on duty "
        "without being driven by rewards."
    )
    dummy_references = [
        Reference(
            source="Bhagavad Gita, Chapter 2, Verse 47",
            link="https://example.com/gita/ch2v47"
        )
    ]
    
    return RegenerateResponse(
        new_response=dummy_response,
        references=dummy_references
    )

"""class QueryRequest(BaseModel):
    query: str

@app.post("/v1/response")
async def generate_response(request: QueryRequest):
    print(f"Received query: {request.query}")
    new_response = pipeline_rag(request.query)
    print(f"Generated response: {new_response}")
    return {'response': new_response}"""


class CacheRequest(BaseModel):
    query: str

@app.post("/v1/response-cache")
async def generate_response_cache(request: CacheRequest):
    print(f"Received query: {request.query}")
    new_response = pipeline_rag_cache(request.query)
    print(f"Generated response: {new_response}")
    return {'response': new_response}

class RecommendationRequest(BaseModel):
    query: str

@app.post("/v1/recommendation")
async def generate_recommendation(request: RecommendationRequest):
    print(f"Received query: {request.query}")
    recommendation = get_recommended_questions(request.query)
    print(f"Generated response: {recommendation}")
    return recommendation

@app.get("/status")
async def status():
    """Check the status of the application."""
    #Implement the logic to check the status of the application, return current time, version number, etc. 
    return {
        "status": "OK",
        "uptime": "5 days, 3 hours",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Default route of the application."""
    return {"message": "Main Page"}


