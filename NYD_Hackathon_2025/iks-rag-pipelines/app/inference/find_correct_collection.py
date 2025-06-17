"""import os
import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel

load_dotenv()

# Load the embedding model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-distilbert-cos-v1")
model = AutoModel.from_pretrained("sentence-transformers/multi-qa-distilbert-cos-v1")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# Connect to Qdrant
qdrant_client = QdrantClient(
    url="https://bbe512e4-6b6e-475e-bfb5-fe04f5797900.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=os.environ.get('QDRANT_API_KEY')
)

def get_best_match(query: str, collection1: str = "yoga_collection", collection2: str = "gita_collection", limit: int = 1):
    query_embedding = get_embedding(query)
    answers1 = qdrant_client.query_points(collection_name=collection1, query=query_embedding, limit=limit)
    answers2 = qdrant_client.query_points(collection_name=collection2, query=query_embedding, limit=limit)
    
    score1 = answers1.points[0].score if answers1.points else 0
    score2 = answers2.points[0].score if answers2.points else 0

    print(score1)
    print(score2)

    if score1 > 0.5 or score2 > 0.5 or ("gita" not in query.lower() and "yoga" not in query.lower()):
        best_collection = collection1 if score1 >= score2 else collection2
    else:
        best_collection = collection2 if query.lower().count("gita") > query.lower().count("yoga") else collection1

    return best_collection
"""