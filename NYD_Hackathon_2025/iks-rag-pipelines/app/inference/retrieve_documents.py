"""import os
import torch
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer
from qdrant_client import QdrantClient

load_dotenv()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-distilbert-cos-v1")
model = AutoModel.from_pretrained("sentence-transformers/multi-qa-distilbert-cos-v1")

def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().tolist()

# Connect to Qdrant
qdrant_client = QdrantClient(
    url="https://bbe512e4-6b6e-475e-bfb5-fe04f5797900.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=os.environ.get('QDRANT_API_KEY')
)

def retrieve_context(query, collection_name):
    query = query.lower()
    query_embedding = encode_text(query)
    
    response = qdrant_client.query_points(collection_name=collection_name, query=query_embedding, limit=15)
    points = response.points  
    translations = [point.payload['translation'] for point in points]
    return " ".join(translations)"""