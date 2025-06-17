from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

# Load the embedding model
model = SentenceTransformer("sentence-transformers/multi-qa-distilbert-cos-v1")

# Connect to Qdrant
qdrant_client = QdrantClient(
    url="https://bbe512e4-6b6e-475e-bfb5-fe04f5797900.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=os.environ.get('QDRANT_API_KEY')
)

def retrieve_context(query, collection_name):
    query = query.lower()
    query_embedding = model.encode(query).tolist()
    
    response = qdrant_client.query_points(collection_name=collection_name, query=query_embedding, limit=15)
    points = response.points  
    
    translations = [point.payload['translation'] for point in points]
    return " ".join(translations)


print(retrieve_context(query = "test", collection_name = "test"))