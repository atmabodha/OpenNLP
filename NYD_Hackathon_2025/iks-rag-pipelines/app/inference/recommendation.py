import pandas as pd
import numpy as np
import sys
from collections import Counter
import os
from dotenv import load_dotenv
from together import Together
from qdrant_client import QdrantClient

sys.path.append(".")
load_dotenv()

client = Together(api_key=os.getenv('TOGETHER_API_KEY'))
def encode_text(query):
    response = client.embeddings.create(
    model="BAAI/bge-base-en-v1.5",
    input=query.lower()
    )
    t = response.data[0].embedding
    return list(t)

qdrant_client = QdrantClient(
    url="https://bbe512e4-6b6e-475e-bfb5-fe04f5797900.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=os.environ.get('QDRANT_API_KEY')
)
df = pd.read_csv("app/dataset/c_d.csv")
cluster_labels = df['Cluster_Label'].to_numpy()
questions = df['question'].to_numpy()

def keyword_search(query):
    query_words = set(query.lower().split())
    
    match_counts = []
    for question in questions:
        question_words = set(question.lower().split())
        match_count = len(query_words & question_words)  
        match_counts.append((question, match_count))
    
    sorted_questions = sorted(match_counts, key=lambda x: x[1], reverse=True)

    top_questions = [q[0] for q in sorted_questions[:5] if q[1] > 0]

    return {"suggestions": top_questions if top_questions else []}

def get_recommended_questions(query):
    if len(query.split(" ")) > 4:
        query_embedding = encode_text(query)

        random_questions = retrieve_questions_cache(query_embedding)

        return {"suggestions": random_questions}
    
    else:
        return keyword_search(query) 

def retrieve_questions_cache(query_embedding, collection_name="QnA_collection", top_k=5):
    response = qdrant_client.query_points(collection_name=collection_name, query=query_embedding, limit=top_k)
    points = response.points
    top_questions = []
    for point in points:
        top_questions.append(point.payload['Question'])
    return top_questions
