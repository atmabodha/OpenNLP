import time
import os
import random
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from dotenv import load_dotenv
from app.inference.pipeline import pipeline_rag

load_dotenv()

def load_data(csv_path, collection_name, start_idx=0, end_idx=None, max_retries=3):
    try:
        # Load CSV data
        data = pd.read_csv(csv_path)
        if end_idx is None or end_idx > len(data):
            end_idx = len(data)
        
        # Initialize Qdrant client
        qdrant_client = QdrantClient(
            url="https://bbe512e4-6b6e-475e-bfb5-fe04f5797900.europe-west3-0.gcp.cloud.qdrant.io:6333", 
            api_key=os.environ.get('QDRANT_API_KEY'),
        )
        
        collections = qdrant_client.get_collections()
        existing_collections = [col.name for col in collections.collections]

        if collection_name not in existing_collections:
            print(f"Collection '{collection_name}' does not exist. Creating new collection...")
            qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
        else:
            print(f"Collection '{collection_name}' already exists. Using existing collection.")

        # Change this according to need can be used with together api
        model_name = "sentence-transformers/multi-qa-distilbert-cos-v1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        def encode_text(text):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0, :].squeeze().tolist()

        for i, idx in enumerate(range(start_idx, end_idx)):
            retries = 0
            while retries < max_retries:
                try:
                    question_text = data['question'][idx]
                    sentence_embedding = encode_text(question_text)
                    answer = pipeline_rag(question_text)
                    
                    # Upsert data into Qdrant
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=[
                            {
                                "id": idx,
                                "vector": sentence_embedding,
                                "payload": {"Question": question_text, "answer": answer}
                            }
                        ]
                    )
                    print(f"Data loaded successfully for question {idx}")
                    break  # Success, exit retry loop
                except Exception as e:
                    retries += 1
                    wait_time = 2 ** retries + random.uniform(0, 1)  # Exponential backoff
                    print(f"Error loading data for question {idx}: {e}. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
            else:
                print(f"Failed to load data for question {idx} after {max_retries} retries.")
    
        print("Data loading completed.")
    except Exception as e:
        print(f"Critical error: {e}")

# Example usage
load_data("/Users/arunkaul/Desktop/yogavivekchatbot/iks-rag-pipelines/app/dataset/c_d.csv", "QnA_collection", start_idx=1682, end_idx=1684)
