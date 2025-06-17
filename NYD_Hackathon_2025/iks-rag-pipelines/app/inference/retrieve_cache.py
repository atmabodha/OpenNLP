import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from together import Together


load_dotenv()

client = Together(api_key=os.getenv('TOGETHER_API_KEY'))
def encode_text(query):
    response = client.embeddings.create(
    model="BAAI/bge-base-en-v1.5",
    input=query.lower()
    )
    t = response.data[0].embedding
    return list(t)

def mock_response():
    """Returns a default response when no relevant answer is found."""
    return {
        "summary_answer": "**Thank you for your question!** We will get back to you soon with the answer.",
        "detailed_answer": (
            "  In the meantime, you can explore related topics such as:\n\n"
            "* **The significance of karma** in the Bhagavad Gita\n"
            "* **The four paths of yoga** described in the Gita\n"
            "* **The first sutra** of Patanjali Yoga Sutras and its meaning\n"
            "* Patanjali's concept of **'Chitta Vritti Nirodha'**\n\n"
            "_Stay tuned — we'll provide your answer shortly!_"
        ),
        "references": [
            {
                "source": "No Source",
                "chapter": "1",
                "verse": "1",
                "text": "No relevant verse available for this query."
            }
        ]
    }

# Connect to Qdrant
qdrant_client = QdrantClient(
    url="https://bbe512e4-6b6e-475e-bfb5-fe04f5797900.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=os.environ.get('QDRANT_API_KEY')
)

def retrieve_context_cache(query, collection_name="QnA_collection"):
    query = query.lower()
    query_embedding = encode_text(query)
    
    response = qdrant_client.query_points(collection_name=collection_name, query=query_embedding, limit=1)
    points = response.points
    
    if not points:
        return mock_response()

    score = points[0].score
    print(points[0].id)
    print(score)

    if score > 0.27:
        return points[0].payload['answer']
    else:
        return mock_response()