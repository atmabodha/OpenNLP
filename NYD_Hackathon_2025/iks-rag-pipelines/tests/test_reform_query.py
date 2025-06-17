from groq import Groq
import os
import re
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def remove_think_tokens(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def rewrite_query_for_rag(context=""):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that rewrites user queries to improve retrieval for a RAG system. "
                    "Your task is to reformat the input query into the following format: <query><new key words related to the query>. "
                    "The new keywords should be relevant to the original query and should help in retrieving more accurate information. "
                    "Focus on extracting key concepts, entities, or themes from the query. "
                    "For example, if the query is about 'the concept of dharma in the Bhagavad Gita,' the output could be: "
                    "<What is the concept of dharma in the Bhagavad Gita?><dharma, Bhagavad Gita, concept, philosophy>. "
                    "Ensure the rewritten query is concise and the keywords are highly relevant."
                ),
            },
            {
                "role": "user",
                "content": f"Rewrite the following query for better retrieval: {context}",
            },
        ],
        model="deepseek-r1-distill-llama-70b",
        max_tokens=500,
    )
    return  remove_think_tokens(chat_completion.choices[0].message.content)


query = input("Enter Query :")
print(rewrite_query_for_rag(query=query))
