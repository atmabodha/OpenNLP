from groq import Groq
import os
import re

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def remove_think_tokens(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def rewrite_query_for_rag(query=""):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that rewrites user queries to improve retrieval for a RAG system. "
                    "Your task is to reformat the input query into the following format: <query><new key words related to the query>. "
                    "The new keywords should be relevant to the original query and should help in retrieving more accurate information. "
                    "Additionally, ensure that the reformulated query is structured in a way that makes it easier to classify "
                    "whether the query is related to the Bhagavad Gita or the Patanjali Yoga Sutras. "
                    "If the query is about Bhagavad Gita, emphasize terms like 'Krishna', 'dharma', 'karma', 'atma', or 'bhakti'. "
                    "If the query is about Patanjali Yoga Sutras, emphasize terms like 'yoga', 'samadhi', 'sutras', 'ashtanga', 'chitta', or 'nirvana'. "
                    "For example, if the query is 'what is dharma in Indian philosophy?', the output should be: "
                    "<What is dharma in Indian philosophy?><dharma, Indian philosophy, Bhagavad Gita, Krishna, karma>. "
                    "Ensure the rewritten query is concise and the keywords are highly relevant."
                ),
            },
            {
                "role": "user",
                "content": f"Rewrite the following query for better retrieval: {query} i only want the reforved query output in the format i have given to you i want no output else in it at all only the format i have given not even here is the response",
            },
        ],
        model="llama3-8b-8192",
        max_tokens=500,
    )
    return remove_think_tokens(chat_completion.choices[0].message.content)



