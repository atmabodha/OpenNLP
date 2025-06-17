"""import time
from groq import Groq
import os
from better_profanity import profanity
from dotenv import load_dotenv
import sys
import json

sys.path.append('..')
from app.inference.query_filter import *
from app.inference.query_reform import rewrite_query_for_rag
from app.inference.find_correct_collection import *
from app.inference.retrieve_documents import *
from app.inference.response_gen import *
from app.inference.validation import check_valid_answer

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def extract_json_dict(s):
    start = s.find('{')
    end = s.rfind('}')
    
    if start == -1 or end == -1 or start > end:
        return None  # No valid JSON structure found
    
    json_str = s[start:end+1]
    
    try:
        return json.loads(json_str)  # Convert to dictionary
    except json.JSONDecodeError:
        return None  # Invalid JSON format

def get_json_response1():
    data = {
        "summary_answer": "#  Invalid Query\n   We can only answer questions related to **Bhagavad Gita** or **Patanjali Yoga Sutras**.",
        "detailed_answer": (
            "## Example Questions\n\n"
            "  You can try asking questions like:\n\n"
            "* What is the significance of karma in the **Bhagavad Gita**?\n"
            "* What are the four paths of yoga described in the **Gita**?\n" 
            "* What is the first sutra of **Patanjali Yoga Sutras** and its meaning?\n"
            "* How does Patanjali describe the concept of **'Chitta Vritti Nirodha'**?\n\n"
            "_Feel free to ask anything related to these scriptures!_"
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
    return data

def pipeline_rag(query, max_retries=3):
    start_time = time.time()
    query = query.lower()
    print(f"Step 1: Query converted to lowercase - {time.time() - start_time:.4f} sec")

    # Step 2: Offensive Language Check
    if check_offensive_language(query) == 1:
        print(f"Step 2: Offensive language detected - {time.time() - start_time:.4f} sec")
        return get_json_response1()
    print(f"Step 2: Offensive language check passed - {time.time() - start_time:.4f} sec")

    # Step 3: Query Validity Check
    is_valid = check_valid(query)
    if int(is_valid) == 0:
        print(f"Step 3: Query validity check failed - {time.time() - start_time:.4f} sec")
        return get_json_response1()
    print(f"Step 3: Query validity check passed - {time.time() - start_time:.4f} sec")

    retries = 0
    while retries < max_retries:
        query_reform = rewrite_query_for_rag(query).lower()
        print(f"Step 4: Query reformulated (Attempt {retries+1}) - {time.time() - start_time:.4f} sec")
        
        collection_name = get_best_match(query=query_reform)
        print(f"Step 5: Best-matching collection found - {time.time() - start_time:.4f} sec")

        collection = "Patanjali Yoga Sutras" if collection_name == "yoga_collection" else "Bhagavad Gita"
        print(f"Step 6: Using collection - {collection}")

        context = retrieve_context(query_reform, collection_name)
        print(f"Step 7: Context retrieved - {time.time() - start_time:.4f} sec")

        answer = get_bot_response(context, query, collection)
        print(f"Step 8: Response generated - {time.time() - start_time:.4f} sec")

        extracted_json = extract_json_dict(answer)
        if extracted_json:  # If valid JSON is extracted, validate it
            validation = check_valid_answer(q=query, a = extracted_json, c=context)
            print(f"Step 9: Validation check - {validation} - {time.time() - start_time:.4f} sec")
            if int(validation) == 1:
                return extracted_json  # Return validated answer

        print("Validation failed. Retrying query reformulation...")
        retries += 1

    print(f"Step 10: All retries failed, returning default response - {time.time() - start_time:.4f} sec")
    return get_json_response1()  # Return default response after max retries
"""