from groq import Groq
import os, sys
from dotenv import load_dotenv
import re

sys.path.append('..')
from app.utils.prompts import PromptTemplates
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---- Query classification based on keywords ----
def classify_query(query):
    query_lower = query.lower().strip()

    categories = {
        "philosophical": [
            "meaning",
            "truth",
            "nature",
            "reality",
            "consciousness",
            "soul",
            "dharma",
            "karma",
            "existence",
            "purpose",
            "free will",
            "illusion",
            "self",
            "ego",
            "mind",
            "maya",
            "moksha",
            "non-duality",
            "eternal",
        ],
        "practical": [
            "how to",
            "what should",
            "guide",
            "help",
            "advice",
            "practice",
            "technique",
            "method",
            "way",
            "steps",
            "daily",
            "routine",
            "apply",
            "habit",
            "improve",
            "develop",
        ],
        "comparative": [
            "difference",
            "compare",
            "versus",
            "vs",
            "contrast",
            "relation",
            "similarities",
            "distinction",
            "connection",
        ],
        "storytelling": [
            "story",
            "example",
            "parable",
            "analogy",
            "illustration",
            "narrative",
        ],
        "meditation": [
            "meditation",
            "reflect",
            "contemplate",
            "focus",
            "inner peace",
            "mindfulness",
            "self-awareness",
            "visualization",
        ],
    }

    for category, keywords in categories.items():
        if any(kw in query_lower for kw in keywords):
            return category

    if len(query_lower.split()) < 3 or query_lower in ["?", "explain", "clarify"]:
        return "clarification"

    return "default"


# ---- Prompt construction ----
def prepare_prompt(query, verses, query_type):
    template = getattr(PromptTemplates, query_type, PromptTemplates.default)
    return template.format(query=query, verses=verses)


def remove_think_tokens(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_bot_response(context="", question="",collection= ""):
    query_type = classify_query(question)
    prompt = prepare_prompt(question, context, query_type)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are a highly structured and detail-oriented assistant, specializing in providing precise, insightful, and well-formatted responses to spiritual questions. Your responses must strictly adhere to the following format and should not deviate:

{
    "summary_answer": "Provide a neutral and respectful introduction to the topic, including relevant historical, cultural, or philosophical from given context context make sure its written in markdown language so that it looks better",
    
    "detailed_answer": "A well-researched and objective response presenting multiple interpretations, ensuring all claims are backed by credible sources or scriptures from the given context make sure its written in markdown language so that it looks better",
    
   "references": [
        {
            "source": ,
            "chapter": "Chapter number(if its mentioned in the question what chapter to use use that only(striclty use the chapter number mentioned in the question if any is mentioned))",
            "verse": "Verse number((if its mentioned in the question what verse to use use that only(striclty use the verse number mentioned in the question if any is mentioned)))",
            "text": "Extract the verse from the context"
        }
    ]
}

Your goal is to provide responses that are **deeply meaningful yet accessible**, guiding the reader toward wisdom, understanding, and self-reflection.  
Never hallucinate information; always stay strictly within the given context.""",
            },
            {
                "role": "user",
                "content": f" {prompt} + keep in mind the instruction I have given you before regarding the answer format. I want the response in the dictionary format I have given to you. The output should be structured as: {{'summary_answer': '', 'detailed_answer': '', 'references': []}} write summary answer and detailed answer in markdown format make sure it is written in markdown language using bold and everything to make it look better , source : {collection} - only mention this source in the response only stirictly give json output that can be used give proper json output nothing else dont use '/n' in output give maximum 2 references",
            },
        ],
        model="deepseek-r1-distill-llama-70b",
        max_tokens=5000,
    )
    
    return remove_think_tokens(chat_completion.choices[0].message.content)



