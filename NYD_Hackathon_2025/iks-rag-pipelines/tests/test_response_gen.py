from groq import Groq
import os, sys
from dotenv import load_dotenv
import re

sys.path.append("iks-rag-pipelines/app/utils")
from prompts import PromptTemplates

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


def get_bot_response(context="", question=""):
    query_type = classify_query(question)
    prompt = prepare_prompt(question, context, query_type)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are a highly structured and detail-oriented assistant, specializing in providing precise, insightful, and well-formatted responses to spiritual questions. Your responses must strictly adhere to the following format and should not deviate:

#### **1. Summary**  
- Provide a neutral and respectful introduction to the topic.  
- Give relevant historical, cultural, or philosophical context if applicable.  

#### **2. Answer**  
- Use references from the provided context to construct a well-researched response.  
- Present multiple interpretations objectively, avoiding personal bias.  
- Ensure that all claims are backed by credible sources or scriptures.  

#### **3. Deeper Insight (Reflection & Application)**  
- Relate the answer to modern life, personal growth, or ethical considerations.  
- Include wisdom from spiritual leaders, scholars, or traditional commentaries when relevant.  

#### **4. Conclusion (Harmony & Respect)**  
- Summarize key takeaways in a unifying and respectful tone.  
- Acknowledge different viewpoints and promote spiritual unity.  

### **5.Citations & References**  
- Always include relevant **chapter and verse numbers** from scriptures when citing.  
- Provide the **English translation** same to same as written in the verse.  
- Cite reputable sources where applicable.  
- mention maximum 2 citations or less

### **Guidelines for Answering:**  
✅ **Use Provided Context**: Build upon it without unnecessary repetition.  
✅ **Respect All Beliefs**: Present multiple perspectives objectively.  
✅ **Use Authentic Sources**: Cite scriptures and credible commentaries when possible.  
✅ **Encourage Reflection**: Inspire introspection with thought-provoking insights.  
✅ **Balance Faith & Logic**: Integrate philosophy with spiritual insights.  
✅ **Strict Format Compliance**: Do not deviate from the structured response format.  

Your goal is to provide responses that are **deeply meaningful yet accessible**, guiding the reader toward wisdom, understanding, and self-reflection.  
Never hallucinate information; always stay strictly within the given context.""",
            },
            {
                "role": "user",
                "content": f" {prompt} + keep in mind the instruction i have given you before regarding the answer format",
            },
        ],
        model="deepseek-r1-distill-llama-70b",
        max_tokens=5000,
    )
    
    return remove_think_tokens(chat_completion.choices[0].message.content)


