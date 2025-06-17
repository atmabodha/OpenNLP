from groq import Groq
import os
from better_profanity import profanity
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))


def check_valid_answer(q="", a="", c=" "):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict text classifier that evaluates answers for correctness. "
                    "Your response must be strictly '1' or '0'. "
                    "Output '1' if and only if the answer is: "
                    "1. Grammatically correct. "
                    "2. Directly relevant to the given question. "
                    "3. Strictly derived from the provided context, with no additional information beyond what is given. "
                    "4. Free from hallucinations or fabricated details. "
                    "5. Not misleading or speculative in any way. "
                    "If the answer violates any of these rules—such as introducing external knowledge, misrepresenting facts, "
                    "or deviating from the provided context—output '0'. "
                    "Your response must only be '1' or '0', with no explanation or additional text."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {q}\nAnswer: {a}\nGiven Context: {c}\nClassify the response:",
            },
        ],
        model="llama3-8b-8192",
        max_tokens=3,
    )
    return chat_completion.choices[0].message.content

