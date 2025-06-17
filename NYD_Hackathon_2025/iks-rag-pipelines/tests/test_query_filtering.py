from groq import Groq
import os
from better_profanity import profanity
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def check_offensive_language(text):
    profanity.load_censor_words()
    if profanity.contains_profanity(text):
        return 1
    else:
        return 0

def check_valid(context=""):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an advanced text classifier for a Bhagavad Gita and Yoga Sutras chatbot. "
                    "Your task is to determine if a given input is relevant to these scriptures and should be processed by the chatbot.\n\n"

                    "Strictly classify the input as either:\n"
                    "1️⃣ **Output '1' (Relevant)** if the sentence is directly or indirectly related to the Bhagavad Gita or Yoga Sutras. This includes:\n"
                    "   - Genuine **questions seeking enlightenment, wisdom, or self-realization**.\n"
                    "   - Queries about **dharma (righteous duty), karma (action and consequence), moksha (liberation), atman (soul), bhakti (devotion), or jnana (knowledge)**.\n"
                    "   - Requests for **explanations of verses, teachings, or philosophical insights** from these texts.\n"
                    "   - Questions on **how to apply Bhagavad Gita or Yoga Sutra principles in daily life**.\n"
                    "   - Inquiries about **meditation, mind control, detachment, or self-discipline** as taught in these scriptures.\n"
                    "   - Comparisons between **Bhagavad Gita and other spiritual or philosophical traditions**, as long as the focus remains on its teachings.\n\n"

                    "2️⃣ **Output '0' (Irrelevant)** if the sentence is unrelated, offensive, or inappropriate. This includes:\n"
                    "   - Any **foul language, offensive words, or disrespectful content**.\n"
                    "   - Topics unrelated to Bhagavad Gita, Yoga Sutras, or their philosophies (e.g., general fitness, politics, pop culture).\n"
                    "   - Generic discussions on yoga that only focus on physical exercise (asanas) without any philosophical or spiritual depth.\n"
                    "   - Motivational or religious statements that do not directly relate to Bhagavad Gita or Yoga Sutras.\n"
                    "   - Casual greetings, small talk, or unrelated chit-chat (e.g., 'How’s your day?', 'What’s the weather like?').\n\n"

                    "⚠️ **Strictly follow the classification format:** Your response must be either **'0'** or **'1'**, with no additional text."
                ),
            },
            {
                "role": "user",
                "content": f"'{context}' - Classify the sentence as 0 or 1:",
            },
        ],
        model="llama3-8b-8192",
        max_tokens=3,
    )
    return chat_completion.choices[0].message.content


query = input("Enter Query :")
if(check_offensive_language(query) == False):
    print(check_valid(query))
else:
    print("Retry")