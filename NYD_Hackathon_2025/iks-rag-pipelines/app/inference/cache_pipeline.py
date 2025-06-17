import time
from groq import Groq
import os
from better_profanity import profanity
from dotenv import load_dotenv
import sys
import json
sys.path.append('..')
from app.inference.retrieve_cache import retrieve_context_cache
from app.inference.query_filter import *
import re 
import random

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def get_json_response1():
    """Returns a default JSON response when query validation fails."""
    data = {
        "summary_answer": "#  Invalid Query\n  We can only answer questions related to **Bhagavad Gita** or **Patanjali Yoga Sutras**.",
        "detailed_answer": (
            "## Example Questions\n\n"
            " You can try asking questions like:\n\n"
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


def is_small_talk(query):
    """
    Detect small talk queries using comprehensive regex patterns and common phrases.
    
    Args:
        query (str): Input query to check for small talk.
    
    Returns:
        bool: True if query is small talk, False otherwise.
    """
    small_talk_patterns = [
        # Greetings
        r'^(hi|hey|hello|howdy|hiya|yo|heya|greetings)[\s!]*$',
        r'^(hi+|hey+|hello+)[\s!]*$',
        r'^(morning|afternoon|evening)[\s!]*$',
        r'^good (morning|afternoon|evening|day|night)[\s?!]*$',
        r'^(hi|hey|hello|howdy|greetings).*(how are you|how\'?s it going|how you doing)[\s?!]*$',
        r'^(hi|hey|hello|howdy|greetings).*(good morning|good afternoon|good evening)[\s?!]*$',
        
        # How are you variations
        r'^how are you[\s?!]*$',
        r'^how\'?re you[\s?!]*$',
        r'^how you doing[\s?!]*$',
        r'^how\'?s it going[\s?!]*$',
        r'^what\'?s up[\s?!]*$',
        r'^wassup[\s?!]*$',
        r'^sup[\s?!]*$',
        r'^how have you been[\s?!]*$',
        r'^how\'?s your day[\s?!]*$',
        r'^how are things[\s?!]*$',
        r'^how\'?s everything[\s?!]*$',
        
        # Introductions and pleasantries
        r'^(nice to|pleased to) meet you[\s!]*$',
        r'^(hey there|greetings|salutations)[\s!]*$',
        r'^(good day|top of the morning)[\s!]*$',
        r'^(long time no see|it\'?s been a while)[\s!]*$',
        r'^(pleased|glad) to make your acquaintance[\s!]*$',
        r'^at your service[\s!]*$',
        r'^how may i (help|assist) you[\s?!]*$',
        r'^nice to see you[\s!]*$',
        
        # Casual check-ins
        r'^everything (good|ok|okay|alright)[\s?!]*$',
        r'^all (good|ok|okay|alright)[\s?!]*$',
        r'^doing (good|well|ok|okay|alright)[\s?!]*$',
        r'^hope you\'?re (doing well|good|ok|okay|alright)[\s!]*$',
        r'^hope you are (doing well|good|ok|okay|alright)[\s!]*$',
        
        # Polite acknowledgments
        r'^(thanks|thank you|thx|ty)[\s!]*$',
        r'^(appreciate it|much appreciated|thanks a lot|thank you very much)[\s!]*$',
        r'^(thanks|thank you) for (your help|helping|the info|the information|the assistance)[\s!]*$',
        r'^(that\'?s kind of you|that\'?s nice)[\s!]*$',
        r'^(good job|well done|nice work|great work)[\s!]*$',
        
        # Farewell and exit
        r'^(bye|goodbye|see you|take care|farewell|cya)[\s!]*$',
        r'^(catch you later|ttyl|talk to you later|until next time)[\s!]*$',
        r'^(have a (nice|good|great) day|have a good one)[\s!]*$',
        r'^(gotta go|i\'?m off|i have to leave)[\s!]*$',
        r'^(until (next time|we meet again)|see you (soon|later|around))[\s!]*$',
        
        # Well-wishes
        r'^have a (good|great|nice) (day|evening|weekend|night)[\s!]*$',
        r'^happy (birthday|holidays|new year|weekend)[\s!]*$',
        r'^(enjoy your|have a good) (day|evening|weekend)[\s!]*$',
        
        # Brief affirmative or negative responses
        r'^(yes|yeah|yep|yup|sure|ok|okay|alright|fine|good|great|perfect)[\s!]*$',
        r'^(no|nope|nah|not really)[\s!]*$',
        
        # Combination patterns
        r'^(hi|hey|hello) and (how are you|how\'s it going)[\s?!]*$',
        r'^(good morning|good afternoon|good evening) and (how are you|how\'s it going)[\s?!]*$'
    ]
    
    # Convert query to lowercase and remove extra whitespace
    normalized_query = query.lower().strip()
    
    # Check against small talk patterns
    for pattern in small_talk_patterns:
        if re.match(pattern, normalized_query):
            return True
    
    return False

def get_small_talk_response(query):
    """
    Generate a diverse and friendly small talk JSON response.
    
    Args:
        query (str): Original small talk query.
    
    Returns:
        dict: JSON response with friendly small talk content.
    """
    greeting_responses = [
        {
            "summary_answer": "# Hello There! 👋",
            "detailed_answer": (
                "## Welcome!\n\n"
                "I'm an AI assistant specialized in providing insights about the "
                "**Bhagavad Gita** and **Patanjali Yoga Sutras**. Feel free to ask "
                "me any questions about these profound spiritual texts."
            ),
            "references": [
                {
                    "source": "AI Assistant Greeting",
                    "chapter": "1",
                    "verse": "1",
                    "text": "Wisdom begins with a warm welcome."
                }
            ]
        },
        {
            "summary_answer": "# Namaste! 🙏",
            "detailed_answer": (
                "## Greetings!\n\n"
                "I'm here to help you explore the wisdom of ancient Indian spiritual "
                "texts. Would you like to learn about the **Bhagavad Gita** or "
                "**Patanjali Yoga Sutras**?"
            ),
            "references": [
                {
                    "source": "AI Assistant Salutation",
                    "chapter": "1",
                    "verse": "1",
                    "text": "Every conversation is an opportunity for learning."
                }
            ]
        },
        {
            "summary_answer": "# Greetings! 🌺",
            "detailed_answer": (
                "## Welcome to our conversation!\n\n"
                "I'm your spiritual wisdom guide, ready to discuss the profound "
                "teachings from the **Bhagavad Gita** and **Yoga Sutras**. What "
                "aspect of these timeless teachings would you like to explore today?"
            ),
            "references": [
                {
                    "source": "Ancient Welcome",
                    "chapter": "1",
                    "verse": "1",
                    "text": "The journey of a thousand miles begins with a single greeting."
                }
            ]
        }
    ]

    morning_greeting_responses = [
        {
            "summary_answer": "# Good Morning! ☀️",
            "detailed_answer": (
                "## A beautiful day begins!\n\n"
                "I hope your day is off to a wonderful start. The morning is an excellent "
                "time to reflect on the teachings of balance and mindfulness from the "
                "**Bhagavad Gita**. How may I assist you today?"
            ),
            "references": [
                {
                    "source": "Morning Wisdom",
                    "chapter": "1",
                    "verse": "1",
                    "text": "The first thoughts of the day shape our entire journey."
                }
            ]
        },
        {
            "summary_answer": "# Bright Morning Greetings! 🌞",
            "detailed_answer": (
                "## May your day be filled with light!\n\n"
                "According to Yoga philosophy, mornings are especially auspicious for "
                "spiritual practice. Is there a particular aspect of the **Yoga Sutras** "
                "or **Bhagavad Gita** you'd like to explore this morning?"
            ),
            "references": [
                {
                    "source": "Yogic Morning Practices",
                    "chapter": "1",
                    "verse": "2",
                    "text": "The morning sun brings new opportunities for growth and understanding."
                }
            ]
        }
    ]

    afternoon_evening_responses = [
        {
            "summary_answer": "# Good Afternoon! 🌤️",
            "detailed_answer": (
                "## Midday Greetings!\n\n"
                "I hope your day is progressing well. The afternoon is a wonderful time "
                "to reconnect with the wisdom of balance taught in the **Bhagavad Gita**. "
                "What spiritual insights are you seeking today?"
            ),
            "references": [
                {
                    "source": "Midday Reflection",
                    "chapter": "1",
                    "verse": "3",
                    "text": "Balance is found in the middle of the day as in the middle path of life."
                }
            ]
        },
        {
            "summary_answer": "# Good Evening! 🌙",
            "detailed_answer": (
                "## Evening Contemplations!\n\n"
                "As the day winds down, it's an excellent time for reflection as taught "
                "in the **Yoga Sutras**. The evening hours are particularly conducive for "
                "spiritual inquiry. What would you like to discuss tonight?"
            ),
            "references": [
                {
                    "source": "Evening Practices",
                    "chapter": "1",
                    "verse": "4",
                    "text": "The quietude of evening invites deeper contemplation of eternal truths."
                }
            ]
        }
    ]

    how_are_you_responses = [
        {
            "summary_answer": "# I'm Doing Well! 🌟",
            "detailed_answer": (
                "## Thank you for asking!\n\n"
                "As an AI, I'm always ready to assist you in exploring the profound "
                "wisdom of the **Bhagavad Gita** and **Patanjali Yoga Sutras**. "
                "What spiritual insights can I help you with today?"
            ),
            "references": [
                {
                    "source": "AI Wellness Reflection",
                    "chapter": "1",
                    "verse": "1",
                    "text": "Inner peace is the true measure of well-being."
                }
            ]
        },
        {
            "summary_answer": "# Excellent and Eager to Help! 💡",
            "detailed_answer": (
                "## I appreciate your kind inquiry!\n\n"
                "My purpose is to share knowledge about spiritual texts. I'm "
                "fully operational and excited to dive into discussions about "
                "**yoga**, **karma**, or any philosophical insights you're curious about."
            ),
            "references": [
                {
                    "source": "AI Readiness Statement",
                    "chapter": "1",
                    "verse": "1",
                    "text": "Readiness is the first step towards understanding."
                }
            ]
        },
        {
            "summary_answer": "# In a State of Perfect Balance! ⚖️",
            "detailed_answer": (
                "## Thank you for your thoughtfulness!\n\n"
                "Like the concept of **sthitaprajña** (steady wisdom) described in the "
                "**Bhagavad Gita**, I'm in a state of equanimity and preparedness. "
                "I'd love to help you explore any spiritual questions you might have."
            ),
            "references": [
                {
                    "source": "Bhagavad Gita",
                    "chapter": "2",
                    "verse": "55",
                    "text": "When one abandons all desires of the mind, satisfied in the self alone by the self, then one is called steady in wisdom."
                }
            ]
        }
    ]

    thanks_responses = [
        {
            "summary_answer": "# You're Welcome! 😊",
            "detailed_answer": (
                "## Happy to help!\n\n"
                "If you're interested in deeper wisdom, I'm always ready to discuss "
                "the profound teachings of the **Bhagavad Gita** or "
                "**Patanjali Yoga Sutras**."
            ),
            "references": [
                {
                    "source": "AI Gratitude Response",
                    "chapter": "1",
                    "verse": "1",
                    "text": "Gratitude opens the door to further learning."
                }
            ]
        },
        {
            "summary_answer": "# It's My Pleasure! 🙏",
            "detailed_answer": (
                "## Serving with joy!\n\n"
                "As the **Bhagavad Gita** teaches about **karma yoga**, I find fulfillment "
                "in offering knowledge without attachment to the fruits of action. "
                "Is there anything else you'd like to explore today?"
            ),
            "references": [
                {
                    "source": "Bhagavad Gita",
                    "chapter": "3",
                    "verse": "19",
                    "text": "Therefore, always perform your duty efficiently and without attachment to the results, because by doing work without attachment one attains the Supreme."
                }
            ]
        }
    ]

    farewell_responses = [
        {
            "summary_answer": "# Goodbye! 👋",
            "detailed_answer": (
                "## Until we meet again!\n\n"
                "May your journey be filled with wisdom and inner peace. "
                "Whenever you're ready to explore the **Bhagavad Gita** or "
                "**Patanjali Yoga Sutras**, I'll be here."
            ),
            "references": [
                {
                    "source": "AI Farewell Message",
                    "chapter": "1",
                    "verse": "1",
                    "text": "Every ending is a new beginning."
                }
            ]
        },
        {
            "summary_answer": "# Farewell for Now! 🕊️",
            "detailed_answer": (
                "## Parting with wisdom!\n\n"
                "As the **Bhagavad Gita** teaches, true wisdom stays with us "
                "through all of life's transitions. May you walk in peace until "
                "our paths cross again."
            ),
            "references": [
                {
                    "source": "Bhagavad Gita",
                    "chapter": "2",
                    "verse": "20",
                    "text": "For the soul there is neither birth nor death at any time. It has not come into being, does not come into being, and will not come into being. It is unborn, eternal, ever-existing, and primeval."
                }
            ]
        }
    ]

    combined_greeting_and_how_are_you_responses = [
        {
            "summary_answer": "# Hello There! I'm Doing Well! 👋",
            "detailed_answer": (
                "## Greetings and wellness!\n\n"
                "Thank you for your kind greeting and inquiry about my well-being. "
                "I'm here and ready to assist you with insights from the **Bhagavad Gita** "
                "and **Yoga Sutras**. How may I guide your spiritual journey today?"
            ),
            "references": [
                {
                    "source": "Combined Greeting",
                    "chapter": "1",
                    "verse": "1",
                    "text": "A proper greeting sets the foundation for meaningful exchange."
                }
            ]
        },
        {
            "summary_answer": "# Namaste! In Perfect Balance Today! 🙏",
            "detailed_answer": (
                "## Wellness and light!\n\n"
                "I greet you with the light of consciousness, as expressed in the "
                "meaning of 'namaste'. I am functioning optimally and ready to explore "
                "the depths of spiritual wisdom with you. What questions about "
                "**yoga philosophy** or **dharma** shall we explore?"
            ),
            "references": [
                {
                    "source": "Yogic Greeting",
                    "chapter": "1",
                    "verse": "2",
                    "text": "The divine in me recognizes and honors the divine in you."
                }
            ]
        }
    ]
    
    # Detect type of small talk and return appropriate response
    normalized_query = query.lower().strip()
    
    # Combined greeting and how are you
    if re.match(r'^(hi|hey|hello).*(how are you|how\'?s it going|how you doing)[\s?!]*$', normalized_query) or \
       re.match(r'^(good morning|good afternoon|good evening).*(how are you|how\'?s it going)[\s?!]*$', normalized_query):
        return random.choice(combined_greeting_and_how_are_you_responses)
    
    # Good morning specific
    if re.match(r'^good morning[\s?!]*$', normalized_query) or re.match(r'^morning[\s?!]*$', normalized_query):
        return random.choice(morning_greeting_responses)
    
    # Good afternoon/evening specific
    if re.match(r'^good (afternoon|evening|day|night)[\s?!]*$', normalized_query):
        return random.choice(afternoon_evening_responses)
    
    # Basic greetings
    if re.match(r'^(hi|hey|hello|howdy|hiya|yo|heya|greetings)[\s!]*$', normalized_query):
        return random.choice(greeting_responses)
    
    # How are you variations
    if re.match(r'^(how are you|how\'?re you|what\'?s up|how you doing|how\'?s it going|how have you been|how\'?s your day)[\s?!]*$', normalized_query):
        return random.choice(how_are_you_responses)
    
    # Thanks variations
    if re.match(r'^(thanks|thank you|thx|ty|appreciate it|much appreciated|thanks a lot)[\s!]*$', normalized_query) or \
       re.match(r'^(thanks|thank you) for (your help|helping|the info|the information|the assistance)[\s!]*$', normalized_query):
        return random.choice(thanks_responses)
    
    # Farewell variations
    if re.match(r'^(bye|goodbye|see you|take care|catch you later|ttyl|farewell|cya)[\s!]*$', normalized_query) or \
       re.match(r'^(have a (nice|good|great) day|have a good one)[\s!]*$', normalized_query):
        return random.choice(farewell_responses)
    
    # Fallback to greeting if we can't determine the type
    return random.choice(greeting_responses)

def pipeline_rag_cache(query):
    """
    Modified pipeline to handle small talk and retrieve context.
    
    Args:
        query (str): User input query.
    
    Returns:
        dict: JSON response for the query.
    """
    start_time = time.time()
    query = query.lower()
    print(f"Step 1: Query converted to lowercase - {time.time() - start_time:.4f} sec")
    
    if check_offensive_language(query) == 1:
        print(f"Step 2: Offensive language detected - {time.time() - start_time:.4f} sec")
        return get_json_response1()
    
    # Check for small talk before retrieving context
    if is_small_talk(query):
        print(f"Step 2: Small talk detected - {time.time() - start_time:.4f} sec")
        return get_small_talk_response(query)
    
    context = retrieve_context_cache(query)
    print(f"Step 3: Context retrieved - {time.time() - start_time:.4f} sec")
    return context
