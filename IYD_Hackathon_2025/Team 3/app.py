import os
import sys 
import re
import requests
import time
import random
import pandas as pd
import logging
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Union
from enum import Enum
from tqdm import tqdm
from better_profanity import profanity
# import json
# import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fact_checker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API Configuration
MISTRAL_API_KEY = "Sx9wjoosQ6rNbjoNTXlQus1HSAjUublB"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.LOTpaZOycm_u2Dj_FsoWyAbSCjEs_AQf0yyYCcjKtgk"
QDRANT_HOST = "19208b69-1b41-4830-9b4d-66b3e1181237.us-east4-0.gcp.cloud.qdrant.io"

collection_name = "ramayana_facts"

# Redis configuration (currently disabled because of client configuration issues)
# REDIS_HOST = "localhost"
# REDIS_PORT = 6379
# REDIS_PASSWORD = None
# INDEX_NAME = "idx:ramayana_anchors"
# VECTOR_FIELD = "embedding"
# VECTOR_DIM = 384

# since redis client is not working this has no significance as it is introduced to 
# bypass that step
# Load glossary data (currently disabled)
# with open("./data/glossary.json", "r") as f:
#     glossary = json.load(f)

# Initialize profanity filter
profanity.load_censor_words()
logger.info("Profanity filter initialized")

# Model Caching System
class ModelCache:
    """Singleton class to cache models in memory for reuse across fact-checking operations"""
    _instance = None
    _initialized = False

    def __new__(cls):
        """Ensure only one instance exists (Singleton pattern)"""
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
            logger.info("ModelCache singleton instance created")
        return cls._instance

    def __init__(self):
        """Initialize model storage if not already done"""
        if not self._initialized:
            self.embedding_model = None
            self.reranker_tokenizer = None
            self.reranker_model = None
            self._initialized = True
            logger.info("ModelCache initialized with empty model slots")

    def get_embedding_model(self):
        """Get cached embedding model, load if not cached"""
        if self.embedding_model is None:
            logger.info("Loading embedding model...")
            model_paths = get_model_paths()
            embedding_path = model_paths["embedding"]

            if os.path.exists(embedding_path):
                logger.info(f"Loading cached embedding model from {embedding_path}")
                self.embedding_model = SentenceTransformer(embedding_path)
            else:
                logger.info("Loading embedding model from HuggingFace (BAAI/bge-base-en)")
                self.embedding_model = SentenceTransformer("BAAI/bge-base-en")

            logger.info("Embedding model loaded and cached in memory")
        return self.embedding_model

    def get_reranker_models(self):
        """Get cached reranker models, load if not cached"""
        if self.reranker_tokenizer is None or self.reranker_model is None:
            logger.info("Loading reranker models...")
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            model_paths = get_model_paths()
            reranker_path = model_paths["reranker"]

            if os.path.exists(reranker_path):
                logger.info(f"Loading cached reranker models from {reranker_path}")
                self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_path)
                self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_path)
            else:
                logger.info("Loading reranker models from HuggingFace (BAAI/bge-reranker-base)")
                self.reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
                self.reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")

            self.reranker_model.eval()  # Set to evaluation mode
            logger.info("Reranker models loaded and cached in memory")

        return self.reranker_tokenizer, self.reranker_model

# Global model cache instance
model_cache = ModelCache()
logger.info("Global model cache instance created")

# Reasoning prompt template for fact verification
REASONING_STEPS = f"""
Provide concise step-by-step reasoning:
1. Check for direct mentions of the claim
2. Look for indirect references or implications
3. Cross-validate across passages
4. Consider alternative phrasings
5. Assess evidence sufficiency

Then provide:
- If the claim is not explicitly stated but can be inferred from roles, actions, or relationships, mark it TRUE with justification. Else FALSE.
"""

# Data Models & Enums
class EvidenceQuality(str, Enum):
    """Enumeration for evidence quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"
    ERROR = "error"

class QdrantPoint:
    """Represents a document point from Qdrant vector database"""
    def __init__(self, id: str, payload: dict, score: Optional[float] = None):
        self.id = id
        self.payload = payload
        self.score = score

class GraphState(TypedDict, total=False):
    """State dictionary for the fact-checking workflow graph"""
    query: str                                                    # Input statement to verify
    valid: Optional[str]                                         # Validation result
    issue: Optional[str]                                         # Validation issues
    docs: Optional[List[QdrantPoint]]                           # Retrieved documents
    reranked: Optional[List[tuple[QdrantPoint, float]]]         # Reranked documents with scores
    verdict: Optional[str]                                       # Final TRUE/FALSE verdict
    reasoning: Optional[str]                                     # Reasoning explanation
    needs_retry: Optional[bool]                                  # Whether retry is needed
    feedback: Optional[str]                                      # Quality feedback
    confidence_score: Optional[float]                           # Confidence in result
    evidence_quality: Optional[Union[str, EvidenceQuality]]     # Quality of evidence
    retry_count: Optional[int]                                   # Number of retries attempted
    message: Optional[str]                                       # Status messages
    halt: Optional[bool]                                         # Whether to halt processing
    error: Optional[str]                                         # Error messages

# Base Classes
class BaseAgent:
    """Abstract base class for all processing agents in the fact-checking pipeline"""

    def observe(self, state: dict) -> dict:
        """Extract relevant inputs from the current state"""
        raise NotImplementedError

    def act(self, inputs: dict) -> dict:
        """Perform the agent's specific processing task"""
        raise NotImplementedError

    def update_state(self, state: dict, outputs: dict) -> dict:
        """Update the state with the agent's outputs"""
        raise NotImplementedError

    def run(self, state: dict) -> dict:
        """Execute the full agent workflow: observe -> act -> update"""
        inputs = self.observe(state)
        outputs = self.act(inputs)
        return self.update_state(state, outputs)


class DecisionController:
    """Controls workflow decisions for retry logic and termination"""

    def __init__(self, max_retries: int = 1):
        self.max_retries = max_retries
        logger.info(f"DecisionController initialized with max_retries={max_retries}")

    def decide_next(self, state: dict) -> str:
        """Determine the next step in the workflow based on current state"""
        retry_count = state.get("retry_count", 0)
        needs_retry = state.get("needs_retry", False)

        if state.get("halt") or state.get("error"):
            logger.info("Workflow halted due to error or validation failure")
            return "end"

        if retry_count >= self.max_retries:
            logger.info(f"Maximum retries ({self.max_retries}) reached")
            return "end"

        if needs_retry:
            logger.info(f"Retry needed, attempt {retry_count + 1}")
            return "retry_generate"

        return "end"

# Agent Classes
class QueryValidatorAgent(BaseAgent):
    """Validates input queries for profanity, relevance, and format"""

    def observe(self, state: dict) -> dict:
        """Extract query from state"""
        return {"query": state.get("query")}

    def act(self, inputs: dict) -> dict:
        """Validate the query and return any issues"""
        query = inputs["query"]
        logger.info(f"Validating query: {query[:50]}...")
        issue = validate_query(query)
        if issue:
            logger.warning(f"Query validation failed: {issue}")
        else:
            logger.info("Query validation passed")
        return {"issue": issue}

    def update_state(self, state: dict, outputs: dict) -> dict:
        """Update state with validation results"""
        if outputs["issue"]:
            return {
                **state,
                "halt": True,
                "valid": outputs["issue"]
            }
        return {**state}


class RetrieverAgent(BaseAgent):
    """Retrieves relevant documents from Qdrant vector database"""

    def __init__(self, k: int = 30):
        self.k = k
        logger.info(f"RetrieverAgent initialized with k={k}")

    def observe(self, state: dict) -> dict:
        """Extract query for document retrieval"""
        return {"query": state.get("query")}

    def act(self, inputs: dict) -> dict:
        """Retrieve top-k documents from vector database"""
        query = inputs["query"]
        logger.info(f"Retrieving top {self.k} documents for query")
        docs = retrieve_top_k(query, self.k)
        logger.info(f"Retrieved {len(docs)} documents")
        return {"docs": docs}

    def update_state(self, state: dict, outputs: dict) -> dict:
        """Update state with retrieved documents"""
        return {
            **state,
            "docs": outputs["docs"]
        }


class RerankerAgent(BaseAgent):
    """Reranks retrieved documents using a reranking model for better relevance"""

    def __init__(self, top: int = 15):
        self.top = top
        logger.info(f"RerankerAgent initialized with top={top}")

    def observe(self, state: dict) -> dict:
        """Extract query and documents for reranking"""
        return {
            "query": state.get("query"),
            "docs": state.get("docs", [])
        }

    def act(self, inputs: dict) -> dict:
        """Rerank documents based on relevance to query"""
        query = inputs["query"]
        docs = inputs["docs"]
        logger.info(f"Reranking {len(docs)} documents, keeping top {self.top}")
        reranked = rerank_results(query, docs, self.top)
        logger.info(f"Reranking completed, {len(reranked)} documents selected")
        return {"reranked": reranked}

    def update_state(self, state: dict, outputs: dict) -> dict:
        """Update state with reranked documents"""
        return {
            **state,
            "reranked": outputs["reranked"]
        }


class GeneratorAgent(BaseAgent):
    """Generates fact-checking responses using Mistral AI"""

    def observe(self, state: dict) -> dict:
        """Extract query, reranked documents, and feedback for generation"""
        return {
            "query": state.get("query"),
            "reranked": state.get("reranked"),
            "feedback": state.get("feedback")
        }

    def act(self, inputs: dict) -> dict:
        """Generate fact-checking response using Mistral AI"""
        query = inputs["query"]
        reranked = inputs["reranked"]
        feedback = inputs.get("feedback")

        logger.info(f"Generating fact-check response for query")
        if feedback:
            logger.info(f"Using feedback: {feedback}")

        result = mistral_verify_fact(query, reranked, feedback=feedback)
        verdict = extract_verdict(result)

        logger.info(f"Generated verdict: {verdict}")
        return {
            "verdict": verdict,
            "reasoning": result,
            "retry_count": 0
        }

    def update_state(self, state: dict, outputs: dict) -> dict:
        """Update state with generated response"""
        updated_state = {**state, **outputs}
        return updated_state


class EvaluatorAgent(BaseAgent):
    """Evaluates the quality of generated responses and determines if retry is needed"""

    def observe(self, state: dict) -> dict:
        """Extract components needed for evaluation"""
        query = state.get("query")
        reasoning = state.get("reasoning")
        reranked = state.get("reranked", [])
        retry_count = state.get("retry_count", 0)

        return {
            "query": query,
            "reasoning": reasoning,
            "reranked": reranked,
            "retry_count": retry_count
        }

    def act(self, inputs: dict) -> dict:
        """Evaluate response quality and calculate metrics"""
        query = inputs["query"]
        reasoning = inputs["reasoning"]
        reranked = inputs["reranked"]

        logger.info("Evaluating response quality")
        feedback = evaluate_fact_quality(query, reasoning)
        confidence_score = calculate_confidence_score(reranked, feedback)
        # currently deciding needs_retry based on feedback (since threshold for confidence score is debatable)
        needs_retry = feedback.lower() in ["uncertain", "hallucinated"]
        evidence_quality = calculate_evidence_quality(reranked)

        logger.info(f"Evaluation results - Feedback: {feedback}, Needs retry: {needs_retry}")
        logger.info(f"Confidence: {confidence_score:.3f}, Evidence quality: {evidence_quality}")

        return {
            "feedback": feedback,
            "needs_retry": needs_retry,
            "confidence_score": confidence_score,
            "evidence_quality": evidence_quality
        }

    def update_state(self, state: dict, outputs: dict) -> dict:
        """Update state with evaluation results"""
        updated_state = {
            **state,
            "feedback": outputs["feedback"],
            "needs_retry": outputs["needs_retry"],
            "confidence_score": outputs["confidence_score"],
            "evidence_quality": outputs["evidence_quality"]
        }
        return updated_state


class RetryGenerateAgent(BaseAgent):
    """Regenerates fact-checking responses when evaluation flags issues"""

    def observe(self, state: dict) -> dict:
        query = state.get("query")
        reranked = state.get("reranked", [])
        feedback = state.get("feedback")
        retry_count = state.get("retry_count", 0)

        return {
            "query": query,
            "reranked": reranked,
            "feedback": feedback,
            "retry_count": retry_count
        }

    def act(self, inputs: dict) -> dict:
        current_retry_count = inputs["retry_count"]
        new_retry_count = current_retry_count + 1
        feedback = inputs.get("feedback", "")

        result = mistral_verify_fact(
            inputs["query"],
            inputs["reranked"],
            feedback=feedback
        )

        verdict = extract_verdict(result)

        return {
            "reasoning": result,
            "verdict": verdict,
            "retry_count": new_retry_count
        }

    def update_state(self, state: dict, outputs: dict) -> dict:
        state.update({
            "reasoning": outputs.get("reasoning"),
            "verdict": outputs.get("verdict"),
            "retry_count": outputs.get("retry_count")
        })

        return state

# Utility Functions
def download_and_cache_models():
    """Download and cache required models locally, then load them into memory"""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    # Download embedding model if not exists
    embedding_model_path = os.path.join(models_dir, "bge-base-en")
    if not os.path.exists(embedding_model_path):
        print("Downloading embedding model (first time only)...")
        model = SentenceTransformer("BAAI/bge-base-en")
        model.save(embedding_model_path)
        print(f"Embedding model cached at: {embedding_model_path}")

    # Download reranker model if not exists
    reranker_model_path = os.path.join(models_dir, "bge-reranker-base")
    if not os.path.exists(reranker_model_path):
        print("Downloading reranker model (first time only)...")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
        model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
        tokenizer.save_pretrained(reranker_model_path)
        model.save_pretrained(reranker_model_path)
        print(f"Reranker model cached at: {reranker_model_path}")

    # Pre-load models into memory cache for faster access
    print("Pre-loading models into memory cache...")
    model_cache.get_embedding_model()
    model_cache.get_reranker_models()
    print("All models loaded and ready for use!")


def get_model_paths():
    """Get local model paths"""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    return {
        "embedding": os.path.join(models_dir, "bge-base-en"),
        "reranker": os.path.join(models_dir, "bge-reranker-base")
    }

# 
def calculate_confidence_score(reranked_passages, feedback):
    if not reranked_passages:
        return 0.0

    scores = [score for _, score in reranked_passages]

    if not scores:
        return 0.0

    avg_rerank_score = sum(scores) / len(scores)

    feedback_multiplier = {
        "reliable": 1.0,
        "uncertain": 0.7,
        "hallucinated": 0.3
    }
    multiplier = feedback_multiplier.get(feedback.lower(), 0.5)
    confidence = avg_rerank_score * multiplier

    return max(0.0, min(1.0, confidence))


def calculate_evidence_quality(reranked_passages):
    if not reranked_passages:
        return "low"

    scores = [score for _, score in reranked_passages]

    if not scores:
        return "low"

    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    passage_count = len(scores)

    if avg_score >= 0.8 and max_score >= 0.9 and passage_count >= 3:
        return "high"
    elif avg_score >= 0.6 and max_score >= 0.7 and passage_count >= 2:
        return "medium"
    else:
        return "low"


def evaluate_fact_quality(query: str, result: str) -> str:
    if not query or not result:
        return "uncertain"

    prompt = (
        "You are a factuality evaluator. Determine whether the following answer is factually correct for the query.\n"
        "Reply with exactly one word: reliable, hallucinated, or uncertain.\n\n"
        f"Query: {query}\n"
        f"Answer: {result}"
    )

    max_retries = 1
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {MISTRAL_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "mistral-medium",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2
                },
                timeout=60
            )
            response.raise_for_status()
            api_result = response.json()

            content = api_result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().lower()
            keyword = content.split()[0] if content else "uncertain"

            if keyword in {"reliable"}:
                return "reliable"
            elif keyword in {"hallucinated"}:
                return "hallucinated"
            elif keyword in {"uncertain"}:
                return "uncertain"
            else:
                return "uncertain"

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                else:
                    return "uncertain"
            else:
                return "uncertain"

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            else:
                return "uncertain"

        except (KeyError, IndexError, Exception) as e:
            return "uncertain"

    return "uncertain"


def mistral_verify_fact(fact, reranked_passages, feedback=None):
    top_contexts = [doc.payload["content"] for doc, _ in reranked_passages[:5]]
    context_texts = "\n\n".join(top_contexts)

    if feedback:
        prompt = f"""
        Previous answer flagged for: "{feedback}". Re-evaluate carefully.

        Claim: {fact}
        Passages: {context_texts}

        {REASONING_STEPS}
        """
    else:
        prompt = f"""
        You are a Ramayana expert and logical reasoning analyst. Determine whether the claim is logically or textually supported by the passages. If exact phrases aren't found, reason whether the meaning is implied.

        Claim: {fact}

        Passages: {context_texts}

        {REASONING_STEPS}
        """

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "mistral-medium",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,  # Low temperature for consistency
        "max_tokens": 512,   # Limit for concise responses
    }

    max_retries = 1
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()

            # Validate response structure
            if "choices" not in result or len(result["choices"]) == 0:
                raise Exception(f"API Error: {result.get('message', 'Unexpected response structure')}")

            content = result["choices"][0]["message"]["content"]
            return content

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit error
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter for rate limits
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(f"Mistral API Rate Limit Error: {str(e)}")
            else:
                raise Exception(f"Mistral API HTTP Error: {str(e)}")

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                # Exponential backoff for connection errors
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            else:
                raise Exception(f"Mistral API Request Error: {str(e)}")

        except KeyError as e:
            raise Exception(f"API Response Error: Missing expected field {str(e)}")
        except Exception as e:
            raise Exception(f"Mistral API Unexpected Error: {str(e)}")


def rerank_results(query, retrieved_docs, top):
    """Rerank results using cached models for better performance"""
    if not retrieved_docs:
        return []

    try:
        import torch

        # Use cached models
        tokenizer, model = model_cache.get_reranker_models()

        valid_docs = [doc for doc in retrieved_docs if hasattr(doc, 'payload') and doc.payload.get("content", "").strip()]
        passages = [doc.payload["content"] for doc in valid_docs]

        inputs = tokenizer(
            [query] * len(passages),
            passages,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(-1)

        scores = logits.tolist()

        reranked = list(zip(valid_docs, scores))
        reranked_sorted = sorted(reranked, key=lambda x: x[1], reverse=True)
        return reranked_sorted[:top]
    except Exception as e:
        print(f"Reranking failed: {e}")
        return retrieved_docs[:top] if retrieved_docs else []


def get_embedding(query):
    """Get embedding using cached model for better performance"""
    model = model_cache.get_embedding_model()
    prompt = "Represent this sentence for retrieving supporting documents: " + query
    return model.encode(prompt, normalize_embeddings=True).tolist()


def retrieve_top_k(query, k):
    try:
        client = QdrantClient(
            host=QDRANT_HOST,
            port=443,
            api_key=QDRANT_API_KEY,
            https=True
        )

        query_vector = get_embedding(query)
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=k
        )

        return search_results

    except Exception as e:
        raise Exception(f"Document retrieval failed: {str(e)}")


def clean_query(q):
    q = q.strip()                    # Remove leading/trailing whitespace
    q = re.sub(r"\s+", " ", q)      # Normalize multiple spaces to single space
    q = q.lower()                   # Convert to lowercase for consistency
    return q


# def is_entity_present(query, glossary_dict):
#     """Check if query contains any Ramayana-related entities from glossary"""
#     return not set(query.split()).isdisjoint({e.lower() for g in glossary_dict.values() for e in g})


# def is_semantically_relevant(query, threshold, top_k):
#     """
#     Check semantic relevance using Redis vector similarity search.
#     Compares query embedding against anchor queries with cosine similarity.
#     Falls back to True if Redis is unavailable.
#     """
#     try:
#         if redis_client is None:
#             # Graceful fallback when Redis is unavailable
#             return True

#         # Generate query embedding using same model as Redis data
#         embedding = model.encode(query, normalize_embeddings=True)
#         embedding_bytes = float32_to_bytes(embedding)

#         # Perform KNN search in Redis
#         redis_query = f"*=>[KNN {top_k} @{VECTOR_FIELD} $vec_param AS score]"

#         res = redis_client.ft(INDEX_NAME).search(
#             redis_query,
#             query_params={"vec_param": embedding_bytes}
#         )

#         if not res.docs:
#             return False

#         # Calculate average similarity score
#         top_scores = [float(doc.score) for doc in res.docs if hasattr(doc, "score")]
#         if not top_scores:
#             return False

#         avg_cosine_distance = sum(top_scores) / len(top_scores)
#         avg_cosine_similarity = 1 - avg_cosine_distance

#         return avg_cosine_similarity >= threshold

#     except redis.exceptions.ResponseError as e:
#         # Redis query error - fallback to allowing the query
#         return True
#     except Exception as e:
#         # Any other Redis error - fallback to allowing the query
#         return True


def validate_query(query):
    # Clean and normalize the query string
    query = clean_query(query)

    # Reject queries that are too short to be meaningful
    if len(query) <= 3:
        return "NOT RELEVANT"

    # Profanity filtering using better_profanity
    if profanity.contains_profanity(query):
        return "NOT RELEVANT"

    # Entity presence check (fast validation)
    # Uncomment and use if glossary-based entity validation is needed
    # if is_entity_present(query, glossary):
    #     return None

    # Semantic relevance check (requires Redis, currently disabled)
    # Uncomment and use if Redis-based semantic validation is enabled
    # if not is_semantically_relevant(query, threshold, top_k):
    #     return "Your query seems unrelated to our fact-checking purpose."

    # If all checks pass, query is considered valid
    return None


def extract_verdict(result_text: str) -> Optional[str]:
    """
    Extract TRUE/FALSE verdict from model response using regex patterns.
    Supports multiple formats and prioritizes explicit verdict statements.
    """
    text_lower = result_text.lower()

    # Patterns for TRUE verdicts (ordered by specificity)
    true_patterns = [
        r'verdict:\s*true',
        r'answer:\s*true',
        r'result:\s*true',
        r'\btrue\b'  # Standalone true word
    ]

    # Patterns for FALSE verdicts (ordered by specificity)
    false_patterns = [
        r'verdict:\s*false',
        r'answer:\s*false',
        r'result:\s*false',
        r'\bfalse\b'  # Standalone false word
    ]

    # Check for TRUE patterns first
    for pattern in true_patterns:
        if re.search(pattern, text_lower):
            return "TRUE"

    # Check for FALSE patterns
    for pattern in false_patterns:
        if re.search(pattern, text_lower):
            return "FALSE"

    # No verdict found
    return "NOT RELEVANT"

# Agent Instances
validator_agent = QueryValidatorAgent()
retriever_agent = RetrieverAgent()
reranker_agent = RerankerAgent()
generator_agent = GeneratorAgent()
evaluator_agent = EvaluatorAgent()
retry_agent = RetryGenerateAgent()

# Graph Node Functions
def validate_node(state: dict) -> dict:
    """Validates query for profanity, relevance, and format"""
    return validator_agent.run(state)

def retrieve_node(state: dict) -> dict:
    """Retrieves top-k relevant documents from Qdrant"""
    return retriever_agent.run(state)

def rerank_node(state: dict) -> dict:
    """Reranks retrieved documents for better relevance"""
    return reranker_agent.run(state)

def generate_node(state: dict) -> dict:
    """Generates fact-checking response using Mistral AI"""
    return generator_agent.run(state)

def retry_generate_node(state: dict) -> dict:
    """Regenerates response with feedback from evaluator"""
    return retry_agent.run(state)

def evaluate_result_node(state: dict) -> dict:
    """Evaluates response quality and determines if retry is needed"""
    return evaluator_agent.run(state)

def should_continue_after_validation(state: dict) -> str:
    """Routes to end if validation fails, otherwise continues to retrieval"""
    if state.get("halt") or state.get("error"):
        return "end"
    return "retrieve"

# Graph Creation & Setup
def create_graph():
    builder = StateGraph(GraphState)
    decision_controller = DecisionController()

    # Add all processing nodes
    builder.add_node("validate", validate_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("generate", generate_node)
    builder.add_node("evaluate", evaluate_result_node)
    builder.add_node("retry_generate", retry_generate_node)

    # Set starting point
    builder.set_entry_point("validate")

    # Conditional routing after validation
    builder.add_conditional_edges(
        "validate",
        should_continue_after_validation,
        path_map={
            "retrieve": "retrieve",
            "end": END
        }
    )

    # Linear flow through main pipeline
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "generate")
    builder.add_edge("generate", "evaluate")

    # Conditional routing after evaluation (retry or end)
    builder.add_conditional_edges(
        "evaluate",
        decision_controller.decide_next,
        path_map={
            "retry_generate": "retry_generate",
            "end": END
        }
    )

    # Retry loop back to evaluation
    builder.add_edge("retry_generate", "evaluate")
    return builder.compile()


graph = create_graph()

# Main Processing Functions
def generate_predictions_csv(input_csv, output_csv=None):
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        return pd.DataFrame()

    if 'Statement' not in df.columns:
        return pd.DataFrame()

    if output_csv:
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    processed_statements = set()
    if output_csv and os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            processed_statements = set(existing_df['Statement'].tolist())
        except Exception as e:
            pass

    # Initialize output CSV with headers if it doesn't exist
    if output_csv and not os.path.exists(output_csv):
        header_df = pd.DataFrame(columns=["Statement", "Prediction"])
        header_df.to_csv(output_csv, index=False)

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        statement = row['Statement']

        if statement in processed_statements:
            continue

        if i > 0:  # Add delay after first iteration
            time.sleep(2)

        try:
            state = {"query": statement}
            response = graph.invoke(state)

            predicted_truth = response.get('verdict', 'NOT RELEVANT')
            if predicted_truth is None:
                predicted_truth = 'NOT RELEVANT'

            prediction_row = {
                "Statement": statement,
                "Prediction": predicted_truth
            }

        except Exception as e:
            prediction_row = {
                "Statement": statement,
                "Prediction": "NOT RELEVANT"
            }

        # Write each prediction immediately to CSV
        if output_csv:
            prediction_df = pd.DataFrame([prediction_row])
            prediction_df.to_csv(output_csv, mode='a', header=False, index=False)

    # Return the final results
    if output_csv and os.path.exists(output_csv):
        return pd.read_csv(output_csv)
    else:
        return pd.DataFrame()


def main(file_req):
    # Download and cache models on first run
    download_and_cache_models()

    if os.path.isfile(file_req):
        print(f"File '{file_req}' exists in the current folder.")
    else:
        print(f"File '{file_req}' does NOT exist in the current folder.")
        sys.exit(1)
        
    generate_predictions_csv(file_req, "Predictions.csv")
    print(f"Predictions completed and saved to Predictions.csv")


if __name__ == "__main__":
    # file from which claims are read
    file_req = "RamanayaFinal.csv"
    main(file_req)
