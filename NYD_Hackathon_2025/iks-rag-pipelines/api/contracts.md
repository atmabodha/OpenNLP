### **API Contracts for RAG Chatbot Backend**
These API contracts define how the frontend (React + Next.js) communicates with the backend (FastAPI). The API provides endpoints for query prediction, autocomplete suggestions, response generation, and reference retrieval.

---

## **Base URL**
```plaintext
https://api.yourdomain.com/v1/
```

---

## **1. Query Autocomplete**
🔹 **Purpose:** Helps users autocomplete their query based on partially typed input.  
🔹 **Endpoint:** `POST /autocomplete`  
🔹 **Request:**  
```json
{
  "partial_query": "What is the meaning of"
}
```
🔹 **Response:**  
```json
{
  "suggestions": [
    "What is the meaning of life?",
    "What is the meaning of yoga?",
    "What is the meaning of dharma?"
  ]
}
```
🔹 **Notes:**  
- Uses an NLP model or retrieval-based approach to suggest possible queries.  
- Supports real-time user input.

---

## **2. Query Prediction**
🔹 **Purpose:** Predicts the user’s intent and refines the query.  
🔹 **Endpoint:** `POST /predict-query`  
🔹 **Request:**  
```json
{
  "query": "How does karma work in"
}
```
🔹 **Response:**  
```json
{
  "predicted_query": "How does karma work in Bhagavad Gita?",
  "confidence": 0.92
}
```
🔹 **Notes:**  
- Uses transformer models like Llama 3.2 to enhance query understanding.  
- The confidence score helps determine if prediction is accurate.

---

## **3. Generate Response (Chat Completion)**
🔹 **Purpose:** Generates an answer based on RAG (Retrieval-Augmented Generation).  
🔹 **Endpoint:** `POST /chat`  
🔹 **Request:**  
```json
{
  "query": "What does Bhagavad Gita say about self-discipline?",
  "top_k": 3
}
```
🔹 **Response:**  
```json
{
  "response": "The Bhagavad Gita emphasizes self-discipline as an essential virtue in Chapter 6, where Lord Krishna discusses meditation and control over desires...",
  "references": [
    {
      "source": "Bhagavad Gita, Chapter 6, Verse 5",
      "link": "https://example.com/gita/ch6v5"
    },
    {
      "source": "Bhagavad Gita, Chapter 2, Verse 47",
      "link": "https://example.com/gita/ch2v47"
    }
  ],
  "confidence": 0.97
}
```
🔹 **Notes:**  
- Retrieves documents from FAISS/Qdrant.  
- Uses a reranker to prioritize relevant verses.  
- References include verse details and links to full texts.

---

## **4. Retrieve References**
🔹 **Purpose:** Fetches relevant verses or texts supporting a response.  
🔹 **Endpoint:** `POST /retrieve-references`  
🔹 **Request:**  
```json
{
  "query": "Tell me about karma in Yoga Sutras.",
  "top_k": 3
}
```
🔹 **Response:**  
```json
{
  "references": [
    {
      "text": "Karma yoga is the path of selfless action, without attachment to results.",
      "source": "Patanjali Yoga Sutras, Verse 2.1",
      "link": "https://example.com/yoga-sutras/v2-1"
    },
    {
      "text": "The practice of karma yoga purifies the mind and leads to spiritual wisdom.",
      "source": "Patanjali Yoga Sutras, Verse 3.5",
      "link": "https://example.com/yoga-sutras/v3-5"
    }
  ]
}
```
🔹 **Notes:**  
- Retrieves and ranks relevant references from indexed documents.  
- Helps users verify AI-generated responses.

---

## **5. Regenerate Response**
🔹 **Purpose:** Re-generates a response if validation fails.  
🔹 **Endpoint:** `POST /regenerate`  
🔹 **Request:**  
```json
{
  "query": "What is the essence of detachment in Gita?",
  "previous_response": "Detachment means renouncing the world.",
  "feedback": "The response lacks depth. Please provide a detailed explanation."
}
```
🔹 **Response:**  
```json
{
  "new_response": "Detachment in Bhagavad Gita is about acting without attachment to results. It is discussed in Chapter 2, where Krishna advises Arjuna to focus on duty without being driven by rewards.",
  "references": [
    {
      "source": "Bhagavad Gita, Chapter 2, Verse 47",
      "link": "https://example.com/gita/ch2v47"
    }
  ]
}
```
🔹 **Notes:**  
- If a response is flagged as insufficient, this endpoint allows AI to generate a more detailed answer.

---

## **6. Health Check**
🔹 **Purpose:** Check if the API is running.  
🔹 **Endpoint:** `GET /health`  
🔹 **Response:**  
```json
{
  "status": "OK",
  "uptime": "5 days, 3 hours",
  "version": "1.0.0"
}
```

---

### **Summary of API Endpoints**
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/autocomplete` | `POST` | Suggests query completions |
| `/predict-query` | `POST` | Predicts refined query |
| `/chat` | `POST` | Generates chatbot response with references |
| `/retrieve-references` | `POST` | Fetches supporting references |
| `/regenerate` | `POST` | Improves response based on feedback |
| `/health` | `GET` | Checks API health |

---

### **Best Practices for API Design**
- **Use FastAPI for performance & async support.**
- **Enable caching for repeated queries.**
- **Use API versioning (`/v1/`) to prevent breaking changes.**
- **Return structured JSON responses with confidence scores.**
- **Enable logging & monitoring (e.g., Prometheus, Grafana).**

Would you like OpenAPI/Swagger documentation for this? 🚀
