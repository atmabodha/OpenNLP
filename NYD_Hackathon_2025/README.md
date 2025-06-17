# RAG Chatbot

### 🚀 Retrieval-Augmented Generation (RAG) Chatbot

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** that provides insightful responses using verses from the **Bhagavad Gita** and **Patanjali Yoga Sutras**. The chatbot supports **query auto-completion, query prediction, document retrieval, reranking, dynamic response generation, and response validation**.

---

## 🌟 Features
- **🔍 Query Auto-Completion:** Predicts and suggests possible queries.
- **🧠 Intelligent Query Prediction:** Refines user queries for better results.
- **📄 Contextual Document Retrieval:** Uses **FAISS/Qdrant** to fetch relevant verses.
- **📊 Reranking & Filtering:** Ensures the best-matching results are prioritized.
- **🤖 Dynamic Prompt Templates:** Generates structured and contextually rich responses.
- **✅ Response Review & Validation:** Ensures quality by re-generating responses if validation fails.
- **📡 API-Driven Architecture:** Backend built with **FastAPI** and frontend with **React + Next.js**.
- **🐳 Docker-Ready:** Easily deployable as separate **frontend** and **backend** services.

---

## 📂 Project Structure

### **Backend: RAG Pipeline (`iks-rag-pipelines/` - Python + FastAPI)**
```
iks-rag-pipelines/
│── src/
│   ├── data/				  # Data Cleaning & Preprocessing
│   ├── vectorizer/		  # Embedding & Vectorization (FAISS/Qdrant)
│   ├── retrieval/		  # Query Filtering & Document Retrieval
│   ├── rerank/			  # Reranking Logic
│   ├── response_gen/	  # Response Generation & Prompting
│   ├── validation/		  # Response Review & Validation
│   ├── main.py			  # FastAPI Entry Point
│
│── tests/				  # Unit & Integration Tests
│── requirements.txt	  # Python Dependencies
│── Dockerfile			  # Backend Docker Setup
```

### **Frontend: Chat UI (`iks-rag-ui/` - React + Next.js)**
```
iks-rag-ui/
│── src/
│   ├── components/		  # Reusable UI Components
│   ├── pages/			  # Next.js API & Page Routes
│   ├── hooks/			  # Custom React Hooks
│   ├── services/		  # API Call Functions
│   ├── styles/			  # CSS & Tailwind Styling
│   ├── App.tsx			  # Main React App Component
│
│── public/				  # Static Assets
│── package.json		  # Frontend Dependencies
│── Dockerfile			  # Frontend Docker Setup
```

---

## ⚡ API Endpoints (FastAPI Backend)

### 🔹 **1. Query Auto-Completion**
```http
POST /autocomplete
```
**Request:**
```json
{
  "partial_query": "What is karma"
}
```
**Response:**
```json
{
  "suggestions": ["What is karma yoga?", "What is karma in Gita?"]
}
```

### 🔹 **2. Predict Query**
```http
POST /predict-query
```
**Request:**
```json
{
  "query": "Explain dharma in"
}
```
**Response:**
```json
{
  "predicted_query": "Explain dharma in Bhagavad Gita?",
  "confidence": 0.95
}
```

### 🔹 **3. Chatbot Response**
```http
POST /chat
```
**Request:**
```json
{
  "query": "What does Gita say about self-discipline?",
  "top_k": 3
}
```
**Response:**
```json
{
  "response": "The Bhagavad Gita emphasizes self-discipline through detachment from results...",
  "references": [
    { "source": "Bhagavad Gita, Chapter 6, Verse 5", "link": "https://example.com/gita/ch6v5" }
  ],
  "confidence": 0.97
}
```

### 🔹 **4. Retrieve References**
```http
POST /retrieve-references
```
**Request:**
```json
{
  "query": "Tell me about karma in Yoga Sutras.",
  "top_k": 3
}
```
**Response:**
```json
{
  "references": [
    { "text": "Karma yoga is the path of selfless action.", "source": "Patanjali Yoga Sutras, Verse 2.1", "link": "https://example.com/yoga-sutras/v2-1" }
  ]
}
```

---

## 🛠️ Installation & Setup

### **1️⃣ Backend Setup (FastAPI)**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload
```

### **2️⃣ Frontend Setup (React + Next.js)**
```bash
cd frontend
npm install
npm run dev
```

### **3️⃣ Run with Docker (Recommended)**
```bash
# Backend
cd iks-rag-pipelines
docker build -t iks-rag-pipelines .
docker run -p 8000:8000 iks-rag-pipelines

# Frontend
cd iks-rag-ui
docker build -t iks-rag-ui .
docker run -p 3000:3000 iks-rag-ui
```

---

## 🚀 Deployment
### **With Docker Compose**
```yaml
version: '3.8'
services:
  backend:
    build: ./iks-rag-pipelines
    ports:
      - "8000:8000"
    environment:
      - ENV=production
  
  frontend:
    build: ./iks-rag-ui
    ports:
      - "3000:3000"
    depends_on:
      - iks-rag-pipelines
```
```bash
docker-compose up --build
```

---

## ✅ Best Practices Followed
✔️ **Modular architecture** (Separate concerns for retrieval, ranking, generation).  
✔️ **Scalable API** (Using **FastAPI** for async support).  
✔️ **Efficient vector search** (Using **FAISS / Qdrant** for document retrieval).  
✔️ **React + Next.js** (Server-side rendering for faster UI loading).  
✔️ **Containerized setup** (Docker for seamless deployment).  
✔️ **Unit & Integration Tests** (Ensuring API reliability).  
✔️ **OpenAPI Documentation** (Auto-generating Swagger docs).  

---

## 📜 License
This project is licensed under the **MIT License**. Feel free to contribute! 🚀

---
## 🎯 Contributors
- Aryan Kaul
- To be Updated

💡 *Open to contributions! Feel free to fork and improve!* 😃

