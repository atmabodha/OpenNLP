# RAG Chatbot

# NYD Hackathon 2025

Hackathon held around **National Youth Day (January 12, 2025)**

### 🚀 Retrieval-Augmented Generation (RAG) Chatbot

This project was jointly implemented by the members of the top 5 teams who participated in the NYD Hackathon 2025. It implements a **Retrieval-Augmented Generation (RAG) chatbot** that provides insightful responses using verses from the **Bhagavad Gita** and **Patanjali Yoga Sutras**. The chatbot supports **query auto-completion, query prediction, document retrieval, reranking, dynamic response generation, and response validation**.

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
    {
      "source": "Bhagavad Gita, Chapter 6, Verse 5",
      "link": "https://example.com/gita/ch6v5"
    }
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
    {
      "text": "Karma yoga is the path of selfless action.",
      "source": "Patanjali Yoga Sutras, Verse 2.1",
      "link": "https://example.com/yoga-sutras/v2-1"
    }
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
version: "3.8"
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

## Hackathon Task & Guidelines

1. Find a few relevant shlokas (verses) from the Bhagavad Gita and Patanjali Yoga Sutras (PYS) for a user query using LLMs and other information retrieval techniques. You can use this [dataset](https://www.github.com/atmabodha/Vedanta_Datasets) as a starting point to build upon. Feel free to explore various strategies for chunking, reranking, fine-tuning, etc.

2. Feed the retrieved shlokas along with the user query to an open source LLM like LLaMA to generate a summary of the answer. You are expected to work on creating a suitable prompt for this purpose that minimizes hallucinations.

3. Generate the output in a suitable JSON format.

4. Need to identify irrelevant or inappropriate user queries.

5. Do a thorough analysis of the generated answers for a wide variety of user queries.

6. Evaluation Criteria:

- Accuracy of the top verse retrieved for these curated questions from Gita and PYS.

- Quality of the prompt written and summarised answers generated using an open source LLM.

- Depth and quality of the analysis of the results.

- Cost of generating answer per query and lean architecture of the pipeline.

---

## Hackathon Rules

- Each team should have 1-3 members.

- Team members can be students or working professionals.

- All the code or ideas used from elsewhere must be properly cited in the submission report.

- Use only Open Source LLMs like SBERT, LLaMA, etc for all tasks like embeddings, text generation, etc.

- The code submitted for final evaluation must be made openly available for anyone to use.

- Incomplete or inappropriate submissions will be rejected.

- Prize money will be distributed through UPI or as Amazon Gift Vouchers to the team lead.

- Decision of the judges will be final.

---

## Important Dates

December 20, 2024: [Register on Unstop](https://unstop.com/hackathons/the-nyd-hackathon-2025-the-yoga-vivek-group-1281825)

December 22, 2024: First Webinar for registered participants

Dec 29 and Jan 05: Progress monitoring meetings

January 12, 2025: Final submission

January 19, 2025: Presentation of top 10 submissions

January 26, 2025: Prize Announcement

---

## Prizes

- 1st Prize : INR 20k

Kabir Arora, Aryan Kaul & Bhavya Pratap Singh [ Punjab Engineering College, Chandigarh]

- 2nd Prize : INR 10k

Anushree Ghosh, Agniva Saha & Srinjoy Das [ IIT Kharagpur]

- 3rd Prize : INR 5k

Rakshit Sawarn & Ananya Priyaroop [ IIT Bombay]

- 4th Prize : INR 1k

Hritish Maikap [ Vishwakarma Institute of Technology, Pune]

- 5th Prize : INR 1k

Nikhil Yadav, Sanjay VP & Nikhil Raj Soni [SRMU, Lucknow]

---

## Organisers

- [Dr. Kushal Shah, UTMT](https://www.utmt.org)
- [Mr. Vishal Manchanda, Infosys](https://www.linkedin.com/in/vishal-manchanda-097a6643/)
- Other well wishers
