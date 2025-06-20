# Ramayana Fact-Checking RAG Pipeline

A modular, verifiable Retrieval-Augmented Generation (RAG) pipeline tailored for validating factual claims pertaining to the **Ramayana**. Constructed using **LangGraph** and **open-source models**, the system emphasizes **accuracy**, **latency**, **efficiency**, and **modularity**.

---

## Setup Instructions

### Virtual Environment

create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### Dependencies

Required libraries:

```txt
requests>=2.31.0
sentence-transformers>=2.2.2
transformers>=4.30.0
torch>=2.0.0
qdrant-client>=1.6.0
langgraph>=0.0.40
pandas>=2.0.0
tqdm>=4.65.0
better-profanity>=0.7.0
```

Install via:

```bash
pip install -r requirements.txt
```

Run app.py

```bash
  python app.py
```

---

## System Architecture

### Pipeline Overview

The architecture facilitates dynamic flow control with conditional transitions. It adapts in real-time to system signals—halting on errors or retrying on low-confidence outputs—thus ensuring operational efficiency and factual integrity.

Each agent operates independently, enabling granular debugging, seamless scalability, and robust maintenance.

```
(Validator) → [ Retriever | end ] → (Reranker) → (Generator) → (Evaluator) → (Decision Controller) → [ Retry | end ] → (Evaluator)
```

---

## Components

### I. Validator Agent

- Applies profanity filters and length-based exclusion.
- Utilizes a Ramayana-specific glossary to bypass irrelevant queries checking.
- Employs a high-speed Redis store containing 100 synthetically generated reference sentences to filter out irrelevant or malformed questions.

  - **Embedding Model**: `all-MiniLM-L6-v2` (384-dim)

  - Implements KNN search (Top-5, threshold: 0.25).
  - Offers low-latency, semantically coherent embeddings.

This stage performs computationally inexpensive validation tasks upfront to minimize unnecessary API invocations.

### II. Retriever Agent

- **Embedding Model**: [`BAAI/bge-base-en`](https://huggingface.co/BAAI/bge-base-en) (Bi-encoder)

  - Supervised contrastive learning for refined relevance detection.
  - Optimized for English with prompt-tuning capabilities.
  - prompt-tuning support for retrieval optimisation

- **Database**: Qdrant

  - HNSW-based Approximate Nearest Neighbour search.
  - Payload Support

- **Retrieved Documents**: 30

### III. Reranker Agent

- **Model**: [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) (Cross-encoder)

  - Utilizes cross-encoding for heightened semantic precision.
  - Same family with identical tokenizers and training corpus ( alignment in tokenization, embedding space, and reference understanding )

- **Retained Documents**: Top 15

### IV. Generator Agent

- **Model**: [`Mistral-medium`](https://docs.mistral.ai/getting-started/models/models_overview/)(open-source LLM)

  - Configured for logical reasoning and evidential validation.
  - Utilizes structured prompting strategies (e.g., step-wise deduction).
    - direct/indirect evidence
    - cross-validation
    - alternative phrasing
    - sufficiency of proof

- **Output**: Factual verdict accompanied by structured justification.

### V. Evaluator Agent

- **Model**: [`Mistral-medium`](https://docs.mistral.ai/getting-started/models/models_overview/)

  - Assesses generated responses for hallucinations and semantic integrity.
  - Classification: `Reliable`, `Uncertain`, `Hallucinated`
  - Implements a confidence feedback mechanism with capped retries.
  - `Confidence Score =
min ( 1.0, max ( 0.0, (sum of all reranked scores ÷ number of reranked scores) × feedback multiplier) )`

### VI. Retry Generator (fallback)

- Activated upon API failures or evaluator-detected low confidence.
- Applies exponential backoff:
  - `delay = base + 2 * attempt`

---

## Evaluation Metrics

| Metric          | Value         |
| --------------- | ------------- |
| Accuracy        | 83%+          |
| Average Latency | 15–30 seconds |
| Retrieval Depth | 30            |
| Reranked Depth  | 15            |
| Total Verses    | 18,454        |

---

## Data Collection

- **Primary Source**: [Valmikiramayan.net](https://valmikiramayan.net)
- **Tooling**: Selenium WebDriver
- **Crawl Structure**: Navigation from Links → Introductions + Verses
- **Postprocessing**: Data cleaning and normalization for semantic ingestion

---

## Future Enhancements

1. **Claim Decomposition**: Disaggregate compound queries into atomic factual units
2. **Hybrid Reranking**: Integrate dense and sparse scoring strategies
3. **Prompt Resilience**: Improve robustness against paraphrasing and syntactic variation
4. **Vocabulary Expansion**: Broaden Redis glossary for domain-specific filtering using extracted introductions

---

## Implementation Notes

- Redis integration is currently non-functional; relevant sections, including glossary-based filters, are commented out.

# Contact 

For bug or error reports and feature requests, please write an email to : sanjayvp08@gmail.com