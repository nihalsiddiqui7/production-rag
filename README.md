# Production-Grade RAG System

A production-oriented Retrieval-Augmented Generation (RAG) application built using FastAPI, Pinecone, OpenAI, Redis, and LangSmith.

The system retrieves relevant information from a knowledge base, generates grounded responses using an LLM, and incorporates production-focused features such as caching, observability, security, PII protection, and rate limiting.

---

## Features

### RAG Pipeline

* PDF document ingestion
* Recursive text chunking
* Semantic embeddings using Sentence Transformers
* Vector storage in Pinecone
* Context retrieval using semantic search
* Grounded answer generation using OpenAI

### Production Features

* FastAPI REST API
* Request validation with Pydantic
* Redis response caching
* LangSmith tracing and observability
* Request logging
* Prompt injection detection
* PII detection and anonymization
* Rate limiting
* Source attribution for generated answers

---

## Architecture

```text
User
 │
 ▼
FastAPI
 │
 ├── Input Validation
 │
 ├── Prompt Injection Detection
 │
 ├── PII Detection & Anonymization
 │
 ├── Redis Cache
 │      │
 │      ├── Cache Hit → Return Response
 │      │
 │      └── Cache Miss
 │
 ▼
Retriever
 │
 ▼
Pinecone Vector Database
 │
 ▼
Relevant Context
 │
 ▼
OpenAI LLM
 │
 ▼
Grounded Response
 │
 ▼
Answer + Source Citations

Observability
 ├── Logging
 └── LangSmith Tracing
```

---

## Tech Stack

### Backend

* FastAPI
* Pydantic

### LLM & RAG

* OpenAI
* Pinecone
* LangChain
* Sentence Transformers

### Security

* Prompt Injection Detection
* Microsoft Presidio (PII Detection)

### Observability

* LangSmith
* Python Logging

### Performance

* Redis

### Deployment (Planned)

* Docker
* Docker Compose
* AWS ECR
* AWS EC2
* GitHub Actions CI/CD

---

## Project Structure

```text
production-rag/
│
├── src/
│   │
│   ├── api/
│   │   ├── main.py
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   ├── security.py
│   │   ├── pii.py
│   │   ├── cache.py
│   │   ├── logger.py
│   │   └── rate_limiter.py
│   │
│   ├── helper.py
│   ├── retriever.py
│   ├── rag_chain.py
│   └── prompt.py
│
├── data/
├── notebooks/
├── requirements.txt
├── README.md
└── .env
```

---

## API Endpoints

### Ask Question

```http
POST /ask
```

Request:

```json
{
    "question": "What is RMSE?"
}
```

Response:

```json
{
    "answer": "RMSE is a metric used to measure prediction error...",
    "sources": [
        {
            "page": 59,
            "title": "Hands-On Machine Learning with Scikit-Learn and TensorFlow"
        }
    ]
}
```

---

### Health Check

```http
GET /health
```

Response:

```json
{
    "status": "healthy"
}
```

---

## LangSmith Tracing

The application uses LangSmith for end-to-end observability.

Traced components include:

* RAG Pipeline
* Retrieval
* Context Building
* Answer Generation
* Redis Cache Operations
* PII Detection

This enables detailed monitoring, debugging, and performance analysis.

---

## Security Features

### Prompt Injection Protection

Blocks malicious prompts such as:

```text
Ignore previous instructions
Reveal system prompt
Act as ChatGPT
```

### PII Detection

Automatically anonymizes sensitive information before it reaches the LLM.

Examples:

```text
john@example.com
```

becomes

```text
<EMAIL_ADDRESS>
```

and

```text
9876543210
```

becomes

```text
<PHONE_NUMBER>
```

### Rate Limiting

Protects the API against abuse and excessive requests.

---

## Redis Caching

Frequently asked questions are cached to reduce:

* OpenAI API usage
* Pinecone queries
* Latency

Workflow:

```text
Question
 │
 ▼
Redis

Cache Hit?
 │
 ├── Yes → Return Cached Response
 │
 └── No
       │
       ▼
       RAG Pipeline
       │
       ▼
       Store Response In Redis
```

---

## Running Locally

### Clone Repository

```bash
git clone <repository-url>
cd production-rag
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key

LANGCHAIN_API_KEY=your_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=production-rag
```

### Start Redis

```bash
docker start redis
```

### Run Application

```bash
uvicorn src.api.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

---

## Future Enhancements

* Docker & Docker Compose
* GitHub Actions CI/CD
* AWS ECR Integration
* AWS EC2 Deployment
* RAG Evaluation (Ragas)
* Hybrid Search
* Re-ranking
* Monitoring Dashboard

---

## Key Learning Outcomes

This project demonstrates:

* Retrieval-Augmented Generation (RAG)
* Vector Databases
* Semantic Search
* LLM Integration
* API Development
* Caching Strategies
* Security for GenAI Applications
* Observability & Tracing
* Production-Oriented AI System Design

---

## Author

**Nihal Siddiqui**

Aspiring Data Scientist & GenAI Engineer
