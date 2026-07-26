# Production-Grade RAG System for Technical Knowledge Retrieval

A production-oriented Retrieval-Augmented Generation (RAG) application designed to provide accurate, source-grounded answers from technical documents using Large Language Models, semantic search, and modern MLOps practices.

The system combines vector search, LLM-powered generation, observability, caching, security controls, and API engineering principles to deliver reliable and scalable question-answering capabilities.

---

## Overview

Traditional LLMs are limited by static training data and may generate hallucinated responses when asked about domain-specific documents.

This project addresses that challenge using Retrieval-Augmented Generation (RAG), where relevant document chunks are retrieved from a vector database and supplied to the language model as context before generating a response.

The application exposes a production-ready REST API built with FastAPI and includes observability, security, caching, and operational features commonly found in real-world AI systems.

---

## Key Features

### Retrieval-Augmented Generation (RAG)

- PDF document ingestion
- **Hierarchical parent-child chunking** — small child chunks for precise semantic retrieval, parent chunks for rich generation context
- Semantic embedding generation
- Vector storage using Pinecone
- Similarity-based retrieval
- Context-aware answer generation
- Source-grounded responses

### Production API Features

- FastAPI-based REST API
- Request validation with Pydantic
- Health monitoring endpoint
- Structured logging
- Error handling
- Source attribution

### Security Features

- Prompt injection detection
- Input validation
- Personally Identifiable Information (PII) detection
- Automatic PII anonymization
- API rate limiting

### Performance Optimizations

- Redis response caching
- Reduced OpenAI API calls
- Lower latency for repeated queries

### Observability

- LangSmith tracing
- End-to-end request tracking
- Retrieval inspection
- Prompt monitoring
- LLM response monitoring

---

## System Architecture

```
                        +---------------------+
                        |      Client         |
                        +----------+----------+
                                   |
                                   v
                     +--------------------------+
                     |        FastAPI API       |
                     +----------+---------------+
                                |
               +----------------+----------------+
               |                |                |
               v                v                v

     Prompt Injection      PII Detection    Rate Limiting
        Protection       & Anonymization

                                |
                                v

                      +------------------+
                      |    Redis Cache   |
                      +--------+---------+
                               |
                Cache Hit -----+
                               |
                               v
                      +------------------+
                      |    Retriever     |
                      +--------+---------+
                               |
                               v
                      +------------------+
                      |     Pinecone     |
                      +--------+---------+
                               |
                               v
                      Relevant Context
                               |
                               v
                      +------------------+
                      |      OpenAI      |
                      +--------+---------+
                               |
                               v
                     Grounded Response

Observability Layer
---------------------------------
LangSmith Tracing
Structured Logging
```

---

## Technology Stack

### AI / LLM

- OpenAI GPT-4.1 Mini
- LangChain

### Vector Database

- Pinecone

### Embeddings

- Sentence Transformers
- all-MiniLM-L6-v2

### Backend

- FastAPI
- Pydantic

### Security

- Presidio
- Custom Prompt Injection Filters

### Caching

- Redis

### Observability

- LangSmith
- Python Logging

### Deployment

- Docker
- Docker Compose
- AWS EC2 (Planned)
- AWS ECR (Planned)
- GitHub Actions (Planned)

---

## API Endpoints

### Ask Question

```http
POST /ask
```

**Request**

```json
{
    "question": "What is RMSE?"
}
```

**Response**

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

**Response**

```json
{
    "status": "healthy"
}
```

---

## Chunking Strategy

The system uses **parent-child hierarchical chunking** for optimal retrieval and generation:

- **Child chunks** (small, precise) — embedded and indexed in Pinecone for high-accuracy semantic retrieval
- **Parent chunks** (larger, contextual) — stored as full document sections and returned as context to the LLM
- **Mapping** — each child chunk maps back to its parent, ensuring the LLM receives rich, coherent context while retrieval stays precise

This approach balances retrieval precision with generation quality, a pattern used in production RAG systems by companies like SAP and MongoDB.

---

## Caching Workflow

The system uses Redis to cache frequently requested responses.

```
User Question
      |
      v
   Redis

Cache Hit?
 |
 +-- Yes
 |     |
 |     v
 |  Return Cached Answer
 |
 +-- No
       |
       v
    Retrieve Context
       |
       v
      OpenAI
       |
       v
   Store In Redis
       |
       v
   Return Response
```

This significantly reduces:

- API costs
- Response latency
- Repeated retrieval operations

---

## Security Workflow

```
Incoming Request
       |
       v

Prompt Injection Detection
       |
       v

PII Detection
       |
       v

PII Anonymization
       |
       v

Rate Limiting
       |
       v

RAG Pipeline
```

Examples of protected inputs:

- Prompt injection attempts
- Email addresses
- Phone numbers
- Sensitive user information

---

## Observability

The application uses LangSmith for tracing critical components:

- Retrieval operations
- Context generation
- LLM calls
- Redis cache operations
- PII processing
- End-to-end RAG execution

This enables rapid debugging and performance monitoring.

---

## Running the Application

### Clone Repository

```bash
git clone <repository-url>
cd production-rag
```

### Configure Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=YOUR_KEY
PINECONE_API_KEY=YOUR_KEY

LANGCHAIN_API_KEY=YOUR_KEY
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=production-rag
```

### Run with Docker

```bash
docker compose up --build
```

### Access API Documentation

```
http://localhost:8000/docs
```

---

## Future Enhancements

- Hybrid Search (BM25 + Vector Search)
- CI/CD Pipeline with GitHub Actions
- AWS ECR Integration
- AWS EC2 Deployment
- Monitoring Dashboard
- Automated Testing Suite

---

## Skills Demonstrated

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Vector Databases
- Semantic Search
- API Development
- Caching Systems
- AI Security
- Observability & Monitoring
- Docker & Containerization
- Production AI System Design

---

## Author

### Nihal Siddiqui

AI/ML Engineer | Generative AI Enthusiast

Focused on building production-grade AI applications using modern MLOps and LLM engineering practices.
