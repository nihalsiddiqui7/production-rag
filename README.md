# Production RAG

> A source-grounded question-answering API for machine-learning knowledge, built with FastAPI, Pinecone, parent-child retrieval, OpenAI, Redis, Presidio, LangSmith, Docker, and Ragas.

![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![Pinecone](https://img.shields.io/badge/Pinecone-vector%20db-3B82F6)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)
![License](https://img.shields.io/badge/license-MIT-green)

This project takes a RAG prototype and hardens it into a service with explicit retrieval design, evaluation, security controls, caching, observability, and operational boundaries. It answers questions from *Hands-On Machine Learning with Scikit-Learn and TensorFlow* without asking the model to rely on its general training knowledge: the generation prompt requires answers to come only from retrieved context, and the system returns a controlled refusal when the documents do not contain the answer.

**Measured quality (in-domain, Ragas):** faithfulness `0.963` - answer relevancy `0.894` - context recall `1.000`. Hybrid retrieval was implemented, A/B tested, and deliberately kept behind a flag after it lost on faithfulness and recall. This README treats what was measured, not what sounds impressive.

## Table of Contents

- [What this is](#what-this-is)
- [Why this project](#why-this-project)
- [Architecture](#architecture)
- [Real output example](#real-output-example)
- [Core design decisions](#core-design-decisions)
- [API](#api)
- [Evaluation](#evaluation)
- [Local setup](#local-setup)
- [Delivery workflow](#delivery-workflow)
- [Document ingestion](#document-ingestion)
- [Project layout](#project-layout)
- [Operational verification](#operational-verification)
- [Known limitations and next improvements](#known-limitations-and-next-improvements)
- [Skills demonstrated](#skills-demonstrated)
- [Author](#author)

## What this is

The system answers questions about machine learning with answers grounded in a 564-page textbook:

1. The relevant passages are **retrieved** from a Pinecone vector index.
2. The passages are fed to `gpt-4.1-mini` with a grounding prompt.
3. The API returns the **answer plus the exact pages** it was drawn from.

The production wrapper around that loop is the point of this repository:

- FastAPI service with validated request/response contracts
- Parent-child retrieval that separates search granularity from generation context
- Prompt-injection defense and Presidio-based PII anonymization tuned for ML jargon
- Redis response caching with a one-hour TTL
- LangSmith tracing across the whole pipeline
- Ragas-based evaluation with timestamped, per-question reports
- Docker Compose packaging and GitHub Actions CI/CD to Amazon ECR

## Why this project

Many RAG demos stop at:

```text
PDF -> chunks -> embeddings -> vector search -> LLM
```

This project focuses on the production questions that appear after the demo works:

- How do we preserve enough context without sending entire documents to the LLM?
- How do we return citations a user can inspect?
- What happens when the user sends an injection attempt or PII?
- How do we avoid paying for repeated questions?
- How do we observe retrieval, generation, and cache behavior?
- How do we prove that a retrieval change improved the system instead of assuming it did?

## Architecture

```text
                         Client
                           |
                           v
                    FastAPI /ask
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
    Input validation   Injection check   Rate limit (10/min)
          |
          v
    Presidio PII anonymization
          |
          v
    Redis cache lookup -------- cache hit ------> response
          |
       cache miss
          v
    Parent-child retriever
          |
          +--> Dense: child vectors in Pinecone -> deduplicated parents
          |
          +--> Optional hybrid: dense + BM25 -> weighted RRF -> top 3
          |
          v
    Grounded prompt with retrieved context
          |
          v
    OpenAI gpt-4.1-mini (temperature 0)
          |
          v
    Answer + page/title/parent_id sources

    LangSmith traces the request and major pipeline operations.
```

## Real output example

The following is verbatim API output captured from a live run. The answer is truncated and the source list is abbreviated for readability.

```json
{
  "question": "What is RMSE?",
  "answer": "RMSE (Root Mean Square Error) is a typical performance measure for regression problems that measures the standard deviation of the errors the system makes in its predictions. It corresponds to the Euclidean norm (l2 norm) and is calculated as the square root of the average of the squared differences between the predicted values and the actual values. For example, an RMSE equal to 50,000 means that about 68% of the system's predictions fall within $50,000 of the actual value, and about 95% fall within $100,000 of the actual value. ...",
  "sources": [
    {
      "page": 58,
      "title": "Hands-On Machine Learning with Scikit-Learn and TensorFlow",
      "parent_id": "294c0838-..."
    },
    {
      "page": 60,
      "title": "Hands-On Machine Learning with Scikit-Learn and TensorFlow",
      "parent_id": "012dc144-..."
    },
    {
      "page": 128,
      "title": "Hands-On Machine Learning with Scikit-Learn and TensorFlow",
      "parent_id": "79a6ef79-..."
    }
  ]
}
```

The live response contained seven deduplicated source parents across pages 58, 59, 60, 89, and 128. Each source entry carries the page, the document title, and the parent chunk ID, so every claim in an answer can be traced back to the document.

## Core design decisions

### Parent-child retrieval

Ingestion creates two representations of the source document:

1. Smaller **child chunks** (400 chars) are embedded and indexed in Pinecone for high-recall semantic search.
2. Larger **parent chunks** (1600 chars) are stored in `parent_store.json` and returned to the LLM for richer context.

At query time, the retriever searches children, deduplicates results by `parent_id`, and resolves them back to parent text. This separates search granularity from generation context size instead of forcing one chunk size to serve both jobs. It is the same strategy used by LlamaIndex and production RAG systems.

### Dense retrieval is the production default

The production default is semantic parent-child retrieval. Hybrid retrieval remains available through an environment switch:

```env
RETRIEVAL_MODE=dense    # default
RETRIEVAL_MODE=hybrid   # dense + BM25 + weighted RRF
```

Hybrid retrieval was evaluated rather than accepted on intuition. On the recorded in-domain comparison:

| Metric | Dense | Hybrid |
| --- | ---: | ---: |
| Faithfulness | **0.963** | 0.821 |
| Answer relevancy | **0.894** | 0.858 |
| Context precision | 0.739 | **0.769** |
| Context recall | **1.000** | 0.909 |

Hybrid improved context precision slightly, but BM25 also promoted noisy index/reference pages for queries such as "bagging and boosting". That demoted useful explanatory context and caused a measurable faithfulness and recall regression. Dense therefore remains the safe default, while hybrid is isolated behind a flag for further tuning.

This is an intentional engineering decision: retrieval quality is measured per metric, failure cases are inspected, and the more complex approach is not shipped merely because it sounds more advanced.

The two modes also bound context differently:

- **Dense** returns deduplicated parents bounded by the child search width (`k=10`).
- **Hybrid** caps the fused result at `MAX_PARENTS=3` before prompt assembly.

### Grounded generation

The generation prompt instructs the model to:

- use only retrieved context;
- avoid unsupported claims;
- return `I don't know based on the provided documents.` when context is insufficient.

The API returns both the answer and source metadata, including page number, document title, and parent chunk ID.

### Cost and latency controls

- Redis caches sanitized questions and responses for one hour.
- Cache hits skip both retrieval and the OpenAI generation call.
- Retrieval width and parent caps keep prompt size bounded.
- Evaluation can be restricted to the in-domain subset instead of spending tokens on refusal-only cases.

### Security boundaries

- Pydantic rejects questions shorter than 3 or longer than 1,000 characters.
- Prompt injection patterns are rejected with HTTP 400.
- Presidio anonymizes high-risk PII before caching, tracing, retrieval, and generation.
- ML terminology such as `Adam`, `BERT`, `GPT`, and `LSTM` is whitelisted to avoid damaging retrieval through false-positive PERSON detection.
- The `/ask` endpoint is limited to 10 requests per minute.
- Unexpected exceptions are logged server-side and returned as a generic HTTP 500 response.

## API

### `GET /health`

Response:

```json
{
  "status": "healthy"
}
```

### `POST /ask`

Request:

```json
{
  "question": "What is RMSE?"
}
```

Response:

```json
{
  "answer": "RMSE is ...",
  "sources": [
    {
      "page": 59,
      "title": "Hands-On Machine Learning with Scikit-Learn and TensorFlow",
      "parent_id": "..."
    }
  ]
}
```

Interactive API documentation is available at `http://localhost:8000/docs`.

## Evaluation

The evaluation harness uses Ragas and records per-question CSV reports under `reports/`.

Metrics:

- **Faithfulness**: whether the response is supported by retrieved context.
- **Answer relevancy**: whether the response addresses the question.
- **Context precision**: whether retrieved context is relevant rather than noise.
- **Context recall**: whether the retrieved context contains the information needed for the reference answer.

Run the dense baseline:

```powershell
$env:RETRIEVAL_MODE="dense"
venv\Scripts\python.exe -m src.evaluation.run_ragas dense in_domain
```

Run the hybrid comparison:

```powershell
$env:RETRIEVAL_MODE="hybrid"
venv\Scripts\python.exe -m src.evaluation.run_ragas hybrid in_domain
```

The `in_domain` argument focuses the comparison on questions with document-grounded reference answers and skips questions that expect a controlled refusal. Reports are timestamped, for example:

```text
reports/ragas_report_dense_<timestamp>.csv
reports/ragas_report_hybrid_<timestamp>.csv
```

The evaluation uses `gpt-4.1-mini` as both the application generator and Ragas judge. This keeps the setup reproducible and inexpensive, but it is also a known limitation: an independent judge model would provide stronger validation for a production benchmark.

## Local setup

### Prerequisites

- Python 3.11
- A Pinecone account and index access
- An OpenAI API key
- Docker Desktop, if running Redis through Compose

### Install

```powershell
git clone <repository-url>
cd production-rag

python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env` from `.env.example` and provide real credentials:

```env
OPENAI_API_KEY=...
PINECONE_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=production-rag
REDIS_HOST=localhost
```

### Run Redis and the API locally

```powershell
docker run -d --name production-rag-redis -p 6379:6379 redis:7-alpine
venv\Scripts\python.exe -m uvicorn src.api.main:app --reload --port 8000
```

### Run the complete stack with Docker Compose

```powershell
docker compose up --build
```

Compose starts the API on port `8000` and Redis on port `6379`. Inside Compose, use `REDIS_HOST=redis`.

## Delivery workflow

GitHub Actions provides two repository workflows on the `deploy` branch:

- `docker-build.yml` creates a safe example `.env`, validates the Compose configuration, and builds the stack on pushes and pull requests.
- `deploy.yml` authenticates to AWS, builds the API image, tags it, and publishes it to Amazon ECR.

The current automation publishes the image to ECR; it does not claim to provision or restart an EC2/ECS service. That boundary is documented intentionally rather than presenting image publishing as full infrastructure deployment.

## Document ingestion

The repository expects the source PDF under the configured documents directory. The ingestion script:

1. loads and cleans the PDF;
2. creates parent and child chunks;
3. writes the parent lookup store to `parent_store.json`;
4. creates or reuses Pinecone index `ml-chatbot-v2` with 384-dimensional cosine vectors;
5. uploads the child chunks with `all-MiniLM-L6-v2` embeddings.

Run ingestion when rebuilding the index:

```powershell
venv\Scripts\python.exe store_index.py
```

The runtime needs both the Pinecone child index and the matching `parent_store.json`. A mismatch between those artifacts can produce missing or incorrect source resolution.

## Project layout

```text
.
|-- src/
|   |-- api/
|   |   |-- main.py          FastAPI application and exception handlers
|   |   |-- routes.py        /health and /ask request lifecycle
|   |   |-- schemas.py       Request/response validation
|   |   |-- security.py      Prompt injection checks
|   |   |-- pii.py           Presidio anonymization and ML whitelist
|   |   |-- cache.py         Redis response cache
|   |   `-- rate_limiter.py  SlowAPI limiter
|   |-- evaluation/
|   |   |-- questions.json   Reference evaluation set
|   |   `-- run_ragas.py     Ragas evaluation and timestamped reports
|   |-- rag_chain.py         Retrieval, grounded prompt, generation, sources
|   |-- retriever.py         Dense parent-child and optional hybrid retrieval
|   `-- prompt.py            Grounding and refusal instructions
|-- store_index.py           PDF ingestion and Pinecone indexing
|-- parent_store.json        Runtime parent-chunk lookup store
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
`-- .env.example
```

## Operational verification

The API has been smoke-tested for:

- healthy response from `/health`;
- validation rejection for undersized questions;
- HTTP 400 for prompt injection patterns;
- successful grounded `/ask` responses;
- HTTP 429 after the configured rate limit is exceeded;
- parent store loading and retriever initialization;
- valid Docker Compose configuration.

The Ragas reports provide the quality comparison; smoke tests provide service-level confidence. These are deliberately treated as different types of verification.

## Known limitations and next improvements

This repository is production-oriented, not a claim that every production concern is complete.

- Pinecone and OpenAI are external dependencies, so availability and cost controls are still required at deployment time.
- The evaluation set is small and hand-authored; a larger stratified benchmark would make regressions more statistically meaningful.
- Ragas currently uses the same model family for generation and judging; an independent judge should be added for stronger evaluation.
- Hybrid search needs BM25 corpus filtering or better weighting before it should replace dense retrieval.
- Cross-encoder reranking is a logical next experiment, but it should be accepted only after another measured A/B comparison.
- Authentication, tenant isolation, secret management, and distributed rate limiting would be required for a public multi-user deployment.
- The current Redis key is the sanitized question; a production deployment should include model, prompt, retriever, and document-version identifiers in the cache key.

## Skills demonstrated

- RAG system design beyond a basic vector-search demo
- Parent-child chunking and source resolution
- Pinecone vector search and embedding pipelines
- Retrieval evaluation with Ragas
- Evidence-based retriever selection using A/B metrics
- FastAPI API design and Pydantic contracts
- Prompt-injection defenses and PII anonymization
- Redis caching for cost and latency control
- LangSmith tracing for pipeline observability
- Docker and Docker Compose service packaging
- Failure analysis and explicit production tradeoffs

## Author

**Nihal Siddiqui**

AI and machine-learning engineer focused on reliable, observable, and production-oriented generative AI systems.
