# Creator Analytics Engine: Omni-Channel RAG Pipeline

A full-stack Retrieval-Augmented Generation (RAG) system designed to ingest, process, and compare social media performance across YouTube and Instagram.

The platform enables creators to analyze engagement metrics, compare content strategies, evaluate hook effectiveness, and generate evidence-based recommendations using transcript retrieval, vector search, and LLM reasoning.

---

# Features

* Ingest YouTube and Instagram video URLs
* Extract video metadata dynamically
* Generate transcripts using Deepgram
* Compute engagement rates automatically
* Chunk and embed transcript content
* Store vectors in Qdrant
* Retrieval-Augmented Generation (RAG)
* LangGraph-powered query routing
* Real-time streaming responses
* Session-based conversational memory
* Source-aware answers with timestamps
* Side-by-side creator comparison interface

---

# Architecture Overview

```text
YouTube URL          Instagram URL
      │                     │
      └──────────┬──────────┘
                 ▼
      Metadata + Transcript Layer
      (YT API / Apify / Deepgram)
                 │
                 ▼
        Chunking + FastEmbed
                 │
                 ▼
          Qdrant Vector DB
                 │
                 ▼
         LangGraph Router
                 │
                 ▼
        Groq Llama 3.1 8B
                 │
                 ▼
          Next.js Frontend
```

---

# Tech Stack

## Frontend

* Next.js
* React
* Tailwind CSS
* Server-Sent Events (SSE)

## Backend

* FastAPI
* AsyncIO

## AI & Retrieval

* LangGraph
* FastEmbed
* BAAI/bge-small-en-v1.5
* Qdrant
* Groq (Llama 3.1 8B Instant)

## Data Extraction

* yt-dlp
* Deepgram Nova-2
* Apify

## Infrastructure

* Docker
* Redis (optional)
* Celery (background worker prototype)

---

# Key Architectural Decisions

## 1. Audio Ingestion Strategy

### Problem

Traditional yt-dlp pipelines often rely on FFmpeg transcoding before transcription.

This introduces:

* CPU overhead
* Disk I/O overhead
* Additional latency

### Solution

The system downloads native audio streams directly and forwards them to Deepgram without local transcoding.

### Result

* Reduced ingestion latency
* No FFmpeg dependency
* Simplified deployment

---

## 2. Why BGE-Small?

### Local Execution

Embeddings are generated locally using FastEmbed.

### Cost Efficiency

No recurring embedding API costs.

### Resource Efficiency

Low memory footprint while maintaining strong retrieval quality for social-media transcripts.

---

## 3. Why Qdrant?

### Payload Filtering

Every chunk stores:

* video_id
* platform
* start_time
* end_time

allowing filtered retrieval before similarity search.

### Predictable Retrieval

Pre-filtering reduces search scope and improves retrieval efficiency as data volume grows.

### Deployment 

Qdrant is deployed locally through Docker and accessed via:

```python
QdrantClient(url="http://localhost:6333")
```

### Migration Path

The same architecture can migrate directly to Qdrant Cloud with minimal code changes.

---

## 4. Why LangGraph?

LangGraph enables conditional execution paths rather than a single linear chain.

### Conversational Requests

Examples:

```text
Hi
Hello
Thanks
```

These bypass retrieval entirely.

### Analytical Requests

Examples:

```text
Compare hooks
Why did Video B perform better?
```

These trigger vector retrieval and analysis.

This reduces unnecessary vector searches and token consumption.

---

# Graceful Degradation Strategy

Social media platforms frequently restrict metadata visibility and transcript availability.

The system is designed to continue functioning when partial data becomes unavailable.

## YouTube

1. Native transcript extraction
2. Deepgram fallback

## Instagram

1. Apify extraction
2. Caption fallback

## Final Fallback

If transcript extraction fails entirely, available metadata and descriptions are converted into structured text blocks so the RAG pipeline remains operational.

This ensures:

* Engagement calculations still work
* Metadata questions remain answerable
* Vector retrieval remains functional

---

# Known Limitations

## Instagram Restrictions

Instagram may hide:

* Follower counts
* Like counts
* Other engagement metrics

depending on platform restrictions and scraper availability.

## Transcript Dependency

Retrieval quality depends on transcript quality.

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/Aswanthlal/rag-handler.git

cd rag-handler
```

---

## 2. Configure Environment Variables

Create:

```text
rag-handler-backend/.env
```

Add:

```env
DEEPGRAM_API_KEY=your_deepgram_api_key
GROQ_API_KEY=your_groq_api_key
APIFY_API_TOKEN=your_apify_api_token
```

---

## 3. Start Infrastructure

```bash
docker compose up -d
```

This starts:

* Qdrant
* Redis

---

## 4. Start Backend

```bash
cd rag-handler-backend

python -m venv venv311

# Windows
venv311\Scripts\activate

pip install -r requirements.txt

uvicorn main:app --reload --port 9000
```

---

## 5. Start Frontend

```bash
cd rag-handler-frontend

npm install

npm run dev
```

---

## 6. Open Application

```text
http://localhost:3000
```
