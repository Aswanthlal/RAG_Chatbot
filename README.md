# Creator Analytics Engine: Omni-Channel RAG Pipeline

A full-stack analytical engine designed to ingest, process, and compare social media performance metrics and transcripts across YouTube and Instagram.

The system implements a Retrieval-Augmented Generation (RAG) architecture that enables creators to ask contextual questions regarding engagement, hook effectiveness, creator performance, and content strategy using real video metadata and transcript data.

---

## Features

* Ingest YouTube and Instagram video URLs
* Extract metadata and transcripts dynamically
* Compute engagement metrics automatically
* Chunk and embed transcript content
* Store vectors in Qdrant
* Perform retrieval-augmented analysis using LangGraph
* Stream responses in real time
* Maintain conversational memory across turns
* Compare content across platforms
* Provide source-aware answers with timestamp references

---

## Architecture Overview

```text
YouTube URL          Instagram URL
      │                     │
      └──────────┬──────────┘
                 ▼
      Metadata + Transcript Layer
                 │
                 ▼
        Chunking + Embedding
                 │
                 ▼
              Qdrant
                 │
                 ▼
            LangGraph
                 │
                 ▼
          Groq Llama 3.1
                 │
                 ▼
         Next.js Frontend
```

---

## Tech Stack

### Frontend

* Next.js
* React
* Tailwind CSS
* Server-Sent Events (SSE)

### Backend

* FastAPI
* AsyncIO

### Retrieval & AI

* LangGraph
* FastEmbed
* BAAI/bge-small-en-v1.5
* Qdrant
* Groq (Llama 3.1 8B Instant)

### Transcript Processing

* yt-dlp
* Deepgram Nova-2
* Apify Instagram Services

---

## Architectural Decisions & Scalability Analysis

The architecture was designed around a theoretical workload of approximately:

```text
1,000 creators/day
2,000 videos/day
```

while maintaining low infrastructure costs.

### 1. Audio Ingestion Strategy

#### Problem

Traditional yt-dlp pipelines often rely on FFmpeg transcoding before transcription.

This introduces:

* Additional CPU load
* Disk I/O overhead
* Increased processing latency

#### Solution

The pipeline downloads native audio streams directly and forwards them to Deepgram without local transcoding.

#### Result

* Reduced ingestion latency
* No FFmpeg dependency
* Simplified deployment

---

### 2. Why BGE-Small?

#### Local Execution

Embeddings are generated locally using FastEmbed.

#### Cost Efficiency

No per-request embedding API costs.

#### Resource Efficiency

The model offers strong semantic retrieval quality while maintaining a small memory footprint suitable for standard server deployments.

---

### 3. Why Qdrant?

#### Payload Filtering

Every transcript chunk stores:

* video_id
* platform
* start_time
* end_time

This enables filtered retrieval before vector search.

#### Reduced Search Space

Filtering limits retrieval to relevant video chunks before similarity search.

#### Persistence

The implementation uses:

```python
QdrantClient(path="./qdrant_storage")
```

allowing local persistence across server restarts.

#### Migration Path

The same architecture can be migrated to Qdrant Cloud with minimal configuration changes.

---

### 4. Why LangGraph?

LangGraph was selected instead of a simple linear chain because the system contains multiple execution paths.

#### Conditional Routing

Conversational requests:

```text
Hello
Thanks
Good job
```

bypass retrieval entirely.

Analytical requests:

```text
Compare hooks
Why did Video B perform better?
```

trigger vector retrieval.

This reduces unnecessary database lookups and token consumption.

---

## Graceful Degradation Strategy

Social platforms frequently impose restrictions on metadata access.

The pipeline is designed to continue operating even when partial data becomes unavailable.

### YouTube

1. Transcript API (when available)
2. yt-dlp + Deepgram fallback

### Instagram

1. Apify transcript extraction
2. Caption fallback

### Final Fallback

If transcript extraction fails entirely, the system converts available metadata and descriptions into a structured text block so that:

* Engagement calculations remain available
* Metadata questions still function
* The vector pipeline remains operational

---

## Known Limitations

### Instagram Restrictions

Instagram may hide:

* Follower counts
* Like counts
* Other engagement metrics

depending on platform restrictions and scraper availability.


## Environment Variables

Create a `.env` file inside the backend directory.

```env
DEEPGRAM_API_KEY=your_deepgram_api_key
GROQ_API_KEY=your_groq_api_key
APIFY_API_TOKEN=your_apify_api_token
```

---

## Backend Setup

```bash
cd rag-handler-backend

python -m venv venv311

# Linux / Mac
source venv311/bin/activate

# Windows
venv311\Scripts\activate

pip install -r requirements.txt

uvicorn main:app --reload --port 9000
```

The application automatically initializes:

```text
./qdrant_storage
```

for persistent local vector storage.

---

## Frontend Setup

```bash
cd rag-handler-frontend

npm install

npm run dev
```

Open:

```text
http://localhost:3000
```

---