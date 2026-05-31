import os
import re
import json
import asyncio
import subprocess
from typing import Dict, Any, List, Literal, TypedDict
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from apify_client import ApifyClient
from fastapi import FastAPI, BackgroundTasks
from tasks import ingest_and_embed_video_task
from deepgram import AsyncDeepgramClient
import tempfile
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi


import os
from pathlib import Path
from dotenv import load_dotenv
from deepgram import AsyncDeepgramClient

# ML Models
from faster_whisper import WhisperModel
from fastembed import TextEmbedding

# Vector DB
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

# LangGraph & LangChain
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq

# resolve absolute path to the .env file
env_path = Path(__file__).parent / ".env"

# 2. Force load
load_dotenv(dotenv_path=env_path, override=True)
raw_key = os.getenv("DEEPGRAM_API_KEY")

# 4. Fail loudly if missing
if not raw_key:
    raise ValueError(f"🚨 CRITICAL: Checked {env_path}, but DEEPGRAM_API_KEY is missing or empty!")

clean_key = raw_key.strip(' "\'')

# 6. Initialize client 
deepgram = AsyncDeepgramClient(api_key=clean_key)

load_dotenv()

app = FastAPI(title="Creator Analytics Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# Connect to dockerized qdrant Server over the network
qdrant_client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "creator_analytics"

embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

if not qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE) 
    )

METADATA_DATABASE: Dict[str, Dict[str, Any]] = {}
SESSION_STORAGE: Dict[str, List[Dict[str, str]]] = {}

#  ingestion, extraction   chumking

def clean_url_to_id(url: str, platform: str) -> str:
    if platform == "youtube":
        match = re.search(r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:[^/]+/.+/|(?:v|e(?:mbed)?)|watch\?v=|shorts/)|youtu\.be/)([^"&?/\s]{11})', url)
        return match.group(1) if match else "fallback_yt"
    else:
        match = re.search(r'/(?:reel|p|reels)/([A-Za-z0-9_-]+)', url)
        return match.group(1) if match else "fallback_ig"

from groq import Groq

# Initialize Groq client
groq_client = Groq()

apify_client = ApifyClient(os.getenv("APIFY_API_TOKEN"))

# Force load the .env file explicitly
load_dotenv()

# Grab the key and strip any hidden quotes/spaces
api_key = os.getenv("DEEPGRAM_API_KEY").strip(' "\'')

# Pass the key explicitly
deepgram = AsyncDeepgramClient(api_key=api_key)

async def extract_metadata_and_audio(url: str, platform: Literal["youtube", "instagram"]) -> Dict[str, Any]:
    video_id = clean_url_to_id(url, platform)
    audio_path = None
    
    if platform == "instagram":
        print(f"[IG]: Using Apify for Metadata + Deepgram for Transcription...")
        try:
            # === Apify for Metadata ===
            meta_run = await asyncio.to_thread(
                apify_client.actor("apify/instagram-scraper").call,
                run_input={"directUrls": [url]}
            )
            meta_id = getattr(meta_run, "default_dataset_id", None)
            meta_items = apify_client.dataset(meta_id).list_items().items
            real_meta = meta_items[0] if meta_items else {}

            # === Download Audio ===
            audio_path = await download_audio(url, video_id, platform)
            
            # === Deepgram Transcription ===
            transcript_data = await transcribe_with_deepgram(audio_path)
            
            # Clean up temp audio file after successful processing
            if audio_path and os.path.exists(audio_path):
                try: os.remove(audio_path)
                except: pass

            return {
                "video_id": video_id,
                "platform": "Instagram",
                "creator": real_meta.get("ownerUsername", "Unknown"),
                "follower_count": real_meta.get("ownerFollowersCount") or real_meta.get("owner", {}).get("followersCount"),
                "thumbnail_url": real_meta.get("displayUrl") or real_meta.get("thumbnailUrl"),
                "views": real_meta.get("videoViewCount") or real_meta.get("playCount") or 1,
                "likes": real_meta.get("likesCount", 0),
                "comments": real_meta.get("commentsCount", 0),
                "duration": real_meta.get("videoDuration", 0),
                "upload_date": real_meta.get("timestamp", "Unknown"),
                "hashtags": real_meta.get("hashtags", []),
                "apify_transcript_text": transcript_data["text"],
                "apify_transcript_segments": transcript_data["segments"],
                "metadata_unavailable": False
            }

        except Exception as e:
            print(f"[IG] Error: {str(e)}")
            # Safeguard cleanup on failure
            if audio_path and os.path.exists(audio_path):
                try: os.remove(audio_path)
                except: pass
            return {"video_id": video_id, "platform": "Instagram", "metadata_unavailable": True, "error": str(e)}

    elif platform == "youtube":
        print(f"[YT]: Using yt-dlp + Deepgram...")
        try:
            # Try native captions first (fastest & free)
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                for t in transcript_list:
                    caption_text = " ".join([item['text'] for item in t.fetch()])
                    if caption_text.strip():
                        # CHANGE HERE: Add await
                        return await build_youtube_metadata(video_id, caption_text, None)
            except Exception:
                print(f"[YT]: Native captions unavailable for {video_id}. Falling back to Deepgram...")

            # Download audio + Deepgram
            audio_path = await download_audio(url, video_id, platform)
            transcript_data = await transcribe_with_deepgram(audio_path)
            
            # Clean up temp audio file after successful processing
            if audio_path and os.path.exists(audio_path):
                try: os.remove(audio_path)
                except: pass

            return await build_youtube_metadata(video_id, transcript_data["text"], transcript_data["segments"])

        except Exception as e:
            print(f"[YT] Error: {str(e)}")
            # Safeguard cleanup on failure
            if audio_path and os.path.exists(audio_path):
                try: os.remove(audio_path)
                except: pass
            return {"video_id": video_id, "platform": "YouTube", "metadata_unavailable": True, "error": str(e)}
        
async def download_audio(url: str, video_id: str, platform: str) -> str:
    """Download audio using yt-dlp. Bypasses FFmpeg requirement by using native formats."""
    temp_dir = tempfile.gettempdir()
    
    ydl_opts = {
        'format': 'bestaudio/best',
        # Let yt-dlp determine the correct native file extension (e.g., .webm or .m4a)
        'outtmpl': f"{temp_dir}/{video_id}_{platform}.%(ext)s",
        'quiet': True,
        'no_warnings': True,
        # WE REMOVED THE FFMPEG POSTPROCESSOR!
    }
    
    def download():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Download and extract the info dictionary simultaneously
            info = ydl.extract_info(url, download=True)
            # Ask yt-dlp exactly what it named the final saved file
            return ydl.prepare_filename(info)
    
    # Run the download in a worker thread and capture the REAL path
    actual_audio_path = await asyncio.to_thread(download)
    return actual_audio_path

async def transcribe_with_deepgram(audio_path: str) -> Dict[str, Any]:
    """Transcribe using Deepgram Nova models with native v7 async support."""
    try:
        with open(audio_path, "rb") as audio_file:
            payload = audio_file.read()

        # Notice the 'await' keyword here. No thread pools needed anymore!
        response = await deepgram.listen.v1.media.transcribe_file(
            request=payload,
            model="nova-2",      
            smart_format=True,
            utterances=True,     
            punctuate=True,
            language="en",
        )

        # The response parsing remains exactly the same
        channel = response.results.channels[0].alternatives[0]
        
        segments = []
        for utterance in getattr(channel, 'utterances', []):
            segments.append({
                "start": utterance.start,
                "end": utterance.end,
                "text": utterance.transcript
            })

        return {
            "text": channel.transcript,
            "segments": segments or [{"start": 0.0, "end": 60.0, "text": channel.transcript}]
        }

    except Exception as e:
        print(f"Deepgram Runtime Error: {e}")
        return {"text": "Transcription failed.", "segments": []}
    
    

async def build_youtube_metadata(video_id: str, transcript_text: str, segments: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extracts rich YouTube metadata using yt-dlp safely via a separate thread
    and formats it to match the unified schema used by the downstream pipeline.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,  # Required to parse detailed metrics like subscriber/like counts
    }
    
    def fetch_info():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=False)
            
    try:
        # Offload network I/O out of the main async thread
        info = await asyncio.to_thread(fetch_info)
    except Exception as e:
        print(f"[YT Metadata Extraction Error]: {e}")
        info = {}

    # Normalize the YYYYMMDD string format from yt-dlp to YYYY-MM-DD
    raw_date = info.get("upload_date", "")
    upload_date = (
        f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}" 
        if isinstance(raw_date, str) and len(raw_date) == 8 
        else "Unknown"
    )

    # Calculate defensive duration fallback
    duration = float(info.get("duration", 0))

    # Match the unified payload schema precisely
    return {
        "video_id": video_id,
        "platform": "YouTube",
        "title": info.get("title", "Unknown"),
        "creator": info.get("uploader", "Unknown"),
        "follower_count": info.get("channel_follower_count") or info.get("subscriber_count") or 0,
        "thumbnail_url": info.get("thumbnail"),
        "views": info.get("view_count", 0),
        "likes": info.get("like_count", 0),
        "comments": info.get("comment_count", 0),
        "duration": duration,
        "upload_date": upload_date,
        "hashtags": info.get("tags", []),
        "apify_transcript_text": transcript_text,
        # Ensure segments never fallback to None; provide a bounded window if empty
        "apify_transcript_segments": segments or [{"start": 0.0, "end": duration or 60.0, "text": transcript_text}],
        "metadata_unavailable": False if info else True
    }



def transcribe_and_embed(item: dict, label: str):
    if item.get("metadata_unavailable"): 
        print(f"[{label}] Metadata unavailable, skipping.")
        return
        
    points = []
    aggregated_chunks = []
    
    #INSTAGRAM
    if item["platform"] == "Instagram" and item.get("apify_transcript_segments"):
        print(f"🧬 [{label}]: Using Apify Pre-Transcribed Segments...")
        segments = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in item["apify_transcript_segments"]]
        aggregated_chunks = aggregate_segments(segments, max_duration=15.0)
        
    elif item["platform"] == "Instagram" and item.get("apify_transcript_text"):
        print(f"🧬 [{label}]: Using Apify Pre-Transcribed Text/Caption...")
        text_payload = item["apify_transcript_text"]
        aggregated_chunks = [{"start": 0.0, "end": 60.0, "text": text_payload}]
        
    #YOUTUBE
    elif item["platform"] == "YouTube" and item.get("apify_transcript_text"):
        print(f"🧬 [{label}]: Using YouTube Native Transcript / Text Fallback...")
        text_payload = item["apify_transcript_text"]
        
        if text_payload.startswith("Title:"):
            # Split the title/description payload
            aggregated_chunks = [
                {"start": 0.0, "end": 15.0, "text": f"[Early Video Details] {text_payload[:300]}"},
                {"start": 15.0, "end": float(item.get("duration", 60.0)), "text": text_payload}
            ]
        else:
            aggregated_chunks = [{"start": 0.0, "end": float(item.get("duration", 60.0)), "text": text_payload}]
        
    # 4. Legacy fallback for Groq Audio
    elif item.get("audio_path") and os.path.exists(item["audio_path"]):
        print(f"⚡ [{label}]: Sending YouTube Audio to Groq Whisper API...")
        with open(item["audio_path"], "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(item["audio_path"], file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
        aggregated_chunks = aggregate_segments(transcription.segments, max_duration=15.0)
        try: os.remove(item["audio_path"])
        except: pass
    else:
        print(f"[{label}]: No transcript, caption, or audio found. Skipping embedding.")
        return

    # Unified Batch Embed and Upsert 
    valid_chunks = [c for c in aggregated_chunks if c.get("text", "").strip()]
    chunk_texts = [c["text"] for c in valid_chunks]
    
    if chunk_texts:
        all_vectors = list(embedding_model.embed(chunk_texts))
        for idx, chunk in enumerate(valid_chunks):
            vector = all_vectors[idx]
            points.append(
                models.PointStruct(
                    id=hash(f"{item['video_id']}_{idx}") & 0xFFFFFFFF,
                    vector=vector.tolist(),
                    payload={
                        "text": chunk["text"],
                        "video_label": label,
                        "video_id": item["video_id"],
                        "platform": item["platform"],
                        "start_time": round(chunk["start"], 2),
                        "end_time": round(chunk["end"], 2)
                    }
                )
            )
            
    if points: 
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"[{label}]: Successfully upserted {len(points)} chunks to Qdrant.")


def aggregate_segments(segments, max_duration=15.0):
    chunks, current_chunk = [], None
    for seg in segments:
        # Convert Groq segment to dict 
        start = seg["start"] if isinstance(seg, dict) else seg.start
        end = seg["end"] if isinstance(seg, dict) else seg.end
        
        raw_text = seg["text"] if isinstance(seg, dict) else seg.text
        clean_text = raw_text.strip().replace("\n", " ").replace("  ", " ")
        
        if not current_chunk:
            current_chunk = {"start": start, "end": end, "text": clean_text}
        elif end - current_chunk["start"] <= max_duration:
            current_chunk["end"] = end
            current_chunk["text"] += " " + clean_text
        else:
            chunks.append(current_chunk)
            current_chunk = {"start": start, "end": end, "text": clean_text}
            
    if current_chunk: chunks.append(current_chunk)
    return chunks

# LANGGRAPH ORCHESTRATION 

class ChatState(TypedDict):
    session_id: str
    video_a_id: str
    video_b_id: str
    messages: List[BaseMessage]
    context_str: str

def retrieve_node(state: dict):
    query = state["messages"][-1].content
    video_a_id = state["video_a_id"]
    video_b_id = state["video_b_id"]
    
    # fetch first 15s
    hook_results, _ = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="start_time", range=models.Range(lte=15.0))
            ],
            should=[
                models.FieldCondition(key="video_id", match=models.MatchValue(value=video_a_id)),
                models.FieldCondition(key="video_id", match=models.MatchValue(value=video_b_id))
            ]
        ),
        limit=10,
        with_payload=True,
        with_vectors=False
    )

    sorted_hooks = sorted(hook_results, key=lambda x: x.payload.get('start_time', 0))
    hook_chunks = [f"[{p.payload['platform']} HOOK | {p.payload['start_time']}s - {p.payload['end_time']}s]: {p.payload['text']}" for p in sorted_hooks]
    
    # SEMANTIC SEARCh
    query_vector = list(embedding_model.embed([query]))[0]
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=3
    )
    semantic_chunks = [f"[{r.payload['platform']} MATCH | {r.payload['start_time']}s - {r.payload['end_time']}s]: {r.payload['text']}" for r in search_results.points]
    
    # Merge 
    context_str = "--- GUARANTEED VIDEO HOOKS (First 15s) ---\n" + "\n".join(hook_chunks) + "\n\n--- RELEVANT SEMANTIC MATCHES ---\n" + "\n".join(semantic_chunks)
    return {"context_str": context_str}


# SEMANTIC ROUTER 

def route_query(state: dict) -> str:
    query = state["messages"][-1].content.lower()
    

    conversational_triggers = ["hi", "hello", "hey", "thanks", "thank you", "summarize", "who are you", "ok", "got it", "awesome"]
    
    if any(query.startswith(trigger) or query == trigger for trigger in conversational_triggers):
        print("🚦 [Router]: Conversational intent detected. Bypassing Vector DB...")
        return "bypass"
        
    print("🚦 [Router]: Analytical intent detected. Routing to Qdrant...")
    return "retrieve"

# GRAPH COMPILATION
workflow = StateGraph(ChatState)
workflow.add_node("retrieve", retrieve_node)

workflow.set_conditional_entry_point(
    route_query,
    {
        "retrieve": "retrieve", 
        "bypass": END    
    }
)

workflow.add_edge("retrieve", END)
agent_executor = workflow.compile()

#  API ROUTES

class IngestRequest(BaseModel):
    youtube_url: str
    instagram_url: str

class ChatRequest(BaseModel):
    message: str
    session_id: str
    video_a_id: str
    video_b_id: str


@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    #  checks the Redis queue for the Celery task status
    from tasks import celery_app
    task = celery_app.AsyncResult(task_id)
    
    if task.state == "SUCCESS":
# fetch the final metadata from Qdrant/Database
        return {"status": "SUCCESS", "metadata": task.result}
    
    return {"status": task.state}

@app.post("/api/clear")
async def clear_database():
    print("🧹 Wiping Qdrant Database and Session Storage...")
    try:

        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.Filter()
        )

        METADATA_DATABASE.clear()
        SESSION_STORAGE.clear()
        return {"status": "success", "message": "Database wiped clean."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/ingest")
async def ingest_pipeline(payload: IngestRequest):
    print(f"Starting Synchronous Ingestion...")
    yt_data, ig_data = await asyncio.gather(
        extract_metadata_and_audio(payload.youtube_url, "youtube"),
        extract_metadata_and_audio(payload.instagram_url, "instagram")
    )
    
    for item, label in [(yt_data, "A"), (ig_data, "B")]:
        if not item.get("metadata_unavailable"):
            v = item.get("views", 1)
            
            real_likes = 0 if item.get("likes") == -1 else item.get("likes", 0)
            item["engagement_rate"] = round(((real_likes + item.get("comments", 0)) / v) * 100, 2)
            
        METADATA_DATABASE[item["video_id"]] = item
        
        await asyncio.to_thread(transcribe_and_embed, item, label)
            
    print("Ingestion complete. Sending data back to UI.")
    
    # Return exactly the 'a' and 'b' keys
    return {
        "status": "success",
        "video_a_id": yt_data["video_id"],
        "video_b_id": ig_data["video_id"],
        "a": yt_data,
        "b": ig_data
    }
@app.post("/api/chat")
async def chat_interaction(req: ChatRequest):
    # Manage session history
    if req.session_id not in SESSION_STORAGE:
        SESSION_STORAGE[req.session_id] = []
        
    history = SESSION_STORAGE[req.session_id][-4:]
    messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in history]
    messages.append(HumanMessage(content=req.message))

    # Run Retrieval
    state = agent_executor.invoke({
        "session_id": req.session_id, 
        "video_a_id": req.video_a_id, 
        "video_b_id": req.video_b_id, 
        "messages": messages, 
        "context_str": ""
    })
    
    meta_a = METADATA_DATABASE.get(req.video_a_id, {})
    meta_b = METADATA_DATABASE.get(req.video_b_id, {})
    context = state.get("context_str", "")


# formatting strings
    def clean_metric(val):
        if isinstance(val, str):
            val = val.replace(",", "").replace(" ", "").strip()
            return int(val) if val.isdigit() else val
        return val

    clean_meta_a = {k: clean_metric(v) for k, v in meta_a.items()}
    clean_meta_b = {k: clean_metric(v) for k, v in meta_b.items()}

    # DYNAMIC PROMPT ROUTING
    if not context:
        # Conversational Mode
        system_prompt = (
            "You are a helpful social media co-pilot. The user is greeting you or chatting. "
            "Respond cordially, concisely, and ask how you can help them analyze their creative video data today."
        )
    else:
        # Elite Analyst Mode
        system_prompt = f"""You are an elite, highly critical social media data analyst. Base EVERYTHING strictly on the provided metadata and transcript context.

METADATA FOR VIDEO A (YouTube):
{json.dumps(clean_meta_a, indent=2)}

METADATA FOR VIDEO B (Instagram):
{json.dumps(clean_meta_b, indent=2)}

TRANSCRIPT CONTEXT:
{context}

CRITICAL RULES - VIOLATION NOT ALLOWED:
1. Calculate engagement rate EXACTLY as: ((likes + comments) / views) * 100. Round to 2 decimal places. Never invent different formulas.
2. NEVER repeat paragraphs. Be concise. Use bullet points.
3. If follower count is unavailable or null, say exactly: "Follower count unavailable".
4. For hooks: Clearly distinguish between spoken audio, captions, and textual metadata. Do not claim visual elements unless in transcript.
5. If the premise is wrong (e.g. "why A got more engagement" when B has higher ER), correct it immediately in the first sentence.
6. Do not invent CTAs, logos, or branding that are not explicitly in the transcript context.
"""

    final_messages = [SystemMessage(content=system_prompt)] + state["messages"]
    SESSION_STORAGE[req.session_id].append({"role": "user", "content": req.message})

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        model_kwargs={
            "presence_penalty": 0.5,     
            "frequency_penalty": 0.5,  
        }
    )
    
    async def generate():
        full_response = ""
        try:
            async for chunk in llm.astream(final_messages):
                if chunk.content:
                    full_response += chunk.content
                    yield f"data: {chunk.content}\n\n"

            SESSION_STORAGE[req.session_id].append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            print(f"LLM Generation Error: {str(e)}")
            yield f"data: \n\n[Error generating response: {str(e)}]\n\n"
            
        finally:
            yield "data: [DONE]\n\n"
            
    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)