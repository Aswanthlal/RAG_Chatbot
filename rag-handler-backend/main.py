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

# Real Open-Source ML Models
from faster_whisper import WhisperModel
from fastembed import TextEmbedding

# Vector DB
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

# LangGraph & LangChain (Mandatory Tech Stack)
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

app = FastAPI(title="Creator Analytics Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# -------------------------------------------------------------------------
# 1. LOCAL MODELS & PERSISTENT DB
# -------------------------------------------------------------------------
qdrant_client = QdrantClient(path="./qdrant_storage")
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

# -------------------------------------------------------------------------
# 2. INGESTION, EXTRACTION & SMART CHUNKING
# -------------------------------------------------------------------------
def clean_url_to_id(url: str, platform: str) -> str:
    if platform == "youtube":
        match = re.search(r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:[^/]+/.+/|(?:v|e(?:mbed)?)|watch\?v=|shorts/)|youtu\.be/)([^"&?/\s]{11})', url)
        return match.group(1) if match else "fallback_yt"
    else:
        match = re.search(r'/(?:reel|p|reels)/([A-Za-z0-9_-]+)', url)
        return match.group(1) if match else "fallback_ig"

from groq import Groq

# Initialize Groq client (reads from GROQ_API_KEY in your .env)
groq_client = Groq()

apify_client = ApifyClient(os.getenv("APIFY_API_TOKEN"))

async def extract_metadata_and_audio(url: str, platform: Literal["youtube", "instagram"]) -> Dict[str, Any]:
    video_id = clean_url_to_id(url, platform)
    
    # ---------------------------------------------------------
    # INSTAGRAM: Apify Microservices
    # ---------------------------------------------------------
    if platform == "instagram":
        print(f"🕵️‍♂️ [IG]: Fanning out to Apify Microservices (Metadata + Transcript)...")
        try:
            # Task 1: Official Scraper (Fast, reliable)
            meta_task = asyncio.to_thread(
                apify_client.actor("apify/instagram-scraper").call, 
                run_input={"directUrls": [url]}
            )
            
            # Task 2: Transcript Scraper (Prone to Meta blocks, adding 20s timeout)
            transcript_task = asyncio.to_thread(
                apify_client.actor("apple_yang/instagram-transcripts-scraper").call, 
                run_input={"videoUrl": url},
                timeout_secs=20
            )
            
            # Run concurrently. return_exceptions=True prevents one failure from killing the other!
            results = await asyncio.gather(meta_task, transcript_task, return_exceptions=True)
            meta_run, transcript_run = results[0], results[1]
            
            # 1. Parse Metadata (Mandatory)
            if isinstance(meta_run, Exception):
                raise meta_run # If the main scraper fails, we must abort
            
            meta_id = getattr(meta_run, "default_dataset_id", None)
            meta_items = apify_client.dataset(meta_id).list_items().items
            if not meta_items: raise Exception("Apify Metadata Scraper returned empty.")
            real_meta = meta_items[0]
            
            # 2. Parse Transcript (Optional / Fallback)
            trans_data = {}
            if not isinstance(transcript_run, Exception):
                trans_id = getattr(transcript_run, "default_dataset_id", None)
                if trans_id:
                    trans_items = apify_client.dataset(trans_id).list_items().items
                    if trans_items:
                        trans_data = trans_items[0]
            else:
                print(f" [IG]: Transcript scraper timed out/blocked. Falling back to caption.")

            # Calculate Fallbacks safely
            final_text = trans_data.get("text") or real_meta.get("caption", "No transcript available.")
            real_views = real_meta.get("videoViewCount") or real_meta.get("playCount") or real_meta.get("viewCount") or 1
            owner_node = real_meta.get("owner", {})
            follower_count = (
                real_meta.get("ownerFollowersCount") or 
                owner_node.get("followersCount") or 
                owner_node.get("edge_followed_by", {}).get("count") or 
                real_meta.get("user", {}).get("follower_count") or 
                0  # Default to 0 so the UI knows it was blocked
            )
            
            return {
                "video_id": video_id,
                "platform": "Instagram",
                "creator": real_meta.get("ownerUsername", trans_data.get("userName", "Unknown")),
                "follower_count": follower_count,
                "views": real_views, 
                "likes": real_meta.get("likesCount", trans_data.get("likeCount", 0)),
                "comments": real_meta.get("commentsCount", trans_data.get("commentCount", 0)),
                "duration": real_meta.get("videoDuration", trans_data.get("duration", 0)),
                "upload_date": real_meta.get("timestamp", "Unknown"), 
                "hashtags": real_meta.get("hashtags", []), 
                "audio_path": None, 
                "apify_transcript_text": final_text, 
                "apify_transcript_segments": trans_data.get("segments", []), 
                "metadata_unavailable": False
            }
        except Exception as e:
            print(f"APIFY ERROR: {e}")
            return {"video_id": video_id, "platform": "Instagram", "metadata_unavailable": True, "error": str(e)}
    # ---------------------------------------------------------
    # YOUTUBE: yt-dlp 
    # ---------------------------------------------------------
    elif platform == "youtube":
        output_audio_path = f"temp_{video_id}.mp3"
        ydl_cmd = ["yt-dlp", "--skip-download", "--dump-json", "--no-warnings", url]
        
        try:
            process = await asyncio.create_subprocess_exec(*ydl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, _ = await process.communicate()
            if process.returncode != 0: raise Exception("yt-dlp metadata extraction failed")
                
            import json
            info = json.loads(stdout.decode())
            
            audio_cmd = ["yt-dlp", "-x", "--audio-format", "mp3", "--audio-quality", "0", "-o", output_audio_path, url]
            audio_process = await asyncio.create_subprocess_exec(*audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            await audio_process.communicate()
            
            return {
                "video_id": video_id,
                "platform": "YouTube",
                "creator": info.get("uploader", "Unknown"),
                "follower_count": info.get("channel_follower_count", 0), # This IS the subscriber count!
                "views": info.get("view_count", 0) or 1,
                "likes": info.get("like_count", 0) or 0,
                "comments": info.get("comment_count", 0) or 0,
                "duration": info.get("duration", 0), # NEW: Video length in seconds
                "upload_date": info.get("upload_date", "Unknown"), # NEW: Format YYYYMMDD
                "hashtags": info.get("tags", []), # NEW: Array of video tags
                "audio_path": output_audio_path,
                "apify_transcript_text": "",
                "apify_transcript_segments": [],
                "metadata_unavailable": False
            }
        except Exception as e:
            return {"video_id": video_id, "platform": "YouTube", "metadata_unavailable": True, "error": str(e)}
            
    return {"video_id": video_id, "platform": platform, "metadata_unavailable": True, "error": "Unknown Platform"}
    

def transcribe_and_embed(item: dict, label: str):
    if item.get("metadata_unavailable"): 
        print(f"[{label}] Metadata unavailable, skipping.")
        return
        
    points = []
    aggregated_chunks = [] # Initialize here so the batcher below always has a list
    
    # 1. IF IT'S INSTAGRAM (Apify returned precise timestamp segments)
    if item["platform"] == "Instagram" and item.get("apify_transcript_segments"):
        print(f"🧬 [{label}]: Using Apify Pre-Transcribed Segments...")
        segments = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in item["apify_transcript_segments"]]
        aggregated_chunks = aggregate_segments(segments, max_duration=15.0)
        
    # 2. IF IT'S INSTAGRAM (Apify returned flat text/caption without timestamps)
    elif item["platform"] == "Instagram" and item.get("apify_transcript_text"):
        print(f"🧬 [{label}]: Using Apify Pre-Transcribed Text/Caption...")
        text_payload = item["apify_transcript_text"]
        # Wrap it in a single chunk so it feeds perfectly into the batcher below
        aggregated_chunks = [{"start": 0.0, "end": 60.0, "text": text_payload}]
        
    # 3. IF IT'S YOUTUBE (We use Groq API on the audio file)
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

    # ---------------------------------------------------------
    # Unified Batch Embed and Upsert 
    # (Handles Groq chunks, Apify chunks, AND Apify flat text)
    # ---------------------------------------------------------
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
        print(f"[{label}]: Successfully upserted {len(points)} chunks to Qdrant.")# Fix the aggregate_segments helper to handle dictionaries from Groq


def aggregate_segments(segments, max_duration=15.0):
    chunks, current_chunk = [], None
    for seg in segments:
        # Convert Groq segment to dict if it isn't one already
        start = seg["start"] if isinstance(seg, dict) else seg.start
        end = seg["end"] if isinstance(seg, dict) else seg.end
        text = seg["text"] if isinstance(seg, dict) else seg.text
        
        if not current_chunk:
            current_chunk = {"start": start, "end": end, "text": text.strip()}
        elif end - current_chunk["start"] <= max_duration:
            current_chunk["end"] = end
            current_chunk["text"] += " " + text.strip()
        else:
            chunks.append(current_chunk)
            current_chunk = {"start": start, "end": end, "text": text.strip()}
    if current_chunk: chunks.append(current_chunk)
    return chunks

# -------------------------------------------------------------------------
# 3. LANGGRAPH ORCHESTRATION 
# -------------------------------------------------------------------------
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
    
    # 1. THE GUARANTEED HOOKS: Always fetch the first 15 seconds of both videos
    hook_results, _ = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="start_time",
                    range=models.Range(lte=15.0) # Anything under 15 seconds
                )
            ]
        ),
        limit=20,
        with_payload=True,
        with_vectors=False
    )
    
    # Sort chronologically so the LLM reads it naturally
    sorted_hooks = sorted(hook_results, key=lambda x: (x.payload.get('platform', ''), x.payload.get('start_time', 0)))
    hook_chunks = [f"[{p.payload['platform']} HOOK | {p.payload['start_time']}s - {p.payload['end_time']}s]: {p.payload['text']}" for p in sorted_hooks]
    
    # 2. SEMANTIC SEARCH: Look for specific contextual answers to the user's prompt
    query_vector = list(embedding_model.embed([query]))[0]
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=3 # Keep this small so we don't overflow the context window
    )
    semantic_chunks = [f"[{r.payload['platform']} MATCH | {r.payload['start_time']}s - {r.payload['end_time']}s]: {r.payload['text']}" for r in search_results.points]
    
    # Merge them into a bulletproof context block
    context_str = "--- GUARANTEED VIDEO HOOKS (First 15s) ---\n" + "\n".join(hook_chunks) + "\n\n--- RELEVANT SEMANTIC MATCHES ---\n" + "\n".join(semantic_chunks)
    
    # FIX: Ensure this key matches your ChatState dictionary exactly ('context_str')
    return {"context_str": context_str}


# ---------------------------------------------------------
# THE SEMANTIC ROUTER (Gatekeeper)
# ---------------------------------------------------------
def route_query(state: dict) -> str:
    query = state["messages"][-1].content.lower()
    
    # Fast, compute-free heuristic for conversational chat

    conversational_triggers = ["hi", "hello", "hey", "thanks", "thank you", "summarize", "who are you", "ok", "got it", "awesome"]
    
    if any(query.startswith(trigger) or query == trigger for trigger in conversational_triggers):
        print("🚦 [Router]: Conversational intent detected. Bypassing Vector DB...")
        return "bypass"
        
    print("🚦 [Router]: Analytical intent detected. Routing to Qdrant...")
    return "retrieve"


# ---------------------------------------------------------
# GRAPH COMPILATION
# ---------------------------------------------------------
workflow = StateGraph(ChatState)
workflow.add_node("retrieve", retrieve_node)

# Use conditional entry point instead of a direct edge from START
workflow.set_conditional_entry_point(
    route_query,
    {
        "retrieve": "retrieve", # Route to Qdrant if analytical
        "bypass": END           # Route to END if conversational (Skip retrieval)
    }
)

workflow.add_edge("retrieve", END) # The LLM stream is handled in the route for SSE compatibility
agent_executor = workflow.compile()

# -------------------------------------------------------------------------
# 4. API ROUTES
# -------------------------------------------------------------------------
class IngestRequest(BaseModel):
    youtube_url: str
    instagram_url: str

class ChatRequest(BaseModel):
    message: str
    session_id: str
    video_a_id: str
    video_b_id: str

@app.post("/api/ingest")
async def ingest_pipeline(payload: IngestRequest):
    yt_data, ig_data = await asyncio.gather(
        extract_metadata_and_audio(payload.youtube_url, "youtube"),
        extract_metadata_and_audio(payload.instagram_url, "instagram")
    )
    
    for item, label in [(yt_data, "A"), (ig_data, "B")]:
        if not item.get("metadata_unavailable"):
            v = item["views"]
            item["engagement_rate"] = round(((item["likes"] + item["comments"]) / v) * 100, 2)
            
        METADATA_DATABASE[item["video_id"]] = item
        
        # Offload the sync execution to a separate worker thread dynamically
        # This keeps the main event loop entirely responsive to other network traffic!
        await asyncio.to_thread(transcribe_and_embed, item, label)
            
    return {
        "status": "success",
        "video_a_id": yt_data["video_id"],
        "video_b_id": ig_data["video_id"],
        "data": {"video_A": yt_data, "video_B": ig_data}
    }

@app.post("/api/chat")
async def chat_interaction(req: ChatRequest):
    # Manage session history
    if req.session_id not in SESSION_STORAGE:
        SESSION_STORAGE[req.session_id] = []
        
    history = SESSION_STORAGE[req.session_id][-4:]
    messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in history]
    messages.append(HumanMessage(content=req.message))

    # Run LangGraph Retrieval
    state = agent_executor.invoke({
        "session_id": req.session_id, 
        "video_a_id": req.video_a_id, 
        "video_b_id": req.video_b_id, 
        "messages": messages, 
        "context_str": ""
    })
    
    # 🚨 FIX: Extract safe variables directly from the Request, NOT the State!
    meta_a = METADATA_DATABASE.get(req.video_a_id, {})
    meta_b = METADATA_DATABASE.get(req.video_b_id, {})
    context = state.get("context_str", "")

    # ---------------------------------------------------------
    # DYNAMIC PROMPT ROUTING
    # ---------------------------------------------------------
    if not context:
        # Conversational Mode (Bypassed Qdrant)
        system_prompt = (
            "You are a helpful social media co-pilot. The user is greeting you or chatting. "
            "Respond cordially, concisely, and ask how you can help them analyze their creative video data today."
        )
    else:
        # Elite Analyst Mode (Data retrieved)
        system_prompt = f"""You are an elite, highly critical social media data analyst. 
You are comparing two videos based on their engagement metrics and transcripts.

METADATA FOR VIDEO A (YouTube):
{json.dumps(meta_a, indent=2)}

METADATA FOR VIDEO B (Instagram):
{json.dumps(meta_b, indent=2)}

TRANSCRIPT CONTEXT:
{context}

CRITICAL RULES:
1. NEVER invent or hallucinate hypothetical data or transcripts. If it is not in the context, state that explicitly.
2. AGGRESSIVELY VERIFY PREMISES: If the user states a false premise (e.g., asking why a video got more engagement when the metadata proves it actually got less), you MUST correct them immediately in your opening sentence before answering.
3. Base all your strategic analysis strictly on the provided metadata and transcript context.
"""

    final_messages = [SystemMessage(content=system_prompt)] + state["messages"]
    SESSION_STORAGE[req.session_id].append({"role": "user", "content": req.message})

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, streaming=True)
    
    async def generate():
        full_response = ""
        try:
            # Safely stream the response
            async for chunk in llm.astream(final_messages):
                if chunk.content:
                    full_response += chunk.content
                    yield f"data: {chunk.content}\n\n"
            
            # Only save to history if the generation successfully finishes
            SESSION_STORAGE[req.session_id].append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            # If Groq crashes (e.g., context window exceeded), print the error to the terminal
            print(f"🚨 LLM Generation Error: {str(e)}")
            # And send a graceful error message to the UI!
            yield f"data: \n\n[Error generating response: {str(e)}]\n\n"
            
        finally:
            # The 'finally' block GUARANTEES the stream closes properly so the frontend doesn't hang
            yield "data: [DONE]\n\n"
            
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/api/clear")
async def clear_index():
    if qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        qdrant_client.create_collection(collection_name=COLLECTION_NAME, vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    METADATA_DATABASE.clear()
    SESSION_STORAGE.clear()
    return {"status": "cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)