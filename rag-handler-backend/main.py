import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Clean, serverless LangChain & Pinecone Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

app = FastAPI(title="RAG Chatbot Ingestion Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pinecone and Gemini Embedding Client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# CHANGE THIS LINE:
# embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Change the name so Pinecone knows to build a new one
# Force a brand new index name
INDEX_NAME = "social-media-rag-3072"

# Auto-create the Pinecone Index if it doesn't exist
if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072, # Dimension matches the new embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

class VideoIngestRequest(BaseModel):
    youtube_url: str
    instagram_url: str

class VideoMetadata(BaseModel):
    video_id: str
    platform: str
    creator: str
    follower_count: int
    views: int
    likes: int
    comments: int
    engagement_rate: float
    transcript: str
    hashtags: list[str]
    duration: int

def extract_youtube_data(url: str) -> dict:
    ydl_opts = {'skip_download': True, 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            video_id = info.get('id', 'yt_fallback')
            views = info.get('view_count', 0)
            likes = info.get('like_count', 0)
            comments = info.get('comment_count', 0)
            creator = info.get('uploader', 'Unknown Creator')
            duration = info.get('duration', 0)
            tags = info.get('tags', []) or []
            
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = " ".join([t['text'] for t in transcript_list])
            except Exception:
                transcript_text = "Transcript extraction failed or captions are disabled."

            return {
                "video_id": video_id,
                "platform": "YouTube",
                "creator": creator,
                "follower_count": 0,
                "views": views,
                "likes": likes,
                "comments": comments,
                "transcript": transcript_text,
                "hashtags": tags,
                "duration": duration
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process YouTube URL: {str(e)}")

def extract_instagram_data(url: str) -> dict:
    mock_id = url.split("/reel/")[-1].split("/")[0] if "/reel/" in url else "ig_reel_1"
    return {
        "video_id": mock_id,
        "platform": "Instagram",
        "creator": "instagram_creator_demo",
        "follower_count": 150000,
        "views": 50000,
        "likes": 4200,
        "comments": 350,
        "transcript": "Hey everyone! Today I am showing you the absolute fastest way to optimize your code structure. Make sure you watch until the end for the hidden trick.",
        "hashtags": ["coding", "developer", "growth"],
        "duration": 15
    }

def chunk_and_vectorize(video_label: str, metadata: dict):
    transcript = metadata["transcript"]
    if not transcript or "failed" in transcript:
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(transcript)
    
    vectors_to_upsert = []
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"{metadata['video_id']}_chunk_{i}"
        
        # Generates embedding via cloud API request, completely bypassing local torch requirements
        embedding = embeddings.embed_query(chunk)
        
        vectors_to_upsert.append({
            "id": chunk_id,
            "values": embedding,
            "metadata": {
                "text": chunk,
                "video_label": video_label,
                "video_id": metadata["video_id"],
                "platform": metadata["platform"],
                "creator": metadata["creator"],
                "engagement_rate": metadata["engagement_rate"]
            }
        })
        
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)

@app.post("/api/ingest")
async def ingest_videos(payload: VideoIngestRequest):
    yt_data = extract_youtube_data(payload.youtube_url)
    ig_data = extract_instagram_data(payload.instagram_url)
    
    for data in [yt_data, ig_data]:
        views = data["views"]
        er = ((data["likes"] + data["comments"]) / views) * 100 if views > 0 else 0.0
        data["engagement_rate"] = round(er, 2)
        
    chunk_and_vectorize("video_A", yt_data)
    chunk_and_vectorize("video_B", ig_data)
        
    return {
        "status": "success",
        "message": "Videos successfully ingested, split, embedded, and indexed in Pinecone using Gemini API.",
        "data": {
            "video_A": yt_data,
            "video_B": ig_data
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

class ChatMessage(BaseModel):
    message: str
    history: list[dict]
    # In a full app, you'd pass a list of previous messages here for memory

@app.post("/api/chat")
async def chat_with_agent(req: ChatMessage):
    # 1. Connect to our specific Pinecone Index as a LangChain VectorStore
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text" # This matches the metadata key where we stored the chunk string
    )
    
    # 2. Retrieve the top 5 most relevant chunks based on the user's question
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(req.message)
    
    # 3. Format the context with explicit source citations for the LLM
    context = ""
    for idx, d in enumerate(docs):
        video_id = d.metadata.get('video_id', 'Unknown')
        platform = d.metadata.get('platform', 'Unknown')
        er = d.metadata.get('engagement_rate', 'Unknown')
        context += f"\n--- Chunk {idx+1} (Source: {platform} - Video ID: {video_id} | Engagement: {er}%) ---\n{d.page_content}\n"

    # 4. Initialize the Gemini LLM for streaming
    # 4. Initialize the Gemini LLM for streaming (NO PREFIX!)
    # 4. Initialize the Gemini LLM for streaming
    # Initialize the ultra-fast Groq Llama 3 model
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0.3,
        streaming=True
    )
    
    formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in req.history])
    # 5. Build the strict RAG Prompt
    system_prompt = f"""You are a highly analytical social media strategist. 
    Answer the user's question based ONLY on the provided context chunks.
    You MUST explicitly cite your sources by mentioning the Platform and Video ID in your response.
    Compare the engagement rates and content hooks if asked.
    If the answer is not contained in the context, tell the user you don't have enough data.
    
    CHAT HISTORY:
    {formatted_history}

    CONTEXT:
    {context}
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=req.message)
    ]
    
    # 6. Async Generator to stream text tokens directly to the frontend
    async def generate_response():
        async for chunk in llm.astream(messages):
            if chunk.content:
                # Yielding in SSE (Server-Sent Events) format
                yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_response(), media_type="text/event-stream")