import os
from celery import Celery
from apify_client import ApifyClient

# Initialize Celery to Redis queue
celery_app = Celery(
    "video_tasks",
    broker="redis://localhost:6380/0", 
    backend="redis://localhost:6380/0" 
)

@celery_app.task(name="tasks.ingest_and_embed_video")
def ingest_and_embed_video_task(video_url: str, platform: str, video_id: str):
    """
    This runs entirely in the background as a separate distributed worker process.
    """
    print(f" Worker picked up ingestion job for {platform} video: {video_id}")
    
    print(f"Background job completed successfully for {video_id}")
    return {"status": "SUCCESS", "video_id": video_id}