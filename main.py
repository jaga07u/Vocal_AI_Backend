import os
import json
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
# from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from google import genai
from google.genai import types
from dotenv import load_dotenv
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from pymongo import MongoClient
from pymongo.errors import PyMongoError


load_dotenv()

app = FastAPI(title="Vocab AI Backend")
MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is missing from .env")

mongo_client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=5000
)
mongo_client.admin.command("ping")
print("MongoDB connected!")
db = mongo_client["vocab_ai"]
vocab_collection = db["vocabulary"]

# One cached document per video + level
vocab_collection.create_index(
    [("video_id", 1), ("level", 1)],
    unique=True
)
# Allow extension to communicate with localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Cache results so we don't call Gemini repeatedly for the same video
cache = {}

def get_video_transcript(video_id: str):
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)

        # Prefer English transcripts, including auto-generated ones.
        try:
            transcript = transcript_list.find_transcript([
                "en", "en-US", "en-GB", "en-IN"
            ])
            return transcript.fetch()

        except NoTranscriptFound:
            pass

        # Try translating another available transcript into English.
        for transcript in transcript_list:
            if transcript.is_translatable:
                return transcript.translate("en").fetch()

        return None

    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"Transcript unavailable: {e}")
        return None

    except Exception as e:
        print(f"Transcript fetching error: {type(e).__name__}: {e}")
        return None
    
def analyze_with_gemini(transcript_chunk: list, target_level: str):
    """
    Sends timestamped lines to Gemini to extract CEFR-level vocabulary.
    """
    # Keep only start time and text to save tokens
    formatted_transcript = [
    {
        "start": round(entry.start, 1),
        "text": entry.text
    }
    for entry in transcript_chunk
]

    prompt = f"""
    You are an expert language teacher.
    Analyze the timestamped transcript below.
    Extract key vocabulary words that are at or around or above CEFR level '{target_level}' (e.g., B2, C1).
    
    Transcript:
    {json.dumps(formatted_transcript)} 

    Rules:
    1. Select only meaningful words (ignore basic conversational words like 'the', 'however', 'because').
    2. 'timestamp' MUST strictly match the 'start' second when that word appears in the transcript.
    3. Keep definitions short (under 10 words) and contextual to what is being spoken.
    4. Provide 'duration' in seconds for how long the popup card should stay (between 3.0 and 5.0).

    Output STRICTLY a JSON list of objects matching this format:
    [
      {{
        "word": "eloquent",
        "cefr": "C1",
        "definition": "fluent or persuasive in speaking or writing",
        "timestamp": 14.2,
        "duration": 4.0
      }}
    ]
    """

    try:
        # We request application/json output directly from Gemini
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        print("Gemini response:", response.text)
        vocab_data = json.loads(response.text)
        print("Parsed vocabulary:", vocab_data)
        
        # In case the model returns {"words": [...]} instead of [...]
        if isinstance(vocab_data, dict):
            for key in ["words", "vocabulary", "data"]:
                if key in vocab_data:
                    return vocab_data[key]
            return []
            
        return vocab_data

    except Exception as e:
     print("Gemini analysis error:", type(e).__name__, str(e))
    return []


@app.get("/api/vocab")
def get_vocab(
    video_id: str = Query(..., description="YouTube Video ID"),
    level: str = Query("B2")
):
    level = level.upper()

    # 1. Check MongoDB first
    try:
        cached_doc = vocab_collection.find_one(
            {
                "video_id": video_id,
                "level": level
            },
            {"_id": 0, "vocabulary": 1}
        )
    except PyMongoError as e:
        print("MongoDB read error:", repr(e))
        raise HTTPException(
            status_code=503,
            detail="Vocabulary database is unavailable."
        )

    # 2. If found, return immediately
    if cached_doc is not None:
        print(f"MongoDB cache HIT: {video_id}, {level}")
        return cached_doc["vocabulary"]

    print(f"MongoDB cache MISS: {video_id}, {level}")

    # 3. Fetch YouTube transcript
    transcript = get_video_transcript(video_id)

    if not transcript:
        raise HTTPException(
            status_code=404,
            detail="No usable transcript found for this video."
        )

    # 4. Generate vocabulary using Gemini
    vocab_list = analyze_with_gemini(transcript, level)

    if not vocab_list:
        raise HTTPException(
            status_code=502,
            detail="Vocabulary generation failed or returned no words."
        )

    # 5. Save results in MongoDB
    try:
        vocab_collection.update_one(
            {
                "video_id": video_id,
                "level": level
            },
            {
                "$set": {
                    "video_id": video_id,
                    "level": level,
                    "vocabulary": vocab_list
                }
            },
            upsert=True
        )

        print(f"MongoDB cache SAVED: {video_id}, {level}")

    except PyMongoError as e:
        # Gemini succeeded, so still return the generated result
        print("MongoDB save error:", repr(e))

    # 6. Return the newly generated result
    return vocab_list

@app.get("/api/vocab/cache-status")
def get_vocab_cache_status(
    video_id: str = Query(..., description="YouTube Video ID"),
    level: str = Query("B2")
):
    level = level.upper()

    try:
        cached_doc = vocab_collection.find_one(
            {
                "video_id": video_id,
                "level": level
            },
            {
                "_id": 0,
                "vocabulary": 1
            }
        )
    except PyMongoError as e:
        print("MongoDB cache-status read error:", repr(e))
        raise HTTPException(
            status_code=503,
            detail="Vocabulary database is unavailable."
        )

    if cached_doc is not None:
        print(f"MongoDB cache HIT: {video_id}, {level}")

        return {
            "cached": True,
            "vocabulary": cached_doc.get("vocabulary", [])
        }

    print(f"MongoDB cache MISS: {video_id}, {level}")

    return {
        "cached": False,
        "vocabulary": []
    }

