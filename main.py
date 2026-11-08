import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

# -----------------------------------------------------------------------------
# Configuration and logging
# -----------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vocab_ai")

MONGODB_URI = os.getenv("MONGODB_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is missing from environment variables")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from environment variables")

# -----------------------------------------------------------------------------
# FastAPI application
# -----------------------------------------------------------------------------
app = FastAPI(title="Vocab AI Backend")

# The extension's MV3 service worker uses host_permissions for API access.
# CORS is retained for possible web clients. Do not combine wildcard origins
# with credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("Incoming request: %s %s", request.method, request.url.path)
    try:
        response = await call_next(request)
        logger.info(
            "Completed request: %s %s -> %s",
            request.method,
            request.url.path,
            response.status_code,
        )
        return response
    except Exception:
        logger.exception("Unhandled error while processing %s", request.url.path)
        raise

# -----------------------------------------------------------------------------
# MongoDB
# -----------------------------------------------------------------------------
try:
    mongo_client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=5000,
    )
    mongo_client.admin.command("ping")
    logger.info("MongoDB connected")

    db = mongo_client["vocab_ai"]
    vocab_collection = db["vocabulary"]
    vocab_collection.create_index(
        [("video_id", 1), ("level", 1)],
        unique=True,
    )
except PyMongoError:
    logger.exception("Could not connect to MongoDB")
    raise

# -----------------------------------------------------------------------------
# Gemini
# -----------------------------------------------------------------------------
client = genai.Client(api_key=GEMINI_API_KEY)

# -----------------------------------------------------------------------------
# YouTube transcript retrieval
# -----------------------------------------------------------------------------
def get_video_transcript(video_id: str):
    try:
        ytt_api = YouTubeTranscriptApi()

        # Fetch English transcript only
        transcript = ytt_api.fetch(video_id)

        captions = [
            {
                "text": snippet.text,
                "start": snippet.start,
                "duration": snippet.duration
            }
            for snippet in transcript
        ]

        return captions or None

    except Exception as exc:
        logger.exception("English transcript fetch failed: %s", exc)
        return None
# -----------------------------------------------------------------------------
# Gemini vocabulary analysis
# -----------------------------------------------------------------------------
def analyze_with_gemini(
    transcript_chunk: list[dict[str, Any]],
    target_level: str,
) -> list[dict[str, Any]]:
    """Extract CEFR vocabulary and return a validated list of word objects."""
    formatted_transcript = [
        {
            "start": round(float(entry["start"]), 1),
            "text": entry["text"],
        }
        for entry in transcript_chunk
    ]

    prompt = f"""
You are an expert English language teacher.
Analyze the timestamped transcript below.
Extract useful vocabulary at, around, or above CEFR level {target_level}.

Transcript:
{json.dumps(formatted_transcript, ensure_ascii=False)}

Rules:
1. Select meaningful vocabulary; ignore basic words such as 'the', 'and', and 'because'.
2. The timestamp must match the start time of the transcript segment where the word appears.
3. Keep each definition under 10 words and relevant to the context.
4. Set duration between 3.0 and 5.0 seconds.
5. Return only words that actually appear in the transcript.

Return strictly a JSON array in this format:
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
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response")

        vocab_data = json.loads(response.text)

        # Support either a JSON array or an object containing the array.
        if isinstance(vocab_data, dict):
            for key in ("words", "vocabulary", "data"):
                if isinstance(vocab_data.get(key), list):
                    vocab_data = vocab_data[key]
                    break
            else:
                raise ValueError("Gemini JSON did not contain a vocabulary array")

        if not isinstance(vocab_data, list):
            raise ValueError("Gemini response must be a JSON array")

        # Validate and normalize fields before storing/returning them.
        validated_words = []
        for item in vocab_data:
            if not isinstance(item, dict):
                continue

            word = item.get("word")
            definition = item.get("definition")
            timestamp = item.get("timestamp")
            duration = item.get("duration", 4.0)

            if not isinstance(word, str) or not word.strip():
                continue
            if not isinstance(definition, str) or not definition.strip():
                continue

            try:
                timestamp = float(timestamp)
                duration = float(duration)
            except (TypeError, ValueError):
                continue

            if timestamp < 0:
                continue

            duration = min(max(duration, 3.0), 5.0)

            validated_words.append(
                {
                    "word": word.strip(),
                    "cefr": str(item.get("cefr", target_level)).upper(),
                    "definition": definition.strip(),
                    "timestamp": timestamp,
                    "duration": duration,
                }
            )

        logger.info("Gemini generated %d vocabulary items", len(validated_words))
        return validated_words

    except Exception as exc:
        logger.exception("Gemini vocabulary analysis failed: %s", exc)
        raise

# -----------------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Vocab AI Backend is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/vocab/cache-status")
def get_vocab_cache_status(
    video_id: str = Query(..., min_length=1),
    level: str = Query("B2"),
):
    level = level.upper()

    try:
        cached_doc = vocab_collection.find_one(
            {"video_id": video_id, "level": level},
            {"_id": 0, "vocabulary": 1},
        )
    except PyMongoError as exc:
        logger.exception("MongoDB cache-status read failed")
        raise HTTPException(
            status_code=503,
            detail="Vocabulary database is unavailable.",
        ) from exc

    if cached_doc is not None:
        logger.info("MongoDB cache HIT: %s, %s", video_id, level)
        return {
            "cached": True,
            "vocabulary": cached_doc.get("vocabulary", []),
        }

    logger.info("MongoDB cache MISS: %s, %s", video_id, level)
    return {"cached": False, "vocabulary": []}

@app.get("/api/vocab")
def get_vocab(
    video_id: str = Query(..., min_length=1),
    level: str = Query("B2"),
):
    level = level.upper()

    try:
        cached_doc = vocab_collection.find_one(
            {"video_id": video_id, "level": level},
            {"_id": 0, "vocabulary": 1},
        )
    except PyMongoError as exc:
        logger.exception("MongoDB vocabulary read failed")
        raise HTTPException(
            status_code=503,
            detail="Vocabulary database is unavailable.",
        ) from exc

    if cached_doc is not None:
        logger.info("MongoDB cache HIT: %s, %s", video_id, level)
        return cached_doc.get("vocabulary", [])

    logger.info("MongoDB cache MISS: %s, %s", video_id, level)

    transcript = get_video_transcript(video_id)
    if not transcript:
        raise HTTPException(
            status_code=404,
            detail="No usable transcript found for this video. YouTube may not provide accessible captions.",
        )

    try:
        vocab_list = analyze_with_gemini(transcript, level)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Vocabulary generation failed: {type(exc).__name__}",
        ) from exc

    if not vocab_list:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned no usable vocabulary words.",
        )

    try:
        vocab_collection.update_one(
            {"video_id": video_id, "level": level},
            {
                "$set": {
                    "video_id": video_id,
                    "level": level,
                    "vocabulary": vocab_list,
                }
            },
            upsert=True,
        )
        logger.info("MongoDB cache SAVED: %s, %s", video_id, level)
    except PyMongoError:
        # Return generated vocabulary even if saving the cache fails.
        logger.exception("MongoDB cache save failed for %s", video_id)

    return vocab_list
