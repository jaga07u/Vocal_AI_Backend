import json
import logging
import os
from typing import Any, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vocab_ai")

MONGODB_URI = os.getenv("MONGODB_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is missing")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="Vocab AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(
        "Incoming request: %s %s",
        request.method,
        request.url.path
    )

    try:
        response = await call_next(request)

        logger.info(
            "Completed request: %s %s -> %s",
            request.method,
            request.url.path,
            response.status_code
        )

        return response

    except Exception:
        logger.exception(
            "Unhandled request error: %s",
            request.url.path
        )
        raise


# ============================================================
# MONGODB
# ============================================================

try:
    mongo_client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=5000
    )

    mongo_client.admin.command("ping")

    logger.info("MongoDB connected")

    db = mongo_client["vocab_ai"]

    vocab_collection = db["vocabulary"]

    vocab_collection.create_index(
        [
            ("video_id", 1),
            ("level", 1)
        ],
        unique=True
    )

except PyMongoError:
    logger.exception("MongoDB connection failed")
    raise


# ============================================================
# GEMINI
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# REQUEST MODELS
# ============================================================

class CaptionItem(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=5000
    )

    start: float = Field(
        ge=0
    )

    duration: float = Field(
        default=3.0,
        ge=0,
        le=60
    )


class ProcessTranscriptRequest(BaseModel):
    video_id: str = Field(
        min_length=1,
        max_length=30
    )

    level: str = Field(
        default="A1",
        min_length=2,
        max_length=2
    )

    captions: List[CaptionItem] = Field(
        min_length=1,
        max_length=20000
    )


# ============================================================
# CEFR LEVEL VALIDATION
# ============================================================

VALID_LEVELS = {
    "A1",
    "A2",
    "B1",
    "B2",
    "C1",
    "C2"
}


def validate_level(level: str) -> str:
    level = level.strip().upper()

    if level not in VALID_LEVELS:
        raise HTTPException(
            status_code=422,
            detail="Invalid CEFR level. Use A1, A2, B1, B2, C1, or C2."
        )

    return level


# ============================================================
# TRANSCRIPT NORMALIZATION
# ============================================================

def normalize_transcript(
    captions: List[CaptionItem]
) -> list[dict[str, Any]]:

    transcript = []

    for caption in captions:

        text = " ".join(
            caption.text.replace("\n", " ").split()
        )

        if not text:
            continue

        transcript.append(
            {
                "text": text,
                "start": round(
                    float(caption.start),
                    2
                ),
                "duration": round(
                    float(caption.duration),
                    2
                )
            }
        )

    transcript.sort(
        key=lambda item: item["start"]
    )

    return transcript


# ============================================================
# GEMINI VOCABULARY ANALYSIS
# ============================================================

def analyze_with_gemini(
    transcript_chunk: list[dict[str, Any]],
    target_level: str
) -> list[dict[str, Any]]:

    formatted_transcript = [
        {
            "start": round(
                float(entry["start"]),
                1
            ),
            "text": entry["text"]
        }
        for entry in transcript_chunk
    ]

    prompt = f"""
You are an expert English language teacher.

Analyze the timestamped YouTube transcript below.

Extract useful English vocabulary appropriate for
CEFR level {target_level}.

Transcript:

{json.dumps(formatted_transcript, ensure_ascii=False)}

Rules:

1. Select meaningful vocabulary.
2. Ignore basic words such as "the", "and", "is", and "because".
3. Extract words that actually appear in the transcript.
4. Do not invent vocabulary.
5. Use the timestamp of the transcript segment where
   the word appears.
6. Keep each definition under 10 words.
7. Definitions must explain the word in this context.
8. Duration must be between 3 and 5 seconds.
9. Include the CEFR level of each word.
10. Avoid duplicate words.
11. Return only valid JSON.

Return a JSON array in this format:

[
  {{
    "word": "remarkable",
    "cefr": "B2",
    "definition": "unusual or worthy of attention",
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
                temperature=0.2
            )
        )

        if not response.text:
            raise ValueError(
                "Gemini returned an empty response"
            )

        vocab_data = json.loads(
            response.text
        )

        if isinstance(vocab_data, dict):

            for key in (
                "words",
                "vocabulary",
                "data"
            ):

                if isinstance(
                    vocab_data.get(key),
                    list
                ):

                    vocab_data = vocab_data[key]
                    break

            else:

                raise ValueError(
                    "Gemini response did not contain a vocabulary array"
                )

        if not isinstance(vocab_data, list):

            raise ValueError(
                "Gemini response must be a JSON array"
            )

        validated_words = []
        seen_words = set()

        for item in vocab_data:

            if not isinstance(item, dict):
                continue

            word = item.get("word")
            definition = item.get("definition")

            timestamp = item.get("timestamp")
            duration = item.get("duration", 4.0)

            if not isinstance(word, str):
                continue

            if not isinstance(definition, str):
                continue

            word = word.strip()
            definition = definition.strip()

            if not word or not definition:
                continue

            word_key = word.lower()

            if word_key in seen_words:
                continue

            try:

                timestamp = float(timestamp)
                duration = float(duration)

            except (TypeError, ValueError):

                continue

            if timestamp < 0:
                continue

            duration = min(
                max(duration, 3.0),
                5.0
            )

            cefr = str(
                item.get("cefr", target_level)
            ).upper()

            if cefr not in VALID_LEVELS:
                cefr = target_level

            validated_words.append(
                {
                    "word": word,
                    "cefr": cefr,
                    "definition": definition,
                    "timestamp": timestamp,
                    "duration": duration
                }
            )

            seen_words.add(word_key)

        validated_words.sort(
            key=lambda item: item["timestamp"]
        )

        logger.info(
            "Gemini generated %d vocabulary words",
            len(validated_words)
        )

        return validated_words

    except Exception:

        logger.exception(
            "Gemini vocabulary analysis failed"
        )

        raise


# ============================================================
# CACHE HELPER
# ============================================================

def get_cached_vocabulary(
    video_id: str,
    level: str
):

    try:

        return vocab_collection.find_one(
            {
                "video_id": video_id,
                "level": level
            },
            {
                "_id": 0,
                "vocabulary": 1
            }
        )

    except PyMongoError as exc:

        logger.exception(
            "MongoDB cache read failed"
        )

        raise HTTPException(
            status_code=503,
            detail="Vocabulary database is unavailable."
        ) from exc


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Vocab AI Backend is running"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# ============================================================
# CACHE STATUS
# ============================================================

@app.get("/api/vocab/cache-status")
def get_vocab_cache_status(
    video_id: str = Query(
        ...,
        min_length=1
    ),
    level: str = Query("A1")
):

    level = validate_level(level)

    cached_doc = get_cached_vocabulary(
        video_id,
        level
    )

    if cached_doc is not None:

        logger.info(
            "MongoDB cache HIT: %s, %s",
            video_id,
            level
        )

        return {
            "cached": True,
            "vocabulary": cached_doc.get(
                "vocabulary",
                []
            )
        }

    logger.info(
        "MongoDB cache MISS: %s, %s",
        video_id,
        level
    )

    return {
        "cached": False,
        "vocabulary": []
    }


# ============================================================
# PROCESS FRONTEND TRANSCRIPT
# ============================================================

@app.post("/api/vocab/process")
def process_frontend_transcript(
    payload: ProcessTranscriptRequest
):
    print(payload.captions)
    video_id = payload.video_id.strip()

    level = validate_level(
        payload.level
    )

    transcript = normalize_transcript(
        payload.captions
    )

    if not transcript:

        raise HTTPException(
            status_code=422,
            detail="Transcript contains no usable captions."
        )

    # Check cache again before running Gemini.
    # This prevents duplicate processing if another
    # request has already generated the vocabulary.

    cached_doc = get_cached_vocabulary(
        video_id,
        level
    )

    if cached_doc is not None:

        logger.info(
            "Cache HIT during transcript processing: %s, %s",
            video_id,
            level
        )

        return {
            "cached": True,
            "vocabulary": cached_doc.get(
                "vocabulary",
                []
            )
        }

    # Process the captions sent by the extension.
    # Render does not fetch the YouTube transcript.

    try:

        vocabulary = analyze_with_gemini(
            transcript,
            level
        )

    except Exception as exc:

        logger.exception(
            "Vocabulary generation failed"
        )

        raise HTTPException(
            status_code=502,
            detail="Vocabulary generation failed."
        ) from exc

    if not vocabulary:

        raise HTTPException(
            status_code=502,
            detail="No usable vocabulary was generated."
        )

    # Save the generated vocabulary to MongoDB.

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
                    "vocabulary": vocabulary,
                    "transcript_source": "youtube_extension"
                }
            },
            upsert=True
        )

        logger.info(
            "MongoDB cache SAVED: %s, %s",
            video_id,
            level
        )

    except PyMongoError:

        # Return generated vocabulary even if
        # MongoDB cache saving fails.

        logger.exception(
            "MongoDB cache save failed"
        )

    return {
        "cached": False,
        "vocabulary": vocabulary
    }


# ============================================================
# LEGACY ENDPOINT
# ============================================================

@app.get("/api/vocab")
def get_vocab(
    video_id: str = Query(
        ...,
        min_length=1
    ),
    level: str = Query("A1")
):

    level = validate_level(level)

    cached_doc = get_cached_vocabulary(
        video_id,
        level
    )

    if cached_doc is not None:

        return cached_doc.get(
            "vocabulary",
            []
        )

    raise HTTPException(
        status_code=410,
        detail=(
            "Transcript retrieval from Render is disabled. "
            "Update the extension to send captions using "
            "POST /api/vocab/process."
        )
    )
