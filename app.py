from youtube_transcript_api import YouTubeTranscriptApi

video_id = "l8YUAtZjxIo"

ytt_api = YouTubeTranscriptApi()

transcript = ytt_api.fetch(video_id)

# Convert transcript into normal Python dictionaries
captions = [
    {
        "text": snippet.text,
        "start": snippet.start,
        "duration": snippet.duration
    }
    for snippet in transcript
]

print(captions)