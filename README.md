# Transcribr

A Flask web app that transcribes Zoom recordings and extracts deduplicated screenshots, with real-time progress updates via WebSocket.

## Features

- **Audio Transcription** — Uses OpenAI Whisper (local) to produce timestamped transcripts
- **Screenshot Extraction** — Captures frames every 10 seconds from video, deduplicates visually similar frames using perceptual hashing
- **Web UI** — Upload files, watch progress in real-time, view results side-by-side
- **Job History** — Browse past transcriptions by job ID
- **ZIP Download** — Download transcript + screenshots in one archive

## Requirements

- Python 3.13+
- ffmpeg (`brew install ffmpeg`)

## Setup

```bash
# Create virtual environment
python3.13 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
source venv/bin/activate
python app.py
```

Open http://127.0.0.1:5000 in your browser, upload a `.mp4` video and `.m4a` audio file from a Zoom recording, and wait for processing to complete.

## How It Works

1. Upload `.mp4` (video) and `.m4a` (audio) files from a Zoom call
2. Whisper transcribes the audio into timestamped text
3. ffmpeg extracts frames every 10 seconds from the video
4. Perceptual hashing removes visually duplicate frames
5. Results are displayed side-by-side: transcript on the left, screenshots on the right
6. Download everything as a ZIP archive

## Tech Stack

- **Flask** + **Flask-SocketIO** — Web framework with WebSocket support
- **OpenAI Whisper** — Local speech-to-text transcription
- **ffmpeg** — Video frame extraction
- **Pillow** + **imagehash** — Image processing and perceptual hash deduplication
