# Transcribr Design

## Overview

A Flask web app that transcribes Zoom recordings and extracts deduplicated screenshots, with a web UI for uploading, viewing results, and downloading output.

## Input

- `.mp4` video file (for screenshot extraction)
- `.m4a` audio file (for transcription)
- Files are from the same Zoom call

## Core Processing

### Transcription
- Uses **openai-whisper** (local, `medium` model) on the `.m4a` file
- Output: timestamped `.txt` file with `[HH:MM:SS]` prefixed segments

### Screenshot Extraction
- Extracts frames every **10 seconds** from the `.mp4` using ffmpeg
- Deduplicates using **perceptual hashing** (imagehash library) — consecutive frames with hamming distance below threshold are discarded
- Filenames include timestamps (e.g., `frame_001_00m30s.png`) for cross-referencing with the transcript

## Web App

### Stack
- **Flask** + **Flask-SocketIO** for WebSocket live progress updates
- No database — flat file storage in `jobs/` directory

### Routes
- `GET /` — Upload page + job history list
- `POST /upload` — Accepts files, creates job, redirects to job page
- `GET /job/<job-id>` — Results page with transcript, screenshots, zip download
- `GET /job/<job-id>/download` — Zip download

### Job Processing Flow
1. User uploads `.mp4` and `.m4a` files
2. Server creates a job with a short UUID (e.g., `a3f8c2`)
3. Redirects to `/job/<job-id>` which connects via WebSocket
4. Background thread processes the files, emitting progress events
5. On completion, page renders transcript + screenshot timeline + zip download button

### File Structure
```
jobs/<job-id>/
├── job.json              # Metadata (status, filename, created_at, etc.)
├── transcription.txt     # Timestamped transcript
├── screenshots/          # Deduplicated PNG frames
│   ├── frame_001_00m30s.png
│   └── ...
└── transcribr_<job-id>.zip
```

### Pages
- **Home (`/`)**: File upload form + list of past jobs sorted by date
- **Job (`/job/<job-id>`)**: Progress bar during processing, then transcript + screenshot timeline side by side, zip download button

## Dependencies
- `flask`, `flask-socketio` — web framework + WebSocket
- `openai-whisper` — local transcription
- `Pillow`, `imagehash` — screenshot dedup
- `ffmpeg` — frame extraction (system dependency via `brew install ffmpeg`)

## Decisions
- Whisper `medium` model — good accuracy/speed balance for English
- 10-second screenshot interval with perceptual hash dedup
- Scene detection threshold: 0.3 (ffmpeg default)
- Perceptual hash similarity threshold: configurable
- Short UUIDs for job IDs
- Flat file storage (no database)
