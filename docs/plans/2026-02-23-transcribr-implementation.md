# Transcribr Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Flask web app that transcribes Zoom recordings and extracts deduplicated screenshots, with WebSocket progress updates, job history, and zip downloads.

**Architecture:** Flask + Flask-SocketIO backend with flat-file job storage. Core processing split into a `transcriber.py` module (Whisper transcription + ffmpeg screenshot extraction + perceptual hash dedup). Background threads handle processing while WebSocket pushes live status to the client.

**Tech Stack:** Python 3.13, Flask, Flask-SocketIO, openai-whisper, ffmpeg, Pillow, imagehash

---

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `app.py` (minimal Flask hello world)

**Step 1: Install ffmpeg**

Run: `brew install ffmpeg`
Expected: ffmpeg available at `/opt/homebrew/bin/ffmpeg`

**Step 2: Create a virtual environment**

Run:
```bash
cd /Users/cmerrill/Projects/Transcribr
python3.13 -m venv venv
source venv/bin/activate
```

**Step 3: Create requirements.txt**

```
flask==3.1.*
flask-socketio==5.5.*
openai-whisper
Pillow
imagehash
```

**Step 4: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully

**Step 5: Create minimal app.py**

```python
from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "Transcribr is running"

if __name__ == "__main__":
    app.run(debug=True, port=5000)
```

**Step 6: Verify the app starts**

Run: `python app.py`
Expected: Flask dev server starts on http://127.0.0.1:5000

**Step 7: Commit**

```bash
git init
git add requirements.txt app.py docs/
git commit -m "feat: project setup with Flask scaffold and design docs"
```

---

### Task 2: Core Transcription Module

**Files:**
- Create: `transcriber.py`
- Create: `tests/test_transcriber.py`

**Step 1: Write the failing test for `format_timestamp`**

```python
# tests/test_transcriber.py
from transcriber import format_timestamp

def test_format_timestamp_zero():
    assert format_timestamp(0.0) == "00:00:00"

def test_format_timestamp_seconds():
    assert format_timestamp(65.5) == "00:01:05"

def test_format_timestamp_hours():
    assert format_timestamp(3661.0) == "01:01:01"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_transcriber.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement format_timestamp in transcriber.py**

```python
# transcriber.py

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_transcriber.py -v`
Expected: 3 PASS

**Step 5: Write the failing test for `transcribe_audio`**

```python
# tests/test_transcriber.py (append)
import os
import tempfile
from unittest.mock import patch, MagicMock
from transcriber import transcribe_audio

def test_transcribe_audio_writes_timestamped_file():
    """Test that transcribe_audio calls whisper and writes formatted output."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "segments": [
            {"start": 0.0, "end": 5.0, "text": " Hello world"},
            {"start": 5.0, "end": 10.0, "text": " Second line"},
        ]
    }

    with patch("transcriber.whisper") as mock_whisper, \
         tempfile.TemporaryDirectory() as tmpdir:
        mock_whisper.load_model.return_value = mock_model
        output_path = os.path.join(tmpdir, "transcription.txt")

        transcribe_audio("fake.m4a", output_path, progress_callback=None)

        content = open(output_path).read()
        assert "[00:00:00]" in content
        assert "Hello world" in content
        assert "[00:00:05]" in content
        assert "Second line" in content
```

**Step 6: Run test to verify it fails**

Run: `python -m pytest tests/test_transcriber.py::test_transcribe_audio_writes_timestamped_file -v`
Expected: FAIL — `ImportError`

**Step 7: Implement transcribe_audio**

```python
# transcriber.py (append)
import whisper

def transcribe_audio(audio_path: str, output_path: str, progress_callback=None, model_name: str = "medium"):
    """Transcribe audio file using Whisper and write timestamped text."""
    if progress_callback:
        progress_callback("Loading Whisper model...")

    model = whisper.load_model(model_name)

    if progress_callback:
        progress_callback("Transcribing audio...")

    result = model.transcribe(audio_path)

    with open(output_path, "w") as f:
        for segment in result["segments"]:
            timestamp = format_timestamp(segment["start"])
            text = segment["text"].strip()
            f.write(f"[{timestamp}] {text}\n")

    if progress_callback:
        progress_callback("Transcription complete.")
```

**Step 8: Run test to verify it passes**

Run: `python -m pytest tests/test_transcriber.py -v`
Expected: All PASS

**Step 9: Commit**

```bash
git add transcriber.py tests/
git commit -m "feat: add transcribe_audio with timestamped output"
```

---

### Task 3: Screenshot Extraction Module

**Files:**
- Modify: `transcriber.py`
- Modify: `tests/test_transcriber.py`

**Step 1: Write the failing test for `extract_screenshots`**

```python
# tests/test_transcriber.py (append)
import tempfile
from unittest.mock import patch, call
from transcriber import extract_screenshots

def test_extract_screenshots_calls_ffmpeg(tmp_path):
    """Test that extract_screenshots invokes ffmpeg correctly."""
    with patch("transcriber.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        # Create fake raw frames so the function has something to process
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        with patch("transcriber._extract_raw_frames") as mock_extract:
            mock_extract.return_value = []
            screenshots_dir = tmp_path / "screenshots"
            screenshots_dir.mkdir()

            extract_screenshots(
                "fake.mp4",
                str(screenshots_dir),
                interval=10,
                progress_callback=None
            )

            mock_extract.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_transcriber.py::test_extract_screenshots_calls_ffmpeg -v`
Expected: FAIL — `ImportError`

**Step 3: Implement _extract_raw_frames and extract_screenshots**

```python
# transcriber.py (append)
import subprocess
import os
from PIL import Image
import imagehash

def _extract_raw_frames(video_path: str, output_dir: str, interval: int = 10, progress_callback=None):
    """Extract frames from video at regular intervals using ffmpeg."""
    if progress_callback:
        progress_callback("Extracting frames from video...")

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps=1/{interval}",
        "-frame_pts", "1",
        os.path.join(output_dir, "raw_%06d.png"),
        "-y", "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)

    frames = []
    for i, filename in enumerate(sorted(os.listdir(output_dir))):
        if filename.startswith("raw_") and filename.endswith(".png"):
            seconds = i * interval
            filepath = os.path.join(output_dir, filename)
            frames.append((filepath, seconds))

    if progress_callback:
        progress_callback(f"Extracted {len(frames)} raw frames.")

    return frames


def _deduplicate_frames(frames: list, threshold: int = 5, progress_callback=None):
    """Remove visually similar consecutive frames using perceptual hashing."""
    if not frames:
        return []

    if progress_callback:
        progress_callback("Deduplicating frames...")

    unique = [frames[0]]
    prev_hash = imagehash.phash(Image.open(frames[0][0]))

    for filepath, seconds in frames[1:]:
        curr_hash = imagehash.phash(Image.open(filepath))
        if abs(curr_hash - prev_hash) >= threshold:
            unique.append((filepath, seconds))
            prev_hash = curr_hash

    if progress_callback:
        progress_callback(f"Kept {len(unique)} unique frames from {len(frames)} total.")

    return unique


def extract_screenshots(video_path: str, output_dir: str, interval: int = 10, hash_threshold: int = 5, progress_callback=None):
    """Extract and deduplicate screenshots from video."""
    import tempfile

    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as raw_dir:
        frames = _extract_raw_frames(video_path, raw_dir, interval, progress_callback)
        unique_frames = _deduplicate_frames(frames, hash_threshold, progress_callback)

        if progress_callback:
            progress_callback("Saving unique frames...")

        saved = []
        for i, (filepath, seconds) in enumerate(unique_frames, 1):
            mins = seconds // 60
            secs = seconds % 60
            out_name = f"frame_{i:03d}_{mins:02d}m{secs:02d}s.png"
            out_path = os.path.join(output_dir, out_name)

            img = Image.open(filepath)
            img.save(out_path)
            saved.append((out_name, seconds))

        if progress_callback:
            progress_callback(f"Saved {len(saved)} screenshots.")

        return saved
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_transcriber.py -v`
Expected: All PASS

**Step 5: Write a deduplication unit test**

```python
# tests/test_transcriber.py (append)
from transcriber import _deduplicate_frames
from PIL import Image

def test_deduplicate_removes_identical_frames(tmp_path):
    """Identical images should be deduplicated down to one."""
    img = Image.new("RGB", (100, 100), color="red")
    paths = []
    for i in range(5):
        p = tmp_path / f"frame_{i}.png"
        img.save(str(p))
        paths.append((str(p), i * 10))

    result = _deduplicate_frames(paths, threshold=5)
    assert len(result) == 1

def test_deduplicate_keeps_different_frames(tmp_path):
    """Visually different images should all be kept."""
    colors = ["red", "green", "blue", "yellow"]
    paths = []
    for i, color in enumerate(colors):
        img = Image.new("RGB", (100, 100), color=color)
        p = tmp_path / f"frame_{i}.png"
        img.save(str(p))
        paths.append((str(p), i * 10))

    result = _deduplicate_frames(paths, threshold=5)
    assert len(result) == 4
```

**Step 6: Run tests**

Run: `python -m pytest tests/test_transcriber.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add transcriber.py tests/
git commit -m "feat: add screenshot extraction with perceptual hash dedup"
```

---

### Task 4: Flask App with Upload and Job Management

**Files:**
- Modify: `app.py`
- Create: `templates/index.html`
- Create: `templates/job.html`
- Create: `static/style.css`

**Step 1: Implement app.py with all routes and SocketIO**

```python
# app.py
import os
import json
import uuid
import zipfile
from datetime import datetime
from threading import Thread

from flask import Flask, render_template, request, redirect, url_for, send_file, abort
from flask_socketio import SocketIO, emit

from transcriber import transcribe_audio, extract_screenshots

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["JOBS_FOLDER"] = os.path.join(os.path.dirname(__file__), "jobs")

socketio = SocketIO(app)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["JOBS_FOLDER"], exist_ok=True)


def get_job_dir(job_id):
    return os.path.join(app.config["JOBS_FOLDER"], job_id)


def load_job(job_id):
    job_file = os.path.join(get_job_dir(job_id), "job.json")
    if not os.path.exists(job_file):
        return None
    with open(job_file) as f:
        return json.load(f)


def save_job(job_id, data):
    job_dir = get_job_dir(job_id)
    os.makedirs(job_dir, exist_ok=True)
    with open(os.path.join(job_dir, "job.json"), "w") as f:
        json.dump(data, f, indent=2)


def get_all_jobs():
    jobs_dir = app.config["JOBS_FOLDER"]
    jobs = []
    if os.path.exists(jobs_dir):
        for job_id in os.listdir(jobs_dir):
            job = load_job(job_id)
            if job:
                job["id"] = job_id
                jobs.append(job)
    return sorted(jobs, key=lambda j: j.get("created_at", ""), reverse=True)


def process_job(job_id, video_path, audio_path):
    """Run transcription and screenshot extraction in a background thread."""
    job_dir = get_job_dir(job_id)
    screenshots_dir = os.path.join(job_dir, "screenshots")
    transcript_path = os.path.join(job_dir, "transcription.txt")

    def progress(msg):
        socketio.emit("progress", {"job_id": job_id, "message": msg}, namespace="/", to=job_id)

    try:
        # Update status
        job = load_job(job_id)
        job["status"] = "processing"
        save_job(job_id, job)

        # Transcribe
        progress("Starting transcription...")
        transcribe_audio(audio_path, transcript_path, progress_callback=progress)

        # Extract screenshots
        progress("Starting screenshot extraction...")
        saved_frames = extract_screenshots(video_path, screenshots_dir, interval=10, progress_callback=progress)

        # Build zip
        progress("Creating zip archive...")
        zip_path = os.path.join(job_dir, f"transcribr_{job_id}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(transcript_path, "transcription.txt")
            for frame_name, _ in saved_frames:
                frame_path = os.path.join(screenshots_dir, frame_name)
                zf.write(frame_path, f"screenshots/{frame_name}")

        # Update job metadata
        job["status"] = "complete"
        job["screenshots"] = [{"filename": name, "seconds": secs} for name, secs in saved_frames]
        save_job(job_id, job)

        progress("Done!")
        socketio.emit("complete", {"job_id": job_id}, namespace="/", to=job_id)

    except Exception as e:
        job = load_job(job_id)
        job["status"] = "error"
        job["error"] = str(e)
        save_job(job_id, job)
        progress(f"Error: {e}")
        socketio.emit("error", {"job_id": job_id, "message": str(e)}, namespace="/", to=job_id)

    finally:
        # Clean up uploaded files
        for path in [video_path, audio_path]:
            if os.path.exists(path):
                os.remove(path)


@app.route("/")
def index():
    jobs = get_all_jobs()
    return render_template("index.html", jobs=jobs)


@app.route("/upload", methods=["POST"])
def upload():
    video = request.files.get("video")
    audio = request.files.get("audio")

    if not video or not audio:
        return "Both video (.mp4) and audio (.m4a) files are required.", 400

    job_id = uuid.uuid4().hex[:8]
    job_dir = get_job_dir(job_id)
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(os.path.join(job_dir, "screenshots"), exist_ok=True)

    video_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{job_id}_video.mp4")
    audio_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{job_id}_audio.m4a")
    video.save(video_path)
    audio.save(audio_path)

    save_job(job_id, {
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "video_filename": video.filename,
        "audio_filename": audio.filename,
    })

    thread = Thread(target=process_job, args=(job_id, video_path, audio_path))
    thread.daemon = True
    thread.start()

    return redirect(url_for("job_page", job_id=job_id))


@app.route("/job/<job_id>")
def job_page(job_id):
    job = load_job(job_id)
    if not job:
        abort(404)
    job["id"] = job_id

    transcript = ""
    transcript_path = os.path.join(get_job_dir(job_id), "transcription.txt")
    if os.path.exists(transcript_path):
        with open(transcript_path) as f:
            transcript = f.read()

    return render_template("job.html", job=job, transcript=transcript)


@app.route("/job/<job_id>/download")
def download_zip(job_id):
    zip_path = os.path.join(get_job_dir(job_id), f"transcribr_{job_id}.zip")
    if not os.path.exists(zip_path):
        abort(404)
    return send_file(zip_path, as_attachment=True)


@app.route("/job/<job_id>/screenshots/<filename>")
def serve_screenshot(job_id, filename):
    filepath = os.path.join(get_job_dir(job_id), "screenshots", filename)
    if not os.path.exists(filepath):
        abort(404)
    return send_file(filepath)


@socketio.on("join")
def handle_join(data):
    from flask_socketio import join_room
    job_id = data.get("job_id")
    if job_id:
        join_room(job_id)


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)
```

**Step 2: Run the app to check for import errors**

Run: `python -c "from app import app; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Flask app with upload, job management, and WebSocket progress"
```

---

### Task 5: HTML Templates and CSS

**Files:**
- Create: `templates/index.html`
- Create: `templates/job.html`
- Create: `static/style.css`

**Step 1: Create templates/index.html**

The home page with file upload form and job history table.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transcribr</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <h1>Transcribr</h1>
        <p class="subtitle">Upload a Zoom recording to get a transcription and screenshots.</p>

        <form action="/upload" method="post" enctype="multipart/form-data" class="upload-form">
            <div class="form-group">
                <label for="video">Video file (.mp4)</label>
                <input type="file" id="video" name="video" accept=".mp4" required>
            </div>
            <div class="form-group">
                <label for="audio">Audio file (.m4a)</label>
                <input type="file" id="audio" name="audio" accept=".m4a" required>
            </div>
            <button type="submit" class="btn">Upload & Process</button>
        </form>

        {% if jobs %}
        <h2>Job History</h2>
        <table class="job-table">
            <thead>
                <tr>
                    <th>Job ID</th>
                    <th>Video File</th>
                    <th>Date</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {% for job in jobs %}
                <tr onclick="window.location='/job/{{ job.id }}'">
                    <td><code>{{ job.id }}</code></td>
                    <td>{{ job.video_filename }}</td>
                    <td>{{ job.created_at[:16] }}</td>
                    <td><span class="status status-{{ job.status }}">{{ job.status }}</span></td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}
    </div>
</body>
</html>
```

**Step 2: Create templates/job.html**

The job results page with WebSocket progress, transcript, and screenshots.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Job {{ job.id }} - Transcribr</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <header>
            <a href="/" class="back-link">&larr; Back</a>
            <h1>Job <code>{{ job.id }}</code></h1>
            <p class="meta">{{ job.video_filename }} &mdash; {{ job.created_at[:16] }}</p>
        </header>

        {% if job.status in ['queued', 'processing'] %}
        <div class="progress-section" id="progress-section">
            <div class="spinner"></div>
            <p id="progress-message">Connecting...</p>
        </div>
        {% endif %}

        {% if job.status == 'error' %}
        <div class="error-banner">
            <strong>Error:</strong> {{ job.error }}
        </div>
        {% endif %}

        <div class="actions" id="actions" style="{% if job.status != 'complete' %}display:none{% endif %}">
            <a href="/job/{{ job.id }}/download" class="btn">Download ZIP</a>
        </div>

        <div class="results" id="results" style="{% if job.status != 'complete' %}display:none{% endif %}">
            <div class="transcript-panel">
                <h2>Transcript</h2>
                <pre id="transcript">{{ transcript }}</pre>
            </div>
            <div class="screenshots-panel">
                <h2>Screenshots</h2>
                <div class="screenshot-grid" id="screenshot-grid">
                    {% if job.screenshots %}
                    {% for shot in job.screenshots %}
                    <div class="screenshot-card">
                        <img src="/job/{{ job.id }}/screenshots/{{ shot.filename }}" alt="{{ shot.filename }}">
                        <span class="timestamp">{{ '%02d:%02d:%02d' % (shot.seconds // 3600, (shot.seconds % 3600) // 60, shot.seconds % 60) }}</span>
                    </div>
                    {% endfor %}
                    {% endif %}
                </div>
            </div>
        </div>
    </div>

    {% if job.status in ['queued', 'processing'] %}
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.5/socket.io.min.js"></script>
    <script>
        const socket = io();
        const jobId = "{{ job.id }}";

        socket.on("connect", () => {
            socket.emit("join", { job_id: jobId });
            document.getElementById("progress-message").textContent = "Waiting for processing to start...";
        });

        socket.on("progress", (data) => {
            if (data.job_id === jobId) {
                document.getElementById("progress-message").textContent = data.message;
            }
        });

        socket.on("complete", (data) => {
            if (data.job_id === jobId) {
                window.location.reload();
            }
        });

        socket.on("error", (data) => {
            if (data.job_id === jobId) {
                document.getElementById("progress-message").textContent = "Error: " + data.message;
                document.querySelector(".spinner").style.display = "none";
            }
        });
    </script>
    {% endif %}
</body>
</html>
```

**Step 3: Create static/style.css**

Clean, minimal CSS for the app.

```css
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f5;
    color: #333;
    line-height: 1.6;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

h1 { margin-bottom: 0.25rem; }
h2 { margin: 1.5rem 0 1rem; }
.subtitle { color: #666; margin-bottom: 2rem; }
.meta { color: #666; margin-bottom: 1rem; }
.back-link { color: #0066cc; text-decoration: none; display: inline-block; margin-bottom: 1rem; }

/* Upload form */
.upload-form {
    background: white;
    padding: 2rem;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    max-width: 500px;
}
.form-group { margin-bottom: 1.5rem; }
.form-group label { display: block; font-weight: 600; margin-bottom: 0.5rem; }
.form-group input[type="file"] { width: 100%; }

.btn {
    display: inline-block;
    background: #0066cc;
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 1rem;
    text-decoration: none;
}
.btn:hover { background: #0052a3; }

/* Job table */
.job-table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.job-table th, .job-table td { padding: 0.75rem 1rem; text-align: left; }
.job-table thead { background: #f8f8f8; }
.job-table tbody tr { cursor: pointer; border-top: 1px solid #eee; }
.job-table tbody tr:hover { background: #f0f7ff; }

.status { padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.85rem; }
.status-complete { background: #d4edda; color: #155724; }
.status-processing { background: #fff3cd; color: #856404; }
.status-queued { background: #e2e3e5; color: #383d41; }
.status-error { background: #f8d7da; color: #721c24; }

/* Progress */
.progress-section { text-align: center; padding: 3rem; }
.spinner {
    width: 40px; height: 40px; margin: 0 auto 1rem;
    border: 4px solid #e0e0e0; border-top-color: #0066cc;
    border-radius: 50%; animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

.error-banner { background: #f8d7da; color: #721c24; padding: 1rem; border-radius: 8px; margin: 1rem 0; }

/* Results layout */
.actions { margin-bottom: 1.5rem; }
.results { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }

.transcript-panel pre {
    background: white;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    white-space: pre-wrap;
    word-wrap: break-word;
    max-height: 80vh;
    overflow-y: auto;
    font-size: 0.9rem;
    line-height: 1.8;
}

.screenshot-grid {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    max-height: 80vh;
    overflow-y: auto;
}

.screenshot-card {
    background: white;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    overflow: hidden;
}
.screenshot-card img { width: 100%; display: block; }
.screenshot-card .timestamp {
    display: block;
    padding: 0.5rem;
    text-align: center;
    font-family: monospace;
    background: #f8f8f8;
    color: #555;
}

@media (max-width: 768px) {
    .results { grid-template-columns: 1fr; }
}
```

**Step 4: Verify templates render**

Run: `python -c "from app import app; client = app.test_client(); r = client.get('/'); print(r.status_code, 'index' if b'Transcribr' in r.data else 'FAIL')"`
Expected: `200 index`

**Step 5: Commit**

```bash
mkdir -p templates static
git add templates/ static/
git commit -m "feat: add HTML templates and CSS for upload, job view, and history"
```

---

### Task 6: Integration Test with Real Files

**Files:**
- Modify: `tests/test_transcriber.py`

**Step 1: Write an integration test (skippable if no ffmpeg)**

```python
# tests/test_transcriber.py (append)
import shutil
import pytest

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_extract_screenshots_integration(tmp_path):
    """Integration test with a real (tiny) video file generated by ffmpeg."""
    # Generate a 30-second test video with color changes
    test_video = str(tmp_path / "test.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i",
        "color=c=red:s=320x240:d=10,drawtext=text='Part 1':fontsize=40:fontcolor=white:x=10:y=10[v1];"
        "color=c=blue:s=320x240:d=10,drawtext=text='Part 2':fontsize=40:fontcolor=white:x=10:y=10[v2];"
        "color=c=green:s=320x240:d=10,drawtext=text='Part 3':fontsize=40:fontcolor=white:x=10:y=10[v3];"
        "[v1][v2][v3]concat=n=3:v=1:a=0",
        test_video
    ], check=True, capture_output=True)

    screenshots_dir = str(tmp_path / "screenshots")
    result = extract_screenshots(test_video, screenshots_dir, interval=10)

    # Should have ~3 unique frames (red, blue, green)
    assert len(result) >= 2  # At minimum red and one other
    for filename, seconds in result:
        assert os.path.exists(os.path.join(screenshots_dir, filename))
        assert "m" in filename and "s" in filename  # Has timestamp in name
```

**Step 2: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS (integration test may skip if ffmpeg not installed)

**Step 3: Commit**

```bash
git add tests/
git commit -m "test: add integration test for screenshot extraction"
```

---

### Task 7: Final Verification

**Step 1: Install ffmpeg if not present**

Run: `brew install ffmpeg`

**Step 2: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

**Step 3: Start the app and manually test**

Run: `python app.py`
Then open http://127.0.0.1:5000 in browser, upload the Zoom files, verify:
- Progress updates appear in real-time
- Transcript displays with timestamps
- Screenshots display with timestamps
- Zip downloads correctly
- Job appears in history on home page

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: Transcribr v1 — Zoom transcription and screenshot extraction web app"
```
