import os
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

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
        job = load_job(job_id)
        job["status"] = "processing"
        save_job(job_id, job)

        progress("Starting transcription...")
        transcribe_audio(audio_path, transcript_path, progress_callback=progress)

        progress("Starting screenshot extraction...")
        saved_frames = extract_screenshots(video_path, screenshots_dir, interval=10, progress_callback=progress)

        progress("Creating zip archive...")
        zip_path = os.path.join(job_dir, f"transcribr_{job_id}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(transcript_path, "transcription.txt")
            for frame_name, _ in saved_frames:
                frame_path = os.path.join(screenshots_dir, frame_name)
                zf.write(frame_path, f"screenshots/{frame_name}")

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
    socketio.run(app, debug=True, port=5001, allow_unsafe_werkzeug=True)
