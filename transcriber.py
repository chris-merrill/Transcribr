# transcriber.py
import whisper


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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
