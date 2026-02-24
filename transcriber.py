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
