import os
import tempfile
from unittest.mock import patch, MagicMock

from transcriber import format_timestamp

def test_format_timestamp_zero():
    assert format_timestamp(0.0) == "00:00:00"

def test_format_timestamp_seconds():
    assert format_timestamp(65.5) == "00:01:05"

def test_format_timestamp_hours():
    assert format_timestamp(3661.0) == "01:01:01"


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
