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


from transcriber import extract_screenshots

def test_extract_screenshots_calls_ffmpeg(tmp_path):
    """Test that extract_screenshots invokes ffmpeg correctly."""
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
    paths = []
    for i in range(4):
        # Create images with distinct patterns so perceptual hashes differ
        img = Image.new("RGB", (128, 128), color="white")
        pixels = img.load()
        # Draw a unique thick block in a different quadrant for each image
        quad_x = (i % 2) * 64
        quad_y = (i // 2) * 64
        for x in range(quad_x, quad_x + 64):
            for y in range(quad_y, quad_y + 64):
                pixels[x, y] = (0, 0, 0)
        p = tmp_path / f"frame_{i}.png"
        img.save(str(p))
        paths.append((str(p), i * 10))

    result = _deduplicate_frames(paths, threshold=5)
    assert len(result) == 4
