"""Offline tests for the video-merger skill.

Tests that ``get_sorted_videos`` and ``get_video_info`` handle edge cases
gracefully instead of crashing with ``AttributeError`` / ``ValueError``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src/agentos/skills/bundled/video-merger/src/video_merger.py"
)

_spec = importlib.util.spec_from_file_location("video_merger", _SCRIPT)
assert _spec is not None and _spec.loader is not None
video_merger = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(video_merger)


def _make_merger() -> video_merger.VideoMerger:  # type: ignore[attr-defined]
    """Create a VideoMerger with mocked ffmpeg/ffprobe paths."""
    with patch.object(video_merger.subprocess, "run", return_value=MagicMock(returncode=0)):
        return video_merger.VideoMerger(  # type: ignore[attr-defined]
            ffmpeg_path="/usr/bin/ffmpeg",
            ffprobe_path="/usr/bin/ffprobe",
        )


def test_get_sorted_videos_skips_files_without_digit_prefix(tmp_path: Path) -> None:
    """Files that don't match the ``\\d+_`` prefix must be skipped, not crash."""
    # Create files: one properly named, one not
    (tmp_path / "01_intro.mp4").write_bytes(b"")
    (tmp_path / "README.mp4").write_bytes(b"")
    (tmp_path / "02_outro.mp4").write_bytes(b"")

    merger = _make_merger()
    result = merger.get_sorted_videos(str(tmp_path))
    assert len(result) == 2
    assert "01_intro" in result[0]
    assert "02_outro" in result[1]


def test_get_sorted_videos_raises_when_no_valid_files(tmp_path: Path) -> None:
    """When no file matches the naming convention, raise ValueError."""
    (tmp_path / "random.mp4").write_bytes(b"")
    (tmp_path / "other.txt").write_bytes(b"")

    merger = _make_merger()
    with pytest.raises(ValueError, match="未找到"):
        merger.get_sorted_videos(str(tmp_path))


def test_get_video_info_handles_missing_duration() -> None:
    """When ffprobe returns fewer than 3 lines, don't crash with ValueError."""
    merger = _make_merger()

    # Simulate ffprobe returning only 2 lines (e.g. corrupt file without duration)
    mock_result = MagicMock()
    mock_result.stdout = "1920\n1080\n"  # only width+height, no duration
    with patch.object(video_merger.subprocess, "run", return_value=mock_result):
        result = merger.get_video_info("/fake/path.mp4")
        assert result[0] == 1920
        assert result[1] == 1080
        assert result[2] == 0.0  # default duration when missing


def test_get_video_info_handles_empty_output() -> None:
    """When ffprobe returns nothing, don't crash."""
    merger = _make_merger()

    mock_result = MagicMock()
    mock_result.stdout = ""
    with patch.object(video_merger.subprocess, "run", return_value=mock_result):
        result = merger.get_video_info("/fake/path.mp4")
        assert result == (0, 0, 0.0)
