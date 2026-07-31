"""Audio generation module using edge-tts.

Converts Vietnamese narration text into:
- narration.mp3 audio file
- subtitles.srt subtitle file with word-level timing
"""

import asyncio
import math
import os
import re
import edge_tts

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_VOICE = "vi-VN-HoaiMyNeural"  # Vietnamese female voice
DEFAULT_OUTPUT_DIR = "output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def _estimate_chunk_duration_seconds(text: str, wpm: int = 160) -> float:
    """Estimate how long a Vietnamese text chunk takes to speak.

    Uses words-per-minute as a rough heuristic.  Vietnamese is typically
    spoken at 140-180 wpm.
    """
    word_count = len(text.split())
    return (word_count / wpm) * 60.0


def _format_srt_time(seconds: float) -> str:
    """Convert seconds (float) to SRT time format HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _generate_srt(
    scenes: list[dict],
    full_audio_duration: float | None = None,
    output_path: str = "output/subtitles.srt",
) -> str:
    """Generate an SRT subtitle file from storyboard scenes and timings.

    Parameters
    ----------
    scenes : list[dict]
        Each scene must have keys: scene_id, narration_chunk.
    full_audio_duration : float or None
        Total duration of the full narration audio in seconds.
        If None, it will be estimated.
    output_path : str
        Where to write the .srt file.

    Returns
    -------
    str
        Path to the generated .srt file.
    """
    _ensure_dir(os.path.dirname(output_path))

    # Estimate durations for each chunk proportionally
    if full_audio_duration is not None and full_audio_duration > 0:
        total_estimated = sum(
            _estimate_chunk_duration_seconds(s["narration_chunk"]) for s in scenes
        )
        if total_estimated <= 0:
            total_estimated = 1.0
        scale = full_audio_duration / total_estimated
    else:
        scale = 1.0

    lines = []
    current_time = 0.0

    for scene in scenes:
        chunk_duration = _estimate_chunk_duration_seconds(scene["narration_chunk"]) * scale
        if chunk_duration < 1.0:
            chunk_duration = 1.0  # minimum 1 second per scene

        start_time = current_time
        end_time = current_time + chunk_duration

        lines.append(str(scene["scene_id"]))
        lines.append(
            f"{_format_srt_time(start_time)} --> {_format_srt_time(end_time)}"
        )
        lines.append(scene["narration_chunk"])
        lines.append("")  # blank line separates entries

        current_time = end_time

    srt_content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def _generate_audio_async(
    text: str,
    voice: str = DEFAULT_VOICE,
    output_path: str = "output/narration.mp3",
) -> tuple[str, float]:
    """Async helper: generate speech audio with edge-tts.

    Returns
    -------
    tuple[str, float]
        (output_path, duration_seconds)
    """
    _ensure_dir(os.path.dirname(output_path))

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

    # Estimate duration from file size (rough heuristic)
    # edge-tts doesn't directly return duration, so we estimate
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        # MP3 at ~16 kbps for speech: bytes -> bits -> seconds
        estimated_duration = (file_size * 8) / 16000
    else:
        estimated_duration = _estimate_chunk_duration_seconds(text)

    return output_path, estimated_duration


def generate_audio_and_subtitles(
    storyboard: dict,
    voice: str = DEFAULT_VOICE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Generate narration audio and subtitle files from a storyboard.

    Parameters
    ----------
    storyboard : dict
        Must contain 'full_narration' (str) and 'storyboard_scenes' (list of dict).
    voice : str
        edge-tts voice name.  Defaults to vi-VN-HoaiMyNeural.
    output_dir : str
        Directory to write output files.

    Returns
    -------
    dict
        {
            "audio_path": str,
            "subtitles_path": str,
            "duration_seconds": float,
            "scenes": list[dict]  # original scenes with added timing info
        }
    """
    _ensure_dir(output_dir)

    full_narration = storyboard.get("full_narration", "")
    scenes = storyboard.get("storyboard_scenes", [])

    audio_path = os.path.join(output_dir, "narration.mp3")
    subtitles_path = os.path.join(output_dir, "subtitles.srt")

    # Generate audio using asyncio event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    audio_path, duration = loop.run_until_complete(
        _generate_audio_async(full_narration, voice, audio_path)
    )

    # Generate SRT subtitles
    subtitles_path = _generate_srt(scenes, duration, subtitles_path)

    # Add timing info to scenes
    if duration > 0 and scenes:
        total_estimated = sum(
            _estimate_chunk_duration_seconds(s["narration_chunk"]) for s in scenes
        )
        if total_estimated <= 0:
            total_estimated = 1.0
        scale = duration / total_estimated
        current_time = 0.0
        for scene in scenes:
            chunk_dur = _estimate_chunk_duration_seconds(scene["narration_chunk"]) * scale
            if chunk_dur < 1.0:
                chunk_dur = 1.0
            scene["start_time"] = current_time
            scene["end_time"] = current_time + chunk_dur
            current_time += chunk_dur

    return {
        "audio_path": audio_path,
        "subtitles_path": subtitles_path,
        "duration_seconds": duration,
        "scenes": scenes,
    }