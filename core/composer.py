"""Video composer module using MoviePy.

Stitches the final review video by overlaying:
1. Narration audio track
2. B-roll video clips (one per scene)
3. Burn-in subtitles (SRT hardcoded onto video)
"""

import os
import subprocess

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_OUTPUT_FILENAME = "final_review_video.mp4"
DEFAULT_VIDEO_SIZE = (1024, 576)
DEFAULT_FPS = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _burn_subtitles_ffmpeg(
    video_path: str,
    subtitles_path: str,
    output_path: str,
) -> str:
    """Burn subtitles into a video using FFmpeg's subtitles filter.

    This produces a new video file with hardcoded (burned-in) subtitles.

    Parameters
    ----------
    video_path : str
        Path to the input video (without subtitles).
    subtitles_path : str
        Path to the .srt subtitle file.
    output_path : str
        Path for the output video with burned-in subtitles.

    Returns
    -------
    str
        Path to the output video with subtitles.
    """
    # Escape special characters in paths for FFmpeg filter
    # FFmpeg subtitles filter requires escaped paths on Windows
    escaped_subs = subtitles_path.replace(":", "\\:").replace("'", "'\\\\''")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"subtitles='{escaped_subs}'",
        "-c:a", "aac",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[composer] Subtitles burned into: {output_path}")
    except subprocess.CalledProcessError as exc:
        print(f"[composer] FFmpeg subtitle burn failed: {exc.stderr}")
        # Fallback: copy input as output (no subtitles)
        import shutil
        shutil.copy2(video_path, output_path)
        print(f"[composer] Copied video without subtitles to: {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compose_final_video(
    clip_paths: list[str],
    audio_path: str,
    subtitles_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
    video_size: tuple[int, int] = DEFAULT_VIDEO_SIZE,
    fps: int = DEFAULT_FPS,
) -> str:
    """Compose the final review video from clips, audio, and subtitles.

    The pipeline:
    1. Concatenate all B-roll video clips into a single video track.
    2. Overlay the narration audio onto the concatenated video.
    3. Burn subtitles into the final video.

    Parameters
    ----------
    clip_paths : list of str
        Paths to the individual B-roll video clips (MP4).
    audio_path : str
        Path to the narration audio file (MP3).
    subtitles_path : str
        Path to the SRT subtitle file.
    output_dir : str
        Directory to write the final output.
    output_filename : str
        Name of the final output video file.
    video_size : tuple of (width, height)
        Output video resolution.
    fps : int
        Output video frame rate.

    Returns
    -------
    str
        Path to the final composed video file.
    """
    _ensure_dir(output_dir)
    output_path = os.path.join(output_dir, output_filename)
    temp_video_no_subs = os.path.join(output_dir, "temp_no_subs.mp4")

    # ------------------------------------------------------------------
    # Step 1: Concatenate video clips and add audio
    # ------------------------------------------------------------------
    if not clip_paths:
        print("[composer] No video clips provided. Creating a blank video.")
        # Create a blank video with the audio duration
        try:
            # Get audio duration
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            audio_duration = float(result.stdout.strip())
        except Exception:
            audio_duration = 30.0  # fallback

        # Create a blank video with audio
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=0x1E1E3C:s={video_size[0]}x{video_size[1]}:d={audio_duration}:r={fps}",
            "-i", audio_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            "-shortest",
            temp_video_no_subs,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    else:
        # Create a concat file for FFmpeg
        concat_file = os.path.join(output_dir, "concat_list.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            for clip_path in clip_paths:
                if os.path.exists(clip_path):
                    # Use absolute path to avoid issues
                    abs_path = os.path.abspath(clip_path)
                    f.write(f"file '{abs_path}'\n")

        # Concatenate videos and add audio
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-i", audio_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            "-shortest",  # match duration to shortest input (audio or video)
            temp_video_no_subs,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"[composer] Concatenated video + audio: {temp_video_no_subs}")
        except subprocess.CalledProcessError as exc:
            print(f"[composer] FFmpeg concat failed: {exc.stderr}")
            # Fallback: use MoviePy
            try:
                from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips

                print("[composer] Falling back to MoviePy for concatenation...")
                clips = []
                for cp in clip_paths:
                    if os.path.exists(cp):
                        clips.append(VideoFileClip(cp))

                if clips:
                    final_clip = concatenate_videoclips(clips, method="compose")
                    audio = AudioFileClip(audio_path)
                    final_clip = final_clip.set_audio(audio)
                    final_clip.write_videofile(
                        temp_video_no_subs,
                        codec="libx264",
                        audio_codec="aac",
                        fps=fps,
                    )
                    for c in clips:
                        c.close()
                    final_clip.close()
                    audio.close()
                else:
                    raise ValueError("No valid clips to concatenate")
            except Exception as mp_err:
                print(f"[composer] MoviePy fallback also failed: {mp_err}")
                # Ultimate fallback: copy audio to a blank video
                cmd_fallback = [
                    "ffmpeg", "-y",
                    "-f", "lavfi",
                    "-i", f"color=c=0x1E1E3C:s={video_size[0]}x{video_size[1]}:d=30:r={fps}",
                    "-i", audio_path,
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-pix_fmt", "yuv420p",
                    "-shortest",
                    temp_video_no_subs,
                ]
                subprocess.run(cmd_fallback, check=True, capture_output=True, text=True)

        # Cleanup concat file
        try:
            os.remove(concat_file)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Step 2: Burn subtitles
    # ------------------------------------------------------------------
    if os.path.exists(subtitles_path):
        final_path = _burn_subtitles_ffmpeg(temp_video_no_subs, subtitles_path, output_path)
    else:
        print("[composer] No subtitles file found, skipping subtitle burn.")
        import shutil
        shutil.copy2(temp_video_no_subs, output_path)
        final_path = output_path

    # ------------------------------------------------------------------
    # Step 3: Cleanup temp file
    # ------------------------------------------------------------------
    try:
        if os.path.exists(temp_video_no_subs):
            os.remove(temp_video_no_subs)
    except OSError:
        pass

    print(f"[composer] Final video created: {final_path}")
    return final_path