#!/usr/bin/env python3
"""CLI entry point for the Faceless Review Video Pipeline.

Consolidates the full pipeline into a single executable Python script.

Usage:
    # Full pipeline (one-shot)
    python run_pipeline.py --topic "Đánh giá iPhone 15 Pro Max"

    # Specify models
    python run_pipeline.py --topic "Review Samsung Galaxy" --openrouter-model "deepseek/deepseek-r1:free" --video-model ltx_video

    # Launch Gradio web UI
    python run_pipeline.py --ui

    # Use existing storyboard (skip OpenRouter)
    python run_pipeline.py --storyboard output/storyboard.json

    # Generate storyboard only
    python run_pipeline.py --topic "Top 5 laptop" --storyboard-only

Environment Variables:
    OPENROUTER_API_KEY    Your OpenRouter API key (or pass via --api-key)
"""

import argparse
import json
import os
import sys
import traceback

# ---------------------------------------------------------------------------
# Ensure core package is importable
# ---------------------------------------------------------------------------
_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from core.script_gen import generate_storyboard
from core.audio_gen import generate_audio_and_subtitles
from core.image_gen import generate_broll_images
from core.video_gen import render_video_clips
from core.composer import compose_final_video

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-r1:free"
DEFAULT_VIDEO_MODEL = "wan_2_1"
DEFAULT_TTS_VOICE = "vi-VN-HoaiMyNeural"


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Faceless Review Video Pipeline — CLI entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_pipeline.py --topic \"Đánh giá iPhone 15\"\n"
            "  python run_pipeline.py --topic \"Top 5 laptop\" --video-model ltx_video\n"
            "  python run_pipeline.py --ui\n"
            "  python run_pipeline.py --storyboard output/storyboard.json\n"
        ),
    )

    # Pipeline mode
    parser.add_argument(
        "--topic", "-t",
        type=str,
        default="",
        help="Review topic (Vietnamese, e.g. 'Đánh giá iPhone 15 Pro Max')",
    )
    parser.add_argument(
        "--storyboard", "-s",
        type=str,
        default="",
        help="Path to existing storyboard JSON file (skip OpenRouter generation)",
    )

    # Model selection
    parser.add_argument(
        "--openrouter-model", "-om",
        type=str,
        default=DEFAULT_OPENROUTER_MODEL,
        help=f"OpenRouter model ID (default: {DEFAULT_OPENROUTER_MODEL})",
    )
    parser.add_argument(
        "--video-model", "-vm",
        type=str,
        default=DEFAULT_VIDEO_MODEL,
        choices=["wan_2_1", "ltx_video"],
        help=f"Video generation model (default: {DEFAULT_VIDEO_MODEL})",
    )
    parser.add_argument(
        "--tts-voice", "-tv",
        type=str,
        default=DEFAULT_TTS_VOICE,
        choices=["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"],
        help=f"Edge-TTS voice (default: {DEFAULT_TTS_VOICE})",
    )

    # API key
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
    )

    # Modes
    parser.add_argument(
        "--ui", "-u",
        action="store_true",
        help="Launch Gradio web UI instead of running the pipeline",
    )
    parser.add_argument(
        "--storyboard-only", "-so",
        action="store_true",
        help="Generate storyboard only, then exit",
    )

    # Output
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    return parser


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    topic: str = "",
    storyboard_path: str = "",
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL,
    video_model: str = DEFAULT_VIDEO_MODEL,
    tts_voice: str = DEFAULT_TTS_VOICE,
    api_key: str | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    storyboard_only: bool = False,
) -> str:
    """Run the full video generation pipeline.

    Parameters
    ----------
    topic : str
        Review topic (used if no storyboard_path provided).
    storyboard_path : str
        Path to existing storyboard JSON (skips OpenRouter if provided).
    openrouter_model : str
        OpenRouter model identifier.
    video_model : str
        Video model name ("wan_2_1" or "ltx_video").
    tts_voice : str
        Edge-TTS voice name.
    api_key : str or None
        OpenRouter API key.
    output_dir : str
        Output directory for all generated assets.
    storyboard_only : bool
        If True, only generate storyboard and exit.

    Returns
    -------
    str
        Path to the final composed video, or empty string on failure.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load or generate storyboard
    # ------------------------------------------------------------------
    storyboard = None

    if storyboard_path and os.path.exists(storyboard_path):
        print(f"\n{'='*60}")
        print("📖 STEP 1/5: Loading existing storyboard...")
        print(f"{'='*60}")
        with open(storyboard_path, "r", encoding="utf-8") as f:
            storyboard = json.load(f)
        print(f"   ✅ Loaded storyboard from {storyboard_path}")
        print(f"   📋 Scenes: {len(storyboard.get('storyboard_scenes', []))}")
        topic = storyboard.get("topic", topic)
    else:
        if not topic:
            print("❌ No topic provided and no storyboard file found.")
            print("   Use --topic or --storyboard")
            return ""

        print(f"\n{'='*60}")
        print("🤖 STEP 1/5: Generating storyboard with OpenRouter...")
        print(f"{'='*60}")
        print(f"   Topic: {topic}")
        print(f"   Model: {openrouter_model}")

        storyboard = generate_storyboard(
            topic=topic,
            model=openrouter_model,
            api_key=api_key,
        )

        # Save storyboard
        storyboard_path_out = os.path.join(output_dir, "storyboard.json")
        with open(storyboard_path_out, "w", encoding="utf-8") as f:
            json.dump(storyboard, f, ensure_ascii=False, indent=2)
        print(f"   ✅ Storyboard saved to {storyboard_path_out}")
        print(f"   📋 Scenes: {len(storyboard.get('storyboard_scenes', []))}")

        # If storyboard-only mode, exit here
        if storyboard_only:
            print(f"\n{'='*60}")
            print("📝 Storyboard-only mode. Exiting.")
            print(f"{'='*60}")
            print(f"\n📋 Storyboard preview:\n{json.dumps(storyboard, ensure_ascii=False, indent=2)}")
            return ""

    scenes = storyboard.get("storyboard_scenes", [])
    if not scenes:
        print("❌ No scenes in storyboard. Aborting.")
        return ""

    # ------------------------------------------------------------------
    # Step 2: Generate audio and subtitles
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("🔊 STEP 2/5: Generating narration audio with Edge-TTS...")
    print(f"{'='*60}")
    print(f"   Voice: {tts_voice}")

    try:
        audio_result = generate_audio_and_subtitles(
            storyboard=storyboard,
            voice=tts_voice,
            output_dir=output_dir,
        )
        print(f"   ✅ Audio: {audio_result['audio_path']}")
        print(f"   ✅ Subtitles: {audio_result['subtitles_path']}")
        print(f"   ⏱️  Duration: {audio_result['duration_seconds']:.1f}s")
    except Exception as exc:
        print(f"   ❌ Audio generation failed: {exc}")
        print(f"      {traceback.format_exc()}")
        return ""

    # ------------------------------------------------------------------
    # Step 3: Generate B-roll images
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("🖼️  STEP 3/5: Generating B-roll images with Flux.1-schnell...")
    print(f"{'='*60}")

    broll_prompts = [s.get("broll_prompt", "") for s in scenes]
    scene_ids = [s.get("scene_id", i + 1) for i, s in enumerate(scenes)]

    try:
        image_paths = generate_broll_images(
            prompts=broll_prompts,
            scene_ids=scene_ids,
            output_dir=os.path.join(output_dir, "images"),
            unload_after=True,
        )
        print(f"   ✅ Generated {len(image_paths)} B-roll images")
        for img_path in image_paths:
            print(f"      • {img_path}")
    except Exception as exc:
        print(f"   ❌ Image generation failed: {exc}")
        print(f"      {traceback.format_exc()}")
        return ""

    # ------------------------------------------------------------------
    # Step 4: Render video clips
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"🎬 STEP 4/5: Rendering video clips with {video_model}...")
    print(f"{'='*60}")

    motion_prompts = [s.get("broll_prompt", "") for s in scenes]

    try:
        clip_paths = render_video_clips(
            image_paths=image_paths,
            prompts=motion_prompts,
            scene_ids=scene_ids,
            model=video_model,
            output_dir=os.path.join(output_dir, "clips"),
        )
        print(f"   ✅ Rendered {len(clip_paths)} video clips")
        for clip_path in clip_paths:
            print(f"      • {clip_path}")
    except Exception as exc:
        print(f"   ❌ Video rendering failed: {exc}")
        print(f"      {traceback.format_exc()}")
        return ""

    # ------------------------------------------------------------------
    # Step 5: Compose final video
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("🎞️  STEP 5/5: Composing final video...")
    print(f"{'='*60}")

    try:
        final_video_path = compose_final_video(
            clip_paths=clip_paths,
            audio_path=audio_result["audio_path"],
            subtitles_path=audio_result["subtitles_path"],
            output_dir=output_dir,
        )
        print(f"   ✅ Final video: {final_video_path}")
    except Exception as exc:
        print(f"   ❌ Video composition failed: {exc}")
        print(f"      {traceback.format_exc()}")
        return ""

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print(f"   📝 Topic: {topic}")
    print(f"   🎬 Video model: {video_model}")
    print(f"   📋 Scenes: {len(scenes)}")
    print(f"   ⏱️  Duration: {audio_result['duration_seconds']:.1f}s")
    print(f"   📁 Output: {final_video_path}")
    print(f"\n📥 Output files are in: {os.path.abspath(output_dir)}")
    print(f"   • Final video: {final_video_path}")
    print(f"   • Audio: {audio_result['audio_path']}")
    print(f"   • Subtitles: {audio_result['subtitles_path']}")
    print(f"   • Storyboard: {os.path.join(output_dir, 'storyboard.json')}")
    print(f"   • Images: {os.path.join(output_dir, 'images/')}")
    print(f"   • Clips: {os.path.join(output_dir, 'clips/')}")

    return final_video_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = _build_parser()
    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")

    # Launch Gradio UI mode
    if args.ui:
        print("🚀 Launching Gradio web UI...")
        print(f"   Open http://localhost:7860 in your browser\n")
        from app import build_ui
        demo = build_ui()
        demo.launch(
            share=True,
            debug=False,
            server_name="0.0.0.0",
            server_port=7860,
        )
        return

    # Run pipeline
    result = run_pipeline(
        topic=args.topic,
        storyboard_path=args.storyboard,
        openrouter_model=args.openrouter_model,
        video_model=args.video_model,
        tts_voice=args.tts_voice,
        api_key=api_key or None,
        output_dir=args.output_dir,
        storyboard_only=args.storyboard_only,
    )

    if not result:
        sys.exit(1)


if __name__ == "__main__":
    main()