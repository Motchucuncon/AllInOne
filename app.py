"""Gradio interface for the Faceless Review Video Pipeline.

Provides:
- Topic input
- OpenRouter model selector (DeepSeek, Llama, etc.)
- Video model selector (Wan 2.1 / LTX-Video)
- Storyboard editor (view and edit generated scenes before rendering)
- Full pipeline execution (generate -> render -> compose)
- Video output display and download
"""

import json
import os
import tempfile
import traceback
from typing import Any

import gradio as gr

from core.script_gen import generate_storyboard
from core.audio_gen import generate_audio_and_subtitles
from core.image_gen import generate_broll_images
from core.video_gen import render_video_clips
from core.composer import compose_final_video

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPENROUTER_MODELS = [
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-chat:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemini-2.0-flash-lite-preview:free",
    "qwen/qwen-2.5-72b-instruct:free",
]

VIDEO_MODELS = [
    "wan_2_1",
    "ltx_video",
]

TTS_VOICES = [
    "vi-VN-HoaiMyNeural",
    "vi-VN-NamMinhNeural",
]

OUTPUT_DIR = "output"


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------

def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_storyboard_generation(
    topic: str,
    openrouter_model: str,
    api_key: str,
) -> tuple[str, str]:
    """Step 1: Generate storyboard JSON from topic using OpenRouter."""
    _ensure_output_dir()

    if not topic.strip():
        return "", "⚠️ Please enter a review topic."

    try:
        storyboard = generate_storyboard(
            topic=topic,
            model=openrouter_model,
            api_key=api_key if api_key else None,
        )

        # Save storyboard to disk
        storyboard_path = os.path.join(OUTPUT_DIR, "storyboard.json")
        with open(storyboard_path, "w", encoding="utf-8") as f:
            json.dump(storyboard, f, ensure_ascii=False, indent=2)

        # Format for display in the storyboard editor
        formatted = json.dumps(storyboard, ensure_ascii=False, indent=2)
        return formatted, f"✅ Storyboard generated! ({len(storyboard.get('storyboard_scenes', []))} scenes)"

    except Exception as exc:
        return "", f"❌ Storyboard generation failed: {exc}\n{traceback.format_exc()}"


def run_pipeline(
    topic: str,
    storyboard_json: str,
    openrouter_model: str,
    video_model: str,
    tts_voice: str,
    api_key: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, str]:
    """Run the full pipeline: generate storyboard, audio, images, video, compose.

    Parameters
    ----------
    topic : str
        Review topic.
    storyboard_json : str
        JSON string of the storyboard (editable by user).
    openrouter_model : str
        OpenRouter model identifier.
    video_model : str
        Video model (wan_2_1 or ltx_video).
    tts_voice : str
        Edge-TTS voice name.
    api_key : str
        OpenRouter API key.

    Returns
    -------
    tuple[str, str, str]
        (status_message, video_path, storyboard_json_output)
    """
    _ensure_output_dir()

    # ------------------------------------------------------------------
    # Step 1: Parse storyboard (use edited version if provided)
    # ------------------------------------------------------------------
    progress(0.0, desc="📝 Parsing storyboard...")

    if storyboard_json and storyboard_json.strip():
        try:
            storyboard = json.loads(storyboard_json)
        except json.JSONDecodeError as exc:
            err_msg = f"❌ Invalid storyboard JSON: {exc}"
            return err_msg, "", storyboard_json
    else:
        # Generate storyboard from topic
        progress(0.05, desc="🤖 Generating storyboard with OpenRouter...")
        try:
            storyboard = generate_storyboard(
                topic=topic,
                model=openrouter_model,
                api_key=api_key if api_key else None,
            )
            storyboard_json = json.dumps(storyboard, ensure_ascii=False, indent=2)
        except Exception as exc:
            err_msg = f"❌ Storyboard generation failed: {exc}\n{traceback.format_exc()}"
            return err_msg, "", storyboard_json

    scenes = storyboard.get("storyboard_scenes", [])
    if not scenes:
        return "❌ No scenes in storyboard.", "", storyboard_json

    # ------------------------------------------------------------------
    # Step 2: Generate audio and subtitles
    # ------------------------------------------------------------------
    progress(0.15, desc="🔊 Generating narration audio with Edge-TTS...")
    try:
        audio_result = generate_audio_and_subtitles(
            storyboard=storyboard,
            voice=tts_voice,
            output_dir=OUTPUT_DIR,
        )
        audio_path = audio_result["audio_path"]
        subtitles_path = audio_result["subtitles_path"]
        duration = audio_result["duration_seconds"]
        print(f"[app] Audio generated: {audio_path} ({duration:.1f}s)")
    except Exception as exc:
        err_msg = f"❌ Audio generation failed: {exc}\n{traceback.format_exc()}"
        return err_msg, "", storyboard_json

    # ------------------------------------------------------------------
    # Step 3: Generate B-roll images
    # ------------------------------------------------------------------
    progress(0.30, desc="🖼️ Generating B-roll images with Flux.1-schnell...")
    broll_prompts = [s["broll_prompt"] for s in scenes]
    scene_ids = [s["scene_id"] for s in scenes]

    try:
        image_paths = generate_broll_images(
            prompts=broll_prompts,
            scene_ids=scene_ids,
            output_dir=os.path.join(OUTPUT_DIR, "images"),
            unload_after=True,
        )
        print(f"[app] Generated {len(image_paths)} B-roll images.")
    except Exception as exc:
        err_msg = f"❌ Image generation failed: {exc}\n{traceback.format_exc()}"
        return err_msg, "", storyboard_json

    # ------------------------------------------------------------------
    # Step 4: Render video clips
    # ------------------------------------------------------------------
    progress(0.55, desc="🎬 Rendering video clips...")
    motion_prompts = [s.get("broll_prompt", "") for s in scenes]

    try:
        clip_paths = render_video_clips(
            image_paths=image_paths,
            prompts=motion_prompts,
            scene_ids=scene_ids,
            model=video_model,
            output_dir=os.path.join(OUTPUT_DIR, "clips"),
        )
        print(f"[app] Rendered {len(clip_paths)} video clips.")
    except Exception as exc:
        err_msg = f"❌ Video rendering failed: {exc}\n{traceback.format_exc()}"
        return err_msg, "", storyboard_json

    # ------------------------------------------------------------------
    # Step 5: Compose final video
    # ------------------------------------------------------------------
    progress(0.85, desc="🎞️ Composing final video...")
    try:
        final_video_path = compose_final_video(
            clip_paths=clip_paths,
            audio_path=audio_path,
            subtitles_path=subtitles_path,
            output_dir=OUTPUT_DIR,
        )
    except Exception as exc:
        err_msg = f"❌ Video composition failed: {exc}\n{traceback.format_exc()}"
        return err_msg, "", storyboard_json

    progress(1.0, desc="✅ Done!")

    success_msg = (
        f"✅ Video generated successfully!\n"
        f"   • Topic: {topic}\n"
        f"   • Video model: {video_model}\n"
        f"   • Scenes: {len(scenes)}\n"
        f"   • Duration: {duration:.1f}s\n"
        f"   • Output: {final_video_path}"
    )

    return success_msg, final_video_path, storyboard_json


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks interface."""

    css = """
    .gradio-container { max-width: 1200px; margin: auto; }
    .storyboard-box { font-family: monospace; font-size: 13px; }
    """

    with gr.Blocks(
        title="Faceless Review Video Generator",
        css=css,
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# 🎬 Faceless Review Video Generator\n"
            "Generate automated review videos with AI narration, B-roll, and subtitles."
        )

        with gr.Row():
            with gr.Column(scale=1):
                # --- Input Section ---
                gr.Markdown("### 📝 Input")
                topic_input = gr.Textbox(
                    label="Review Topic",
                    placeholder="e.g., iPhone 15 review, Top 5 laptop 2024, ...",
                    lines=2,
                )

                with gr.Row():
                    openrouter_model = gr.Dropdown(
                        choices=OPENROUTER_MODELS,
                        value=OPENROUTER_MODELS[0],
                        label="OpenRouter Model",
                    )
                    video_model = gr.Dropdown(
                        choices=VIDEO_MODELS,
                        value=VIDEO_MODELS[0],
                        label="Video Model",
                    )

                tts_voice = gr.Dropdown(
                    choices=TTS_VOICES,
                    value=TTS_VOICES[0],
                    label="TTS Voice",
                )

                api_key = gr.Textbox(
                    label="OpenRouter API Key (optional, or set OPENROUTER_API_KEY env)",
                    type="password",
                    placeholder="sk-or-v1-...",
                )

                with gr.Row():
                    generate_btn = gr.Button("🚀 Generate Storyboard", variant="secondary")
                    run_btn = gr.Button("▶️ Run Full Pipeline", variant="primary", size="lg")

                status_output = gr.Textbox(
                    label="Status",
                    lines=4,
                    interactive=False,
                )

            with gr.Column(scale=1):
                # --- Storyboard Editor ---
                gr.Markdown("### 📋 Storyboard Editor")
                gr.Markdown(
                    "Edit the JSON storyboard before running the full pipeline. "
                    "Change narration text, B-roll prompts, or scene structure."
                )
                storyboard_editor = gr.Textbox(
                    label="Storyboard JSON",
                    lines=20,
                    elem_classes=["storyboard-box"],
                )

        # --- Output Section ---
        gr.Markdown("### 🎥 Output Video")
        video_output = gr.Video(
            label="Final Video",
            height=400,
        )

        # ------------------------------------------------------------------
        # Event handlers
        # ------------------------------------------------------------------

        # Generate storyboard only
        generate_btn.click(
            fn=run_storyboard_generation,
            inputs=[topic_input, openrouter_model, api_key],
            outputs=[storyboard_editor, status_output],
        )

        # Run full pipeline
        run_btn.click(
            fn=run_pipeline,
            inputs=[
                topic_input,
                storyboard_editor,
                openrouter_model,
                video_model,
                tts_voice,
                api_key,
            ],
            outputs=[status_output, video_output, storyboard_editor],
        )

        # Examples
        gr.Markdown("### 💡 Examples")
        gr.Examples(
            examples=[
                ["Đánh giá iPhone 15 Pro Max"],
                ["Top 5 quán cà phê đẹp nhất Sài Gòn"],
                ["Review Samsung Galaxy S24 Ultra"],
                ["Đánh giá xe máy điện VinFast"],
            ],
            inputs=[topic_input],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        share=True,
        debug=False,
        server_name="0.0.0.0",
        server_port=7860,
    )