"""Gradio Web UI for the Faceless Review Video Pipeline.

Single-click interface:
1. Enter OpenRouter API Key + Review Topic
2. Select Video Model (Wan 2.1 / LTX-Video)
3. Click "🚀 Bắt đầu tạo Video Review"
4. Pipeline runs: Script → Audio/Sub → Flux B-roll → Wan/LTX → Composer
5. Output: Audio file, B-roll clips, Final Review Video

Usage:
    python app.py
"""

import json
import os
import sys
import traceback
from typing import Any

import gradio as gr

# Ensure core package is importable
_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from core.script_gen import generate_storyboard
from core.audio_gen import generate_audio_and_subtitles
from core.image_gen import generate_broll_images
from core.video_gen import render_video_clips
from core.composer import compose_final_video

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VIDEO_MODELS = ["wan_2_1", "ltx_video"]
TTS_VOICES = ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"]
OUTPUT_DIR = "output"
OPENROUTER_MODELS = [
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-chat:free",
    "deepseek/deepseek-v4-flash",
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemini-2.0-flash-lite-preview:free",
    "qwen/qwen-2.5-72b-instruct:free",
]


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "clips"), exist_ok=True)


# ---------------------------------------------------------------------------
# Main pipeline function (called by Gradio)
# ---------------------------------------------------------------------------

def generate_review_video(
    api_key: str,
    topic: str,
    video_model: str,
    tts_voice: str,
    openrouter_model: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str | None, list[str], str | None, str]:
    """Run the full pipeline and return results for display.

    Parameters
    ----------
    api_key : str
        OpenRouter API key.
    topic : str
        Review topic (Vietnamese).
    video_model : str
        "wan_2_1" or "ltx_video".
    tts_voice : str
        Edge-TTS voice name.
    openrouter_model : str
        OpenRouter model ID.

    Returns
    -------
    tuple
        (status_text, audio_path, broll_image_paths, video_path, storyboard_json)
    """
    _ensure_output_dir()

    # --- Validate inputs ---
    if not api_key.strip():
        return "❌ Vui lòng nhập OpenRouter API Key.", None, [], None, ""
    if not topic.strip():
        return "❌ Vui lòng nhập Chủ đề Review.", None, [], None, ""

    # Set API key
    os.environ["OPENROUTER_API_KEY"] = api_key

    # ------------------------------------------------------------------
    # STEP 1: Generate Storyboard
    # ------------------------------------------------------------------
    progress(0.05, desc="📝 Bước 1/5: Tạo Storyboard với OpenRouter...")
    try:
        storyboard = generate_storyboard(
            topic=topic,
            model=openrouter_model,
            api_key=api_key,
        )
        scenes = storyboard.get("storyboard_scenes", [])
        if not scenes:
            return "❌ Không có scene nào trong storyboard.", None, [], None, ""
        storyboard_json = json.dumps(storyboard, ensure_ascii=False, indent=2)
        print(f"[app] ✅ Storyboard: {len(scenes)} scenes")
    except Exception as exc:
        return f"❌ Lỗi Storyboard: {exc}", None, [], None, ""

    # ------------------------------------------------------------------
    # STEP 2: Generate Audio & Subtitles
    # ------------------------------------------------------------------
    progress(0.20, desc="🔊 Bước 2/5: Tạo Audio thuyết minh với Edge-TTS...")
    try:
        audio_result = generate_audio_and_subtitles(
            storyboard=storyboard,
            voice=tts_voice,
            output_dir=OUTPUT_DIR,
        )
        audio_path = audio_result["audio_path"]
        subtitles_path = audio_result["subtitles_path"]
        duration = audio_result["duration_seconds"]
        scenes = audio_result["scenes"]
        print(f"[app] ✅ Audio: {duration:.1f}s — {audio_path}")
    except Exception as exc:
        return f"❌ Lỗi Audio: {exc}", None, [], None, storyboard_json

    # ------------------------------------------------------------------
    # STEP 3: Generate B-roll Images (Flux.1-schnell)
    # ------------------------------------------------------------------
    progress(0.40, desc="🖼️ Bước 3/5: Tạo ảnh B-roll với Flux.1-schnell...")
    broll_prompts = [s.get("broll_prompt", "") for s in scenes]
    scene_ids = [s.get("scene_id", i + 1) for i, s in enumerate(scenes)]
    try:
        image_paths = generate_broll_images(
            prompts=broll_prompts,
            scene_ids=scene_ids,
            output_dir=os.path.join(OUTPUT_DIR, "images"),
            unload_after=True,
        )
        print(f"[app] ✅ {len(image_paths)} B-roll images")
    except Exception as exc:
        return f"❌ Lỗi sinh ảnh B-roll: {exc}", audio_path, [], None, storyboard_json

    # ------------------------------------------------------------------
    # STEP 4: Render Video Clips (Wan 2.1 / LTX-Video)
    # ------------------------------------------------------------------
    progress(0.65, desc=f"🎬 Bước 4/5: Render video clips với {video_model}...")
    try:
        clip_paths = render_video_clips(
            image_paths=image_paths,
            prompts=broll_prompts,
            scene_ids=scene_ids,
            model=video_model,
            output_dir=os.path.join(OUTPUT_DIR, "clips"),
        )
        print(f"[app] ✅ {len(clip_paths)} video clips")
    except Exception as exc:
        return f"❌ Lỗi render video: {exc}", audio_path, image_paths, None, storyboard_json

    # ------------------------------------------------------------------
    # STEP 5: Compose Final Video
    # ------------------------------------------------------------------
    progress(0.85, desc="🎞️ Bước 5/5: Ghép video hoàn chỉnh...")
    try:
        final_video_path = compose_final_video(
            clip_paths=clip_paths,
            audio_path=audio_path,
            subtitles_path=subtitles_path,
            output_dir=OUTPUT_DIR,
        )
        print(f"[app] ✅ Final video: {final_video_path}")
    except Exception as exc:
        return f"❌ Lỗi ghép video: {exc}", audio_path, image_paths, None, storyboard_json

    progress(1.0, desc="✅ Hoàn thành!")

    status = (
        f"✅ **Tạo video thành công!**\n\n"
        f"📝 **Chủ đề:** {topic}\n"
        f"🎬 **Model Video:** {video_model}\n"
        f"📋 **Số scene:** {len(scenes)}\n"
        f"⏱️ **Thời lượng:** {duration:.1f} giây\n"
        f"📁 **File đầu ra:** `{final_video_path}`"
    )

    return status, audio_path, image_paths, final_video_path, storyboard_json


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks UI."""

    css = """
    .gradio-container { max-width: 1000px; margin: auto; }
    footer { display: none !important; }
    """

    with gr.Blocks(
        title="🎬 Faceless Review Video Generator",
        css=css,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    ) as demo:
        # Header
        gr.Markdown(
            """
            # 🎬 Faceless Review Video Generator
            **Tự động tạo video Review với AI — Chỉ cần nhập chủ đề!**
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # ======== INPUT PANEL ========
                gr.Markdown("## ⚙️ Cấu hình đầu vào")

                api_key = gr.Textbox(
                    label="🔑 OpenRouter API Key",
                    placeholder="sk-or-v1-... (https://openrouter.ai/keys)",
                    type="password",
                )

                topic = gr.Textbox(
                    label="📝 Chủ đề Review",
                    placeholder='VD: "Đánh giá iPhone 15 Pro Max", "Top 5 quán cà phê Sài Gòn"',
                    lines=3,
                )

                with gr.Row():
                    video_model = gr.Dropdown(
                        choices=VIDEO_MODELS,
                        value=VIDEO_MODELS[0],
                        label="🎬 Model Video",
                        info="Wan 2.1 (chất lượng cao) / LTX-Video (nhanh hơn)",
                    )
                    openrouter_model = gr.Dropdown(
                        choices=OPENROUTER_MODELS,
                        value=OPENROUTER_MODELS[0],
                        label="🤖 Model AI",
                    )

                tts_voice = gr.Dropdown(
                    choices=TTS_VOICES,
                    value=TTS_VOICES[0],
                    label="🗣️ Giọng đọc",
                    info="HoaiMy (nữ) / NamMinh (nam)",
                )

                run_btn = gr.Button(
                    "🚀 Bắt đầu tạo Video Review",
                    variant="primary",
                    size="lg",
                )

                status = gr.Markdown("👉 _Điền thông tin và nhấn nút để bắt đầu..._")

            with gr.Column(scale=1):
                # ======== OUTPUT PANEL ========
                gr.Markdown("## 🎯 Kết quả đầu ra")

                with gr.Tabs():
                    with gr.TabItem("🎥 Video Review"):
                        video_output = gr.Video(
                            label="Video Review Hoàn Chỉnh",
                            height=360,
                            interactive=False,
                        )

                    with gr.TabItem("🔊 Audio Thuyết Minh"):
                        audio_output = gr.Audio(
                            label="Audio thuyết minh (MP3)",
                            type="filepath",
                            interactive=False,
                        )

                    with gr.TabItem("🖼️ B-roll Clips"):
                        gallery_output = gr.Gallery(
                            label="Ảnh B-roll đã tạo",
                            columns=3,
                            rows=2,
                            height=300,
                            object_fit="contain",
                        )

                    with gr.TabItem("📋 Storyboard JSON"):
                        storyboard_output = gr.JSON(
                            label="Storyboard JSON",
                        )

        # ======== EVENT HANDLER ========
        run_btn.click(
            fn=generate_review_video,
            inputs=[
                api_key,
                topic,
                video_model,
                tts_voice,
                openrouter_model,
            ],
            outputs=[
                status,
                audio_output,
                gallery_output,
                video_output,
                storyboard_output,
            ],
        )

        # ======== EXAMPLES ========
        gr.Markdown("## 💡 Ví dụ")
        gr.Examples(
            examples=[
                ["sk-or-v1-example", "Đánh giá iPhone 15 Pro Max", "wan_2_1", "vi-VN-HoaiMyNeural"],
                ["sk-or-v1-example", "Top 5 quán cà phê đẹp nhất Sài Gòn", "ltx_video", "vi-VN-NamMinhNeural"],
                ["sk-or-v1-example", "Review Samsung Galaxy S24 Ultra", "wan_2_1", "vi-VN-HoaiMyNeural"],
            ],
            inputs=[api_key, topic, video_model, tts_voice],
            label="Click để điền nhanh (API Key mẫu - hãy thay bằng key thật của bạn)",
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point — bắt buộc phải có dòng này để sinh link Gradio public
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=True, server_name="0.0.0.0")