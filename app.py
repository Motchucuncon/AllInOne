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
) -> tuple[str, str | None, list[str], list[str], str | None, str]:
    """Run the full pipeline and return results for display.

    Returns
    -------
    tuple
        (log_text, audio_path, broll_image_paths, clip_paths, video_path, storyboard_json)
    """
    _ensure_output_dir()
    log_lines = []

    def add_log(msg: str):
        print(msg)
        log_lines.append(msg)

    # ===== MỘT TRY...EXCEPT DUY NHẤT BAO BỌC TOÀN BỘ PIPELINE =====
    try:
        # --- Validate inputs ---
        if not api_key.strip():
            raise ValueError("❌ Vui lòng nhập OpenRouter API Key.")
        if not topic.strip():
            raise ValueError("❌ Vui lòng nhập Chủ đề Review.")

        os.environ["OPENROUTER_API_KEY"] = api_key

        # ------------------------------------------------------------------
        # STEP 1: Generate Storyboard
        # ------------------------------------------------------------------
        add_log("\n" + "=" * 60)
        add_log("📝 BƯỚC 1/5: TẠO STORYBOARD VỚI OPENROUTER")
        add_log("=" * 60)
        add_log(f"   • Model AI: {openrouter_model}")
        add_log(f"   • Chủ đề: {topic}")
        progress(0.05, desc="📝 Bước 1/5: Tạo Storyboard với OpenRouter...")

        storyboard = generate_storyboard(
            topic=topic,
            model=openrouter_model,
            api_key=api_key,
        )
        scenes = storyboard.get("storyboard_scenes", [])
        if not scenes:
            raise ValueError("❌ Storyboard không có scene nào.")
        storyboard_json = json.dumps(storyboard, ensure_ascii=False, indent=2)
        add_log(f"   ✅ Đã tạo {len(scenes)} scenes")
        add_log(f"   📝 Narration: {storyboard['full_narration'][:100]}...")
        for scene in scenes:
            add_log(f"\n   --- Scene {scene['scene_id']} ---")
            add_log(f"   📝 Narration: {scene['narration_chunk'][:80]}...")
            add_log(f"   🖼️  B-roll Prompt: {scene['broll_prompt'][:80]}...")

        # ------------------------------------------------------------------
        # STEP 2: Generate Audio & Subtitles
        # ------------------------------------------------------------------
        add_log("\n" + "=" * 60)
        add_log("🔊 BƯỚC 2/5: TẠO AUDIO THUYẾT MINH VỚI EDGE-TTS")
        add_log("=" * 60)
        add_log(f"   • Giọng đọc: {tts_voice}")
        add_log(f"   • Độ dài văn bản: {len(storyboard['full_narration'])} ký tự")
        progress(0.20, desc="🔊 Bước 2/5: Tạo Audio thuyết minh với Edge-TTS...")

        audio_result = generate_audio_and_subtitles(
            storyboard=storyboard,
            voice=tts_voice,
            output_dir=OUTPUT_DIR,
        )
        audio_path = audio_result["audio_path"]
        subtitles_path = audio_result["subtitles_path"]
        duration = audio_result["duration_seconds"]
        scenes = audio_result["scenes"]
        add_log(f"   ✅ Audio: {audio_path}")
        add_log(f"   ✅ Subtitles: {subtitles_path}")
        add_log(f"   ⏱️  Thời lượng: {duration:.1f}s")

        # ------------------------------------------------------------------
        # STEP 3: Generate B-roll Images (Flux.1-schnell)
        # ------------------------------------------------------------------
        broll_prompts = [s.get("broll_prompt", "") for s in scenes]
        scene_ids = [s.get("scene_id", i + 1) for i, s in enumerate(scenes)]

        add_log("\n" + "=" * 60)
        add_log("🖼️  BƯỚC 3/5: TẠO ẢNH B-ROLL VỚI FLUX.1-SCHNELL")
        add_log("=" * 60)
        add_log(f"   • Tổng số ảnh cần tạo: {len(broll_prompts)}")

        for i, (prompt, sid) in enumerate(zip(broll_prompts, scene_ids)):
            add_log(f"\n   --- Đang tạo ảnh Scene {sid} ({i+1}/{len(broll_prompts)}) ---")
            add_log(f"   🖼️  Prompt: {prompt}")
            progress(0.30 + 0.30 * (i / len(broll_prompts)),
                     desc=f"🖼️  Đang tạo ảnh Scene {sid}...")

        image_paths = generate_broll_images(
            prompts=broll_prompts,
            scene_ids=scene_ids,
            output_dir=os.path.join(OUTPUT_DIR, "images"),
            unload_after=True,
        )
        add_log(f"\n   ✅ Đã tạo {len(image_paths)} ảnh B-roll:")
        for img_path in image_paths:
            add_log(f"      • {os.path.basename(img_path)}")

        # ------------------------------------------------------------------
        # STEP 4: Render Video Clips (Wan 2.1 / LTX-Video)
        # ------------------------------------------------------------------
        add_log("\n" + "=" * 60)
        add_log(f"🎬 BƯỚC 4/5: RENDER VIDEO CLIPS VỚI {video_model.upper()}")
        add_log("=" * 60)
        add_log(f"   • Model: {video_model}")
        add_log(f"   • Số clip: {len(image_paths)}")

        for i, (sid, img_path) in enumerate(zip(scene_ids, image_paths)):
            add_log(f"\n   --- Đang render Scene {sid} ({i+1}/{len(image_paths)}) ---")
            add_log(f"   📁 Ảnh đầu vào: {os.path.basename(img_path)}")
            progress(0.60 + 0.20 * (i / len(image_paths)),
                     desc=f"🎬 Đang render clip Scene {sid}...")

        clip_paths = render_video_clips(
            image_paths=image_paths,
            prompts=broll_prompts,
            scene_ids=scene_ids,
            model=video_model,
            output_dir=os.path.join(OUTPUT_DIR, "clips"),
        )
        add_log(f"\n   ✅ Đã render {len(clip_paths)} clips:")
        for clip_path in clip_paths:
            add_log(f"      • {os.path.basename(clip_path)}")

        # ------------------------------------------------------------------
        # STEP 5: Compose Final Video
        # ------------------------------------------------------------------
        add_log("\n" + "=" * 60)
        add_log("🎞️  BƯỚC 5/5: GHÉP VIDEO HOÀN CHỈNH")
        add_log("=" * 60)
        progress(0.85, desc="🎞️ Bước 5/5: Ghép video hoàn chỉnh...")

        final_video_path = compose_final_video(
            clip_paths=clip_paths,
            audio_path=audio_path,
            subtitles_path=subtitles_path,
            output_dir=OUTPUT_DIR,
        )
        add_log(f"\n   ✅ Final video: {final_video_path}")

        progress(1.0, desc="✅ Hoàn thành!")

        add_log("\n" + "=" * 60)
        add_log("🎉 PIPELINE HOÀN THÀNH!")
        add_log("=" * 60)
        add_log(f"   📝 Chủ đề: {topic}")
        add_log(f"   🎬 Model Video: {video_model}")
        add_log(f"   📋 Số scene: {len(scenes)}")
        add_log(f"   ⏱️ Thời lượng: {duration:.1f}s")
        add_log(f"   📁 File đầu ra: {final_video_path}")

        log_html = "<div class='log-box'><pre>" + "\n".join(log_lines) + "</pre></div>"
        return log_html, audio_path, image_paths, clip_paths, final_video_path, storyboard_json

    except Exception:
        # ===== BẮT LỖI: In toàn bộ traceback chi tiết, màu đỏ =====
        import traceback
        tb_text = traceback.format_exc()
        print(tb_text)  # stdout
        log_lines.append("\n❌ " + "=" * 56)
        log_lines.append("❌  PIPELINE THẤT BẠI!")
        log_lines.append("❌ " + "=" * 56)
        log_lines.append("")
        log_lines.extend(tb_text.strip().split("\n"))
        log_lines.append("")
        log_lines.append("❌ " + "=" * 56)
        log_lines.append("❌  VUI LÒNG KIỂM TRA LỖI VÀ THỬ LẠI")
        log_lines.append("❌ " + "=" * 56)

        error_html = (
            "<div class='log-box' style='border: 2px solid #ff4444;'>"
            "<pre style='color: #ff6b6b; white-space: pre-wrap; word-wrap: break-word;'>"
            + "\n".join(log_lines)
            + "</pre></div>"
        )
        return error_html, None, [], [], None, ""


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks UI."""

    css = """
    .gradio-container { max-width: 1200px; margin: auto; }
    footer { display: none !important; }
    .log-box { font-family: monospace; font-size: 13px; background: #1e1e1e; color: #d4d4d4; border-radius: 6px; padding: 10px; height: 300px; overflow-y: auto; }
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
            **Tự động tạo video Review với AI — Chi tiết từng bước!**
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

            with gr.Column(scale=1):
                # ======== LOG OUTPUT ========
                gr.Markdown("## 📋 Log chi tiết")
                log_output = gr.HTML(
                    label="Log chi tiết",
                    value="<div class='log-box'><pre style='color:#888;'>👉 Điền thông tin và nhấn nút để bắt đầu...</pre></div>",
                )

        # ======== OUTPUT PANEL ========
        gr.Markdown("## 🎯 Kết quả đầu ra")
        with gr.Tabs():
            with gr.TabItem("🎥 Video Review"):
                video_output = gr.Video(
                    label="Video Review Hoàn Chỉnh",
                    height=400,
                    interactive=False,
                )

            with gr.TabItem("🖼️ Ảnh B-roll Trung Gian"):
                gallery_output = gr.Gallery(
                    label="Ảnh B-roll đã tạo (theo từng scene)",
                    columns=3,
                    rows=2,
                    height=400,
                    object_fit="contain",
                )

            with gr.TabItem("🎬 Clip Video Trung Gian"):
                clip_gallery = gr.Gallery(
                    label="Video clips đã render (theo từng scene)",
                    columns=2,
                    rows=2,
                    height=400,
                    object_fit="contain",
                )

            with gr.TabItem("🔊 Audio Thuyết Minh"):
                audio_output = gr.Audio(
                    label="Audio thuyết minh (MP3)",
                    type="filepath",
                    interactive=False,
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
                log_output,
                audio_output,
                gallery_output,
                clip_gallery,
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
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=True, server_name="0.0.0.0")