# Architecture: OpenRouter + Multi-Model Video Review Pipeline

## 1. System Overview

This repository implements an Automated Faceless Review Video Generator Engine designed to run entirely on Google Colab Free using a Gradio web interface.

The system accepts a review topic, generates a structured storyboard through an API, creates narration using text-to-speech, produces local B-roll keyframe images, renders motion video clips with a selectable multi-model engine, and stitches the result into a final review video.

The output is focused on:
- Narrator voiceover
- B-roll motion clips
- Burn-in subtitles

No face animation or lipsync is required.

## 2. Engineering Constraints

### API Strategy
- Script generation uses the OpenRouter API (for example DeepSeek-R1 or Llama 3.3).
- Image generation runs locally with Diffusers and Flux.1-schnell on Colab T4 GPU hardware to avoid paid image APIs.
- Voiceover uses edge-tts (Microsoft free TTS).

### Multi-Model Video Engine
- The backend must support both Wan 2.1 (image-to-video) and LTX-Video.
- The selected model is controlled through the Gradio UI.

### Memory Management
- Colab T4 memory must be released aggressively.
- The pipeline must call `torch.cuda.empty_cache()` and `gc.collect()` after every rendered video clip.
- This is required to reduce the risk of CUDA out-of-memory failures during long pipelines.

## 3. Technology Stack

Core dependencies are defined in requirements.txt:
- gradio>=4.0.0
- requests
- diffusers
- transformers
- accelerate
- torch
- torchvision
- edge-tts
- moviepy
- ffmpeg-python
- imageio-ffmpeg

## 4. Repository Structure

```text
faceless-video-pipeline/
├── ARCHITECTURE.md            # System architecture specification
├── requirements.txt           # Python dependencies for Colab
├── core/
│   ├── __init__.py
│   ├── script_gen.py          # OpenRouter API -> parses review topic into JSON storyboard
│   ├── image_gen.py           # Flux.1-schnell local Diffusers -> generates B-roll keyframe images
│   ├── video_gen.py           # Multi-model engine (Wan 2.1 + LTX-Video)
│   ├── audio_gen.py           # Edge-TTS -> generates narration audio and subtitles
│   └── composer.py            # FFmpeg/MoviePy -> stitches final review video
├── app.py                     # Main Gradio Blocks UI
└── run_pipeline.ipynb         # Colab-ready notebook entry point
```

## 5. Data Flow Pipeline

### Input
A user enters a review topic in the Gradio interface.

### Step 1: Script Generation
The script layer calls the OpenRouter API and produces a JSON payload containing:
- `full_narration`: full Vietnamese narration text
- `storyboard_scenes`: an array of scene objects containing:
  - `scene_id`
  - `narration_chunk`
  - `broll_prompt`

### Step 2: Audio Generation
The audio pipeline converts the narration into:
- `narration.mp3`
- `subtitles.srt`

using edge-tts.

### Step 3: B-roll Generation
The image generation layer creates local PNG keyframe images using a local Diffusers pipeline.

### Step 4: Video Rendering
Each storyboard scene is converted into a motion video clip using the selected engine:
- Wan 2.1 (image-to-video)
- LTX-Video

After every clip render, the pipeline clears VRAM using:
- `torch.cuda.empty_cache()`
- `gc.collect()`

### Step 5: Timeline Stitching
The final composer layer overlays narration, B-roll clips, and subtitles into one video using MoviePy and FFmpeg.

The final output is written to:
- `/content/output/final_review_video.mp4`

## 6. Module Responsibilities

### core/script_gen.py
Responsible for interfacing with OpenRouter and converting a review topic into a structured storyboard payload.

### core/image_gen.py
Responsible for generating local B-roll keyframe images using Flux.1-schnell through Diffusers.

### core/video_gen.py
Responsible for converting generated images into short motion clips using either Wan 2.1 or LTX-Video.

### core/audio_gen.py
Responsible for generating narration audio and subtitle files from the final script text.

### core/composer.py
Responsible for combining audio, video clips, and subtitles into a final review video.

### app.py
Responsible for the Gradio UI, model selector, storyboard editor, and output exploration.

## 7. Expected Runtime Flow

1. User enters a review topic.
2. The script generator creates a storyboard JSON.
3. Audio is generated from the full narration text.
4. Keyframe images are generated locally.
5. Each scene is rendered as a short motion clip.
6. The composer stitches clips, audio, and subtitles into one final output.
7. The finished video is displayed or downloaded from the Gradio app.

## 8. Implementation Notes

- Keep all model operations modular so the video backend can swap between Wan 2.1 and LTX-Video without changing the rest of the pipeline.
- Use explicit output folders for intermediate assets and final results.
- Design the pipeline to recover gracefully from failures in image or video generation while preserving partial outputs.
- Prefer deterministic file naming for each scene and asset to simplify debugging in Colab.

## 9. Recommended Development Order

1. Scaffold the repository structure.
2. Implement script generation with OpenRouter.
3. Implement local image generation.
4. Implement the video rendering backend for Wan 2.1 and LTX-Video.
5. Implement audio generation and subtitle export.
6. Implement the video composer.
7. Wire everything into the Gradio application.
