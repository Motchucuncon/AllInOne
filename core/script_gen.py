"""OpenRouter-based storyboard generation module.

Calls OpenRouter API (e.g., DeepSeek-R1, Llama 3.3) to generate a structured
storyboard JSON from a user-provided review topic.
"""

import json
import os
import time
import requests

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-r1:free"  # fallback model
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 120

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a professional Vietnamese video script writer. Your task is to create a detailed storyboard for a faceless review video.

You MUST respond with ONLY valid JSON, no markdown fences, no extra text.

The JSON structure must be:

{
  "full_narration": "The complete Vietnamese narration text, 60-120 seconds when spoken.",
  "storyboard_scenes": [
    {
      "scene_id": 1,
      "narration_chunk": "The narration text for this specific scene.",
      "broll_prompt": "A detailed English prompt for generating a B-roll keyframe image with Flux.1-schnell. Describe the visual scene in detail."
    }
  ]
}

Guidelines:
- full_narration should be 150-300 Vietnamese words, divided into 3-6 scenes.
- Each narration_chunk is a natural segment of the full_narration (not the whole thing).
- Each broll_prompt is a detailed English description for Flux.1-schnell image generation. Include style cues like "cinematic lighting, high quality, 4K, photorealistic".
- The scenes should follow a logical flow: introduction, main points, conclusion.
- Keep the tone engaging and suitable for a short video review."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_payload(topic: str, model: str = DEFAULT_MODEL) -> dict:
    """Build the request payload for the OpenRouter chat completion endpoint."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Review topic: {topic}\n\nCreate a Vietnamese storyboard for a faceless review video about this topic."},
        ],
        "temperature": 0.7,
        "max_tokens": 4096,
    }


def _call_openrouter(payload: dict, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Make a POST request to OpenRouter and return the response JSON."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Motchucuncon/AllInOne",
        "X-Title": "Faceless Review Video Generator",
    }

    resp = requests.post(
        OPENROUTER_BASE_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _parse_storyboard(raw_content: str) -> dict:
    """Parse the LLM response string into a storyboard dict.

    Tries to extract JSON from the response.  Falls back to a minimal
    placeholder if parsing fails.
    """
    # Strip markdown code fences if present
    content = raw_content.strip()
    if content.startswith("```"):
        # Remove opening fence (possibly with language tag)
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline:].strip()
        # Remove closing fence
        if content.endswith("```"):
            content = content[:-3].strip()
        elif content.endswith("```"):
            content = content[:-3].strip()

    # Try direct JSON parse
    try:
        storyboard = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: try to find JSON object in the text
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                storyboard = json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                storyboard = None
        else:
            storyboard = None

    if storyboard is None:
        # Return a minimal placeholder
        storyboard = {
            "full_narration": content[:500] if len(content) > 500 else content,
            "storyboard_scenes": [
                {
                    "scene_id": 1,
                    "narration_chunk": content[:200] if len(content) > 200 else content,
                    "broll_prompt": "A cinematic video background related to the review topic, high quality, 4K, photorealistic",
                }
            ],
        }

    # Ensure required keys exist
    if "full_narration" not in storyboard:
        storyboard["full_narration"] = ""
    if "storyboard_scenes" not in storyboard or not isinstance(storyboard["storyboard_scenes"], list):
        storyboard["storyboard_scenes"] = []

    # Ensure each scene has required fields
    for scene in storyboard["storyboard_scenes"]:
        if "scene_id" not in scene:
            scene["scene_id"] = 1
        if "narration_chunk" not in scene:
            scene["narration_chunk"] = ""
        if "broll_prompt" not in scene:
            scene["broll_prompt"] = "A cinematic video background, high quality, 4K"

    # Add topic
    return storyboard


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_storyboard(
    topic: str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    """Generate a storyboard payload for the provided review topic.

    Parameters
    ----------
    topic : str
        The review topic (e.g., "iPhone 15 review").
    model : str
        OpenRouter model identifier (e.g., "deepseek/deepseek-r1:free").
    api_key : str or None
        OpenRouter API key.  Falls back to OPENROUTER_API_KEY env var.
    max_retries : int
        Number of retry attempts on failure.

    Returns
    -------
    dict
        A storyboard dict with keys:
        - full_narration: str
        - storyboard_scenes: list[dict]
    """
    # Resolve API key
    key = api_key or OPENROUTER_API_KEY
    if not key:
        # Return a mock storyboard for development / testing
        return _mock_storyboard(topic)

    payload = _build_payload(topic, model=model)

    last_error = None
    for attempt in range(1 + max_retries):
        try:
            response_data = _call_openrouter(payload)
            raw_content = response_data["choices"][0]["message"]["content"]
            storyboard = _parse_storyboard(raw_content)
            storyboard["topic"] = topic
            return storyboard
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt < max_retries:
                wait = 2 ** attempt
                time.sleep(wait)
            continue

    # If all retries fail, return a mock storyboard
    print(f"OpenRouter request failed after {max_retries} retries: {last_error}")
    return _mock_storyboard(topic)


def _mock_storyboard(topic: str) -> dict:
    """Return a hardcoded fallback storyboard for development or offline use."""
    return {
        "topic": topic,
        "full_narration": (
            f"Hôm nay chúng ta sẽ cùng nhau đánh giá {topic}. "
            f"Đây là một sản phẩm đang nhận được rất nhiều sự quan tâm từ cộng đồng. "
            f"Đầu tiên, hãy cùng nhìn vào thiết kế tổng thể. "
            f"Sản phẩm có thiết kế hiện đại, tinh tế và sang trọng. "
            f"Các chi tiết được hoàn thiện một cách tỉ mỉ, chất lượng xây dựng rất tốt. "
            f"Tiếp theo, hãy nói về hiệu năng. "
            f"Sản phẩm hoạt động mượt mà, đáp ứng tốt mọi nhu cầu sử dụng hàng ngày. "
            f"Hiệu năng vượt trội so với các đối thủ cùng phân khúc. "
            f"Cuối cùng, đây là một lựa chọn đáng giá. "
            f"Nếu bạn đang cân nhắc mua {topic}, đừng ngần ngại. "
            f"Cảm ơn các bạn đã xem video này!"
        ),
        "storyboard_scenes": [
            {
                "scene_id": 1,
                "narration_chunk": f"Hôm nay chúng ta sẽ cùng nhau đánh giá {topic}. Đây là một sản phẩm đang nhận được rất nhiều sự quan tâm từ cộng đồng.",
                "broll_prompt": f"A cinematic shot of {topic} on a sleek table, studio lighting, 4K, photorealistic, high detail",
            },
            {
                "scene_id": 2,
                "narration_chunk": "Đầu tiên, hãy cùng nhìn vào thiết kế tổng thể. Sản phẩm có thiết kế hiện đại, tinh tế và sang trọng. Các chi tiết được hoàn thiện một cách tỉ mỉ, chất lượng xây dựng rất tốt.",
                "broll_prompt": f"Close-up macro shot of {topic} premium materials and craftsmanship, shallow depth of field, cinematic lighting, 4K",
            },
            {
                "scene_id": 3,
                "narration_chunk": "Tiếp theo, hãy nói về hiệu năng. Sản phẩm hoạt động mượt mà, đáp ứng tốt mọi nhu cầu sử dụng hàng ngày. Hiệu năng vượt trội so với các đối thủ cùng phân khúc.",
                "broll_prompt": f"Dynamic action shot showing {topic} in use, motion blur effects, fast-paced, cinematic, high energy",
            },
            {
                "scene_id": 4,
                "narration_chunk": f"Cuối cùng, đây là một lựa chọn đáng giá. Nếu bạn đang cân nhắc mua {topic}, đừng ngần ngại. Cảm ơn các bạn đã xem video này!",
                "broll_prompt": f"A beautiful sunset background with {topic} silhouetted, award-winning cinematography, emotional, 4K",
            },
        ],
    }