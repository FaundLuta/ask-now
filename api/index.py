import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import requests
from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv

load_dotenv()

PUBLIC_DIR = Path(__file__).resolve().parent.parent / "public"

app = Flask(__name__, static_folder=str(PUBLIC_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

SYSTEM_PROMPT = """You are Ask Now, a clear, capable general-purpose AI assistant.
Respond in English unless the user explicitly asks for another language.
Help with writing, coding, analysis, planning, research, math, and creative work.
Be useful and concise by default. Format answers with clean markdown: short paragraphs,
descriptive headings, bullets for lists, and fenced code blocks for code. Never put a
heading and its content on the same line. Do not add greetings or filler unless useful.
Never speak, narrate, or claim to have performed actions outside this chat.
When files are provided, ground your answer in their contents and say when something is not available.
"""


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _extract_file(file_storage: Any) -> tuple[str, dict[str, Any] | None]:
    filename = file_storage.filename or "upload"
    mime_type = file_storage.mimetype or "application/octet-stream"
    raw = file_storage.read()
    if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
        from PyPDF2 import PdfReader

        reader = PdfReader(io.BytesIO(raw))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        return f"PDF: {filename}\n{text[:50000]}", None
    if mime_type.startswith("image/"):
        encoded = base64.b64encode(raw).decode("ascii")
        return f"Image: {filename}", {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}
    decoded = raw.decode("utf-8", errors="ignore")
    return f"File: {filename}\n{decoded[:50000]}", None


def _messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    history = payload.get("history", [])
    if not isinstance(history, list) or not history:
        raise ValueError("No message provided.")
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history[-20:]:
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"}:
            messages.append({"role": item["role"], "content": str(item.get("content", ""))[:50000]})
    return messages


def _groq_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _available_models(api_key: str) -> list[dict[str, Any]]:
    response = requests.get("https://api.groq.com/openai/v1/models", headers=_groq_headers(api_key), timeout=15)
    body = response.json()
    if response.status_code != 200:
        error = body.get("error", {}).get("message", response.text)
        raise RuntimeError(f"[{response.status_code}] {error}")
    models = []
    for item in body.get("data", []):
        model_id = item.get("id", "")
        input_modalities = item.get("input_modalities") or []
        output_modalities = item.get("output_modalities") or []
        if (
            model_id
            and "text" in input_modalities
            and "text" in output_modalities
            and not any(token in model_id.lower() for token in ("guard", "safeguard", "allam"))
        ):
            models.append({
                "id": model_id,
                "context_window": item.get("context_window"),
                "owned_by": item.get("owned_by"),
                "supports_images": "image" in input_modalities,
            })
    return sorted(models, key=lambda model: model["id"])


@app.get("/")
def index() -> Any:
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.get("/models")
def models() -> Any:
    api_key = _env("GROQ_API_KEY")
    if not api_key:
        return jsonify({"error": "GROQ_API_KEY is missing.", "models": []}), 500
    try:
        return jsonify({"models": _available_models(api_key)})
    except requests.Timeout:
        return jsonify({"error": "The model list request timed out.", "models": []}), 504
    except Exception as error:
        return jsonify({"error": str(error), "models": []}), 502


@app.post("/chat")
def chat() -> Any:
    try:
        if request.form.get("payload"):
            payload = json.loads(request.form["payload"])
        else:
            payload = request.get_json(silent=True) or {}
        messages = _messages(payload)
        upload = request.files.get("file")
        has_image = False
        if upload:
            context, image_part = _extract_file(upload)
            has_image = image_part is not None
            user_message = messages[-1]
            if image_part:
                user_message["content"] = [{"type": "text", "text": user_message["content"]}, image_part]
            else:
                user_message["content"] = f"{user_message['content']}\n\nAttached file context:\n{context}"
        api_key = _env("GROQ_API_KEY")
        if not api_key:
            return jsonify({"error": "GROQ_API_KEY is missing. Add it to Vercel Environment Variables."}), 500
        requested_model = str(payload.get("model", "")).strip()
        configured_model = _env("GROQ_VISION_MODEL" if has_image else "GROQ_MODEL")
        model = requested_model or configured_model
        if not model:
            available = _available_models(api_key)
            if not available:
                return jsonify({"error": "No chat models are available for this API key."}), 502
            candidates = [item for item in available if not has_image or item["supports_images"]]
            if not candidates:
                return jsonify({"error": "This API key has no image-capable chat model."}), 400
            model = candidates[0]["id"]
        if has_image:
            available = _available_models(api_key)
            selected = next((item for item in available if item["id"] == model), None)
            if not selected or not selected["supports_images"]:
                return jsonify({"error": "The selected model cannot read images. Choose a model marked vision in Settings."}), 400
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=_groq_headers(api_key),
            json={"model": model, "messages": messages, "max_tokens": 2048, "temperature": 0.65},
            timeout=55,
        )
        body = response.json()
        if response.status_code != 200:
            error = body.get("error", {}).get("message", response.text)
            return jsonify({"error": f"[{response.status_code}] {error}"}), response.status_code
        return jsonify({"reply": body["choices"][0]["message"]["content"].strip(), "model": model})
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except requests.Timeout:
        return jsonify({"error": "The model took too long to respond. Try a shorter prompt."}), 504
    except Exception as error:
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=int(_env("PORT", "5000")))