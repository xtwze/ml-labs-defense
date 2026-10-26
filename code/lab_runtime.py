"""Shared configuration and media requests for the notebook collection."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import time

import requests
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
BASE_URL = "https://routerai.ru/api/v1"
TEXT_MODEL = "openai/gpt-4.1-nano"
IMAGE_MODEL = "recraft/recraft-v4.1-flash"
VISION_MODEL = TEXT_MODEL
STT_MODEL = "openai/whisper-large-v3-turbo"
TTS_MODEL = "x-ai/grok-voice-tts-1.0"


def api_key():
    value = os.environ.get("ROUTERAI_API_KEY", "").strip()
    if value:
        return value
    for path in [ROOT / ".env", ROOT.parent / ".routerai-labs.env"]:
        if path.exists():
            for line in path.read_text().splitlines():
                if line.startswith("ROUTERAI_API_KEY="):
                    value = line.split("=", 1)[1].strip().strip('\"').strip("'")
                    if value:
                        return value
    try:
        from google.colab import userdata
        value = userdata.get("ROUTERAI_API_KEY")
        if value:
            return value
    except (ImportError, KeyError):
        pass
    from getpass import getpass
    value = getpass("RouterAI API key: ").strip()
    if not value:
        raise RuntimeError("Set ROUTERAI_API_KEY or create a local .env file")
    os.environ["ROUTERAI_API_KEY"] = value
    return value


def make_llm(temperature=0.2, max_tokens=500, model=TEXT_MODEL):
    return ChatOpenAI(model=model, api_key=api_key(), base_url=BASE_URL,
                      temperature=temperature, max_tokens=max_tokens,
                      timeout=90, max_retries=2)


def json_result(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def _record(endpoint, response, data=None):
    record = {"endpoint": endpoint, "status": response.status_code,
              "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "id": response.headers.get("X-Generation-Id")}
    if data:
        record["usage"] = data.get("usage")
        record["id"] = data.get("id", record["id"])
    with (ROOT / "api_requests.jsonl").open("a") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def request_json(endpoint, payload):
    response = requests.post(BASE_URL + endpoint,
        headers={"Authorization": "Bearer " + api_key()}, json=payload, timeout=180)
    data = response.json()
    _record(endpoint, response, data)
    if not response.ok:
        raise RuntimeError(f"RouterAI {response.status_code}: {str(data)[:500]}")
    return data


def generate_image(prompt, filename):
    path = ASSETS / filename
    fingerprint = hashlib.sha256((IMAGE_MODEL + prompt).encode()).hexdigest()
    stamp = path.with_suffix(path.suffix + ".sha256")
    if path.exists() and stamp.exists() and stamp.read_text() == fingerprint:
        # Некоторые API возвращают WebP даже при output_format="png".
        # Пересохраняем кеш в настоящий формат расширения, иначе PyCharm видит
        # MIME image/png, но получает WebP-байты и не может показать картинку.
        from PIL import Image
        expected = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}.get(path.suffix.lower())
        if expected:
            with Image.open(path) as picture:
                if picture.format != expected:
                    converted = picture.convert("RGB") if expected == "JPEG" else picture.convert("RGBA")
                    converted.save(path, format=expected)
        return path
    data = request_json("/images", {"model": IMAGE_MODEL, "prompt": prompt,
                                    "n": 1, "aspect_ratio": "1:1", "output_format": "png"})
    item = data["data"][0]
    if "b64_json" in item:
        raw = base64.b64decode(item["b64_json"])
    else:
        response = requests.get(item["url"], timeout=90)
        response.raise_for_status()
        raw = response.content
    # Сохраняем изображение в формате, соответствующем имени файла, независимо
    # от фактического формата байтов, возвращённых сервисом.
    from PIL import Image
    expected = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}.get(path.suffix.lower())
    if expected:
        with Image.open(io.BytesIO(raw)) as picture:
            converted = picture.convert("RGB") if expected == "JPEG" else picture.convert("RGBA")
            converted.save(path, format=expected)
    else:
        path.write_bytes(raw)
    stamp.write_text(fingerprint)
    return path


def image_message(path, instruction):
    from PIL import Image
    picture = Image.open(path).convert("RGB")
    picture.thumbnail((768, 768))
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG")
    url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    return [{"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": url}}]


def synthesize(text, filename, voice="eve"):
    path = ASSETS / filename
    fingerprint = hashlib.sha256((TTS_MODEL + voice + text).encode()).hexdigest()
    stamp = path.with_suffix(path.suffix + ".sha256")
    if path.exists() and stamp.exists() and stamp.read_text() == fingerprint:
        return path
    response = requests.post(BASE_URL + "/audio/speech",
        headers={"Authorization": "Bearer " + api_key()},
        json={"model": TTS_MODEL, "input": text, "voice": voice,
              "response_format": "mp3"}, timeout=180)
    _record("/audio/speech", response)
    if not response.ok:
        raise RuntimeError(f"TTS {response.status_code}: {response.text[:500]}")
    path.write_bytes(response.content)
    stamp.write_text(fingerprint)
    return path


def transcribe(path):
    cache = ASSETS / (Path(path).stem + "_transcript.json")
    fingerprint = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    stamp = cache.with_suffix(".sha256")
    if cache.exists() and stamp.exists() and stamp.read_text() == fingerprint:
        return json.loads(cache.read_text())["text"]
    with Path(path).open("rb") as audio:
        response = requests.post(BASE_URL + "/audio/transcriptions",
            headers={"Authorization": "Bearer " + api_key()},
            data={"model": STT_MODEL, "language": "ru"},
            files={"file": (Path(path).name, audio, "audio/mpeg")}, timeout=120)
    response.raise_for_status()
    data = response.json()
    _record("/audio/transcriptions", response, data)
    cache.write_text(json.dumps(data, ensure_ascii=False))
    stamp.write_text(fingerprint)
    return data["text"]
