import json
import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

import httpx
import urllib3
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CLIENT_SECRET = os.getenv("CLIENT_SECRET") or os.getenv("MKEY")
VERIFY_SSL = os.getenv("VERIFY_SSL", "false").lower() == "true"
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat-2-Pro")
HTTP_TIMEOUT = httpx.Timeout(90.0, connect=20.0)

AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGA_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
FILE_URL = "https://gigachat.devices.sberbank.ru/api/v1/files"

SYSTEM_PROMPT = """
Ты получаешь аудиофайл с голосом пользователя.
Нужно:
1. Расшифровать содержимое аудио на русском языке.
2. Оценить эмоциональное состояние и тон.
3. Вернуть только JSON без markdown и без пояснений.

Формат ответа:
{
  "transcript": "точная или максимально близкая расшифровка",
  "emotion_summary": "краткое описание эмоций и тона",
  "emotion_label": "calm|neutral|anxious|confident|excited|sad|angry|uncertain",
  "confidence": 0.0,
  "markers": ["наблюдение 1", "наблюдение 2"]
}
"""

JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
GENERIC_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)

app = FastAPI(title="Voice Broker")


class VoiceAnalysisResponse(BaseModel):
    transcript: str = ""
    emotion_summary: str = ""
    emotion_label: str = "neutral"
    confidence: float = 0.0
    markers: list[str] = Field(default_factory=list)
    raw_response: str | None = None


def extract_json_payload(raw_content: Any) -> str | None:
    text = str(raw_content).strip()
    match = JSON_FENCE_RE.search(text) or GENERIC_FENCE_RE.search(text)
    if match:
        return match.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, number)), 3)


def normalize_markers(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_response(data: dict[str, Any], raw_response: str | None = None) -> VoiceAnalysisResponse:
    return VoiceAnalysisResponse(
        transcript=str(data.get("transcript", "")).strip(),
        emotion_summary=str(data.get("emotion_summary", "")).strip(),
        emotion_label=str(data.get("emotion_label", "neutral")).strip() or "neutral",
        confidence=clamp_confidence(data.get("confidence")),
        markers=normalize_markers(data.get("markers")),
        raw_response=raw_response,
    )


def get_token() -> str:
    if not CLIENT_SECRET:
        raise RuntimeError("CLIENT_SECRET не найден в окружении.")

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {CLIENT_SECRET}",
        "RqUID": str(uuid.uuid4()),
    }
    with httpx.Client(timeout=HTTP_TIMEOUT, verify=VERIFY_SSL) as client:
        response = client.post(AUTH_URL, headers=headers, data={"scope": "GIGACHAT_API_PERS"})
    response.raise_for_status()
    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("Не удалось получить access token GigaChat.")
    return token


def convert_to_wav(input_bytes: bytes, suffix: str) -> bytes:
    source_path = None
    target_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".webm") as source:
            source.write(input_bytes)
            source_path = Path(source.name)

        target_path = source_path.with_suffix(".wav")
        process = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "wav",
                str(target_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(process.stderr.strip() or "ffmpeg завершился с ошибкой.")
        return target_path.read_bytes()
    finally:
        if source_path and source_path.exists():
            source_path.unlink(missing_ok=True)
        if target_path and target_path.exists():
            target_path.unlink(missing_ok=True)


def upload_audio(token: str, file_bytes: bytes) -> str:
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": ("voice.wav", file_bytes, "audio/wav")}
    with httpx.Client(timeout=HTTP_TIMEOUT, verify=VERIFY_SSL) as client:
        response = client.post(FILE_URL, headers=headers, files=files, data={"purpose": "general"})
    response.raise_for_status()
    file_id = response.json().get("id")
    if not file_id:
        raise RuntimeError("GigaChat не вернул file id.")
    return file_id


def analyze_with_gigachat(file_id: str, token: str) -> VoiceAnalysisResponse:
    payload = {
        "model": GIGACHAT_MODEL,
        "temperature": 0.1,
        "max_tokens": 1200,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Расшифруй приложенное аудио и оцени эмоции по заданному JSON-формату.",
                "attachments": [file_id],
            },
        ],
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    with httpx.Client(timeout=HTTP_TIMEOUT, verify=VERIFY_SSL) as client:
        response = client.post(GIGA_URL, headers=headers, json=payload)
    response.raise_for_status()

    raw_response = response.json()["choices"][0]["message"]["content"]
    json_payload = extract_json_payload(raw_response)
    if not json_payload:
        return VoiceAnalysisResponse(
            emotion_summary="Не удалось разобрать ответ модели как JSON.",
            emotion_label="uncertain",
            markers=["Ответ модели пришёл в неструктурированном виде."],
            raw_response=str(raw_response),
        )

    try:
        parsed = json.loads(json_payload)
    except Exception:
        return VoiceAnalysisResponse(
            emotion_summary="GigaChat вернул JSON с ошибкой разбора.",
            emotion_label="uncertain",
            markers=["Проверь raw_response для диагностики."],
            raw_response=str(raw_response),
        )
    return normalize_response(parsed, str(raw_response))


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=VoiceAnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)) -> VoiceAnalysisResponse:
    raw_audio = await file.read()
    if not raw_audio:
        raise HTTPException(status_code=400, detail="Пустой аудиофайл.")

    suffix = Path(file.filename or "voice.webm").suffix or ".webm"

    try:
        wav_bytes = convert_to_wav(raw_audio, suffix)
        token = get_token()
        file_id = upload_audio(token, wav_bytes)
        return analyze_with_gigachat(file_id, token)
    except httpx.HTTPStatusError as error:
        raise HTTPException(status_code=502, detail=f"GigaChat HTTP error: {error.response.text}") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8010)
