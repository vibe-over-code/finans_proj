import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from gigachat import GigaChat
from pydantic import BaseModel, Field

load_dotenv()

VERIFY_SSL = os.getenv("GIGACHAT_VERIFY_SSL", "false").strip().lower() in {"1", "true", "yes"}
MODEL_NAME = os.getenv("GIGACHAT_MODEL", "GigaChat-2-Pro")
VOICE_TIMEOUT_SECONDS = float(os.getenv("VOICE_TIMEOUT_SECONDS", "45"))

VOICE_PROMPT = os.getenv(
    "VOICE_PROMPT",
    "You will receive an audio attachment.\n"
    "Return ONLY valid JSON (no markdown, no extra text) with fields:\n"
    "- transcript: string\n"
    "- emotion: short label in English (e.g. anxious, calm, angry, excited)\n"
    "- emotion_index: number from 0.0 to 1.0 (0 calm, 1 very intense)\n"
    "- summary: one short sentence\n",
)


def normalize_credentials(raw_value: str | None) -> str | None:
    if not raw_value:
        return None
    value = raw_value.strip().strip('"').strip("'")
    if value.startswith("CLIENT_SECRET="):
        value = value.split("=", 1)[1].strip()
    return value or None


GIGACHAT_CREDENTIALS = normalize_credentials(
    os.getenv("CLIENT_SECRET") or os.getenv("GIGACHAT_CREDENTIALS") or os.getenv("MKEY")
)


class VoiceAnalysisResponse(BaseModel):
    transcript: str
    emotion: str
    emotion_index: float = Field(ge=0.0, le=1.0)
    summary: str
    raw: str | None = None


app = FastAPI(title="Voice Broker")


def get_gigachat_client() -> GigaChat:
    if not GIGACHAT_CREDENTIALS:
        raise RuntimeError("CLIENT_SECRET not found in environment")

    return GigaChat(
        credentials=GIGACHAT_CREDENTIALS,
        verify_ssl_certs=VERIFY_SSL,
        model=MODEL_NAME,
    )


def _try_parse_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def get_voice_json(audio_path: str | Path) -> VoiceAnalysisResponse:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    with get_gigachat_client() as giga:
        with open(path, "rb") as audio_file:
            uploaded_file = giga.upload_file(audio_file)

        response = giga.chat(
            {
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": VOICE_PROMPT,
                        "attachments": [uploaded_file.id_],
                    }
                ],
            }
        )

    if not response or not response.choices:
        raise RuntimeError("GigaChat returned an empty response")

    raw_text = str(response.choices[0].message.content or "").strip()
    payload = _try_parse_json(raw_text)
    if not payload:
        return VoiceAnalysisResponse(
            transcript="",
            emotion="unknown",
            emotion_index=0.0,
            summary="Failed to parse JSON from model response.",
            raw=raw_text,
        )

    emotion_index_raw = payload.get("emotion_index", 0.0)
    try:
        emotion_index = float(emotion_index_raw)
    except Exception:
        emotion_index = 0.0

    return VoiceAnalysisResponse(
        transcript=str(payload.get("transcript", "")),
        emotion=str(payload.get("emotion", "")),
        emotion_index=emotion_index,
        summary=str(payload.get("summary", "")),
        raw=raw_text,
    )


@app.get("/health")
async def healthcheck() -> dict[str, str | float]:
    return {"status": "ok", "model": MODEL_NAME, "timeout": VOICE_TIMEOUT_SECONDS}


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


@app.post("/analyze", response_model=VoiceAnalysisResponse)
async def api_analyze(file: UploadFile = File(...)) -> VoiceAnalysisResponse:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = Path(tmp.name)

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(get_voice_json, temp_path),
            timeout=VOICE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as error:
        raise HTTPException(status_code=504, detail="Voice analysis timed out") from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    finally:
        temp_path.unlink(missing_ok=True)


def run_cli() -> None:
    print(f"Voice broker uses {MODEL_NAME}")
    while True:
        path = input("Path to audio file (or 'exit'): ").strip().strip('"')
        if path.lower() == "exit":
            return
        if not path:
            continue

        try:
            result = get_voice_json(path)
            print(result.model_dump_json(ensure_ascii=False))
        except Exception as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    import sys
    import uvicorn

    if len(sys.argv) > 1 and sys.argv[1] == "api":
        uvicorn.run(app, host="0.0.0.0", port=8010)
    else:
        run_cli()