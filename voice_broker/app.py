import json
import os
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

load_dotenv()

MODEL_ID = os.getenv("QWEN2_AUDIO_MODEL", "Qwen/Qwen2-Audio-7B-Instruct")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("VOICE_MAX_NEW_TOKENS", "256"))
DEFAULT_SAMPLE_RATE = int(os.getenv("VOICE_SAMPLE_RATE", "16000"))
DEFAULT_MAX_SECONDS = int(os.getenv("VOICE_MAX_SECONDS", "30"))

SYSTEM_PROMPT = """Ты анализируешь голосовые сообщения для инвестиционного голосового помощника.
Нужно по голосу и словам пользователя определить:
- точную расшифровку речи;
- эмоциональный тон;
- уверенность или тревожность;
- есть ли срочность в запросе;
- как лучше ответить помощнику в следующей реплике.

Верни только JSON без markdown в формате:
{
  "transcript": "текст речи пользователя",
  "sentiment": "positive|neutral|negative|mixed",
  "emotion": "calm|confident|uncertain|anxious|excited|frustrated",
  "urgency": "low|medium|high",
  "confidence_score": 0.0,
  "investment_intent": "кратко опиши, что хотел пользователь",
  "tone_summary": "кратко опиши голосовой тон и настроение",
  "assistant_reply_hint": "как помощнику лучше ответить следующим сообщением"
}
"""

USER_PROMPT = """Проанализируй это голосовое сообщение пользователя для инвестиционного ассистента.
Учитывай не только слова, но и эмоциональную подачу голоса.
Ответь строго JSON."""


class VoiceAnalysisResponse(BaseModel):
    transcript: str
    sentiment: str
    emotion: str
    urgency: str
    confidence_score: float
    investment_intent: str
    tone_summary: str
    assistant_reply_hint: str
    raw_model_text: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_id: str
    model_loaded: bool


app = FastAPI(title="Voice Broker Test Service")

_model_lock = threading.Lock()
_processor: AutoProcessor | None = None
_model: Qwen2AudioForConditionalGeneration | None = None
_device: str | None = None


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_model() -> tuple[AutoProcessor, Qwen2AudioForConditionalGeneration, str]:
    global _processor, _model, _device

    with _model_lock:
        if _processor is None or _model is None or _device is None:
            device = get_device()
            print(f"[model] loading {MODEL_ID} on {device} ...")
            _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
            _model = Qwen2AudioForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=get_torch_dtype(),
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            _model.to(device)
            _model.eval()
            _device = device
            print("[model] ready")

    return _processor, _model, _device


def list_microphones() -> list[dict[str, Any]]:
    devices = sd.query_devices()
    result = []
    for index, device in enumerate(devices):
        if int(device["max_input_channels"]) > 0:
            result.append(
                {
                    "index": index,
                    "name": device["name"],
                    "channels": int(device["max_input_channels"]),
                    "default_samplerate": int(device["default_samplerate"]),
                }
            )
    return result


def choose_input_device() -> int | None:
    devices = list_microphones()
    if not devices:
        return None

    print("\nДоступные микрофоны:")
    for device in devices:
        print(
            f"  [{device['index']}] {device['name']} | "
            f"channels={device['channels']} | default_sr={device['default_samplerate']}"
        )

    default_input = sd.default.device[0] if sd.default.device else None
    print(f"\nТекущий input device: {default_input}")
    raw_value = input("Введите индекс микрофона или просто Enter для дефолтного: ").strip()
    if not raw_value:
        return int(default_input) if default_input is not None and int(default_input) >= 0 else None
    return int(raw_value)


def record_microphone_to_wav(
    output_path: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = 1,
    max_seconds: int = DEFAULT_MAX_SECONDS,
    device: int | None = None,
) -> Path:
    audio_chunks: list[np.ndarray] = []
    errors: queue.Queue[Exception] = queue.Queue()
    stop_event = threading.Event()
    started_at = time.monotonic()

    def callback(indata: np.ndarray, frames: int, stream_time: Any, status: sd.CallbackFlags) -> None:
        del frames, stream_time
        if status:
            print(f"[audio] status: {status}")
        audio_chunks.append(indata.copy())
        if time.monotonic() - started_at >= max_seconds:
            stop_event.set()

    def wait_for_stop() -> None:
        input("Нажмите Enter, когда захотите остановить запись...\n")
        stop_event.set()

    print("\nПодготовка к записи.")
    print("1. Проверьте, что микрофон выбран правильно.")
    print("2. Нажмите Enter, чтобы начать запись.")
    print(f"3. Говорите. Автостоп через {max_seconds} сек.")
    input()

    waiter = threading.Thread(target=wait_for_stop, daemon=True)
    waiter.start()

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            callback=callback,
            device=device,
        ):
            print("Запись началась. Говорите в микрофон.")
            while not stop_event.is_set():
                time.sleep(0.05)
    except Exception as error:
        errors.put(error)

    if not errors.empty():
        raise errors.get()

    if not audio_chunks:
        raise RuntimeError("Запись не содержит аудио. Проверьте микрофон и разрешения.")

    audio = np.concatenate(audio_chunks, axis=0)
    sf.write(str(output_path), audio, sample_rate)
    return output_path


def build_analysis_prompt() -> str:
    return (
        "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"{USER_PROMPT}"
    )


def extract_json_payload(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            raw = "\n".join(lines[1:-1]).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Model did not return JSON")
    return json.loads(raw[start : end + 1])


def analyze_audio_file(audio_path: str | Path, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> VoiceAnalysisResponse:
    processor, model, device = load_model()
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)

    target_sr = processor.feature_extractor.sampling_rate
    if sample_rate != target_sr:
        try:
            import librosa
        except ImportError as error:
            raise RuntimeError(
                f"Audio sample rate is {sample_rate}, but model expects {target_sr}. Install librosa or record at {target_sr} Hz."
            ) from error
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr

    prompt = build_analysis_prompt()
    inputs = processor(text=prompt, audio=audio, return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, prompt_length:]
    raw_output = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    payload = extract_json_payload(raw_output)
    payload["raw_model_text"] = raw_output
    return VoiceAnalysisResponse(**payload)


@app.get("/health", response_model=HealthResponse)
async def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok", model_id=MODEL_ID, model_loaded=_model is not None)


@app.get("/api/microphones")
async def microphones() -> dict[str, Any]:
    return {"items": list_microphones()}


@app.post("/api/analyze-voice", response_model=VoiceAnalysisResponse)
async def analyze_voice(file: UploadFile = File(...)) -> VoiceAnalysisResponse:
    suffix = Path(file.filename or "voice.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = Path(tmp.name)

    try:
        return analyze_audio_file(temp_path)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        temp_path.unlink(missing_ok=True)


def print_result(result: VoiceAnalysisResponse) -> None:
    payload = result.model_dump() if hasattr(result, "model_dump") else result.dict()
    print("\nAnalysis result:\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

def run_terminal_mode() -> None:
    print("Тестовый голосовой сервис для инвестиционного ассистента")
    print(f"Модель: {MODEL_ID}")
    print("Режимы:")
    print("  1. Запись с микрофона")
    print("  2. Анализ готового аудиофайла")
    print("  3. Выход")

    while True:
        choice = input("\nВыберите режим [1/2/3]: ").strip()

        if choice == "1":
            device = choose_input_device()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                temp_path = Path(tmp.name)

            try:
                wav_path = record_microphone_to_wav(temp_path, device=device)
                print(f"\nФайл записан: {wav_path}")
                result = analyze_audio_file(wav_path)
                print_result(result)
            except Exception as error:
                print(f"\nОшибка записи или анализа: {error}")
            finally:
                temp_path.unlink(missing_ok=True)

        elif choice == "2":
            raw_path = input("Введите путь к аудиофайлу: ").strip().strip('"')
            if not raw_path:
                print("Путь не указан.")
                continue
            try:
                result = analyze_audio_file(raw_path)
                print_result(result)
            except Exception as error:
                print(f"\nОшибка анализа: {error}")

        elif choice == "3":
            print("Завершаю работу.")
            return

        else:
            print("Нужен выбор 1, 2 или 3.")


if __name__ == "__main__":
    run_terminal_mode()
