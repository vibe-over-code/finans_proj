import json
import os
import queue
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# --- КОНФИГУРАЦИЯ ---
MODEL_ID = os.getenv("QWEN2_AUDIO_MODEL", "Qwen/Qwen2-Audio-7B-Instruct")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("VOICE_MAX_NEW_TOKENS", "256"))
DEFAULT_SAMPLE_RATE = int(os.getenv("VOICE_SAMPLE_RATE", "16000"))
DEFAULT_MAX_SECONDS = int(os.getenv("VOICE_MAX_SECONDS", "30"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

SYSTEM_PROMPT = """Ты — эксперт по анализу речевых эмоций (SER). Твоя задача: оценить психоэмоциональное состояние человека по голосу.
Игнорируй содержание слов, фокусируйся на интонации, громкости, темпе и дрожании голоса.

Выдай результат СТРОГО в формате JSON:
{
  "stability": 0.0-1.0,
  "emotions": "названия ключевых эмоций",
  "acoustic_analysis": "краткое описание: темп, интонационная кривая, наличие стресс-маркеров",
  "risk_level": "low/medium/high"
}

Критерии stability:
1.0 — Монотонный, уверенный, спокойный голос (диктор, профессионал).
0.5 — Обычный разговорный голос с естественными модуляциями.
0.0 — Паника, истерика, сильный гнев или прерывистое дыхание."""

USER_PROMPT = "Проанализируй акустику голоса. Текстовая расшифровка не нужна. Сосредоточься на эмоциональной стабильности."

# --- МОДЕЛИ ДАННЫХ ---
class VoiceAnalysisResponse(BaseModel):
    text: str
    stability_score: float
    emotions: str
    raw_output: str

class HealthResponse(BaseModel):
    status: str
    model_id: str
    model_loaded: bool

app = FastAPI(title="Voice Broker Stability Service")

# --- ГЛОБАЛЬНЫЕ ОБЪЕКТЫ ---
_model_lock = threading.Lock()
_processor: AutoProcessor | None = None
_model: Qwen2AudioForConditionalGeneration | None = None

def load_model() -> tuple[AutoProcessor, Qwen2AudioForConditionalGeneration]:
    global _processor, _model

    with _model_lock:
        if _processor is None or _model is None:
            print(f"[model] Загрузка {MODEL_ID} на {DEVICE} ...")
            _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
            _model = Qwen2AudioForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=DTYPE,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            _model.to(DEVICE)
            _model.eval()
            print("[model] Модель готова")

    return _processor, _model

# --- РАБОТА С МИКРОФОНОМ ---
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
        raise RuntimeError("Запись не содержит аудио. Проверьте микрофон.")

    audio = np.concatenate(audio_chunks, axis=0)
    sf.write(str(output_path), audio, sample_rate)
    return output_path

# --- ЛОГИКА АНАЛИЗА ---
def parse_stability(text: str) -> float:
    match = re.search(r"Стабильность:\s*([\d.]+)", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return 0.5

def analyze_audio_file(audio_path: str | Path, max_new_tokens: int = 128) -> VoiceAnalysisResponse:
    processor, model = load_model()
    path = Path(audio_path)

    # Загрузка и подготовка аудио (16кГц)
    import librosa
    audio, _ = librosa.load(str(path), sr=16000)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "input.wav"}, 
            {"type": "text", "text": USER_PROMPT}
        ]}
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=prompt, audio=audio, sampling_rate=16000, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False  # Для максимальной строгости
        )

    generated_ids = generated_ids[:, inputs["input_ids"].size(1):]
    response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # --- Парсинг JSON ---
    stability = 0.5
    emo_val = "Не определено"
    
    try:
        # Очистка и поиск JSON
        clean_json = re.sub(r'```json\s*|```', '', response_text).strip()
        start = clean_json.find('{')
        end = clean_json.rfind('}') + 1
        
        if start != -1 and end != 0:
            data = json.loads(clean_json[start:end])
            stability = data.get("stability", 0.5)
            # Собираем описание из анализа и списка эмоций
            emo_val = f"{data.get('emotions', '')} | {data.get('acoustic_analysis', '')}"
    except Exception as e:
        print(f"Ошибка парсинга: {e}")

    # Исправляем опечатку в названии переменной и ключа
    return VoiceAnalysisResponse(
        text="[СКРЫТО]", 
        stability_score=float(stability), # Теперь название совпадает с моделью Pydantic
        emotions=str(emo_val),
        raw_output=response_text
    )

# --- API ENDPOINTS ---
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

# --- ТЕРМИНАЛЬНЫЙ ИНТЕРФЕЙС ---
def print_result(result: VoiceAnalysisResponse) -> None:
    print("\n" + "="*40)
    print("РЕЗУЛЬТАТ АНАЛИЗА:")
    print("="*40)
    # Используем .model_dump() для совместимости с Pydantic V2
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    print("="*40 + "\n")

def run_terminal_mode() -> None:
    print("Тестовый голосовой сервис для инвестиционного ассистента")
    print(f"Модель: {MODEL_ID}")
    print("Режимы:")
    print("  1. Запись с микрофона (с ручной остановкой)")
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
                print(f"\nАнализ файла: {wav_path}")
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