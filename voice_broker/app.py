import os
import queue
import tempfile
import threading
import time
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# --- КОНФИГУРАЦИЯ ---
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

# Модель просто описывает звук текстом, не зная про JSON
PROMPT_TEXT = "Describe the emotional state of the speaker and the tone of voice."

# --- СТАНДАРТНЫЙ JSON ДЛЯ ОТПРАВКИ ---
class VoiceAnalysisResponse(BaseModel):
    analysis: str

app = FastAPI()

# --- ГЛОБАЛЬНЫЕ ОБЪЕКТЫ ---
_processor = None
_model = None
_lock = threading.Lock()

def load_model():
    global _processor, _model
    with _lock:
        if _processor is None:
            _processor = AutoProcessor.from_pretrained(MODEL_ID)
            _model = Qwen2AudioForConditionalGeneration.from_pretrained(
                MODEL_ID, device_map="auto"
            )
    return _processor, _model

# --- ЯДРО АНАЛИЗА (ТВОЙ РАБОЧИЙ ВАРИАНТ) ---
def get_raw_analysis(audio_path: str | Path) -> str:
    processor, model = load_model()
    
    # 1. Загрузка (16кГц как просит Qwen)
    audio, _ = librosa.load(str(audio_path), sr=16000)

    # 2. Формирование промпта (Официальный шаблон)
    messages = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": str(audio_path)},
            {"type": "text", "text": PROMPT_TEXT}
        ]}
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audios=audio, return_tensors="pt", padding=True).to(model.device)

    # 3. Генерация (чистая строка)
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=256)
    
    # Отрезаем входной промпт, оставляем только ответ модели
    generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return response.strip()

# --- API ---
@app.post("/analyze", response_model=VoiceAnalysisResponse)
async def api_analyze(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        t_path = Path(tmp.name)
    
    try:
        # Модель дает строку, а мы сами делаем из нее JSON
        result_text = get_raw_analysis(t_path)
        return VoiceAnalysisResponse(analysis=result_text)
    finally:
        t_path.unlink(missing_ok=True)

# --- ТЕРМИНАЛ (ДЛЯ ТЕСТОВ) ---
def run_cli():
    print(f"Загрузка {MODEL_ID}...")
    load_model()
    
    while True:
        path = input("\nВведите путь к файлу (или 'exit'): ").strip().strip('"')
        if path.lower() == 'exit': break
        if not os.path.exists(path): continue
        
        try:
            raw_text = get_raw_analysis(path)
            # Вывод стандартного JSON
            final_json = VoiceAnalysisResponse(analysis=raw_text).model_dump_json(ensure_ascii=False)
            print(f"\nГОТОВЫЙ JSON:\n{final_json}")
        except Exception as e:
            print(f"Ошибка: {e}")

if __name__ == "__main__":
    import sys
    # Если запуск с аргументом 'api', стартуем сервер, иначе терминал
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        run_cli()