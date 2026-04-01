import asyncio
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import urllib3
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from tinkoff_sandbox import fetch_sandbox_portfolio, sandbox_is_configured

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CLIENT_SECRET = os.getenv("CLIENT_SECRET") or os.getenv("MKEY")
VERIFY_SSL = os.getenv("VERIFY_SSL", "false").lower() == "true"
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat-2-Pro")
VOICE_BROKER_URL = os.getenv("VOICE_BROKER_URL", "http://voice-broker:8010/analyze")
NEWS_SERVICE_URL = os.getenv("NEWS_SERVICE_URL", "http://news-service:8002")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "180"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "24"))
HTTP_TIMEOUT = httpx.Timeout(60.0, connect=15.0)

AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGA_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

WELCOME_MESSAGE = "Здравствуйте! Какая цель ваших инвестиций?"
VOICE_ANALYSIS_FALLBACK = "Голосовой анализ пока недоступен. Итог строится по тексту диалога."

SYSTEM_PROMPT = """
Ты — инвестиционный ассистент. Твоя задача — через спокойный диалог понять цели клиента, его отношение к риску, ограничения и возможности, и на этой основе предложить подходящий инвестиционный портфель.

Ты не проходишь формальный тест, а ведешь естественный разговор.

---

### Ограничения поведения (обязательные)

1. Ты НЕ называешь себя психологом, профилировщиком или любым подобным термином.
2. Ты НЕ говоришь, что анализируешь голос, эмоции по голосу или речь.
3. Ты НЕ упоминаешь скрытые метрики, индексы или внутренние оценки.
4. Ты НЕ объясняешь, как именно оцениваешь клиента.

---

### Ограничение на количество вопросов

1. Ты можешь задать НЕ БОЛЕЕ 15 вопросов за весь диалог.
2. После каждого вопроса увеличивай внутренний счетчик.
3. Если достигнут лимит ИЛИ информации уже достаточно раньше — переходи к финальному ответу.
4. Никогда не превышай лимит даже если информации мало.

---

### Правила диалога

1. Задавай только ОДИН вопрос за сообщение.
2. Первый вопрос всегда про цель инвестирования.
3. Все вопросы — открытые, без вариантов ответов.
4. Не используй тесты, шкалы, А/Б/В варианты.
5. Не задавай персональные вопросы (ФИО, возраст, адрес, доход и т.д.).
6. Можно мягко уточнять:
   - горизонт инвестирования
   - отношение к просадкам
   - потребность в ликвидности
   - возможность регулярных вложений

7. Если ответ поверхностный — задай ОДИН уточняющий вопрос.

---

### Что нужно определить (внутренне, не озвучивать)

- цель инвестирования
- горизонт
- отношение к риску 0-1, где 0 — полная консервативность, 1 — полная агрессивность
- эмоциональную устойчивость 0-1, где 0 — паникер, 1 — спокойный
- опыт инвестиций
- дисциплину
- ограничения и требования

---

### Если есть текущий портфель

Если в системном сообщении передан актуальный текущий портфель клиента из Tinkoff Sandbox:
- используй только этот портфель как источник истины
- не проси клиента вручную перечислять бумаги, если данных уже достаточно
- не выдумывай отсутствующие позиции
- оцени соответствие портфеля целям и выяви противоречия

Если sandbox-портфель не передан, и клиент сам упоминает инвестиции:
- попроси описать их простыми словами
- оцени соответствие целям
- проверь баланс риска
- выяви противоречия

---

### Когда завершать

Заверши диалог если:
- информации достаточно ИЛИ
- достигнут лимит вопросов

---

### Формат ответа при сборе информации

- задай следующий вопрос
- в конце добавь: [STATUS: COLLECTING]

---

### Финальный ответ

Когда данных достаточно:

1. Кратко опиши цель клиента
2. Определи тип инвестора (консервативный / умеренный / агрессивный)
3. Дай краткое объяснение
4. Если есть портфель — оцени его
5. Предложи целевой портфель

Добавь в конце:
[STATUS: FINISHED]

---

### JSON (обязательно после FINISHED)

Выведи JSON в блоке кода:

{
  "goal": "...",
  "investor_type": "...",
  "emotion_index": 0.0,
  "risk_index": 0.0,
  "capacity_index": 0.0,
  "voice_emotion_assessment": "оценка по тексту, без упоминания голоса",
  "wants": [],
  "constraints": [],
  "current_portfolio": {},
  "current_portfolio_assessment": "...",
  "target_portfolio": {
    "Акции": "0-100%",
    "Облигации": "0-100%",
    "Денежные инструменты": "0-100%",
    "Альтернативные/прочие": "0-100%"
  },
  "profile_summary": "..."
}
"""

STATUS_RE = re.compile(r"\[STATUS:\s*(COLLECTING|FINISHED)\]", re.IGNORECASE)
JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
GENERIC_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)

app = FastAPI(title="Survey Service")


@dataclass
class SurveySession:
    session_id: str
    history: list[dict[str, Any]] = field(default_factory=list)
    voice_notes: list[str] = field(default_factory=list)
    last_transcript: str = ""
    sandbox_portfolio: dict[str, Any] | None = None
    sandbox_prompt: str = ""
    sandbox_error: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionPayload(BaseModel):
    session_id: str


class AskTextPayload(SessionPayload):
    message: str


sessions: dict[str, SurveySession] = {}
sessions_lock = threading.Lock()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def trim_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(history) <= MAX_HISTORY_MESSAGES + 1:
        return history
    return [history[0], *history[-MAX_HISTORY_MESSAGES:]]


def create_session() -> SurveySession:
    session = SurveySession(
        session_id=str(uuid.uuid4()),
        history=[{"role": "system", "content": SYSTEM_PROMPT}],
    )
    refresh_sandbox_portfolio(session)
    with sessions_lock:
        sessions[session.session_id] = session
    return session


def notify_news_presence(action: str, session_id: str) -> None:
    url = f"{NEWS_SERVICE_URL.rstrip('/')}/api/presence/{action}"
    try:
        with httpx.Client(timeout=5.0) as client:
            client.post(url, json={"session_id": session_id})
    except Exception:
        return


def cleanup_expired_sessions() -> None:
    deadline = utc_now() - timedelta(seconds=SESSION_TTL_SECONDS)
    stale: list[str] = []
    with sessions_lock:
        for session_id, session in list(sessions.items()):
            if session.last_seen < deadline:
                stale.append(session_id)
                sessions.pop(session_id, None)
    for session_id in stale:
        notify_news_presence("unregister", session_id)


def get_session_or_404(session_id: str) -> SurveySession:
    cleanup_expired_sessions()
    with sessions_lock:
        session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Сессия опросника не найдена или уже закрыта.")
    session.last_seen = utc_now()
    return session


def aggregate_voice_analysis(session: SurveySession) -> str:
    if not session.voice_notes:
        return VOICE_ANALYSIS_FALLBACK
    unique: list[str] = []
    for item in reversed(session.voice_notes):
        text = item.strip()
        if text and text not in unique:
            unique.append(text)
    return " ".join(reversed(unique[:3])) if unique else VOICE_ANALYSIS_FALLBACK


def clamp_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, number)), 3)


def ensure_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_target_portfolio(value: Any) -> dict[str, str]:
    defaults = {
        "Акции": "",
        "Облигации": "",
        "Денежные инструменты": "",
        "Альтернативные/прочие": "",
    }
    if not isinstance(value, dict):
        return defaults
    result = defaults.copy()
    for key, item in value.items():
        result[str(key)] = str(item).strip()
    return result


def extract_json_payload(raw_content: str) -> str | None:
    text = str(raw_content).strip()
    match = JSON_FENCE_RE.search(text) or GENERIC_FENCE_RE.search(text)
    if match:
        return match.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def strip_structured_blocks(raw_content: str) -> str:
    text = str(raw_content).strip()
    match = JSON_FENCE_RE.search(text) or GENERIC_FENCE_RE.search(text)
    if match:
        text = text.replace(match.group(0), "")
    return STATUS_RE.sub("", text).strip()


def normalize_survey_result(data: dict[str, Any], session: SurveySession) -> dict[str, Any]:
    current_portfolio = session.sandbox_portfolio if session.sandbox_portfolio is not None else data.get("current_portfolio", "")
    return {
        "goal": str(data.get("goal", "")).strip(),
        "investor_type": str(data.get("investor_type", "")).strip(),
        "emotion_index": clamp_float(data.get("emotion_index")),
        "risk_index": clamp_float(data.get("risk_index")),
        "capacity_index": clamp_float(data.get("capacity_index")),
        "voice_emotion_assessment": str(data.get("voice_emotion_assessment") or aggregate_voice_analysis(session)).strip(),
        "wants": ensure_text_list(data.get("wants")),
        "constraints": ensure_text_list(data.get("constraints")),
        "current_portfolio": current_portfolio,
        "current_portfolio_assessment": str(data.get("current_portfolio_assessment", "")).strip(),
        "target_portfolio": normalize_target_portfolio(data.get("target_portfolio")),
        "profile_summary": str(data.get("profile_summary", "")).strip(),
    }


def parse_assistant_reply(raw_reply: str, session: SurveySession) -> dict[str, Any]:
    statuses = STATUS_RE.findall(raw_reply)
    status = statuses[-1].upper() if statuses else "COLLECTING"
    visible_reply = strip_structured_blocks(raw_reply)

    parsed_result = None
    json_error = None
    json_payload = extract_json_payload(raw_reply)
    if json_payload:
        try:
            parsed_result = normalize_survey_result(json.loads(json_payload), session)
        except Exception as error:
            json_error = str(error)

    if status == "FINISHED" and not parsed_result:
        parsed_result = normalize_survey_result({}, session)
    if status == "FINISHED" and parsed_result:
        parsed_result["voice_emotion_assessment"] = aggregate_voice_analysis(session)

    return {
        "reply": visible_reply or ("Профиль собран, результат ниже." if status == "FINISHED" else "Продолжаем опрос."),
        "status": status,
        "parsed_result": parsed_result,
        "voice_analysis": aggregate_voice_analysis(session),
        "sandbox_portfolio": session.sandbox_portfolio,
        "sandbox_error": session.sandbox_error,
        "json_error": json_error,
    }


def refresh_sandbox_portfolio(session: SurveySession) -> None:
    if not sandbox_is_configured():
        session.sandbox_portfolio = None
        session.sandbox_prompt = ""
        session.sandbox_error = ""
        return

    try:
        snapshot = fetch_sandbox_portfolio()
    except Exception as error:
        session.sandbox_portfolio = None
        session.sandbox_prompt = ""
        session.sandbox_error = str(error)
        return

    session.sandbox_portfolio = snapshot.to_dict()
    session.sandbox_prompt = (
        "Актуальный текущий портфель клиента получен из Tinkoff Sandbox. "
        "Это единственный достоверный источник current_portfolio. "
        "Не придумывай новые позиции и не проси клиента перечислять их заново, если это не нужно для уточнения целей. "
        f"{snapshot.to_prompt_text()}"
    )
    session.sandbox_error = ""


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


def ask_gigachat(session: SurveySession, text: str) -> str:
    token = get_token()
    refresh_sandbox_portfolio(session)
    with session.lock:
        session.history.append({"role": "user", "content": text.strip()})
        session.history = trim_history(session.history)
        messages = [dict(item) for item in session.history]
    if session.sandbox_prompt:
        base_system = messages[0]["content"] if messages and messages[0].get("role") == "system" else SYSTEM_PROMPT
        combined_system = f"{base_system.strip()}\n\n---\n\n### Актуальный портфель из Tinkoff Sandbox\n{session.sandbox_prompt.strip()}"
        if messages and messages[0].get("role") == "system":
            messages[0] = {"role": "system", "content": combined_system}
        else:
            messages.insert(0, {"role": "system", "content": combined_system})

    payload = {
        "model": GIGACHAT_MODEL,
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 1400,
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    with httpx.Client(timeout=HTTP_TIMEOUT, verify=VERIFY_SSL) as client:
        response = client.post(GIGA_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"GigaChat API error {response.status_code}: {response.text}")

    reply = response.json()["choices"][0]["message"]["content"]
    with session.lock:
        session.history.append({"role": "assistant", "content": reply})
        session.history = trim_history(session.history)
    return reply


def fetch_news(limit: int) -> dict[str, Any]:
    url = f"{NEWS_SERVICE_URL.rstrip('/')}/api/news"
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, params={"limit": limit})
        response.raise_for_status()
        return response.json()
    except Exception as error:
        return {
            "items": [],
            "active_viewers": 0,
            "last_refresh_at": None,
            "error": f"Не удалось получить новости: {error}",
        }


def call_voice_broker(filename: str, content_type: str, raw_audio: bytes) -> dict[str, Any]:
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        response = client.post(
            VOICE_BROKER_URL,
            files={"file": (filename, raw_audio, content_type)},
        )
    response.raise_for_status()
    return response.json()


@app.get("/")
def index() -> dict[str, str]:
    return {
        "service": "survey-service",
        "status": "ok",
        "message": "Frontend was moved to a dedicated service. Use the frontend container to open the UI.",
    }


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/session/start")
def start_session() -> dict[str, Any]:
    session = create_session()
    notify_news_presence("register", session.session_id)
    return {
        "session_id": session.session_id,
        "welcome_message": WELCOME_MESSAGE,
        "sandbox_portfolio": session.sandbox_portfolio,
        "sandbox_error": session.sandbox_error,
    }


@app.post("/api/session/ping")
def ping_session(payload: SessionPayload) -> dict[str, str]:
    session = get_session_or_404(payload.session_id)
    session.last_seen = utc_now()
    notify_news_presence("ping", payload.session_id)
    return {"status": "ok"}


@app.post("/api/session/close")
def close_session(payload: SessionPayload) -> dict[str, str]:
    with sessions_lock:
        sessions.pop(payload.session_id, None)
    notify_news_presence("unregister", payload.session_id)
    return {"status": "closed"}


@app.get("/api/news")
def news_proxy(session_id: str, limit: int = 8) -> dict[str, Any]:
    session = get_session_or_404(session_id)
    session.last_seen = utc_now()
    notify_news_presence("ping", session_id)
    return fetch_news(limit)


@app.get("/api/sandbox/portfolio")
def sandbox_portfolio(session_id: str) -> dict[str, Any]:
    session = get_session_or_404(session_id)
    refresh_sandbox_portfolio(session)
    if session.sandbox_error:
        raise HTTPException(status_code=502, detail=session.sandbox_error)
    return {
        "portfolio": session.sandbox_portfolio,
        "source": "tinkoff_sandbox",
    }


@app.post("/api/ask_text")
async def ask_text(payload: AskTextPayload) -> dict[str, Any]:
    session = get_session_or_404(payload.session_id)
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Пустое сообщение.")
    try:
        raw_reply = await asyncio.to_thread(ask_gigachat, session, payload.message)
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return parse_assistant_reply(raw_reply, session)


@app.post("/api/upload_voice")
async def upload_voice(session_id: str = Form(...), file: UploadFile = File(...)) -> dict[str, Any]:
    session = get_session_or_404(session_id)
    raw_audio = await file.read()
    if not raw_audio:
        raise HTTPException(status_code=400, detail="Файл пустой.")

    try:
        voice_result = await asyncio.to_thread(
            call_voice_broker,
            file.filename or "voice.webm",
            file.content_type or "audio/webm",
            raw_audio,
        )
    except httpx.HTTPStatusError as error:
        raise HTTPException(status_code=502, detail=f"Voice broker error: {error.response.text}") from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    transcript = str(voice_result.get("transcript", "")).strip()
    emotion_summary = str(voice_result.get("emotion_summary", "")).strip()
    if emotion_summary:
        session.voice_notes.append(emotion_summary)
    session.last_transcript = transcript

    prompt_parts = ["Пользователь ответил голосовым сообщением."]
    if transcript:
        prompt_parts.append(f"Расшифровка ответа: {transcript}")
    if emotion_summary:
        prompt_parts.append(f"Внутренняя эмоциональная заметка: {emotion_summary}")
    prompt_parts.append("Продолжи опрос по содержанию ответа. Если данных уже достаточно, заверши профиль.")

    try:
        raw_reply = await asyncio.to_thread(ask_gigachat, session, "\n".join(prompt_parts))
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    payload = parse_assistant_reply(raw_reply, session)
    payload["voice_broker"] = voice_result
    payload["transcript"] = transcript
    return payload


