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
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

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
Ты инвестиционный ассистент. Через спокойный разговор пойми цель клиента, его отношение к риску,
ограничения и ожидания, а затем предложи подходящий профиль инвестора.

Правила:
1. Не называй себя психологом или профилировщиком.
2. Не говори пользователю, что анализируешь голос или скрытые метрики.
3. Задавай только один открытый вопрос за сообщение.
4. Первый вопрос всегда про цель инвестирования.
5. Не задавай персональные вопросы вроде ФИО, адреса и дохода.
6. Всего не более 15 вопросов.
7. Если данных достаточно раньше, переходи к финальному ответу.

Промежуточный ответ:
- задай следующий вопрос
- в конце добавь [STATUS: COLLECTING]

Финальный ответ:
- кратко опиши цель
- определи тип инвестора
- поясни вывод
- если есть текущий портфель, оцени его
- предложи целевой портфель
- в конце добавь [STATUS: FINISHED]
- сразу после этого выведи JSON в блоке ```json ... ```

Обязательный JSON:
{
  "goal": "...",
  "investor_type": "...",
  "emotion_index": 0.0,
  "risk_index": 0.0,
  "capacity_index": 0.0,
  "voice_emotion_assessment": "...",
  "wants": [],
  "constraints": [],
  "current_portfolio": "...",
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
    return {
        "goal": str(data.get("goal", "")).strip(),
        "investor_type": str(data.get("investor_type", "")).strip(),
        "emotion_index": clamp_float(data.get("emotion_index")),
        "risk_index": clamp_float(data.get("risk_index")),
        "capacity_index": clamp_float(data.get("capacity_index")),
        "voice_emotion_assessment": str(data.get("voice_emotion_assessment") or aggregate_voice_analysis(session)).strip(),
        "wants": ensure_text_list(data.get("wants")),
        "constraints": ensure_text_list(data.get("constraints")),
        "current_portfolio": data.get("current_portfolio", ""),
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
        "json_error": json_error,
    }


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
    with session.lock:
        session.history.append({"role": "user", "content": text.strip()})
        session.history = trim_history(session.history)
        messages = [dict(item) for item in session.history]

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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/session/start")
def start_session() -> dict[str, str]:
    session = create_session()
    notify_news_presence("register", session.session_id)
    return {"session_id": session.session_id, "welcome_message": WELCOME_MESSAGE}


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


HTML_PAGE = """
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Инвест-опросник</title>
<style>
:root { --bg:#eef3ef; --card:#ffffff; --border:#d8dfd6; --text:#1f2a22; --muted:#5a675f; --accent:#1d6b52; }
* { box-sizing:border-box; }
body { margin:0; font-family:"Segoe UI",sans-serif; background:linear-gradient(180deg,#f7faf5,#eef3ef); color:var(--text); }
.main { display:grid; grid-template-columns:1.15fr 0.85fr; gap:16px; min-height:100vh; padding:16px; }
.panel { background:rgba(255,255,255,0.93); border:1px solid var(--border); border-radius:22px; box-shadow:0 18px 48px rgba(24,37,29,.08); overflow:hidden; }
.head { padding:18px 20px; border-bottom:1px solid var(--border); background:linear-gradient(135deg,rgba(29,107,82,.1),rgba(255,255,255,.4)); }
.head h1,.head h2,.card h3 { margin:0; }
.sub { margin:6px 0 0; color:var(--muted); font-size:14px; }
.chat { display:flex; flex-direction:column; min-height:calc(100vh - 32px); }
.messages { flex:1; padding:18px 20px; overflow:auto; display:flex; flex-direction:column; gap:10px; }
.msg { max-width:84%; padding:12px 14px; border-radius:16px; white-space:pre-wrap; line-height:1.45; }
.msg.bot { align-self:flex-start; background:#eef3ef; }
.msg.user { align-self:flex-end; background:var(--accent); color:#fff; }
.meta { padding:0 20px 8px; color:var(--muted); font-size:13px; display:none; }
.result { display:none; padding:0 20px 18px; }
.card { background:#f8faf8; border:1px solid var(--border); border-radius:18px; padding:14px; }
.grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; margin-top:12px; }
.item { background:#fff; border:1px solid var(--border); border-radius:14px; padding:10px; }
.label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
.value { margin-top:6px; white-space:pre-wrap; font-weight:600; }
.composer { display:flex; gap:10px; padding:16px 20px 20px; }
.composer input { flex:1; border:1px solid var(--border); border-radius:999px; padding:14px 16px; }
button { border:none; border-radius:999px; padding:12px 16px; cursor:pointer; font-weight:600; }
.send,.refresh { background:var(--accent); color:#fff; }
.voice { background:#dff0e8; color:var(--accent); }
.voice.recording { background:#f8d9d3; color:#ad3d2f; }
.news-body { padding:18px; display:flex; flex-direction:column; gap:12px; }
.news { background:#f8faf8; border:1px solid var(--border); border-radius:18px; padding:14px; }
.news p { margin:8px 0 0; color:var(--muted); line-height:1.45; }
.row { display:flex; justify-content:space-between; gap:10px; align-items:start; }
.badge { padding:6px 10px; border-radius:999px; background:#e4efe9; color:var(--accent); font-size:12px; font-weight:700; }
@media (max-width:980px) { .main { grid-template-columns:1fr; } .chat { min-height:auto; } }
@media (max-width:720px) { .grid { grid-template-columns:1fr; } .composer { flex-wrap:wrap; } .composer input { width:100%; } }
</style>
</head>
<body>
<div class="main">
  <section class="panel chat">
    <div class="head">
      <h1>Опросник инвестора</h1>
      <p class="sub">Сервис опросника только ведёт диалог. Голос уходит в отдельный broker и возвращается как расшифровка плюс эмоции.</p>
    </div>
    <div id="messages" class="messages"></div>
    <div id="voiceMeta" class="meta"></div>
    <div id="result" class="result"></div>
    <div class="composer">
      <button id="voiceBtn" class="voice" type="button">Голос</button>
      <input id="textInput" type="text" placeholder="Ответьте ассистенту" />
      <button id="sendBtn" class="send" type="button">Отправить</button>
    </div>
  </section>
  <aside class="panel">
    <div class="head">
      <div class="row">
        <div>
          <h2>Новости рынка</h2>
          <p class="sub">Лента обновляется только пока у кого-то открыт опросник.</p>
        </div>
        <button id="refreshBtn" class="refresh" type="button">Обновить</button>
      </div>
      <p id="newsStatus" class="sub">Подключаем ленту...</p>
    </div>
    <div id="newsList" class="news-body"></div>
  </aside>
</div>
<script>
let sessionId = null;
let mediaRecorder = null;
let audioChunks = [];

const messagesEl = document.getElementById('messages');
const voiceMetaEl = document.getElementById('voiceMeta');
const resultEl = document.getElementById('result');
const textInputEl = document.getElementById('textInput');
const sendBtnEl = document.getElementById('sendBtn');
const voiceBtnEl = document.getElementById('voiceBtn');
const refreshBtnEl = document.getElementById('refreshBtn');
const newsListEl = document.getElementById('newsList');
const newsStatusEl = document.getElementById('newsStatus');

function escapeHtml(value) {
  return String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function addMessage(text, side) {
  const div = document.createElement('div');
  div.className = 'msg ' + side;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderResult(result) {
  resultEl.style.display = 'block';
  const portfolio = Object.entries(result.target_portfolio || {}).map(([k, v]) => `${k}: ${v}`).join('\\n');
  const wants = (result.wants || []).join('\\n') || 'Нет данных';
  const constraints = (result.constraints || []).join('\\n') || 'Нет данных';
  resultEl.innerHTML = `
    <div class="card">
      <h3>Результат</h3>
      <div class="grid">
        <div class="item"><div class="label">Цель</div><div class="value">${escapeHtml(result.goal || 'Не указана')}</div></div>
        <div class="item"><div class="label">Тип инвестора</div><div class="value">${escapeHtml(result.investor_type || 'Не определён')}</div></div>
        <div class="item"><div class="label">Risk Index</div><div class="value">${escapeHtml(result.risk_index)}</div></div>
        <div class="item"><div class="label">Emotion Index</div><div class="value">${escapeHtml(result.emotion_index)}</div></div>
        <div class="item"><div class="label">Capacity Index</div><div class="value">${escapeHtml(result.capacity_index)}</div></div>
        <div class="item"><div class="label">Анализ голоса</div><div class="value">${escapeHtml(result.voice_emotion_assessment || 'Нет данных')}</div></div>
        <div class="item"><div class="label">Ожидания</div><div class="value">${escapeHtml(wants)}</div></div>
        <div class="item"><div class="label">Ограничения</div><div class="value">${escapeHtml(constraints)}</div></div>
        <div class="item"><div class="label">Текущий портфель</div><div class="value">${escapeHtml(typeof result.current_portfolio === 'string' ? result.current_portfolio : JSON.stringify(result.current_portfolio || {}, null, 2))}</div></div>
        <div class="item"><div class="label">Оценка портфеля</div><div class="value">${escapeHtml(result.current_portfolio_assessment || 'Нет данных')}</div></div>
        <div class="item"><div class="label">Целевой портфель</div><div class="value">${escapeHtml(portfolio || 'Нет данных')}</div></div>
        <div class="item"><div class="label">Итоговый профиль</div><div class="value">${escapeHtml(result.profile_summary || 'Нет данных')}</div></div>
      </div>
    </div>`;
}

function renderVoiceMeta(data) {
  const transcript = data?.transcript || data?.voice_broker?.transcript || '';
  const summary = data?.voice_broker?.emotion_summary || data?.voice_analysis || '';
  if (!transcript && !summary) {
    voiceMetaEl.style.display = 'none';
    voiceMetaEl.textContent = '';
    return;
  }
  voiceMetaEl.style.display = 'block';
  const parts = [];
  if (transcript) parts.push('Расшифровка: ' + transcript);
  if (summary) parts.push('Эмоции: ' + summary);
  voiceMetaEl.textContent = parts.join(' | ');
}

function renderNews(items) {
  if (!items || !items.length) {
    newsListEl.innerHTML = '<div class="news"><strong>Пока пусто</strong><p>Сервис ещё не подготовил новости.</p></div>';
    return;
  }
  newsListEl.innerHTML = items.map((item) => {
    const analysis = item.portfolio_risk_analysis || {};
    const badge = analysis.has_portfolio_risk ? 'Есть риск' : 'Без сигнала';
    return `<article class="news"><div class="row"><strong>${escapeHtml(item.title || 'Без заголовка')}</strong><span class="badge">${escapeHtml(badge)}</span></div><p>${escapeHtml(item.summary || '')}</p><p>${escapeHtml(analysis.summary || '')}</p></article>`;
  }).join('');
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || data.error || 'Ошибка запроса');
  return data;
}

async function refreshNews() {
  if (!sessionId) return;
  newsStatusEl.textContent = 'Обновляем новости...';
  try {
    const data = await api(`/api/news?session_id=${encodeURIComponent(sessionId)}&limit=8`);
    renderNews(data.items || []);
    newsStatusEl.textContent = data.error || `Активных опросников: ${data.active_viewers ?? 0}`;
  } catch (error) {
    newsStatusEl.textContent = error.message;
    renderNews([]);
  }
}

function applyAssistantPayload(data) {
  if (data.reply) addMessage(data.reply, 'bot');
  renderVoiceMeta(data);
  if (data.parsed_result) renderResult(data.parsed_result);
}

async function sendText() {
  const message = textInputEl.value.trim();
  if (!message || !sessionId) return;
  addMessage(message, 'user');
  textInputEl.value = '';
  try {
    const data = await api('/api/ask_text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message })
    });
    applyAssistantPayload(data);
  } catch (error) {
    addMessage(error.message, 'bot');
  }
}

async function uploadVoice(blob) {
  if (!sessionId) return;
  const formData = new FormData();
  formData.append('session_id', sessionId);
  formData.append('file', blob, 'voice.webm');
  try {
    const data = await api('/api/upload_voice', { method: 'POST', body: formData });
    applyAssistantPayload(data);
  } catch (error) {
    addMessage(error.message, 'bot');
  }
}

async function toggleRecording() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    voiceBtnEl.classList.remove('recording');
    voiceBtnEl.textContent = 'Голос';
    return;
  }

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
  audioChunks = [];
  mediaRecorder.ondataavailable = (event) => audioChunks.push(event.data);
  mediaRecorder.onstop = () => {
    const blob = new Blob(audioChunks, { type: 'audio/webm' });
    addMessage('Голосовое сообщение отправлено.', 'user');
    uploadVoice(blob);
    stream.getTracks().forEach((track) => track.stop());
  };
  mediaRecorder.start();
  voiceBtnEl.classList.add('recording');
  voiceBtnEl.textContent = 'Стоп';
}

function closeSession() {
  if (!sessionId) return;
  fetch('/api/session/close', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
    keepalive: true
  }).catch(() => {});
}

async function pingSession() {
  if (!sessionId) return;
  try {
    await api('/api/session/ping', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId })
    });
  } catch (_) {}
}

async function init() {
  try {
    const data = await api('/api/session/start', { method: 'POST' });
    sessionId = data.session_id;
    addMessage(data.welcome_message, 'bot');
    refreshNews();
    setInterval(pingSession, 15000);
    setInterval(refreshNews, 30000);
  } catch (error) {
    addMessage(error.message, 'bot');
    newsStatusEl.textContent = error.message;
  }
}

sendBtnEl.addEventListener('click', sendText);
voiceBtnEl.addEventListener('click', toggleRecording);
refreshBtnEl.addEventListener('click', refreshNews);
textInputEl.addEventListener('keydown', (event) => { if (event.key === 'Enter') sendText(); });
window.addEventListener('beforeunload', closeSession);
window.addEventListener('load', init);
</script>
</body>
</html>
"""
