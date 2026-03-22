import json
import logging
import os
import re
import uuid
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from gigachat import GigaChat
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
load_dotenv()

MODEL_NAME = os.getenv("GIGACHAT_MODEL", "GigaChat-2-Pro")
VERIFY_SSL = os.getenv("GIGACHAT_VERIFY_SSL", "false").strip().lower() in {"1", "true", "yes"}
PORTFOLIO_PROFILE_URL = os.getenv("PORTFOLIO_PROFILE_URL", "http://localhost:8000/api/client-profile")
VOICE_SERVICE_URL = os.getenv("VOICE_SERVICE_URL", "http://localhost:8010/analyze")


def normalize_credentials(raw_value: str | None) -> str | None:
    if not raw_value:
        return None
    value = raw_value.strip().strip('"').strip("'")
    if value.startswith("CLIENT_SECRET="):
        value = value.split("=", 1)[1].strip()
    return value or None


GIGACHAT_CREDENTIALS = normalize_credentials(
    os.getenv("CLIENT_SECRET") or os.getenv("GIGACHAT_CREDENTIALS") or os.getenv("MKey")
)


app = FastAPI(title="Survey Web UI")


class StartSessionResponse(BaseModel):
    session_id: str
    message: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    message: str
    finished: bool
    profile: dict[str, Any] | None = None
    voice: dict[str, Any] | None = None


class SessionResetRequest(BaseModel):
    session_id: str


SYSTEM_PROMPT = """
Ты проводишь короткое финансовое интервью и собираешь профиль инвестора для сервиса ребалансировки.

Твои правила:
1. Задавай только один вопрос за раз.
2. Первый вопрос всегда про цель инвестиций.
3. Вопросы короткие, открытые, без вариантов ответа.
4. Если ответ слишком короткий, мягко попроси уточнить.
5. Не упоминай JSON, API, схему данных или внутренние поля.
6. Когда данных достаточно, заверши диалог.

Нужно собрать:
- goal
- investor_type
- risk_index от 0.0 до 1.0
- emotion_index от 0.0 до 1.0
- target_portfolio как JSON-объект с долями в процентах
- current_portfolio как JSON-объект только если данных достаточно
- profile_summary

Требования к портфелям:
1. target_portfolio обязателен.
2. target_portfolio и current_portfolio должны быть объектами вида:
   {
     "Облигации": "50%",
     "Акции": "40%",
     "Золото": "10%"
   }
3. Не используй массивы для портфеля.
4. Если current_portfolio неизвестен, не добавляй его.

Маркеры:
- Во время интервью добавляй [STATUS: COLLECTING]
- В финальном сообщении добавляй [STATUS: FINISHED]

Финал:
Когда данных достаточно, верни короткое завершение и затем JSON в fenced block.
"""


INDEX_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Финансовый опросник</title>
  <style>
    :root {
      --text: #2f241d;
      --muted: #6f6258;
      --panel: rgba(255, 250, 242, 0.94);
      --accent: #0f766e;
      --accent-2: #c2410c;
      --border: rgba(47, 36, 29, 0.12);
      --user: #e7f7f4;
      --bot: #fff8ef;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(194, 65, 12, 0.12), transparent 30%),
        radial-gradient(circle at bottom right, rgba(15, 118, 110, 0.16), transparent 35%),
        linear-gradient(135deg, #efe4d2 0%, #f7f2ea 50%, #e7edea 100%);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }

    .shell {
      width: min(980px, 100%);
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 24px;
      backdrop-filter: blur(12px);
      box-shadow: 0 24px 70px rgba(49, 38, 29, 0.15);
      overflow: hidden;
    }

    .hero {
      padding: 28px 28px 18px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(135deg, rgba(255,255,255,0.45), rgba(255,255,255,0.18));
    }

    h1 {
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 40px);
      line-height: 1.05;
    }

    .subtitle {
      margin: 0;
      color: var(--muted);
      max-width: 60ch;
      line-height: 1.5;
    }

    .toolbar, .voicebar {
      display: flex;
      gap: 12px;
      padding: 18px 28px 0;
      flex-wrap: wrap;
      align-items: center;
    }

    .voicebar {
      padding-top: 10px;
      padding-bottom: 18px;
    }

    button {
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      font: inherit;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease;
    }

    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }

    .primary { background: var(--accent); color: white; }
    .secondary { background: transparent; color: var(--accent-2); border: 1px solid rgba(194, 65, 12, 0.24); }

    .chat {
      padding: 18px 28px;
      height: 54vh;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .bubble {
      max-width: min(760px, 92%);
      padding: 14px 16px;
      border-radius: 18px;
      line-height: 1.5;
      white-space: pre-wrap;
      border: 1px solid var(--border);
    }

    .bot { align-self: flex-start; background: var(--bot); }
    .user { align-self: flex-end; background: var(--user); }
    .system { align-self: center; background: rgba(47, 36, 29, 0.06); color: var(--muted); }

    .composer {
      padding: 0 28px 10px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
    }

    textarea, .file {
      width: 100%;
      border-radius: 18px;
      border: 1px solid var(--border);
      padding: 14px 16px;
      font: inherit;
      background: rgba(255,255,255,0.72);
      color: var(--text);
    }

    textarea { min-height: 84px; resize: vertical; }

    .status {
      padding: 0 28px 18px;
      color: var(--muted);
      min-height: 24px;
      font-size: 14px;
    }

    .report {
      margin: 0 28px 28px;
      padding: 18px;
      border-radius: 20px;
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--border);
      display: none;
      white-space: pre-wrap;
      line-height: 1.5;
    }

    .report.visible { display: block; }

    @media (max-width: 720px) {
      .composer { grid-template-columns: 1fr; }
      .chat { height: 50vh; }
      .bubble { max-width: 100%; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>Локальный опросник инвестора</h1>
      <p class="subtitle">Текст и голос в одном окне. Голос отправляется в voice-broker, возвращается транскрипт и эмоции, после чего опрос продолжается.</p>
    </section>

    <div class="toolbar">
      <button id="startBtn" type="button" class="primary">Начать опрос</button>
      <button id="resetBtn" type="button" class="secondary">Сбросить</button>
      <span id="sessionTag" style="color: var(--muted); font-size: 14px;"></span>
    </div>

    <section id="chat" class="chat">
      <div class="bubble system">Нажмите «Начать опрос», чтобы открыть диалог.</div>
    </section>

    <form id="chatForm" class="composer">
      <textarea id="messageInput" placeholder="Напишите ответ на вопрос..." disabled></textarea>
      <button id="sendBtn" type="submit" class="primary" disabled>Отправить</button>
    </form>

    <div class="voicebar">
      <button id="recordBtn" type="button" class="secondary" disabled>Записать голос</button>
      <button id="stopBtn" type="button" class="secondary" disabled>Стоп</button>
      <input id="fileInput" class="file" type="file" accept="audio/*" disabled />
      <button id="uploadBtn" type="button" class="secondary" disabled>Отправить файл</button>
    </div>

    <div id="status" class="status"></div>
    <pre id="report" class="report"></pre>
  </main>

  <script>
    const chat = document.getElementById("chat");
    const report = document.getElementById("report");
    const statusNode = document.getElementById("status");
    const startBtn = document.getElementById("startBtn");
    const resetBtn = document.getElementById("resetBtn");
    const sendBtn = document.getElementById("sendBtn");
    const chatForm = document.getElementById("chatForm");
    const messageInput = document.getElementById("messageInput");
    const recordBtn = document.getElementById("recordBtn");
    const stopBtn = document.getElementById("stopBtn");
    const fileInput = document.getElementById("fileInput");
    const uploadBtn = document.getElementById("uploadBtn");
    const sessionTag = document.getElementById("sessionTag");

    let sessionId = null;
    let finished = false;
    let recorder = null;
    let streamRef = null;
    let chunks = [];

    function addBubble(text, kind) {
      const node = document.createElement("div");
      node.className = `bubble ${kind}`;
      node.textContent = text;
      chat.appendChild(node);
      chat.scrollTop = chat.scrollHeight;
    }

    function fetchWithTimeout(url, options = {}, timeoutMs = 30000) {
      const controller = new AbortController();
      const id = setTimeout(() => controller.abort(), timeoutMs);
      return fetch(url, { ...options, signal: controller.signal }).finally(() => clearTimeout(id));
    }

    function setControlsDisabled(disabled) {
      sendBtn.disabled = disabled || !sessionId || finished;
      messageInput.disabled = disabled || !sessionId || finished;
      recordBtn.disabled = disabled || !sessionId || finished;
      fileInput.disabled = disabled || !sessionId || finished;
      uploadBtn.disabled = disabled || !sessionId || finished;
      resetBtn.disabled = disabled && !sessionId;
      startBtn.disabled = disabled;
    }

    function setBusy(disabled, text = "") {
      setControlsDisabled(disabled);
      if (!recorder || recorder.state !== "recording") {
        stopBtn.disabled = true;
      }
      statusNode.textContent = text;
    }

    function renderReport(profile) {
      if (!profile) {
        report.classList.remove("visible");
        report.textContent = "";
        return;
      }

      const lines = [
        "Ваш инвестиционный паспорт",
        "",
        `Цель: ${profile.goal ?? ""}`,
        `Тип: ${profile.investor_type ?? ""}`,
        `Индекс риска: ${profile.risk_index ?? ""}`,
        `Эмоциональная устойчивость: ${profile.emotion_index ?? ""}`,
        "Целевой портфель:",
        JSON.stringify(profile.target_portfolio ?? {}, null, 2),
      ];

      if (profile.current_portfolio) {
        lines.push("", "Текущий портфель:", JSON.stringify(profile.current_portfolio, null, 2));
      }

      lines.push("", "Профиль:", profile.profile_summary ?? "");
      report.textContent = lines.join("\n");
      report.classList.add("visible");
    }

    async function startSurvey() {
      setBusy(true, "Запускаю интервью...");
      finished = false;
      report.classList.remove("visible");
      report.textContent = "";

      try {
        const response = await fetchWithTimeout("/api/session/start", { method: "POST" }, 30000);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Не удалось начать опрос");

        sessionId = data.session_id;
        sessionTag.textContent = `session: ${sessionId.slice(0, 8)}`;
        chat.innerHTML = "";
        addBubble(data.message, "bot");
        setBusy(false, "Опрос активен");
        messageInput.focus();
      } catch (error) {
        setBusy(false, `Ошибка старта: ${error}`);
      }
    }

    async function resetSurvey() {
      setBusy(true, "Сбрасываю сессию...");
      try {
        if (sessionId) {
          await fetchWithTimeout(
            "/api/session/reset",
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ session_id: sessionId }),
            },
            10000,
          );
        }
      } finally {
        sessionId = null;
        finished = false;
        sessionTag.textContent = "";
        chat.innerHTML = "";
        addBubble("Нажмите «Начать опрос», чтобы открыть диалог.", "system");
        renderReport(null);
        messageInput.value = "";
        fileInput.value = "";
        setBusy(false, "");
      }
    }

    async function sendText(event) {
      event.preventDefault();
      const message = messageInput.value.trim();
      if (!message || !sessionId || finished) return;

      addBubble(message, "user");
      messageInput.value = "";
      setBusy(true, "Модель отвечает...");

      try {
        const response = await fetchWithTimeout(
          "/api/chat",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId, message }),
          },
          30000,
        );
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Не удалось обработать сообщение");

        addBubble(data.message, "bot");
        finished = Boolean(data.finished);
        renderReport(data.profile || null);
        setBusy(false, finished ? "Опрос завершен" : "Опрос активен");
      } catch (error) {
        addBubble(`Ошибка: ${error}`, "system");
        setBusy(false, "Ошибка при обработке сообщения");
      }
    }

    async function sendVoiceFile(file, label) {
      if (!sessionId || finished || !file) return;

      addBubble(label, "user");
      setBusy(true, "Отправляю голос и жду расшифровку...");

      const form = new FormData();
      form.append("session_id", sessionId);
      form.append("file", file, file.name || "voice.webm");

      try {
        const response = await fetchWithTimeout("/api/voice", { method: "POST", body: form }, 60000);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Не удалось обработать голос");

        if (data.voice && data.voice.transcript) {
          addBubble(`Транскрипт: ${data.voice.transcript}`, "system");
          if (data.voice.emotion) {
            addBubble(`Эмоции: ${data.voice.emotion} (index=${data.voice.emotion_index})`, "system");
          }
        }

        addBubble(data.message, "bot");
        finished = Boolean(data.finished);
        renderReport(data.profile || null);
        setBusy(false, finished ? "Опрос завершен" : "Опрос активен");
      } catch (error) {
        addBubble(`Ошибка голоса: ${error}`, "system");
        setBusy(false, "Ошибка при обработке голоса");
      }
    }

    async function startRecording() {
      if (!sessionId || finished) return;

      try {
        streamRef = await navigator.mediaDevices.getUserMedia({ audio: true });
        chunks = [];
        recorder = new MediaRecorder(streamRef);

        recorder.ondataavailable = (event) => {
          if (event.data && event.data.size > 0) chunks.push(event.data);
        };

        recorder.onstop = async () => {
          try {
            const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
            const voiceFile = new File([blob], "recording.webm", { type: blob.type || "audio/webm" });
            await sendVoiceFile(voiceFile, "Голосовое сообщение отправлено");
          } finally {
            if (streamRef) {
              streamRef.getTracks().forEach((track) => track.stop());
            }
            streamRef = null;
            recorder = null;
            chunks = [];
            stopBtn.disabled = true;
            if (!finished) {
              recordBtn.disabled = false;
            }
          }
        };

        recorder.start();
        recordBtn.disabled = true;
        stopBtn.disabled = false;
        statusNode.textContent = "Идет запись...";
      } catch (error) {
        addBubble(`Не удалось включить микрофон: ${error}`, "system");
      }
    }

    function stopRecording() {
      if (!recorder || recorder.state !== "recording") return;
      stopBtn.disabled = true;
      recorder.stop();
    }

    async function uploadSelectedFile() {
      const selected = fileInput.files && fileInput.files[0];
      if (!selected) {
        addBubble("Сначала выберите аудиофайл.", "system");
        return;
      }
      await sendVoiceFile(selected, "Аудиофайл отправлен");
      fileInput.value = "";
    }

    startBtn.addEventListener("click", startSurvey);
    resetBtn.addEventListener("click", resetSurvey);
    chatForm.addEventListener("submit", sendText);
    recordBtn.addEventListener("click", startRecording);
    stopBtn.addEventListener("click", stopRecording);
    uploadBtn.addEventListener("click", uploadSelectedFile);
  </script>
</body>
</html>
"""


sessions: dict[str, list[dict[str, str]]] = {}


def get_gigachat_client() -> GigaChat:
    if not GIGACHAT_CREDENTIALS:
        raise RuntimeError("CLIENT_SECRET not found in environment")
    return GigaChat(credentials=GIGACHAT_CREDENTIALS, verify_ssl_certs=VERIFY_SSL, model=MODEL_NAME)


async def send_profile_to_portfolio_service(profile: dict[str, Any], session_id: str) -> None:
    payload = {"client_id": f"web_{session_id}", "survey_result": profile}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(PORTFOLIO_PROFILE_URL, json=payload)
            response.raise_for_status()
    except Exception as error:
        logging.exception("Failed to send survey profile: %s", error)


def extract_final_payload(ai_text: str) -> dict[str, Any] | None:
    match = re.search(r"```json\s*(.*?)\s*```", ai_text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip(), strict=False)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from AI: {e}")
        return None


def clean_ai_text(ai_text: str) -> str:
    cleaned = ai_text.replace("[STATUS: COLLECTING]", "")
    cleaned = cleaned.replace("[STATUS: FINISHED]", "")
    cleaned = re.sub(r"```json\s*.*?\s*```", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def create_initial_history() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Привет. Начни интервью."},
    ]


async def complete_chat(messages: list[dict[str, str]]) -> str:
    # Используем асинхронный контекстный менеджер и метод achat
    async with GigaChat(credentials=GIGACHAT_CREDENTIALS, verify_ssl_certs=VERIFY_SSL, model=MODEL_NAME) as giga:
        response = await giga.achat({"model": MODEL_NAME, "messages": messages})
        
    if not response or not response.choices:
        raise RuntimeError("Empty response from GigaChat")
    return str(response.choices[0].message.content or "")


async def analyze_voice_via_service(audio_bytes: bytes, filename: str) -> dict[str, Any]:
    files = {"file": (filename, audio_bytes, "application/octet-stream")}
    async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=5.0)) as client:
        response = await client.post(VOICE_SERVICE_URL, files=files)
        response.raise_for_status()
        return response.json()


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.post("/api/session/start", response_model=StartSessionResponse)
async def start_session() -> StartSessionResponse:
    history = create_initial_history()
    try:
        ai_text = await complete_chat(history)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"GigaChat error: {error}") from error

    history.append({"role": "assistant", "content": ai_text})
    session_id = uuid.uuid4().hex
    sessions[session_id] = history
    return StartSessionResponse(session_id=session_id, message=clean_ai_text(ai_text))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    history = sessions.get(request.session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Session not found. Start a new survey.")

    history.append({"role": "user", "content": request.message})
    try:
        ai_text = await complete_chat(history)
    except Exception as error:
        history.pop()
        raise HTTPException(status_code=502, detail=f"GigaChat error: {error}") from error

    history.append({"role": "assistant", "content": ai_text})
    finished = "[STATUS: FINISHED]" in ai_text

    profile = None
    if finished:
        profile = extract_final_payload(ai_text)
        if profile:
            await send_profile_to_portfolio_service(profile, request.session_id)

    return ChatResponse(message=clean_ai_text(ai_text), finished=finished, profile=profile)


@app.post("/api/voice", response_model=ChatResponse)
async def voice(session_id: str = Form(...), file: UploadFile = File(...)) -> ChatResponse:
    history = sessions.get(session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Session not found. Start a new survey.")

    try:
        audio_bytes = await file.read()
        voice_json = await analyze_voice_via_service(audio_bytes, file.filename or "voice.wav")
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Voice service error: {error}") from error

    transcript = str(voice_json.get("transcript", "")).strip()
    history.append({"role": "system", "content": f"Voice analysis, internal only: {json.dumps(voice_json, ensure_ascii=False)}"})
    history.append({"role": "user", "content": transcript or "(голосовое сообщение без транскрипта)"})

    try:
        ai_text = await complete_chat(history)
    except Exception as error:
        history.pop()
        history.pop()
        raise HTTPException(status_code=502, detail=f"GigaChat error: {error}") from error

    history.append({"role": "assistant", "content": ai_text})
    finished = "[STATUS: FINISHED]" in ai_text

    profile = None
    if finished:
        profile = extract_final_payload(ai_text)
        if profile:
            await send_profile_to_portfolio_service(profile, session_id)

    return ChatResponse(
        message=clean_ai_text(ai_text),
        finished=finished,
        profile=profile,
        voice={
            "transcript": voice_json.get("transcript"),
            "emotion": voice_json.get("emotion"),
            "emotion_index": voice_json.get("emotion_index"),
            "summary": voice_json.get("summary"),
        },
    )


@app.post("/api/session/reset")
async def reset_session(request: SessionResetRequest) -> dict[str, str]:
    sessions.pop(request.session_id, None)
    return {"status": "ok"}