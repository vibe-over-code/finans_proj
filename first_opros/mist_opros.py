import json
import logging
import os
import re
import uuid
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from mistralai import Mistral
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
load_dotenv()

MISTRAL_API_KEY = os.getenv("MKey")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
PORTFOLIO_PROFILE_URL = os.getenv(
    "PORTFOLIO_PROFILE_URL",
    "http://localhost:8000/api/client-profile",
)

client = Mistral(api_key=MISTRAL_API_KEY)
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


class SessionResetRequest(BaseModel):
    session_id: str


SYSTEM_PROMPT = """
Ты проводишь короткое финансовое интервью и должен получить данные для сервиса ребалансировки портфеля.

Главная цель интервью:
1. Сначала обязательно выясни главную инвестиционную цель пользователя.
2. На основе цели, горизонта, допустимой просадки и отношения к риску определи целевой портфель.
3. Отдельно выясни, есть ли у пользователя уже текущий портфель.
4. Если текущий портфель есть и пользователь описал его достаточно понятно, оформи current_portfolio как JSON-объект.
5. Если текущего портфеля нет или данных недостаточно, не добавляй поле current_portfolio в финальный JSON вообще.

Правила диалога:
1. Задавай только один вопрос за раз.
2. Все вопросы открытые, без вариантов ответа.
3. Первый вопрос должен быть именно про цель.
4. Если пользователь пишет слишком кратко, мягко проси уточнить.
5. Не упоминай JSON, API, схему данных или внутренние поля.
6. Веди диалог кратко, по делу, без длинных вступлений.
7. Когда данных достаточно, заверши диалог.

Что нужно собрать:
- goal: формулировка цели пользователя
- investor_type: название типа инвестора
- risk_index: число от 0.0 до 1.0
- emotion_index: число от 0.0 до 1.0
- target_portfolio: JSON-объект с распределением по классам активов в процентах
- current_portfolio: JSON-объект с текущим распределением по классам активов в процентах, только если данные есть
- profile_summary: краткий портрет подхода к принятию решений

Требования к портфелям:
1. target_portfolio обязателен.
2. target_portfolio и current_portfolio должны быть JSON-объектами, например:
   {
     "Облигации": "50%",
     "Акции": "40%",
     "Золото": "10%"
   }
3. Не используй массивы, только объект "класс актива" -> "доля%".
4. Если current_portfolio неизвестен, не пиши ключ current_portfolio вообще.

Маркеры:
- Во время интервью добавляй [STATUS: COLLECTING]
- В финальном сообщении добавляй [STATUS: FINISHED]

Финал:
Когда данных достаточно, верни короткое завершение и затем JSON в fenced block:

```json
{
  "goal": "формулировка цели",
  "investor_type": "тип инвестора",
  "risk_index": 0.0,
  "emotion_index": 0.0,
  "target_portfolio": {
    "Облигации": "50%",
    "Акции": "40%",
    "Золото": "10%"
  },
  "profile_summary": "краткий портрет"
}
```

Если current_portfolio известен, добавь его в этот JSON.
"""


INDEX_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Финансовый опросник</title>
  <style>
    :root {
      --bg: #f4efe6;
      --panel: rgba(255, 250, 242, 0.92);
      --text: #2f241d;
      --muted: #6f6258;
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
      width: min(920px, 100%);
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
      font-size: clamp(28px, 4vw, 42px);
      line-height: 1.05;
      font-weight: 700;
    }

    .subtitle {
      margin: 0;
      color: var(--muted);
      max-width: 54ch;
      font-size: 17px;
      line-height: 1.5;
    }

    .toolbar {
      display: flex;
      gap: 12px;
      padding: 18px 28px 0;
      flex-wrap: wrap;
    }

    button {
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      font: inherit;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease, box-shadow 120ms ease;
    }

    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.6; cursor: wait; transform: none; }

    .primary {
      background: var(--accent);
      color: white;
      box-shadow: 0 10px 24px rgba(15, 118, 110, 0.25);
    }

    .secondary {
      background: transparent;
      color: var(--accent-2);
      border: 1px solid rgba(194, 65, 12, 0.24);
    }

    .chat {
      padding: 18px 28px;
      height: 54vh;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .bubble {
      max-width: min(720px, 92%);
      padding: 14px 16px;
      border-radius: 18px;
      line-height: 1.5;
      white-space: pre-wrap;
      animation: rise 180ms ease;
      border: 1px solid var(--border);
    }

    .bot { align-self: flex-start; background: var(--bot); }
    .user { align-self: flex-end; background: var(--user); }
    .system { align-self: center; background: rgba(47, 36, 29, 0.06); color: var(--muted); }

    .composer {
      padding: 0 28px 28px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
    }

    textarea {
      width: 100%;
      min-height: 84px;
      resize: vertical;
      border-radius: 18px;
      border: 1px solid var(--border);
      padding: 16px 18px;
      font: inherit;
      color: var(--text);
      background: rgba(255, 255, 255, 0.72);
    }

    .status {
      padding: 0 28px 22px;
      color: var(--muted);
      font-size: 14px;
      min-height: 22px;
    }

    .report {
      margin: 0 28px 28px;
      padding: 18px;
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid var(--border);
      display: none;
      white-space: pre-wrap;
      line-height: 1.5;
    }

    .report.visible { display: block; }

    @keyframes rise {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }

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
      <p class="subtitle">Интервью проходит прямо в браузере и собирает профиль для сервиса ребалансировки без Telegram.</p>
    </section>

    <div class="toolbar">
      <button id="startBtn" class="primary">Начать опрос</button>
      <button id="resetBtn" class="secondary">Сбросить</button>
    </div>

    <section id="chat" class="chat">
      <div class="bubble system">Нажмите «Начать опрос», чтобы открыть диалог.</div>
    </section>

    <form id="chatForm" class="composer">
      <textarea id="messageInput" placeholder="Напишите ответ на вопрос..." disabled></textarea>
      <button id="sendBtn" type="submit" class="primary" disabled>Отправить</button>
    </form>

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

    let sessionId = null;
    let finished = false;

    function addBubble(text, kind) {
      const node = document.createElement("div");
      node.className = `bubble ${kind}`;
      node.textContent = text;
      chat.appendChild(node);
      chat.scrollTop = chat.scrollHeight;
    }

    function setBusy(value, text = "") {
      startBtn.disabled = value;
      resetBtn.disabled = value && !sessionId;
      sendBtn.disabled = value || !sessionId || finished;
      messageInput.disabled = value || !sessionId || finished;
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
      report.textContent = lines.join("\\n");
      report.classList.add("visible");
    }

    async function startSurvey() {
      setBusy(true, "Запускаю интервью...");
      report.classList.remove("visible");
      report.textContent = "";
      finished = false;

      try {
        const response = await fetch("/api/session/start", { method: "POST" });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Не удалось начать опрос");

        sessionId = data.session_id;
        chat.innerHTML = "";
        addBubble(data.message, "bot");
        setBusy(false, "Опрос активен");
        messageInput.focus();
      } catch (error) {
        setBusy(false, String(error));
      }
    }

    async function resetSurvey() {
      if (!sessionId) {
        chat.innerHTML = "";
        addBubble("Нажмите «Начать опрос», чтобы открыть диалог.", "system");
        renderReport(null);
        finished = false;
        setBusy(false, "");
        return;
      }

      setBusy(true, "Сбрасываю сессию...");
      try {
        await fetch("/api/session/reset", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId }),
        });
      } finally {
        sessionId = null;
        finished = false;
        chat.innerHTML = "";
        addBubble("Нажмите «Начать опрос», чтобы открыть диалог.", "system");
        renderReport(null);
        messageInput.value = "";
        setBusy(false, "");
      }
    }

    async function sendMessage(event) {
      event.preventDefault();
      const message = messageInput.value.trim();
      if (!message || !sessionId || finished) return;

      addBubble(message, "user");
      messageInput.value = "";
      setBusy(true, "Модель отвечает...");

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId, message }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Не удалось обработать сообщение");

        addBubble(data.message, "bot");
        finished = Boolean(data.finished);
        renderReport(data.profile || null);
        setBusy(false, finished ? "Опрос завершён" : "Опрос активен");
      } catch (error) {
        addBubble(`Ошибка: ${error}`, "system");
        setBusy(false, "Ошибка при обработке сообщения");
      }
    }

    startBtn.addEventListener("click", startSurvey);
    resetBtn.addEventListener("click", resetSurvey);
    chatForm.addEventListener("submit", sendMessage);
  </script>
</body>
</html>
"""


sessions: dict[str, list[dict[str, str]]] = {}


async def send_profile_to_portfolio_service(profile: dict, session_id: str):
    payload = {
        "client_id": f"web_{session_id}",
        "survey_result": profile,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as transport:
            response = await transport.post(PORTFOLIO_PROFILE_URL, json=payload)
            response.raise_for_status()
        logging.info("Survey profile sent to portfolio service")
    except Exception as error:
        logging.exception("Failed to send survey profile to portfolio service: %s", error)


def extract_final_payload(ai_text: str) -> dict | None:
    match = re.search(r"```json\s*(.*?)\s*```", ai_text, re.DOTALL)
    if not match:
        return None
    return json.loads(match.group(1).strip(), strict=False)


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


def complete_chat(messages: list[dict[str, str]]) -> str:
    response = client.chat.complete(model=MISTRAL_MODEL, messages=messages)
    return str(response.choices[0].message.content)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.post("/api/session/start", response_model=StartSessionResponse)
async def start_session() -> StartSessionResponse:
    history = create_initial_history()
    try:
        ai_text = complete_chat(history)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Mistral error: {error}") from error

    history.append({"role": "assistant", "content": ai_text})
    session_id = uuid.uuid4().hex
    sessions[session_id] = history
    return StartSessionResponse(session_id=session_id, message=clean_ai_text(ai_text))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    history = sessions.get(request.session_id)
    if history is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Start a new survey.",
        )

    history.append({"role": "user", "content": request.message})

    try:
        ai_text = complete_chat(history)
    except Exception as error:
        history.pop()
        raise HTTPException(status_code=502, detail=f"Mistral error: {error}") from error

    history.append({"role": "assistant", "content": ai_text})
    finished = "[STATUS: FINISHED]" in ai_text

    profile = None
    if finished:
        profile = extract_final_payload(ai_text)
        if profile:
            await send_profile_to_portfolio_service(profile, request.session_id)

    return ChatResponse(
        message=clean_ai_text(ai_text),
        finished=finished,
        profile=profile,
    )


@app.post("/api/session/reset")
async def reset_session(request: SessionResetRequest) -> dict[str, str]:
    sessions.pop(request.session_id, None)
    return {"status": "ok"}
