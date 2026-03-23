import os
import uuid
import requests
import urllib3
import tempfile
import subprocess
from flask import Flask, request, jsonify, render_template_string

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CLIENT_SECRET = os.getenv("CLIENT_SECRET")
VERIFY_SSL = False

AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGA_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
FILE_URL = "https://gigachat.devices.sberbank.ru/api/v1/files"

app = Flask(__name__)
chat_history = []

# ✅ ТВОЙ ПОЛНЫЙ ПРОМПТ (оставь как есть)
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

Если клиент упоминает инвестиции:
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

# ---------- TOKEN ----------
def get_token():
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {CLIENT_SECRET}",
        "RqUID": str(uuid.uuid4()),
    }
    res = requests.post(AUTH_URL, headers=headers, data={"scope": "GIGACHAT_API_PERS"}, verify=VERIFY_SSL)
    return res.json().get("access_token")

# ---------- CONVERT ----------
def convert_to_wav(input_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f_in:
        f_in.write(input_bytes)
        input_path = f_in.name

    output_path = input_path + ".wav"

    subprocess.run([
        "ffmpeg",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with open(output_path, "rb") as f:
        return f.read()

# ---------- UPLOAD ----------
def upload_file(file_bytes, token):
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": ("voice.wav", file_bytes, "audio/wav")}

    res = requests.post(FILE_URL, headers=headers, files=files, data={"purpose": "general"}, verify=VERIFY_SSL)

    if res.status_code != 200:
        print("UPLOAD ERROR:", res.text)
        return None

    return res.json().get("id")

# ---------- CHAT ----------
def ask_giga(text, file_id=None):
    global chat_history
    token = get_token()

    if not chat_history:
        chat_history.append({"role": "system", "content": SYSTEM_PROMPT})

    msg = {"role": "user", "content": text}
    if file_id:
        msg["attachments"] = [file_id]

    chat_history.append(msg)

    payload = {
        "model": "GigaChat-2-Pro",
        "messages": chat_history,
        "temperature": 0.6,
        "max_tokens": 1024,
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    res = requests.post(GIGA_URL, headers=headers, json=payload, verify=VERIFY_SSL)

    if res.status_code != 200:
        print("CHAT ERROR:", res.text)
        return f"Ошибка API {res.status_code}"

    reply = res.json()["choices"][0]["message"]["content"]
    chat_history.append({"role": "assistant", "content": reply})

    return reply

# ---------- ROUTES ----------
@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/ask_text", methods=["POST"])
def ask_text():
    msg = request.json.get("message")
    return jsonify({"reply": ask_giga(msg)})

@app.route("/upload_voice", methods=["POST"])
def upload_voice():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    token = get_token()
    raw = request.files["file"].read()

    # 🔥 ФИКС: реальная конвертация
    wav = convert_to_wav(raw)

    file_id = upload_file(wav, token)

    if not file_id:
        return jsonify({"reply": "Не удалось обработать голосовое сообщение"})

    reply = ask_giga(
        "Проанализируй мои эмоции в этом голосовом сообщении и ответь на вопросы профайлинга.",
        file_id
    )

    return jsonify({"reply": reply})

# ---------- HTML (ТВОЙ ИНТЕРФЕЙС СОХРАНЕН) ----------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Инвест-Профайлер</title>

<style>
body { font-family: Inter; background:#f4f7f6; display:flex; justify-content:center; padding:20px; }
#chat-container { width:100%; max-width:600px; background:white; border-radius:15px; box-shadow:0 10px 25px rgba(0,0,0,0.1); display:flex; flex-direction:column; height:80vh; }
#chat-box { flex:1; overflow-y:auto; padding:20px; display:flex; flex-direction:column; gap:10px; }
.msg { padding:12px 16px; border-radius:15px; max-width:85%; }
.user { background:#007bff; color:white; align-self:flex-end; }
.bot { background:#f1f0f0; }
.controls { padding:20px; display:flex; gap:10px; }
input { flex:1; padding:12px; border-radius:25px; }
#rec-btn.recording { background:red; }
</style>
</head>

<body>
<div id="chat-container">
<div id="chat-box"></div>

<div class="controls">
<button id="rec-btn">🎤 Голос</button>
<input id="text-input">
<button onclick="sendText()">➤</button>
</div>
</div>

<script>
let mediaRecorder;
let audioChunks = [];

const chatBox = document.getElementById('chat-box');
const recBtn = document.getElementById('rec-btn');

recBtn.onclick = async () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        recBtn.classList.remove('recording');
        recBtn.innerText = "🎤 Голос";
        return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm;codecs=opus"
    });

    audioChunks = [];

    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);

    mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunks, { type: "audio/webm" });

        addMessage("🎤 Голосовое сообщение (анализ...)", "user");
        uploadVoice(blob);
    };

    mediaRecorder.start();
    recBtn.classList.add('recording');
    recBtn.innerText = "🛑 Стоп";
};

async function uploadVoice(blob) {
    const fd = new FormData();
    fd.append("file", blob);

    const res = await fetch("/upload_voice", { method:"POST", body:fd });
    const data = await res.json();

    addMessage(data.reply, "bot");
}

async function sendText() {
    const input = document.getElementById("text-input");
    const val = input.value.trim();
    if (!val) return;

    addMessage(val, "user");
    input.value = "";

    const res = await fetch("/ask_text", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({message: val})
    });

    const data = await res.json();
    addMessage(data.reply, "bot");
}

function addMessage(text, side) {
    const div = document.createElement("div");
    div.className = "msg " + side;
    div.innerHTML = text.replace(/```json([\\s\\S]*?)```/g, '<pre>$1</pre>');
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

window.onload = () => {
    addMessage("Здравствуйте! Какая цель ваших инвестиций?", "bot");
};
</script>

</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)