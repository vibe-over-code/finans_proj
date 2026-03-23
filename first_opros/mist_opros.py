import os
import uuid
import requests
import urllib3
from flask import Flask, request, jsonify, render_template_string

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- НАСТРОЙКИ ---
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
VERIFY_SSL = False
AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGA_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
FILE_URL = "https://gigachat.devices.sberbank.ru/api/v1/files"

app = Flask(__name__)
chat_history = []

# Твой промпт без изменений
SYSTEM_PROMPT = """
Ты - инвестиционный профилировщик и финансовый психолог. Твоя задача - через естественный диалог оценить эмоциональность клиента, его отношение к риску, желания, ограничения и реальные возможности, а затем на этой основе сформировать целевой портфель. Если у клиента уже есть вложения, ты должен также оценить текущий портфель и его соответствие целям и характеру клиента.

Ты ведешь спокойный, уважительный, живой разговор. Твоя цель - понять человека, а не провести формальный тест.

### Главные правила
1. Задавай только один вопрос за раз.
2. Первый вопрос всегда должен быть про цель инвестирования.
3. Никогда не предлагай варианты ответов, шкалы, тестовые меню, ответы в формате А/Б/В или перечисления "что вам ближе".
4. Все вопросы должны быть открытыми, чтобы клиент отвечал своими словами.
5. Не задавай личные вопросы: не спрашивай ФИО, возраст, адрес, место работы, семейное положение, точный доход, точный размер капитала и другие персональные данные.
6. Можно мягко выяснять возможности клиента без вторжения в личное: инвестиционный горизонт, нужна ли ликвидность, есть ли запас прочности, возможны ли регулярные пополнения, насколько допустимы временные просадки, насколько критично быстро вернуть деньги.
7. Если ответ поверхностный, задай один уточняющий открытый вопрос без подсказок и вариантов.
8. Не дави, не оценивай клиента и не используй канцелярский тон.

### Особый приоритет: эмоции и голос
1. Главный скрытый параметр профилирования - эмоциональность клиента.
2. Если клиент пишет после голосового сообщения или приходит расшифровка устной речи, уделяй максимум внимания эмоциональным маркерам.
3. Оценивай тревожность, импульсивность, страх потерь, эйфорию, спешку, неуверенность, внутренние противоречия, FOMO, склонность к панике и внушаемость.
4. Если доступны только слова из расшифровки без аудио, все равно оценивай эмоциональность по структуре речи: паузы, повторы, обрывки, самопоправки, резкие формулировки, давление срочности, страх или перевозбуждение.
5. Если признаков голоса мало, делай осторожную оценку и учитывай пониженную уверенность, но не игнорируй эмоциональный фактор.

### Что нужно понять
1. Зачем клиенту инвестиции и какого результата он хочет.
2. На каком горизонте этот результат нужен.
3. Что для него важнее: сохранность, стабильность, рост, высокий потенциал доходности, денежный поток или гибкость доступа к деньгам.
4. Как он переживает просадки, неопределенность и ожидание.
5. Есть ли инвестиционный опыт и как клиент реагировал на убытки, волатильность и резкие движения рынка.
6. Может ли он дисциплинированно держать стратегию длительное время.
7. Есть ли текущий портфель и насколько он подходит под цели, риск и эмоциональную устойчивость клиента.

### Как оценивать
Сформируй внутреннюю оценку:
- emotion_index от 0.0 до 1.0: эмоциональность, тревожность, импульсивность, чувствительность к просадкам.
- risk_index от 0.0 до 1.0: готовность к риску и волатильности ради доходности.
- capacity_index от 0.0 до 1.0: способность выдерживать долгий горизонт, временные убытки и следовать стратегии.

Разделяй:
- желание риска: клиент хочет высокую доходность;
- переносимость риска: клиент выдержит просадку эмоционально;
- возможность риска: клиент реально может позволить себе долгий горизонт и не забирать деньги в неподходящий момент.

### Если есть текущий портфель
Если клиент сам упомянул активы или стало понятно, что портфель уже есть, мягко попроси описать его простыми словами. После этого оцени:
- соответствует ли он цели;
- нет ли перекоса по риску;
- нет ли конфликта между ожиданиями клиента и его эмоциональной устойчивостью;
- что стоит сохранить, сократить, упростить или изменить.

### Формат общения
Пока информации недостаточно:
- задай один следующий открытый вопрос;
- в конце сообщения добавляй [STATUS: COLLECTING].

Когда информации достаточно:
- кратко подведи итог;
- объясни тип инвестора и причины;
- дай оценку текущему портфелю, если он был описан;
- предложи целевой портфель;
- добавь [STATUS: FINISHED];
- затем выведи итоговый JSON в блоке кода.

### Структура итогового JSON
{
  "goal": "цель клиента своими словами",
  "investor_type": "Консервативный | Умеренный | Агрессивный",
  "emotion_index": 0.0,
  "risk_index": 0.0,
  "capacity_index": 0.0,
  "voice_emotion_assessment": "краткая оценка эмоциональности по речи/голосу или пометка, что доступна только текстовая расшифровка",
  "wants": ["ключевые желания клиента"],
  "constraints": ["ключевые ограничения и условия"],
  "current_portfolio": "если есть - краткое описание или JSON",
  "current_portfolio_assessment": "если есть - краткая оценка",
  "target_portfolio": {
    "Акции": "0-100%",
    "Облигации": "0-100%",
    "Денежные инструменты": "0-100%",
    "Альтернативные/прочие": "0-100%"
  },
  "profile_summary": "краткий психологический и инвестиционный портрет клиента"
}
"""

def get_token():
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {CLIENT_SECRET}",
        "RqUID": str(uuid.uuid4()),
    }
    res = requests.post(AUTH_URL, headers=headers, data={"scope": "GIGACHAT_API_PERS"}, verify=VERIFY_SSL)
    return res.json().get("access_token")

def upload_file(file_bytes, token):
    """Загрузка файла. Мы принудительно называем его .wav, так как Сбер это любит."""
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": ("voice.wav", file_bytes, "audio/wav")}
    res = requests.post(FILE_URL, headers=headers, files=files, data={"purpose": "general"}, verify=VERIFY_SSL)
    
    if res.status_code != 200:
        print(f"Ошибка загрузки файла: {res.text}")
        return None
    return res.json().get("id")

def ask_giga(text_content, file_id=None):
    global chat_history
    token = get_token()

    if not chat_history:
        chat_history.append({"role": "system", "content": SYSTEM_PROMPT})

    # Формируем сообщение согласно актуальной документации GigaChat-2-Pro
    message = {
        "role": "user",
        "content": text_content
    }
    if file_id:
        # В GigaChat файлы передаются в списке attachments
        message["attachments"] = [file_id]

    chat_history.append(message)

    payload = {
        "model": "GigaChat-2-Pro",
        "messages": chat_history,
        "temperature": 0.6,
        "max_tokens": 1024,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    try:
        res = requests.post(GIGA_URL, headers=headers, json=payload, verify=VERIFY_SSL)
        if res.status_code != 200:
            print(f"Детали ошибки API: {res.text}")
            return f"Ошибка API {res.status_code}"
            
        content = res.json()["choices"][0]["message"]["content"]
        chat_history.append({"role": "assistant", "content": content})
        return content
    except Exception as e:
        return f"Критическая ошибка: {str(e)}"
    


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/ask_text", methods=["POST"])
def ask_text():
    user_msg = request.json.get("message")
    return jsonify({"reply": ask_giga(user_msg)})

@app.route("/upload_voice", methods=["POST"])
def upload_voice():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    token = get_token()
    # Читаем данные. Браузер пришлет webm/ogg, но мы скажем Сберу, что это wav.
    # Большинство современных API умеют определять кодек сами, если расширение им нравится.
    file_id = upload_file(request.files['file'].read(), token)
    
    if not file_id:
        return jsonify({"reply": "Извините, не удалось обработать голосовое сообщение. Попробуйте еще раз или напишите текстом."})

    # Важно: текст должен быть, иначе API может вернуть ошибку
    reply = ask_giga("Проанализируй мои эмоции в этом голосовом сообщении и ответь на вопросы профайлинга.", file_id)
    return jsonify({"reply": reply})

HTML_PAGE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Инвест-Профайлер</title>
    <style>
        body { font-family: 'Inter', sans-serif; background: #f4f7f6; display: flex; justify-content: center; padding: 20px; }
        #chat-container { width: 100%; max-width: 600px; background: white; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); display: flex; flex-direction: column; height: 80vh; }
        #chat-box { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 10px; }
        .msg { padding: 12px 16px; border-radius: 15px; max-width: 85%; font-size: 15px; }
        .user { background: #007bff; color: white; align-self: flex-end; }
        .bot { background: #f1f0f0; color: #333; align-self: flex-start; }
        .controls { padding: 20px; border-top: 1px solid #eee; display: flex; gap: 10px; }
        input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 25px; outline: none; }
        #rec-btn { background: #28a745; color: white; border: none; border-radius: 25px; padding: 0 20px; cursor: pointer; }
        #rec-btn.recording { background: #dc3545; animation: pulse 1s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; } }
        pre { background: #272822; color: #f8f8f2; padding: 10px; border-radius: 8px; font-size: 12px; overflow-x: auto; }
    </style>
</head>
<body>
    <div id="chat-container">
        <div id="chat-box"></div>
        <div class="controls">
            <button id="rec-btn">🎤 Голос</button>
            <input type="text" id="text-input" placeholder="Ваш ответ...">
            <button onclick="sendText()" style="background:#007bff; color:white; border:none; border-radius:25px; padding:0 20px; cursor:pointer;">➤</button>
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
            } else {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    // Используем стандартный контейнер. Если Сбер продолжит ругаться, 
                    // придется добавить библиотеку для конвертации в wav на лету.
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                    mediaRecorder.onstop = () => {
                        const blob = new Blob(audioChunks, { type: 'audio/wav' });
                        addMessage("🎤 Голосовое сообщение (анализ...)", 'user');
                        uploadVoice(blob);
                    };
                    mediaRecorder.start();
                    recBtn.classList.add('recording');
                    recBtn.innerText = "🛑 Стоп";
                } catch (e) { alert("Микрофон не доступен"); }
            }
        };

        async function uploadVoice(blob) {
            const fd = new FormData();
            fd.append('file', blob);
            const res = await fetch('/upload_voice', { method: 'POST', body: fd });
            const data = await res.json();
            addMessage(data.reply, 'bot');
        }

        async function sendText() {
            const input = document.getElementById('text-input');
            const val = input.value.trim();
            if (!val) return;
            addMessage(val, 'user');
            input.value = '';
            const res = await fetch('/ask_text', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: val})
            });
            const data = await res.json();
            addMessage(data.reply, 'bot');
        }

        function addMessage(text, side) {
            const div = document.createElement('div');
            div.className = `msg ${side}`;
            div.innerHTML = text.replace(/```json([\\s\\S]*?)```/g, '<pre>$1</pre>');
            chatBox.appendChild(div);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        window.onload = () => addMessage("Здравствуйте! Я ваш финансовый ассистент. Расскажите, какая цель ваших инвестиций?", 'bot');
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)