import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Smart Advisor API")

MISTRAL_API_KEY = os.getenv("MKey")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
client = Mistral(api_key=MISTRAL_API_KEY)
CLIENT_FILE = "client_data.json"

# --- СХЕМЫ ДАННЫХ ДЛЯ ПАРСЕРА ---
class NewsItem(BaseModel):
    title: str
    link: str
    summary: str
    portfolio_risk_analysis: Dict[str, Any]

class PortfolioCheckPayload(BaseModel):
    timestamp: str
    risk_categories: List[str]
    news_items: List[NewsItem]

# --- ФУНКЦИЯ ЗАГРУЗКИ ПРОФИЛЯ ---
def load_client_data():
    if not os.path.exists(CLIENT_FILE):
        return None
    with open(CLIENT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# --- ПРОМПТ ДЛЯ MISTRAL ---
ADVISOR_PROMPT = """Ты — персональный финансовый советник. 
Проанализируй срочную новость, оцени разрыв между текущим и целевым портфелем клиента, и напиши сообщение для Telegram.

ДАННЫЕ КЛИЕНТА:
- Психология: {investor_type} (Устойчивость к панике: {emotion_index}/1.0, Аппетит к риску: {risk_index}/1.0)
- ТЕКУЩИЙ портфель: {current_portfolio}
- ЦЕЛЕВОЙ портфель: {target_portfolio}
- Справка: {profile_summary}

ВХОДЯЩАЯ НОВОСТЬ:
- Заголовок: {news_title}
- Суть риска: {news_summary}
- Затронутые активы: {danger_cats}

ИНСТРУКЦИЯ К ОТВЕТУ:
1. Пиши как сообщение в Telegram (абзацы, немного эмодзи).
2. Адаптируй тон под emotion_index (если он низкий — сначала успокой; если высокий — дай сухие факты).
3. Объясни, как эта новость влияет на необходимость перехода от ТЕКУЩЕГО портфеля к ЦЕЛЕВОМУ. 
4. Дай четкие действия (Action Plan): нужно ли сейчас докупать целевые активы, или лучше подождать из-за новости.
5. Не задавай вопросов в конце сообщения.
"""

@app.post("/api/check_portfolio")
async def process_news_and_advise(payload: PortfolioCheckPayload):
    client_data = load_client_data()
    if not client_data:
        return {"status": "error", "message": "Файл client_data.json не найден"}

    recommendations = []

    for news in payload.news_items:
        analysis = news.portfolio_risk_analysis
        
        # Если парсер пометил новость как опасную
        if analysis.get("has_portfolio_risk"):
            
            formatted_prompt = ADVISOR_PROMPT.format(
                investor_type=client_data["investor_type"],
                emotion_index=client_data["emotion_index"],
                risk_index=client_data["risk_index"],
                current_portfolio=json.dumps(client_data["current_portfolio"], ensure_ascii=False),
                target_portfolio=json.dumps(client_data["target_portfolio"], ensure_ascii=False),
                profile_summary=client_data["profile_summary"],
                news_title=news.title,
                news_summary=analysis.get("summary"),
                danger_cats=", ".join(analysis.get("dangerous_categories", []))
            )
            
            response = client.chat.complete(
                model=MISTRAL_MODEL,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0.3
            )
            
            advice_text = response.choices[0].message.content
            
            recommendations.append({
                "client_id": client_data["client_id"],
                "news_trigger": news.title,
                "telegram_message": advice_text
            })

            # Здесь можно добавить логику прямой отправки в Telegram:
            # await send_to_telegram(client_data["client_id"], advice_text)

    return {
        "status": "success", 
        "alerts_generated": len(recommendations),
        "data": recommendations
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)