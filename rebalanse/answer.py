import os
import asyncio
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Импортируем базовый класс GigaChat (теперь он умеет в асинхронность)
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from dotenv import load_dotenv

load_dotenv()

# --- КОНФИГУРАЦИЯ ---
CLIENT_SECRET = os.getenv("MKEY") or os.getenv("CLIENT_SECRET")
VERIFY_SSL = False # Как ты и просил

app = FastAPI(title="Rebalancer Orchestrator API")

# --- МОДЕЛИ ДАННЫХ (ВХОД) ---

class PortfolioData(BaseModel):
    goal: str
    investor_type: str
    risk_index: float
    target_portfolio: Dict[str, str]
    current_portfolio: Optional[Dict[str, str]] = None
    profile_summary: str

class NewsRisk(BaseModel):
    has_portfolio_risk: bool
    summary: str
    dangerous_categories: List[str]
    category_details: Optional[List[Dict[str, Any]]] = None

class VoiceAnalysis(BaseModel):
    analysis: str

class OrchestratorRequest(BaseModel):
    test_result: PortfolioData
    news_result: NewsRisk
    voice_result: VoiceAnalysis

# --- ЯДРО ОРКЕСТРАТОРА ---

class RebalancerBrain:
    def __init__(self):
        if not CLIENT_SECRET:
            raise ValueError("CLIENT_SECRET не найден в .env")
        
        # 2. Инициализируем GigaChat без приставки Async
        self.giga = GigaChat(
            credentials=CLIENT_SECRET,
            verify_ssl_certs=VERIFY_SSL,
            model="GigaChat-2-Pro" # Убедись, что у твоего токена есть доступ к "GigaChat-2-Pro", иначе замени на "GigaChat-Pro"
        )

    async def get_final_advice(self, data: OrchestratorRequest) -> str:
        # Статичные заглушки, которые просил оставить
        behavioral_mock = {
            "app_opens_today": 12,
            "recent_actions": ["открытие_графика_насдак", "повторный_запрос_баланса"]
        }

        # Формируем "жирный" контекст для GigaChat
        # Мы скармливаем ему результаты работы всех твоих промптов
        system_prompt = (
            "Ты — эмпатичный финансовый эксперт-ребалансировщик. Твоя задача — "
            "свести воедино данные портфеля, рыночные риски и эмоциональное состояние клиента."
        )

        user_content = f"""
        ПРИНЯТЫЕ ДАННЫЕ ОТ СЕРВИСОВ:
        
        1. ПОРТФЕЛЬ И ЦЕЛИ (JSON): 
        {data.test_result.model_dump_json(ensure_ascii=False)}
        
        2. РЫНОЧНЫЕ РИСКИ (ПАРСЕР):
        {data.news_result.model_dump_json(ensure_ascii=False)}
        
        3. СОСТОЯНИЕ ГОЛОСА (QWEN2-AUDIO):
        {data.voice_result.analysis}
        
        4. ПОВЕДЕНЧЕСКИЕ МЕТРИКИ (ЗАГЛУШКА):
        Заходов в приложение: {behavioral_mock['app_opens_today']}. 
        Действия: {", ".join(behavioral_mock['recent_actions'])}.

        ЗАДАНИЕ:
        - Если в новостях есть риск для категорий, которые ЕСТЬ в target_portfolio или current_portfolio — акцентируй на этом внимание.
        - Если голос тревожный или заходов много (>10) — начни с фразы, которая успокоит клиента.
        - Дай конкретный совет по ребалансировке (что купить/продать исходя из Delta между портфелями).
        - Ответ должен быть в стиле короткого сообщения для мессенджера. Только текст.
        """

        try:
            # Используем правильную структуру объектов для GigaChat SDK
            payload = Chat(
                messages=[
                    Messages(role=MessagesRole.SYSTEM, content=system_prompt),
                    Messages(role=MessagesRole.USER, content=user_content)
                ],
                model="GigaChat-2-Pro"
            )
            
            # 3. Вызываем асинхронный метод achat() вместо chat()
            response = await self.giga.achat(payload)
            return response.choices[0].message.content
        except Exception as e:
            return f"Критическая ошибка нейронки: {str(e)}"

# Инициализируем "Мозг" один раз при старте
brain = RebalancerBrain()

# --- API ENDPOINTS ---

@app.post("/rebalance")
async def rebalance_endpoint(request: OrchestratorRequest):
    """
    Основной входной узел. Получает JSON-ы от других сервисов и выдает текст.
    """
    advice = await brain.get_final_advice(request)
    return {"status": "success", "advice": advice}

# --- ЗАПУСК ---
if __name__ == "__main__":
    import uvicorn
    # Запускаем сервер на порту 8080
    uvicorn.run(app, host="0.0.0.0", port=8080)