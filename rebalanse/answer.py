import asyncio
import json
import os
from typing import Dict, List, Optional
from mistralai import Mistral
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

class RebalancerAgent:
    def __init__(self, api_key: Optional[str] = None):
        # Ищем ключ MKEY (как в твоем файле) или MISTRAL_API_KEY
        raw_key = api_key or os.getenv("MKEY") or os.getenv("MISTRAL_API_KEY")
        
        if not raw_key:
            raise ValueError("API ключ не найден. Проверь .env файл или переменные окружения.")
            
        self.api_key = raw_key.strip()
        
        # Проверка на ASCII, чтобы избежать ошибок кодировки в будущем
        try:
            self.api_key.encode('ascii')
        except UnicodeEncodeError:
            raise ValueError("В API-ключе найдены недопустимые символы (проверь, нет ли там кириллицы).")

        self.client = Mistral(api_key=self.api_key)
        self.model = "mistral-large-latest"

    def _calculate_deviations(self, current: Dict[str, float], target: Dict[str, float]) -> Dict[str, float]:
        """
        Математический расчет отклонений.
        $$Delta = Target - Current$$
        """
        deviations = {}
        all_keys = set(current.keys()).union(set(target.keys()))
        for key in all_keys:
            c_val = float(str(current.get(key, 0)).replace('%', ''))
            t_val = float(str(target.get(key, 0)).replace('%', ''))
            deviations[key] = round(t_val - c_val, 2)
        return deviations

    async def generate_advice(
        self, 
        test_data: dict, 
        news_risk: dict, 
        voice_state: dict,
        behavioral_data: dict
    ) -> str:
        current = test_data.get("current_portfolio", {})
        target = test_data.get("target_portfolio", {})
        deviations = self._calculate_deviations(current, target)

        prompt = (
            "Ты — профессиональный финансовый советник с высоким уровнем эмпатии. "
            "Твоя задача: проанализировать данные и составить персонализированный план действий.\n\n"
            f"1. ЦЕЛЬ И ПОРТФЕЛЬ:\n"
            f"- Главная цель: {test_data.get('goal')}\n"
            f"- Текущее состояние: {json.dumps(current, ensure_ascii=False)}\n"
            f"- Математическая разница (Delta): {json.dumps(deviations, ensure_ascii=False)}\n\n"
            f"2. РЫНОЧНЫЙ КОНТЕКСТ:\n"
            f"- Риски: {news_risk.get('summary', 'Стабильно')}\n\n"
            f"3. ПОВЕДЕНИЕ И ЭМОЦИИ:\n"
            f"- Анализ голоса: {voice_state.get('analysis', 'Нейтрально')}\n"
            f"- Активность: {behavioral_data.get('app_opens')} заходов сегодня\n\n"
            "ИНСТРУКЦИЯ:\n"
            "- Если пользователь в стрессе, начни с поддержки.\n"
            "- Если актив в риске, не советуй его покупать.\n"
            "- Пиши кратко, пунктами."
        )

        try:
            response = await self.client.chat.complete_async(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ Ошибка API запроса: {str(e)}"

async def run_rebalancer():
    # Моки данных для теста
    test_context = {
        "goal": "Сохранение капитала",
        "target_portfolio": {"Облигации": 60, "Акции": 30},
        "current_portfolio": {"Облигации": 45, "Акции": 45}
    }
    news_context = {"summary": "Волатильность рынка", "dangerous_categories": ["Акции"]}
    voice_context = {"analysis": "The tone is urgent."}
    behavior_context = {"app_opens": 12, "recent_actions": ["check_balance"]}

    try:
        # Теперь не передаем строку в конструктор, он сам возьмет ключ из .env
        rebalancer = RebalancerAgent()
        
        advice = await rebalancer.generate_advice(
            test_data=test_context,
            news_risk=news_context,
            voice_state=voice_context,
            behavioral_data=behavior_context
        )
        print("\n--- СОВЕТ ДЛЯ КЛИЕНТА ---")
        print(advice)
        
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    asyncio.run(run_rebalancer())