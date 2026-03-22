import asyncio
import json
import os
from typing import Dict, List
from mistralai import Mistral

class RebalancerAgent:
    def __init__(self):
        # Инициализация клиента Mistral
        # Убедись, что токен добавлен в переменные окружения: export MISTRAL_API_KEY="твой_ключ"
        self.api_key = os.environ.get("MKey")
        self.client = Mistral(api_key=self.api_key)
        self.model = "mistral-large-latest" # Можно использовать open-mixtral-8x22b для экономии
        print("[SYSTEM] Ребалансер инициализирован с Mistral SDK.")

    def _calculate_deviations(self, current: Dict[str, float], target: Dict[str, float]) -> Dict[str, float]:
        """Считает математическое отклонение текущего портфеля от целевого"""
        deviations = {}
        all_keys = set(current.keys()).union(set(target.keys()))
        for key in all_keys:
            c_val = float(current.get(key, 0))
            t_val = float(target.get(key, 0))
            deviations[key] = round(t_val - c_val, 2)
        return deviations

    async def generate_advice(
        self, 
        user_id: str, 
        test_data: dict, 
        news_risk: dict, 
        voice_state: dict,
        app_opens_per_day: int,
        recent_actions: List[str]
    ) -> str:
        """
        Собирает финансовые, новостные и поведенческие данные,
        отправляя итоговый контекст в Mistral для генерации ответа.
        """
        print(f"\n[API REB] Сбор данных для пользователя: {user_id}")
        
        current = test_data.get("current_portfolio", {})
        target = test_data.get("target_portfolio", {})
        
        if not current:
            return "У вас пока нет активов в портфеле. Давайте начнем с распределения средств согласно целевому плану."

        # 1. Математика ребалансировки
        deviations = self._calculate_deviations(current, target)
        print(f"[REB MATH] Дельта портфеля: {deviations}")

        # 2. Формирование промпта для Mistral
        prompt = f"""
        Ты — эмпатичный ИИ-советник по инвестициям. Проанализируй данные и напиши короткое сообщение для клиента в чат.

        [ФИНАНСОВЫЕ ДАННЫЕ]
        - Цель клиента: {test_data.get('goal')}
        - Профиль: {test_data.get('investor_type')}
        - Текущие доли: {json.dumps(current, ensure_ascii=False)}
        - Целевые доли: {json.dumps(target, ensure_ascii=False)}
        - Необходимые изменения (%): {json.dumps(deviations, ensure_ascii=False)} (Положительное = докупить, Отрицательное = продать)

        [РЫНОЧНЫЙ ФОН (Парсер)]
        - Наличие риска: {news_risk.get('has_portfolio_risk')}
        - Описание риска: {news_risk.get('summary', 'Всё спокойно')}
        - Опасные активы: {json.dumps(news_risk.get('dangerous_categories', []), ensure_ascii=False)}

        [СОСТОЯНИЕ И ПОВЕДЕНИЕ КЛИЕНТА]
        - Анализ голоса: {voice_state.get('analysis', 'Голосовых сообщений нет')}
        - Заходов в приложение за сегодня: {app_opens_per_day}
        - Последние действия: {json.dumps(recent_actions, ensure_ascii=False)}

        [ПРАВИЛА ОТВЕТА]
        1. Оцени уровень стресса клиента. Если заходов в приложение много (>5) и действия нервные (например, частое чтение новостей, просмотр графиков падения) — начни с сильной психологической поддержки. Напомни про его долгосрочную цель.
        2. Если актив нужно докупить, но по нему есть риск из парсера новостей — посоветуй отложить покупку.
        3. Если актив сильно вырос, предложи зафиксировать прибыль.
        4. Не используй JSON, markdown-таблицы или сложные термины. Форматируй текст легко для чтения.
        """

        # 3. Вызов Mistral API (или мок для локального тестирования без ключа)
        if self.api_key == "mock_key_for_test":
            print("[MISTRAL API MOCK] Имитация ответа от нейросети...")
            await asyncio.sleep(2)
            return self._mock_mistral_response(app_opens_per_day, recent_actions, news_risk)
        
        try:
            # Асинхронный вызов реального API Mistral
            chat_response = await self.client.chat.complete_async(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return chat_response.choices[0].message.content
        except Exception as e:
            return f"Произошла ошибка при обращении к финансовому советнику: {e}"

    def _mock_mistral_response(self, app_opens, actions, news) -> str:
        """Временная заглушка, пока не подключен реальный токен API"""
        if app_opens > 10:
            return (
                "Я вижу, что вы сегодня заходили в приложение уже много раз, активно читая новости о ставке и проверяя графики акций. "
                "Дышите глубже — турбулентность на рынке сейчас действительно высокая, и ваш тревожный голос это подтверждает. "
                "Ваша главная цель — пенсия, и такие колебания на долгом горизонте нормальны. \n\n"
                "Давайте посмотрим на план. Сейчас доля акций превышает целевую. "
                "Учитывая новости о высоких рисках, я бы не советовал сейчас ничего докупать. "
                "Лучшим решением будет продать часть акций (около 20%), чтобы зафиксировать прибыль, и перевести эти средства в консервативные облигации. "
                "Это снизит риск и добавит стабильности вашему портфелю."
            )
        return "Всё идет по плану, портфель сбалансирован."

# --- СИМУЛЯЦИЯ РАБОТЫ API ---
async def simulate_api_requests():
    print("=== ЗАПУСК СИМУЛЯЦИИ API РЕБАЛАНСЕРА (MISTRAL) ===\n")
    rebalancer = RebalancerAgent()

    db_test_data = {
        "goal": "Накопить на пенсию с минимальным стрессом",
        "investor_type": "Консерватор",
        "target_portfolio": {"Облигации": 50, "Акции": 40, "Золото": 10},
        "current_portfolio": {"Облигации": 30, "Акции": 60, "Золото": 10}
    }

    parser_data = {
        "has_portfolio_risk": True,
        "summary": "Резкое повышение ключевой ставки, ожидается коррекция на рынке акций",
        "dangerous_categories": ["Акции"],
        "category_details": [{"category": "Акции", "risk_level": "high", "reason": "Ужесточение ДКП"}]
    }

    voice_data = {
        "analysis": "The speaker's tone is extremely anxious, breathing is shallow, indicating high stress and uncertainty about the market."
    }

    # НОВЫЕ ДАННЫЕ: Имитация действий клиента
    app_opens_today = 1
    user_actions_log = [
        "открыл_вкладку_портфель", 
        "просмотр_графика_Акции_за_месяц" 
    ]

    # Вызов ребалансера
    chat_response = await rebalancer.generate_advice(
        user_id="user_101",
        test_data=db_test_data,
        news_risk=parser_data,
        voice_state=voice_data,
        app_opens_per_day=app_opens_today,
        recent_actions=user_actions_log
    )

    print("\n[OUTPUT] ФИНАЛЬНОЕ СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЮ В ЧАТ:\n")
    print("-" * 60)
    print(chat_response)
    print("-" * 60)

if __name__ == "__main__":
    asyncio.run(simulate_api_requests())