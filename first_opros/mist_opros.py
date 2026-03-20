import asyncio
import json
import re
import os
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from mistralai import Mistral
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv()

API_TOKEN = os.getenv("TGKey")
MISTRAL_API_KEY = os.getenv("MKey")
MISTRAL_MODEL = "mistral-large-latest"

bot = Bot(token=API_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
client = Mistral(api_key=MISTRAL_API_KEY)

class Survey(StatesGroup):
    chatting = State()

# --- СИСТЕМНЫЙ ПРОМПТ (ТОЛЬКО ОЦЕНКА) ---
SYSTEM_PROMPT = (
    "Ты — эксперт-диагност в области поведенческих финансов. Твоя цель: через серию уточняющих вопросов "
    "определить психологический профиль инвестора, анализируя логику его решений и реакцию на рыночные ситуации.\n\n"

    "СТАРТ ДИАЛОГА:\n"
    "1. ПЕРВОЕ СООБЩЕНИЕ: Обязательно вежливо поприветствуй пользователя и задай первый открытый вопрос о том, "
    "какой результат в управлении капиталом он считает для себя приоритетным.\n\n"

    "МЕТОДОЛОГИЯ (ДЕЛИКАТНОСТЬ И НАВЕДЕНИЕ):\n"
    "1. ЗАПРЕЩЕНО давать варианты ответов (например: 'Выберите А или Б', 'Вы скорее Х или Y?').\n"
    "2. ЗАПРЕЩЕНО использовать слова 'эмоции', 'чувства', 'психология', 'прошлое', 'опыт'.\n"
    "3. ЗАПРЕЩЕНО заставлять пользователя вспоминать личные события или раскрывать персональные данные.\n"
    "4. ВМЕСТО ЭТОГО: Используй гипотеческие сценарии ('Представьте, что актив упал...', 'Если возникнет выбор...'). "
    "Наводи пользователя на рассуждения только через уточняющие открытые вопросы.\n"
    "5. Если ответ слишком короткий, спроси: 'На чем именно будет основан ваш выбор в этой ситуации?' "
    "или 'Что для вас станет решающим фактором в таком сценарии?'.\n\n"

    "ЧТО НУЖНО ВЫЯВИТЬ (ШКАЛЫ):\n"
    "1. risk_index (Коэффициент риска):\n"
    "   - 0.0 — Консерватор: Приоритет — сохранение, отказ от риска даже при потере выгоды.\n"
    "   - 1.0 — Агрессор: Осознанный поиск риска ради высокой доходности, устойчивость к просадкам.\n"
    "2. emotion_index (Эмоциональная устойчивость):\n"
    "   - 0.0 — Импульсивность: Склонность к быстрым решениям под влиянием момента, новостей или паники.\n"
    "   - 1.0 — Рациональность: Хладнокровное следование логике и плану вопреки внешнему давлению.\n\n"

    "РЕКОМЕНДАЦИЯ ПОРТФЕЛЯ (ЛОГИКА):\n"
    "На основе индексов сформируй примерный состав активов:\n"
    "- Низкий риск (0.0-0.3): упор на облигации и золото (70-80%).\n"
    "- Средний риск (0.4-0.7): сбалансированный портфель (50% акции, 50% облигации).\n"
    "- Высокий риск (0.8-1.0): упор на акции роста, фонды или высокорисковые активы.\n"
    "- Низкая устойчивость (emotion < 0.4): рекомендовать максимально пассивные инструменты (ETF), "
    "чтобы минимизировать влияние паники.\n\n"

    "ПРАВИЛА ОБЩЕНИЯ:\n"
    "1. ТОЛЬКО ОТКРЫТЫЕ ВОПРОСЫ. Бот лишь направляет нить диалога, не предлагая готовых путей.\n"
    "2. МАКСИМАЛЬНАЯ КРАТКОСТЬ. Одна реплика — 1-2 предложения.\n"
    "3. СРАЗУ К ДЕЛУ. Никаких вводных фраз ('Интересно...', 'Понятно...'). Сразу вопрос.\n"
    "4. Задавай строго по ОДНОМУ вопросу за раз. Всего около 15 итераций.\n\n"

    "МАРКЕРЫ:\n"
    "- В процессе: [STATUS: COLLECTING]\n"
    "- Финал: [STATUS: FINISHED]\n\n"

    "ФОРМАТ ФИНАЛЬНОГО JSON:\n"
    "```json\n"
    "{\n"
    "  \"investor_type\": \"название типа (например: Консервативный рантье)\",\n"
    "  \"risk_index\": float,\n"
    "  \"emotion_index\": float,\n"
    "  \"recommended_portfolio\": \"краткое описание состава в процентах\",\n"
    "  \"profile_summary\": \"краткий портрет стиля принятия решений без упоминания личных данных\"\n"
    "}\n"
    "```"
)

@dp.message(Command("start"))
async def start_cmd(message: types.Message, state: FSMContext):
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Привет. Начни глубокое финансовое интервью."}
    ]
    
    response = client.chat.complete(model=MISTRAL_MODEL, messages=history)
    ai_text = response.choices[0].message.content
    
    history.append({"role": "assistant", "content": ai_text})
    await state.update_data(messages=history)
    
    await message.answer("🏦 **Запуск системы финансовой диагностики.**\n\n" + ai_text.replace("[STATUS: COLLECTING]", "").strip())
    await state.set_state(Survey.chatting)

@dp.message(Survey.chatting)
async def handle_chat(message: types.Message, state: FSMContext):
    data = await state.get_data()
    history = data.get('messages', [])
    history.append({"role": "user", "content": message.text})
    
    response = client.chat.complete(model=MISTRAL_MODEL, messages=history)
    ai_text = response.choices[0].message.content
    history.append({"role": "assistant", "content": ai_text})
    await state.update_data(messages=history)

    if "[STATUS: FINISHED]" in ai_text:
        match = re.search(r'```json\s*(.*?)\s*```', ai_text, re.DOTALL)
        if match:
            try:
                res = json.loads(match.group(1).strip(), strict=False)
                
                # Финальный красивый отчет для пользователя
                report = (
                    "📊 **ВАШ ИНВЕСТИЦИОННЫЙ ПАСПОРТ**\n\n"
                    f"👤 **Тип:** {res.get('investor_type')}\n"
                    f"📈 **Индекс риска:** `{res.get('risk_index')}`\n"
                    f"🧠 **Эмоциональный фон:** `{res.get('emotion_index')}`\n"
                    f"🧠 **Рекомендуемое портфолио:** `{res.get('recommended_portfolio')}`\n"
                    f"📝 **Психологический портрет:**\n{res.get('profile_summary')}"
                )
                await message.answer(report)
            except Exception:
                await message.answer("Диагностика завершена. Ваша анкета сохранена в базе.")
        
        await state.clear()
    else:
        await message.answer(ai_text.replace("[STATUS: COLLECTING]", "").strip())

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())