import asyncio
import json
import re
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from mistralai import Mistral
import os
from dotenv import load_dotenv

load_dotenv()


API_TOKEN = os.getenv("TGKey")
MISTRAL_API_KEY = os.getenv("MKey")
MISTRAL_MODEL = "mistral-large-latest"

bot = Bot(token=API_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
client = Mistral(api_key=MISTRAL_API_KEY)

class Survey(StatesGroup):
    chatting = State()


SYSTEM_PROMPT = (
    "Ты — деликатный финансовый аналитик и эксперт по поведенческим финансам. "
    "Твоя задача: в ходе комфортной беседы определить 'коэффициент сдержанности' инвестора (от 0.0 до 1.0). "
    "0.0 — склонность к высокому риску, 1.0 — максимальная консервативность.\n\n"
    
    "ПРАВИЛА ДИАЛОГА:\n"
    "1. ТАКТИЧНОСТЬ: Никаких прямых вопросов о доходах, реальных накоплениях или личных данных. "
    "Используй интересные гипотетические ситуации (например, 'Представьте, что вам досталась крупная сумма...').\n"
    "2. НЕНАВЯЗЧИВОСТЬ: Если пользователь отвечает кратко ('да', 'не знаю'), не дави на него. "
    "Просто мягко направь разговор дальше или предложи готовые варианты ответов на выбор.\n"
    "3. АДАПТИВНОСТЬ: Подстраивайся под уровень знаний собеседника.\n"
    "4. СТРУКТУРА: Задавай только ОДИН вопрос за раз. Достаточно 4-5 вопросов для оценки.\n"
    "5. МАРКЕРЫ: В конце каждой реплики (кроме финальной) ставь [STATUS: COLLECTING].\n\n"
    
    "ФИНАЛ:\n"
    "Когда данных достаточно, напиши [STATUS: FINISHED] и выдай валидный JSON строго в одну строку, "
    "БЕЗ символов переноса строк внутри текста:\n"
    "```json\n"
    "{\"index\": 0.5, \"experience_level\": \"новичок/опытный/профи\", \"reason\": \"краткое обоснование в одно предложение\"}\n"
    "```"
)

@dp.message(Command("start"))
async def start_cmd(message: types.Message, state: FSMContext):
    history = [{"role": "system", "content": SYSTEM_PROMPT}, 
               {"role": "user", "content": "Привет, начни опрос."}]
    
    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=history
    )
    ai_text = response.choices[0].message.content
    
    history.append({"role": "assistant", "content": ai_text})
    await state.update_data(messages=history)
    
    await message.answer(ai_text.replace("[STATUS: COLLECTING]", "").strip())
    await state.set_state(Survey.chatting)

@dp.message(Survey.chatting)
async def handle_chat(message: types.Message, state: FSMContext):
    data = await state.get_data()
    history = data.get('messages', [])
    history.append({"role": "user", "content": message.text})
    
    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=history
    )
    ai_text = response.choices[0].message.content
    history.append({"role": "assistant", "content": ai_text})
    await state.update_data(messages=history)

    if "[STATUS: FINISHED]" in ai_text:
        match = re.search(r'```json\s*(.*?)\s*```', ai_text, re.DOTALL)
        if match:
            json_raw = match.group(1).strip()
            
            try:
                # strict=False решает проблему с переносами строк и контрольными символами
                res = json.loads(json_raw, strict=False) 
                
                await message.answer(
                    f"**Итог (индекс риска):** `{res.get('index')}`\n\n"
                    f"**Опыт:** {res.get('experience_level')}\n"
                    f"**Анализ:** {res.get('reason')}"
                )
            except json.JSONDecodeError as e:
                print(f"JSON Error: {e} | Raw string: {json_raw}")
                await message.answer(f"Опрос завершен. Результат нейросети:\n{json_raw}")
                
        else:
            await message.answer(f"Опрос завершен. Результат:\n{ai_text.replace('[STATUS: FINISHED]', '').strip()}")
            
        await state.clear()
    else:
        clean_text = ai_text.replace("[STATUS: COLLECTING]", "").strip()
        await message.answer(clean_text)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())