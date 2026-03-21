import asyncio
import json
import logging
import os
import re

import httpx
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv
from mistralai import Mistral

logging.basicConfig(level=logging.INFO)
load_dotenv()

API_TOKEN = os.getenv("TGKey")
MISTRAL_API_KEY = os.getenv("MKey")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
PORTFOLIO_PROFILE_URL = os.getenv("PORTFOLIO_PROFILE_URL", "http://localhost:8000/api/client-profile")

bot = Bot(token=API_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
client = Mistral(api_key=MISTRAL_API_KEY)


class Survey(StatesGroup):
    chatting = State()


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


async def send_profile_to_portfolio_service(profile: dict, telegram_user_id: int | None):
    payload = {
        "client_id": f"tg_{telegram_user_id}" if telegram_user_id else None,
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


def format_portfolio(value) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


@dp.message(Command("start"))
async def start_cmd(message: types.Message, state: FSMContext):
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Привет. Начни интервью."},
    ]

    response = client.chat.complete(model=MISTRAL_MODEL, messages=history)
    ai_text = response.choices[0].message.content

    history.append({"role": "assistant", "content": ai_text})
    await state.update_data(messages=history)

    await message.answer(
        "Запуск финансового интервью.\n\n"
        + str(ai_text).replace("[STATUS: COLLECTING]", "").strip()
    )
    await state.set_state(Survey.chatting)


@dp.message(Survey.chatting)
async def handle_chat(message: types.Message, state: FSMContext):
    data = await state.get_data()
    history = data.get("messages", [])
    history.append({"role": "user", "content": message.text})

    response = client.chat.complete(model=MISTRAL_MODEL, messages=history)
    ai_text = response.choices[0].message.content
    history.append({"role": "assistant", "content": ai_text})
    await state.update_data(messages=history)

    if "[STATUS: FINISHED]" in ai_text:
        try:
            res = extract_final_payload(str(ai_text))
            if not res:
                raise ValueError("Final JSON payload not found")

            report_lines = [
                "Ваш инвестиционный паспорт",
                "",
                f"Цель: {res.get('goal')}",
                f"Тип: {res.get('investor_type')}",
                f"Индекс риска: `{res.get('risk_index')}`",
                f"Эмоциональная устойчивость: `{res.get('emotion_index')}`",
                "Целевой портфель:",
                f"```json\n{format_portfolio(res.get('target_portfolio'))}\n```",
            ]

            if "current_portfolio" in res:
                report_lines.extend(
                    [
                        "Текущий портфель:",
                        f"```json\n{format_portfolio(res.get('current_portfolio'))}\n```",
                    ]
                )

            report_lines.extend(
                [
                    "Профиль:",
                    str(res.get("profile_summary")),
                ]
            )

            await message.answer("\n".join(report_lines))
            await send_profile_to_portfolio_service(
                res,
                message.from_user.id if message.from_user else None,
            )
        except Exception:
            await message.answer("Диагностика завершена, но итоговые данные не удалось оформить.")

        await state.clear()
    else:
        await message.answer(str(ai_text).replace("[STATUS: COLLECTING]", "").strip())


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
