import json
import os
import re
import time
from datetime import datetime

import feedparser
import requests
import schedule
from mistralai import Mistral


DEFAULT_RSS_FEEDS = [
    "https://www.gazeta.ru/export/rss/business.xml",
    "https://www.banki.ru/xml/news.rss",
    "https://lenta.ru/rss/news/economics",
]

RISK_CATEGORIES = [
    "акции",
    "облигации",
    "валюта",
    "золото и сырьё",
    "нефть и газ",
    "банковский сектор",
    "технологический сектор",
    "недвижимость",
    "геополитика",
    "инфляция и ставки",
]

DATA_FILE = os.getenv("DATA_FILE", "data/financial_news.json")
PORTFOLIO_SERVICE_URL = os.getenv(
    "PORTFOLIO_SERVICE_URL",
    "http://localhost:8000/api/check_portfolio",
)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
CHECK_INTERVAL_MINUTES = int(os.getenv("CHECK_INTERVAL_MINUTES", "10"))
MISTRAL_API_KEY = os.getenv("MKey")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

client = Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None

RSS_FEEDS = [
    feed.strip()
    for feed in os.getenv("RSS_FEEDS", ",".join(DEFAULT_RSS_FEEDS)).split(",")
    if feed.strip()
]

ANALYSIS_PROMPT = (
    "Ты анализируешь русскоязычные финансовые новости для риск-мониторинга "
    "портфеля пользователя. Определи, есть ли в новости опасность для одной или "
    "нескольких категорий портфеля.\n\n"
    f"Категории портфеля: {', '.join(RISK_CATEGORIES)}.\n\n"
    "Верни только JSON без markdown в формате:\n"
    "{\n"
    '  "has_portfolio_risk": true,\n'
    '  "summary": "краткое объяснение сути риска",\n'
    '  "dangerous_categories": ["категория 1", "категория 2"],\n'
    '  "category_details": [\n'
    "    {\n"
    '      "category": "название категории",\n'
    '      "risk_level": "low|medium|high",\n'
    '      "reason": "почему новость опасна для категории"\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Если явной опасности нет, верни has_portfolio_risk=false, пустые dangerous_categories "
    "и category_details. Не добавляй категории, которых нет в списке."
)

# Глобальное состояние процесса
seen_links = set()
pending_news = []


def load_existing_news():
    """Загружает новости и восстанавливает набор уже обработанных ссылок."""
    global seen_links

    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as file:
                news_data = json.load(file)
                seen_links = {item["link"] for item in news_data if item.get("link")}
                return news_data
        except json.JSONDecodeError:
            print("Ошибка чтения JSON. Файл будет перезаписан.")

    return []


def save_news_to_json(all_news):
    """Сохраняет новости в JSON-файл."""
    data_dir = os.path.dirname(DATA_FILE)
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(all_news, file, ensure_ascii=False, indent=4)


def clean_html(raw_text):
    """Удаляет HTML-теги и лишние пробелы."""
    if not raw_text:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_text)
    return re.sub(r"\s+", " ", text).strip()


def build_news_text(news_item):
    """Собирает текст новости для анализа моделью."""
    title = news_item.get("title", "")
    summary = clean_html(news_item.get("summary", ""))
    return f"Заголовок: {title}\nКраткое содержание: {summary}"


def fallback_analysis(reason):
    """Возвращает безопасную структуру при недоступности анализа."""
    return {
        "has_portfolio_risk": False,
        "summary": reason,
        "dangerous_categories": [],
        "category_details": [],
    }


def extract_json_payload(raw_content):
    """Достаёт JSON из ответа модели, даже если она вернула лишний текст."""
    text = str(raw_content).strip()
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        return fenced_match.group(1)

    json_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if json_match:
        return json_match.group(1)

    return text


def normalize_analysis(analysis):
    """Нормализует ответ модели до ожидаемой структуры."""
    dangerous_categories = [
        category
        for category in analysis.get("dangerous_categories", [])
        if category in RISK_CATEGORIES
    ]

    category_details = []
    for item in analysis.get("category_details", []):
        category = item.get("category")
        if category not in RISK_CATEGORIES:
            continue
        category_details.append(
            {
                "category": category,
                "risk_level": item.get("risk_level", "medium"),
                "reason": item.get("reason", "").strip(),
            }
        )

    return {
        "has_portfolio_risk": bool(analysis.get("has_portfolio_risk")),
        "summary": str(analysis.get("summary", "")).strip(),
        "dangerous_categories": dangerous_categories,
        "category_details": category_details,
    }


def analyze_news_with_mistral(news_item):
    """Прогоняет новость через Mistral и возвращает риск-анализ."""
    if client is None:
        return fallback_analysis("Mistral API key не задан, анализ пропущен.")

    messages = [
        {"role": "system", "content": ANALYSIS_PROMPT},
        {"role": "user", "content": build_news_text(news_item)},
    ]

    try:
        response = client.chat.complete(model=MISTRAL_MODEL, messages=messages)
        raw_content = response.choices[0].message.content
        if isinstance(raw_content, list):
            raw_content = "".join(
                chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                for chunk in raw_content
            )

        parsed = json.loads(extract_json_payload(raw_content))
        return normalize_analysis(parsed)
    except Exception as error:
        print(f"Ошибка анализа Mistral для новости '{news_item.get('title', '')}': {error}")
        return fallback_analysis(f"Ошибка анализа: {error}")


def notify_portfolio_service(news_batch):
    """Отправляет пачку уже проанализированных новостей в микросервис портфеля."""
    payload = {
        "timestamp": datetime.now().isoformat(),
        "risk_categories": RISK_CATEGORIES,
        "news_items": news_batch,
    }

    try:
        print(f"[{datetime.now()}] Отправка {len(news_batch)} новостей в сервис портфелей...")
        response = requests.post(PORTFOLIO_SERVICE_URL, json=payload, timeout=5)

        if response.status_code == 200:
            print("Успешно доставлено.")
            return True

        print(f"Ошибка API: {response.status_code}")
        return False
    except requests.exceptions.RequestException as error:
        print(f"Не удалось связаться с сервисом портфелей: {error}")
        return False


def fetch_and_process_news():
    """Парсит русские RSS-ленты, анализирует новые новости и отправляет батч дальше."""
    global pending_news

    print(f"[{datetime.now()}] Проверка RSS-лент...")
    all_news = load_existing_news()
    new_items_found = False

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                link = entry.get("link")

                if not link or link in seen_links:
                    continue

                news_item = {
                    "title": clean_html(entry.get("title", "Без заголовка")),
                    "link": link,
                    "published": entry.get("published", ""),
                    "summary": clean_html(entry.get("summary", "")),
                    "source": feed.feed.get("title", feed_url),
                    "source_url": feed_url,
                    "language": "ru",
                    "fetched_at": datetime.now().isoformat(),
                }
                news_item["portfolio_risk_analysis"] = analyze_news_with_mistral(news_item)

                all_news.append(news_item)
                seen_links.add(link)
                pending_news.append(news_item)
                new_items_found = True

        except Exception as error:
            print(f"Ошибка при обработке {feed_url}: {error}")

    if new_items_found:
        save_news_to_json(all_news)
        print(f"Сохранено новых статей: {len(pending_news)}")

    if len(pending_news) >= BATCH_SIZE:
        success = notify_portfolio_service(pending_news)
        if success:
            pending_news.clear()


if __name__ == "__main__":
    load_existing_news()
    print("Сервис новостей запущен. Ожидание расписания...")

    fetch_and_process_news()

    schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(fetch_and_process_news)

    while True:
        schedule.run_pending()
        time.sleep(1)
