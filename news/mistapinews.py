import json
import os
import re
import time
from datetime import datetime

import feedparser
import requests
import schedule
from mistralai import Mistral
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()

# --- Конфигурация ---
DEFAULT_RSS_FEEDS = [
    "https://www.gazeta.ru/export/rss/business.xml",
    "https://www.banki.ru/xml/news.rss",
    "https://lenta.ru/rss/news/economics",
]

RISK_CATEGORIES = [
    "акции", "облигации", "валюта", "золото и сырьё",
    "нефть и газ", "банковский сектор", "технологический сектор",
    "недвижимость", "геополитика", "инфляция и ставки",
]

DATA_FILE = os.getenv("DATA_FILE", "data/financial_news.json")
PORTFOLIO_SERVICE_URL = os.getenv("PORTFOLIO_SERVICE_URL", "http://localhost:8000/api/check_portfolio")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
CHECK_INTERVAL_MINUTES = int(os.getenv("CHECK_INTERVAL_MINUTES", "10"))

# ВАЖНО: Убедись, что в .env файл записано MKey=ваш_ключ
MISTRAL_API_KEY = os.getenv("MKey") or os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-medium-latest")

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
    "Если явной опасности нет, верни has_portfolio_risk=false."
)

seen_links = set()
pending_news = []

# --- Утилиты ---

def load_existing_news():
    global seen_links
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as file:
                news_data = json.load(file)
                seen_links = {item["link"] for item in news_data if item.get("link")}
                return news_data
        except (json.JSONDecodeError, Exception):
            print("Ошибка чтения JSON. Файл будет перезаписан.")
    return []

def save_news_to_json(all_news):
    data_dir = os.path.dirname(DATA_FILE)
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(all_news, file, ensure_ascii=False, indent=4)

def clean_html(raw_text):
    if not raw_text: return ""
    text = re.sub(r"<[^>]+>", " ", raw_text)
    return re.sub(r"\s+", " ", text).strip()

def build_news_text(news_item):
    return f"Заголовок: {news_item.get('title', '')}\nКраткое содержание: {clean_html(news_item.get('summary', ''))}"

def fallback_analysis(reason):
    return {
        "has_portfolio_risk": False,
        "summary": reason,
        "dangerous_categories": [],
        "category_details": [],
    }

def extract_json_payload(raw_content):
    text = str(raw_content).strip()
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match: return fenced_match.group(1)
    json_match = re.search(r"(\{.*\})", text, re.DOTALL)
    return json_match.group(1) if json_match else text

def normalize_analysis(analysis):
    dangerous_categories = [c for c in analysis.get("dangerous_categories", []) if c in RISK_CATEGORIES]
    category_details = [
        {
            "category": item.get("category"),
            "risk_level": item.get("risk_level", "medium"),
            "reason": item.get("reason", "").strip(),
        }
        for item in analysis.get("category_details", [])
        if item.get("category") in RISK_CATEGORIES
    ]
    return {
        "has_portfolio_risk": bool(analysis.get("has_portfolio_risk")),
        "summary": str(analysis.get("summary", "")).strip(),
        "dangerous_categories": dangerous_categories,
        "category_details": category_details,
    }

# --- Логика анализа ---

def analyze_news_with_mistral(news_item):
    if not client:
        return fallback_analysis("Mistral API key не задан, анализ пропущен.")

    messages = [
        {"role": "system", "content": ANALYSIS_PROMPT},
        {"role": "user", "content": build_news_text(news_item)},
    ]

    try:
        response = client.chat.complete(model=MISTRAL_MODEL, messages=messages)
        raw_content = response.choices[0].message.content
        parsed = json.loads(extract_json_payload(raw_content))
        return normalize_analysis(parsed)
    except Exception as error:
        print(f"Ошибка Mistral для '{news_item.get('title')}': {error}")
        return fallback_analysis(f"Ошибка анализа: {error}")

def has_portfolio_analysis(news_item):
    """Проверяет, есть ли у новости НОРМАЛЬНЫЙ анализ (не заглушка)."""
    analysis = news_item.get("portfolio_risk_analysis")
    if not isinstance(analysis, dict) or "has_portfolio_risk" not in analysis:
        return False
    
    # Если в summary текст нашей ошибки - значит анализа по факту нет
    summary = analysis.get("summary", "")
    if "Mistral API key не задан, анализ пропущен." in summary or "Ошибка анализа" in summary:
        return False
    return True

def notify_portfolio_service(news_batch):
    payload = {
        "timestamp": datetime.now().isoformat(),
        "risk_categories": RISK_CATEGORIES,
        "news_items": news_batch,
    }
    try:
        print(f"[{datetime.now()}] Отправка батча ({len(news_batch)} шт.) в сервис...")
        response = requests.post(PORTFOLIO_SERVICE_URL, json=payload, timeout=10)
        if response.status_code == 200:
            print("Успешно доставлено.")
            return True
        print(f"Ошибка API: {response.status_code}")
    except Exception as error:
        print(f"Сервис портфелей недоступен: {error}")
    return False

# --- Основные шаги процесса ---

def backfill_missing_analysis(all_news):
    """Шаг 1: Проверка и доанализ старых новостей из JSON."""
    backfilled = []
    print(f"[{datetime.now()}] Проверка базы на новости без анализа...")
    
    for item in all_news:
        if not has_portfolio_analysis(item):
            print(f"-> Доанализируем: {item.get('title')[:50]}...")
            item["portfolio_risk_analysis"] = analyze_news_with_mistral(item)
            backfilled.append(item)
    
    return backfilled

def fetch_and_process_news():
    """Полный цикл: Сначала база, потом RSS."""
    global pending_news

    # 1. Загружаем то, что есть в файле
    all_news = load_existing_news()
    
    # 2. ШАГ 1: Исправляем старые новости
    backfilled_items = backfill_missing_analysis(all_news)
    if backfilled_items:
        save_news_to_json(all_news)
        pending_news.extend(backfilled_items)
        print(f"Исправлено старых записей: {len(backfilled_items)}")

    # 3. ШАГ 2: Парсим новые новости из интернета
    print(f"[{datetime.now()}] Проверка RSS-лент...")
    new_found = False
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                link = entry.get("link")
                if not link or link in seen_links: continue

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
                
                print(f"-> Новая новость: {news_item['title'][:50]}...")
                news_item["portfolio_risk_analysis"] = analyze_news_with_mistral(news_item)
                
                all_news.append(news_item)
                seen_links.add(link)
                pending_news.append(news_item)
                new_found = True
        except Exception as e:
            print(f"Ошибка RSS {feed_url}: {e}")

    if new_found:
        save_news_to_json(all_news)

    # 4. Отправка накопленного в микросервис (порциями по BATCH_SIZE)
    while len(pending_news) >= BATCH_SIZE:
        batch = pending_news[:BATCH_SIZE]
        if notify_portfolio_service(batch):
            del pending_news[:BATCH_SIZE]
        else:
            break

if __name__ == "__main__":
    if not MISTRAL_API_KEY:
        print("ВНИМАНИЕ: MISTRAL_API_KEY не найден в .env файле!")
    
    print("Сервис запущен. Первый запуск...")
    fetch_and_process_news()

    schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(fetch_and_process_news)

    while True:
        schedule.run_pending()
        time.sleep(1)