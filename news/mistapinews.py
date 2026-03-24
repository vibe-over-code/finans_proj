import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import feedparser
from dotenv import load_dotenv
from fastapi import FastAPI
from mistralai import Mistral
from pydantic import BaseModel

load_dotenv()

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

DATA_FILE = os.getenv("DATA_FILE", "/app/data/financial_news.json")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
CHECK_INTERVAL_MINUTES = int(os.getenv("CHECK_INTERVAL_MINUTES", "10"))
VIEWER_TTL_SECONDS = int(os.getenv("VIEWER_TTL_SECONDS", "45"))
MAX_NEWS_ITEMS = int(os.getenv("MAX_NEWS_ITEMS", "50"))
NEWS_SERVICE_PORT = int(os.getenv("NEWS_SERVICE_PORT", "8002"))

MISTRAL_API_KEY = os.getenv("MKey") or os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-medium-latest")
RSS_FEEDS = [
    feed.strip()
    for feed in os.getenv("RSS_FEEDS", ",".join(DEFAULT_RSS_FEEDS)).split(",")
    if feed.strip()
]

ANALYSIS_PROMPT = (
    "Ты анализируешь русскоязычные финансовые новости для риск-мониторинга портфеля.\n"
    f"Категории: {', '.join(RISK_CATEGORIES)}.\n"
    "Верни только JSON без markdown в формате:\n"
    "{\n"
    '  "has_portfolio_risk": true,\n'
    '  "summary": "краткое объяснение сути риска",\n'
    '  "dangerous_categories": ["категория 1"],\n'
    '  "category_details": [\n'
    "    {\n"
    '      "category": "название категории",\n'
    '      "risk_level": "low|medium|high",\n'
    '      "reason": "почему новость опасна"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Если явной опасности нет, верни has_portfolio_risk=false."
)

client = Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None
app = FastAPI(title="News Service")

state_lock = threading.Lock()
seen_links: set[str] = set()
news_cache: list[dict[str, Any]] = []
active_viewers: dict[str, float] = {}
last_refresh_ts = 0.0
worker_started = False


class PresencePayload(BaseModel):
    session_id: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def clean_html(raw_text: str | None) -> str:
    if not raw_text:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_text)
    return re.sub(r"\s+", " ", text).strip()


def extract_json_payload(raw_content: Any) -> str:
    text = str(raw_content).strip()
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        return fenced_match.group(1)
    json_match = re.search(r"(\{.*\})", text, re.DOTALL)
    return json_match.group(1) if json_match else text


def fallback_analysis(reason: str) -> dict[str, Any]:
    return {
        "has_portfolio_risk": False,
        "summary": reason,
        "dangerous_categories": [],
        "category_details": [],
    }


def normalize_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
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


def build_news_text(news_item: dict[str, Any]) -> str:
    return (
        f"Заголовок: {news_item.get('title', '')}\n"
        f"Краткое содержание: {clean_html(news_item.get('summary', ''))}"
    )


def analyze_news_with_mistral(news_item: dict[str, Any]) -> dict[str, Any]:
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
        return fallback_analysis(f"Ошибка анализа: {error}")


def load_existing_news() -> list[dict[str, Any]]:
    path = Path(DATA_FILE)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def save_news_to_json(all_news: list[dict[str, Any]]) -> None:
    path = Path(DATA_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(all_news, ensure_ascii=False, indent=2), encoding="utf-8")


def hydrate_cache() -> None:
    global news_cache, seen_links
    cached = load_existing_news()
    with state_lock:
        news_cache = cached[-MAX_NEWS_ITEMS:]
        seen_links = {item["link"] for item in cached if item.get("link")}


def cleanup_viewers() -> None:
    deadline = time.time() - VIEWER_TTL_SECONDS
    with state_lock:
        for session_id, ts in list(active_viewers.items()):
            if ts < deadline:
                active_viewers.pop(session_id, None)


def get_active_viewer_count() -> int:
    cleanup_viewers()
    with state_lock:
        return len(active_viewers)


def sorted_news_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: item.get("fetched_at", ""), reverse=True)


def fetch_and_process_news(force: bool = False) -> None:
    global last_refresh_ts
    if not force and get_active_viewer_count() == 0:
        return

    with state_lock:
        local_seen_links = set(seen_links)
        local_news = list(news_cache)

    new_found = False
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
        except Exception:
            continue

        for entry in feed.entries:
            link = entry.get("link")
            if not link or link in local_seen_links:
                continue

            news_item = {
                "title": clean_html(entry.get("title", "Без заголовка")),
                "link": link,
                "published": entry.get("published", ""),
                "summary": clean_html(entry.get("summary", "")),
                "source": feed.feed.get("title", feed_url),
                "source_url": feed_url,
                "language": "ru",
                "fetched_at": utc_now().isoformat(),
            }
            news_item["portfolio_risk_analysis"] = analyze_news_with_mistral(news_item)

            local_news.append(news_item)
            local_seen_links.add(link)
            new_found = True

    if new_found:
        local_news = sorted_news_items(local_news)[0:MAX_NEWS_ITEMS]
        save_news_to_json(local_news)

    with state_lock:
        news_cache[:] = sorted_news_items(local_news)[0:MAX_NEWS_ITEMS]
        seen_links.clear()
        seen_links.update(local_seen_links)
        last_refresh_ts = time.time()


def ensure_fresh_news() -> None:
    interval_seconds = max(30, CHECK_INTERVAL_MINUTES * 60)
    if get_active_viewer_count() == 0:
        return
    if not news_cache or time.time() - last_refresh_ts >= interval_seconds:
        fetch_and_process_news(force=True)


def background_worker() -> None:
    interval_seconds = max(30, CHECK_INTERVAL_MINUTES * 60)
    while True:
        try:
            cleanup_viewers()
            if get_active_viewer_count() > 0 and time.time() - last_refresh_ts >= interval_seconds:
                fetch_and_process_news(force=True)
        except Exception:
            pass
        time.sleep(5)


@app.on_event("startup")
def startup_event() -> None:
    global worker_started, last_refresh_ts
    hydrate_cache()
    last_refresh_ts = time.time() if news_cache else 0.0
    if not worker_started:
        threading.Thread(target=background_worker, daemon=True).start()
        worker_started = True


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/presence/register")
def register_presence(payload: PresencePayload) -> dict[str, int]:
    with state_lock:
        active_viewers[payload.session_id] = time.time()
    ensure_fresh_news()
    return {"active_viewers": get_active_viewer_count()}


@app.post("/api/presence/ping")
def ping_presence(payload: PresencePayload) -> dict[str, int]:
    with state_lock:
        active_viewers[payload.session_id] = time.time()
    ensure_fresh_news()
    return {"active_viewers": get_active_viewer_count()}


@app.post("/api/presence/unregister")
def unregister_presence(payload: PresencePayload) -> dict[str, int]:
    with state_lock:
        active_viewers.pop(payload.session_id, None)
    return {"active_viewers": get_active_viewer_count()}


@app.get("/api/news")
def get_news(limit: int = 8) -> dict[str, Any]:
    ensure_fresh_news()
    with state_lock:
        items = list(news_cache)
        active_count = len(active_viewers)
        refresh_at = last_refresh_ts
    return {
        "items": sorted_news_items(items)[:max(1, min(limit, BATCH_SIZE * 4))],
        "active_viewers": active_count,
        "last_refresh_at": datetime.fromtimestamp(refresh_at, tz=timezone.utc).isoformat() if refresh_at else None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=NEWS_SERVICE_PORT)
