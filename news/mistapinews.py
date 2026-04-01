import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import feedparser
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from mistralai import Mistral
from pydantic import BaseModel

load_dotenv()

DEFAULT_RSS_FEEDS = [
    "http://www.cbr.ru/rss/eventrss",
    "http://www.cbr.ru/rss/RssPress",
    "https://www.banki.ru/xml/news.rss",
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
FEED_TIMEOUT_SECONDS = float(os.getenv("FEED_TIMEOUT_SECONDS", "12"))
FEED_MAX_RETRIES = max(1, int(os.getenv("FEED_MAX_RETRIES", "3")))
FEED_RETRY_BACKOFF_SECONDS = float(os.getenv("FEED_RETRY_BACKOFF_SECONDS", "1.5"))
MISTRAL_MAX_RETRIES = max(1, int(os.getenv("MISTRAL_MAX_RETRIES", "2")))
PER_FEED_ITEM_LIMIT = max(1, int(os.getenv("PER_FEED_ITEM_LIMIT", "8")))
MAX_SUMMARY_LENGTH = max(200, int(os.getenv("MAX_SUMMARY_LENGTH", "700")))

MISTRAL_API_KEY = os.getenv("MKey") or os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-medium-latest")
MISTRAL_REQUEST_DELAY_SECONDS = float(os.getenv("MISTRAL_REQUEST_DELAY_SECONDS", "2"))
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
http_timeout = httpx.Timeout(FEED_TIMEOUT_SECONDS, connect=min(5.0, FEED_TIMEOUT_SECONDS))

state_lock = threading.Lock()
refresh_lock = threading.Lock()
seen_links: set[str] = set()
news_cache: list[dict[str, Any]] = []
active_viewers: dict[str, float] = {}
last_refresh_ts = 0.0
worker_started = False
last_feed_errors: dict[str, str] = {}
last_refresh_error = ""
refresh_in_progress = False

FINANCE_KEYWORDS = [
    "банк",
    "цб",
    "ставк",
    "инфляц",
    "эконом",
    "финанс",
    "рын",
    "акци",
    "облигац",
    "валют",
    "рубл",
    "доллар",
    "евро",
    "нефт",
    "газ",
    "дивиденд",
    "бюджет",
    "кредит",
    "депозит",
    "ipo",
    "moex",
]

NOISY_PATTERNS = [
    r"\bреклама\b",
    r"\bпартнер",
    r"\bpromo\b",
    r"\badvert",
    r"\bгороскоп",
    r"\bшоубиз",
    r"\bсериал",
    r"\bзв[её]зд",
]


class PresencePayload(BaseModel):
    session_id: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def clean_html(raw_text: str | None) -> str:
    if not raw_text:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_text)
    return re.sub(r"\s+", " ", text).strip()


def clamp_text(raw_text: str | None, limit: int = MAX_SUMMARY_LENGTH) -> str:
    text = clean_html(raw_text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def normalize_link(link: str | None) -> str:
    value = (link or "").strip()
    if not value:
        return ""
    value = re.sub(r"#.*$", "", value)
    value = re.sub(r"[?&](utm_[^=&]+|from|rss|rssall)=[^&#]*", "", value, flags=re.IGNORECASE)
    value = value.rstrip("?&/")
    return value


def looks_like_finance_news(title: str, summary: str) -> bool:
    text = f"{title} {summary}".lower()
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in NOISY_PATTERNS):
        return False
    return any(keyword in text for keyword in FINANCE_KEYWORDS)


def fetch_feed(feed_url: str) -> tuple[Any | None, str | None]:
    headers = {
        "User-Agent": "finans-news-service/1.0 (+rss aggregator)",
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
    }
    last_error_text = None

    for attempt in range(1, FEED_MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=http_timeout, follow_redirects=True, headers=headers) as http_client:
                response = http_client.get(feed_url)
            response.raise_for_status()
            payload = response.text.strip()
            payload_lower = payload[:500].lower()
            if "<rss" not in payload_lower and "<feed" not in payload_lower and "<?xml" not in payload_lower:
                raise RuntimeError(f"Unexpected non-XML response from source, status={response.status_code}")

            feed = feedparser.parse(response.content)
            if getattr(feed, "bozo", 0) and not getattr(feed, "entries", []):
                bozo_exc = getattr(feed, "bozo_exception", None)
                raise RuntimeError(f"Invalid feed payload: {bozo_exc or 'unknown parse error'}")
            if not getattr(feed, "entries", []):
                raise RuntimeError("Feed returned no entries.")
            return feed, None
        except Exception as error:
            last_error_text = f"attempt {attempt}/{FEED_MAX_RETRIES}: {error}"
            if attempt < FEED_MAX_RETRIES:
                time.sleep(FEED_RETRY_BACKOFF_SECONDS * attempt)

    return None, last_error_text


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

    last_error = None
    for attempt in range(1, MISTRAL_MAX_RETRIES + 1):
        try:
            if MISTRAL_REQUEST_DELAY_SECONDS > 0:
                time.sleep(MISTRAL_REQUEST_DELAY_SECONDS)
            response = client.chat.complete(model=MISTRAL_MODEL, messages=messages)
            raw_content = response.choices[0].message.content
            parsed = json.loads(extract_json_payload(raw_content))
            return normalize_analysis(parsed)
        except Exception as error:
            last_error = error
            if attempt < MISTRAL_MAX_RETRIES:
                time.sleep(min(3.0, attempt))
    return fallback_analysis(f"Ошибка анализа: {last_error}")


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
    global last_refresh_ts, last_feed_errors, last_refresh_error, refresh_in_progress
    if not force and get_active_viewer_count() == 0:
        return

    if not refresh_lock.acquire(blocking=False):
        return

    refresh_in_progress = True
    try:
        with state_lock:
            local_seen_links = set(seen_links)
            local_news = list(news_cache)

        new_found = False
        collected_errors: dict[str, str] = {}

        for feed_url in RSS_FEEDS:
            feed, error_text = fetch_feed(feed_url)
            if error_text:
                collected_errors[feed_url] = error_text
                continue

            accepted_from_feed = 0
            for entry in feed.entries:
                if accepted_from_feed >= PER_FEED_ITEM_LIMIT:
                    break

                raw_link = entry.get("link")
                link = normalize_link(raw_link)
                if not link or link in local_seen_links:
                    continue

                title = clean_html(entry.get("title", "No title"))
                summary = clamp_text(entry.get("summary", "") or entry.get("description", ""))
                if not title or not looks_like_finance_news(title, summary):
                    continue

                news_item = {
                    "title": title,
                    "link": link,
                    "published": entry.get("published", "") or entry.get("updated", ""),
                    "summary": summary,
                    "source": clean_html(feed.feed.get("title", "")) or feed_url,
                    "source_url": feed_url,
                    "language": "ru",
                    "fetched_at": utc_now().isoformat(),
                }
                news_item["portfolio_risk_analysis"] = analyze_news_with_mistral(news_item)

                local_news.append(news_item)
                local_seen_links.add(link)
                accepted_from_feed += 1
                new_found = True

        local_news = sorted_news_items(local_news)[0:MAX_NEWS_ITEMS]
        if new_found:
            save_news_to_json(local_news)

        with state_lock:
            news_cache[:] = local_news
            seen_links.clear()
            seen_links.update(local_seen_links)
            last_feed_errors = collected_errors
            last_refresh_error = "; ".join(f"{url}: {error}" for url, error in collected_errors.items()) if collected_errors else ""
            last_refresh_ts = time.time()
    except Exception as error:
        with state_lock:
            last_refresh_error = str(error)
            last_refresh_ts = time.time()
    finally:
        refresh_in_progress = False
        refresh_lock.release()


def ensure_fresh_news() -> None:
    interval_seconds = max(30, CHECK_INTERVAL_MINUTES * 60)
    if get_active_viewer_count() == 0:
        return
    if not news_cache or time.time() - last_refresh_ts >= interval_seconds:
        try:
            fetch_and_process_news(force=True)
        except Exception:
            return


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
        refresh_error = last_refresh_error
        feed_errors = dict(last_feed_errors)
    return {
        "items": sorted_news_items(items)[:max(1, min(limit, BATCH_SIZE * 4))],
        "active_viewers": active_count,
        "last_refresh_at": datetime.fromtimestamp(refresh_at, tz=timezone.utc).isoformat() if refresh_at else None,
        "refresh_in_progress": refresh_in_progress,
        "error": refresh_error or None,
        "feed_errors": feed_errors,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=NEWS_SERVICE_PORT)
