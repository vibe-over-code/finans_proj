import feedparser
import requests
import json
import time
import schedule
import os
from datetime import datetime


RSS_FEEDS = [
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664", # Пример: CNBC Finance
    "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml" # Пример: WSJ Business
]

DATA_FILE = "data/financial_news.json"
PORTFOLIO_SERVICE_URL = "http://localhost:8000/api/check_portfolio" # Заглушка API
BATCH_SIZE = 5 # Количество новостей для триггера API
CHECK_INTERVAL_MINUTES = 10 # Как часто проверять RSS

# Глобальные переменные для хранения состояния в памяти
seen_links = set()
pending_news = []

def load_existing_news():
    """Загружает существующие новости для инициализации сета прочитанных ссылок."""
    global seen_links
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                news_data = json.load(f)
                seen_links = {item['link'] for item in news_data}
                return news_data
        except json.JSONDecodeError:
            print("Ошибка чтения JSON. Файл будет перезаписан.")
    return []

def save_news_to_json(all_news):
    """Сохраняет весь массив новостей в JSON-файл."""
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_news, f, ensure_ascii=False, indent=4)

def notify_portfolio_service(news_batch):
    """Отправляет пачку новостей в микросервис проверки портфелей."""
    payload = {
        "timestamp": datetime.now().isoformat(),
        "news_items": news_batch
    }
    
    try:
        print(f"[{datetime.now()}] Отправка {len(news_batch)} новостей в сервис портфелей...")
        # Пока это заглушка, таймаут обязателен, чтобы воркер не завис
        response = requests.post(PORTFOLIO_SERVICE_URL, json=payload, timeout=5)
        
        if response.status_code == 200:
            print("Успешно доставлено.")
            return True
        else:
            print(f"Ошибка API: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Не удалось связаться с сервисом портфелей: {e}")
        return False

def fetch_and_process_news():
    """Основная логика парсинга, сохранения и уведомления."""
    global pending_news
    
    print(f"[{datetime.now()}] Проверка RSS-лент...")
    all_news = load_existing_news()
    new_items_found = False
    
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                link = entry.get('link')
                
                # Дедупликация по URL статьи
                if link and link not in seen_links:
                    news_item = {
                        "title": entry.get('title', 'Без заголовка'),
                        "link": link,
                        "published": entry.get('published', ''),
                        "summary": entry.get('summary', ''),
                        "source": feed_url,
                        "fetched_at": datetime.now().isoformat()
                    }
                    
                    all_news.append(news_item)
                    seen_links.add(link)
                    pending_news.append(news_item)
                    new_items_found = True
                    
        except Exception as e:
            print(f"Ошибка при обработке {feed_url}: {e}")

    # Если нашли что-то новое, сохраняем в базу (JSON)
    if new_items_found:
        save_news_to_json(all_news)
        print(f"Сохранено новых статей: {len(pending_news)}")

    # Проверяем, накопилось ли достаточно новостей для отправки
    if len(pending_news) >= BATCH_SIZE:
        success = notify_portfolio_service(pending_news)
        if success:
            # Очищаем буфер только если другой микросервис успешно принял данные
            pending_news.clear()
        # Если API недоступно, новости останутся в pending_news и отправятся в следующий цикл

if __name__ == "__main__":
    # Инициализируем историю при старте
    load_existing_news()
    print("Микросервис запущен. Ожидание расписания...")
    
    # Запускаем один раз при старте для проверки
    fetch_and_process_news()
    
    # Настраиваем периодичность
    schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(fetch_and_process_news)

    while True:
        schedule.run_pending()
        time.sleep(1)