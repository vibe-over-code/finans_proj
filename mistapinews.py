import requests
from bs4 import BeautifulSoup
import time
import random
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MKey")
CACHE_FILE = "portfolio_memory.json"

USER_PORTFOLIO = {
    "Gazprom": 180,
    "IBM": 100,             # Акции
    "Sberbank": 500,        # Акции
    "Gold (ETF)": 10,       # Защитный актив
    "Cash (RUB)": 50000     # Наличные
}

#парсер
class ResilientParser:
    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) Chrome/121.0.0.0 Safari/537.36'
        ]
        self.session = requests.Session()
        self.history = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"last_update": "", "news_pool": []}

    def _save_cache(self, new_news):
        # Добавляем новые новости, убираем дубли, оставляем последние 50
        existing_titles = {n['title'] for n in self.history["news_pool"]}
        for item in new_news:
            if item['title'] not in existing_titles:
                self.history["news_pool"].append(item)
        
        self.history["news_pool"] = self.history["news_pool"][-50:]
        self.history["last_update"] = datetime.now().isoformat()
        
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=4)

    def get_news_for_portfolio(self, portfolio):
        all_fresh_news = []
        tickers = list(portfolio.keys())
        
        print("🔍 Сбор свежих данных с рынка...")
        for ticker in tickers:
            # Ищем только по названиям активов, кэш пропускаем
            if "Cash" in ticker: continue 
            
            url = f"https://news.google.com/rss/search?q={ticker}+stock+financial+news&hl=en-US&gl=US&ceid=US:en"
            self.session.headers.update({'User-Agent': random.choice(self.user_agents)})
            
            try:
                # Таймаут и защита от 443
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, features="xml")
                items = soup.find_all('item')
                
                # Берем топ-3 свежих заголовка по каждому активу
                for item in items[:3]:
                    all_fresh_news.append({
                        "ticker": ticker,
                        "title": item.title.text,
                        "date": item.pubDate.text
                    })
                time.sleep(random.uniform(1.0, 2.5))
                
            except Exception as e:
                print(f"⚠️ Ошибка сети для {ticker} ({e}). Берем данные из памяти.")
        
        # Если интернет полностью лег, берем из кэша
        if not all_fresh_news:
            print("Переход в автономный режим (используем кэш).")
            all_fresh_news = self.history["news_pool"][-10:]
        else:
            self._save_cache(all_fresh_news)
            
        return all_fresh_news

#Аналитика
def analyze_market_with_mistral(portfolio, news_list):
    print("🧠 Передаем данные в Mistral для анализа...")
    
    # Собираем все заголовки в один текст
    news_text = "\n".join([f"- [{n.get('ticker', 'Market')}] {n['title']}" for n in news_list])
    
    prompt = f"""
    You are an expert AI Portfolio Manager.
    User's current portfolio: {json.dumps(portfolio)}
    
    Latest market news:
    {news_text}
    
    Analyze how these news specifically affect THIS portfolio.
    Provide the output in STRICT JSON format with the following keys:
    1. "current_sentiment": float (0.0 extremely negative for this portfolio, 1.0 extremely positive, 0.5 neutral).
    2. "trend_vector": float (-0.2 for bearish/falling trend, +0.2 for bullish/rising trend, 0.0 for flat).
    3. "reasoning": string (Short explanation of the score based on the assets).
    4. "action_advice": string (One clear sentence of advice: Buy, Hold, or Sell specific assets).
    """
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }
    
    data = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.2
    }
    
    try:
        response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=data, timeout=20)
        response.raise_for_status()
        
        content = response.json()['choices'][0]['message']['content']
        return json.loads(content)
        
    except Exception as e:
        print(f"❌ Ошибка Mistral API: {e}")
        return {
            "current_sentiment": 0.5, 
            "trend_vector": 0.0, 
            "reasoning": "Failed to connect to AI engine.", 
            "action_advice": "Hold positions until connection is restored."
        }

#Start
if __name__ == "__main__":
    print(f"💼 Анализируем портфель: {list(USER_PORTFOLIO.keys())}\n")
    
    # 1. Собираем новости
    parser = ResilientParser(CACHE_FILE)
    fresh_news = parser.get_news_for_portfolio(USER_PORTFOLIO)
    
    if not fresh_news:
        print("Нет данных для анализа.")
    else:
        # 2. Отправляем пачку новостей в нейронку (1 запрос = экономия лимитов)
        analysis_result = analyze_market_with_mistral(USER_PORTFOLIO, fresh_news)
        
        # 3. Выводим красивый результат
        print("\n==================================================")
        print("📊 ОТЧЕТ ИИ-СОВЕТНИКА")
        print("==================================================")
        print(f"🌡 Текущий фон портфеля: {analysis_result.get('current_sentiment')} (0 - крах, 1 - рост)")
        print(f"📈 Вектор тренда:        {analysis_result.get('trend_vector')} (-0.2 медведи, +0.2 быки)")
        
        # Считаем итоговый скор (Фон + Вектор)
        final_score = round(analysis_result.get('current_sentiment', 0.5) + analysis_result.get('trend_vector', 0.0), 2)
        # Ограничиваем рамками 0.0 - 1.0
        final_score = max(0.0, min(1.0, final_score))
        
        print(f"🎯 ИТОГОВЫЙ ИНДЕКС:      {final_score}")
        print("--------------------------------------------------")
        print(f"🧠 Логика ИИ: {analysis_result.get('reasoning')}")
        print(f"💡 Совет:     {analysis_result.get('action_advice')}")
        print("==================================================")