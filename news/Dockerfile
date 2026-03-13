FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Отключаем генерацию .pyc файлов и включаем буферизацию логов (чтобы сразу видеть print в консоли)
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

#чтобы избежать проблем с правами доступа
RUN mkdir -p /app/data

# Запуск скрипта
CMD ["python", "mistapinews.py"]