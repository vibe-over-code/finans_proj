FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mistapinews.py .

#чтобы избежать проблем с правами доступа
RUN mkdir -p /app/data

CMD ["python", "mistapinews.py"]
