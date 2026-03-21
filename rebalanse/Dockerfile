FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY answer.py .
COPY client_data.json .

EXPOSE 8000

CMD ["uvicorn", "answer:app", "--host", "0.0.0.0", "--port", "8000"]
