FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install typing_extensions==4.15.0 && \
    pip install https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl && \
    pip install -r requirements.txt

COPY app.py .

EXPOSE 8010

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8010"]
