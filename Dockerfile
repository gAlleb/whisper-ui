FROM python:3.12-slim

# Отключаем создание кеша .pyc и включаем моментальный вывод логов
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Сначала обновляем pip, затем ставим зависимости без кеша
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn httpx google-genai python-multipart

COPY proxy.py .

CMD ["python", "proxy.py"]
