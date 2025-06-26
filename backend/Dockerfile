"""
Dockerfile — это инструкция для сборки Docker-образа.
Он определяет, как будет создаваться и запускаться контейнер.

"""

# Используем официальный образ Python 3.9 на базе slim-версии Debian
FROM python:3.9-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл с зависимостями в контейнер
COPY requirements.txt .

# Устанавливаем зависимости:
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код приложения (включая основной модуль)
COPY ./app ./app

"""
Команда запуска приложения:
    uvicorn - ASGI-сервер для FastAPI
    app.main:app - указывает модуль (app/main.py) и приложение (app = FastAPI())
    --host 0.0.0.0 - слушаем все сетевые интерфейсы
    --port 8000 - фиксируем порт (должен соответствовать EXPOSE и docker-compose)

"""
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]