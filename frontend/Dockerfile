# Базовый образ - официальный Python 3.9 на slim-версии Debian
FROM python:3.9-slim

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Копируем файл зависимостей в контейнер
COPY requirements.txt .

# Устанавливаем зависимости:
RUN pip install --no-cache-dir -r requirements.txt

# Копируем основной файл приложения
COPY app.py .

# Команда запуска приложения:
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]