# Summarization API with FastAPI and Streamlit

Проект для автоматической суммаризации текстов с использованием:
- **Backend**: FastAPI + Transformers (модель BART-large-CNN)
- **Frontend**: Streamlit интерфейс
- **Infrastructure**: Docker контейнеризация

## 📌 Особенности

✔ Поддержка длинных текстов (разбиение на чанки с перекрытием)  
✔ Автоматическое определение GPU/CPU  
✔ Интерактивный веб-интерфейс - http://localhost:8000 (после запуска)   
✔ REST API с документацией - http://localhost:8000/docs (после запуска)
✔ Логирование и обработка ошибок  

## 🚀 Быстрый старт

### Требования
- Docker Desktop (версия 20.10.7+)
- 4GB+ свободной оперативной памяти
- (Опционально) NVIDIA GPU для ускорения обработки

### Запуск
```bash
git clone https://github.com/KarmaNastigla/summarization.git
cd summarization
docker-compose up --build