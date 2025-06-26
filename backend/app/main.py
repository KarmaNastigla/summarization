"""
Файл main.py — Ядро приложения:
1) Содержит логику работы API (роуты, endpoints)
2) Настраивает запуск приложения (UVicorn, middleware)
3) Обрабатывает HTTP-запросы/ответы
4)Координирует работу всех компонентов

"""

# Импорт необходимых библиотек
from fastapi import FastAPI                       # Веб-фреймворк для создания API
from pydantic import BaseModel                    # Для валидации входных данных
from transformers import pipeline, BartTokenizer  # NLP модели и токенизатор
import torch                                      # Для работы с GPU/CPU
import re                                         # Регулярные выражения
import logging                                    # Логирование работы приложения

# Настройка логирования (уровень INFO для отображения важных событий)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)              # Создание логгера

# Создание FastAPI приложения
app = FastAPI()

# Константы для настройки модели суммаризации
MAX_MODEL_LENGTH = 1024                # Максимальная длина входного текста в токенах
SUMMARY_LENGTH = 150                   # Желаемая длина суммаризированного текста
MIN_SUMMARY_LENGTH = 30                # Минимальная длина суммаризированного текста
OVERLAP_SIZE = 50                      # Размер перекрытия между чанками текста


# Загрузка модели при старте приложения
@app.on_event("startup")
async def load_model():
    global summarizer, tokenizer        # Делаем переменные доступными глобально
    try:
        logger.info("Starting model loading...")
        # Инициализация токенизатора BART для CNN модели
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        logger.info("Tokenizer loaded")

        # Определение доступного устройства (GPU если доступен, иначе CPU)
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Using device: {'GPU' if device == 0 else 'CPU'}")

        # Инициализация пайплайна для суммаризации текста
        summarizer = pipeline(
            "summarization",                   # Тип задачи
            model="facebook/bart-large-cnn",        # Используемая модель
            tokenizer=tokenizer,                    # Токенизатор
            device=device                           # Устройство для вычислений
        )
        logger.info("Model successfully loaded")
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        # Прерываем запуск приложения при ошибке загрузки модели
        raise


# Модель запроса для валидации входных данных
class TextRequest(BaseModel):
    text: str                        # Единственное поле - текст для суммаризации


# Корневой endpoint для проверки работы API
@app.get("/")
async def root():
    return {
        "message": "Summarization API is running",
        "endpoints": {
            "summarize": {
                "method": "POST",
                "path": "/summarize",
                "description": "Generate text summary"
            }
        }
    }


# Функция для разбиения длинного текста на чанки
def split_text(text, tokenizer, max_length=MAX_MODEL_LENGTH, overlap=OVERLAP_SIZE):
    # Разбиваем текст на предложения с помощью регулярных выражений
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    chunks = []                       # Список для хранения чанков
    current_chunk = []                # Текущий формируемый чанк
    current_length = 0                # Длина текущего чанка в токенах

    # Формируем чанки с учетом максимальной длины
    for sentence in sentences:
        sentence_length = len(tokenizer.tokenize(sentence))  # Длина предложения в токенах
        # Если предложение помещается в текущий чанк
        if current_length + sentence_length <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))        # Добавляем готовый чанк
                # Сохраняем перекрытие для сохранения контекста между чанками
                current_chunk = current_chunk[-overlap:] if overlap > 0 else []
                current_chunk.append(sentence)
                current_length = sum(len(tokenizer.tokenize(s)) for s in current_chunk)

    # Добавляем последний чанк, если он не пустой
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


# Основной endpoint для суммаризации текста
@app.post("/summarize")
async def summarize(request: TextRequest):
    try:
        text = request.text                           # Получаем текст из запроса

        # Токенизируем текст для проверки длины
        inputs = tokenizer(text, return_tensors="pt", truncation=False)
        input_length = inputs['input_ids'].shape[1]   # Получаем длину в токенах

        # Если текст короткий - обрабатываем сразу
        if input_length <= MAX_MODEL_LENGTH:
            summary = summarizer(
                text,
                max_length=SUMMARY_LENGTH,      # Макс длина суммаризации
                min_length=MIN_SUMMARY_LENGTH,  # Мин длина суммаризации
                do_sample=False                 # Не использовать случайную генерацию
            )[0]['summary_text']
            return {"summary": summary}

        # Для длинных текстов - разбиваем на чанки
        chunks = split_text(text, tokenizer)
        if not chunks:
            return {"error": "Не удалось разбить текст на части"}

        # Суммаризируем каждый чанк отдельно
        summaries = []
        for chunk in chunks:
            chunk_summary = summarizer(
                chunk,
                max_length=SUMMARY_LENGTH,
                min_length=MIN_SUMMARY_LENGTH,
                do_sample=False
            )
            summaries.append(chunk_summary[0]['summary_text'])

        # Объединяем результаты суммаризации чанков
        combined_summary = ' '.join(summaries)

        # Если объединенная суммаризация слишком длинная - рекурсивно сокращаем
        if len(tokenizer.tokenize(combined_summary)) > MAX_MODEL_LENGTH:
            return await summarize(TextRequest(text=combined_summary))

        return {"summary": combined_summary}
    except Exception as e:
        # Обработка любых ошибок в процессе суммаризации
        return {"error": str(e)}