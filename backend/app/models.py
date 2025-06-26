"""
Файл models.py — Модели данных:
1) Содержит Pydantic-модели для валидации данных
2) Описывает структуры данных (схемы запросов/ответов)
3) Задает типы полей и ограничения

"""

# Импорт необходимых библиотек
from transformers import pipeline, BartTokenizer  # Для работы с моделями NLP
import re                                         # Для работы с регулярными выражениями
import torch                                       # Для определения устройства (GPU/CPU)

# Константы для настройки модели
MAX_MODEL_LENGTH = 1024  # Максимальная длина входного текста в токенах
SUMMARY_LENGTH = 150     # Желаемая длина суммаризированного текста
MIN_SUMMARY_LENGTH = 30  # Минимальная длина суммаризированного текста
OVERLAP_SIZE = 50        # Размер перекрытия между частями текста при разбиении


def load_summarizer():
    """
    Загружает модель для суммаризации и токенизатор.

    Возвращает:
        tuple: (summarizer, tokenizer) - пайплайн для суммаризации и токенизатор

    Исключения:
        Exception: Если произошла ошибка при загрузке модели

    """
    try:
        # Загрузка токенизатора для модели BART-large-CNN
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

        # Создание пайплайна для суммаризации
        summarizer = pipeline(
            "summarization",                         # Тип задачи
            model="facebook/bart-large-cnn",              # Используемая модель
            tokenizer=tokenizer,                          # Токенизатор
            device=0 if torch.cuda.is_available() else -1  # Использование GPU если доступен
        )
        return summarizer, tokenizer
    except Exception as e:
        raise Exception(f"Ошибка загрузки модели: {str(e)}")


def split_text(text, tokenizer, max_length=MAX_MODEL_LENGTH, overlap=OVERLAP_SIZE):
    """
    Разбивает текст на части, не превышающие максимальную длину.

    Аргументы:
        1) text (str): Текст для разбиения
        2) tokenizer: Токенизатор для подсчета длины текста
        3) max_length (int): Максимальная длина части текста
        4) overlap (int): Размер перекрытия между частями текста

    Возвращает:
        list: Список частей текста подходящей длины

    """
    # Разбиваем текст на предложения с сохранением пунктуации
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    chunks = []               # Результирующие части текста
    current_chunk = []        # Текущая формируемая часть
    current_length = 0        # Текущая длина в токенах

    for sentence in sentences:
        # Подсчет длины предложения в токенах
        sentence_length = len(tokenizer.tokenize(sentence))

        # Если предложение помещается в текущую часть
        if current_length + sentence_length <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            if current_chunk:
                # Добавляем готовую часть в результат
                chunks.append(' '.join(current_chunk))
                # Сохраняем перекрытие для контекста
                current_chunk = current_chunk[-overlap:] if overlap > 0 else []
                current_chunk.append(sentence)
                current_length = sum(len(tokenizer.tokenize(s)) for s in current_chunk)

    # Добавляем последнюю часть, если она не пустая
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


def summarize_long_text(text, summarizer, tokenizer, max_model_length=MAX_MODEL_LENGTH):
    """
    Генерирует суммаризацию текста, при необходимости разбивая его на части.

    Аргументы:
        1) text (str): Текст для суммаризации
        2) summarizer: Модель для суммаризации
        3) tokenizer: Токенизатор
        4) max_model_length (int): Максимальная длина обрабатываемого текста

    Возвращает:
        str: Суммаризированный текст или сообщение об ошибке

    """
    # Проверяем длину текста в токенах
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_length = inputs['input_ids'].shape[1]

    # Если текст короткий - обрабатываем целиком
    if input_length <= max_model_length:
        result = summarizer(
            text,
            max_length=SUMMARY_LENGTH,
            min_length=MIN_SUMMARY_LENGTH,
            do_sample=False                 # Детерминированная генерация
        )
        return result[0]['summary_text']

    # Разбиваем длинный текст на части
    chunks = split_text(text, tokenizer)
    if not chunks:
        return "Не удалось разбить текст на части"

    # Суммаризируем каждую часть отдельно
    summaries = []
    for chunk in chunks:
        chunk_summary = summarizer(
            chunk,
            max_length=SUMMARY_LENGTH,
            min_length=MIN_SUMMARY_LENGTH,
            do_sample=False
        )
        summaries.append(chunk_summary[0]['summary_text'])

    # Объединяем результаты
    combined_summary = ' '.join(summaries)

    # Если объединенная суммаризация слишком длинная - рекурсивно сокращаем
    if len(tokenizer.tokenize(combined_summary)) > max_model_length:
        return summarize_long_text(combined_summary, summarizer, tokenizer)

    return combined_summary