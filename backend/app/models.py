from transformers import pipeline, BartTokenizer
import re
import torch

MAX_MODEL_LENGTH = 1024
SUMMARY_LENGTH = 150
MIN_SUMMARY_LENGTH = 30
OVERLAP_SIZE = 50

def load_summarizer():
    try:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        return summarizer, tokenizer
    except Exception as e:
        raise Exception(f"Ошибка загрузки модели: {str(e)}")


def split_text(text, tokenizer, max_length=MAX_MODEL_LENGTH, overlap=OVERLAP_SIZE):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(tokenizer.tokenize(sentence))
        if current_length + sentence_length <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = current_chunk[-overlap:] if overlap > 0 else []
                current_chunk.append(sentence)
                current_length = sum(len(tokenizer.tokenize(s)) for s in current_chunk)

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


def summarize_long_text(text, summarizer, tokenizer, max_model_length=MAX_MODEL_LENGTH):
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_length = inputs['input_ids'].shape[1]

    if input_length <= max_model_length:
        return summarizer(
            text,
            max_length=SUMMARY_LENGTH,
            min_length=MIN_SUMMARY_LENGTH,
            do_sample=False
        )[0]['summary_text']

    chunks = split_text(text, tokenizer)
    if not chunks:
        return "Не удалось разбить текст на части"

    summaries = []
    for chunk in chunks:
        chunk_summary = summarizer(
            chunk,
            max_length=SUMMARY_LENGTH,
            min_length=MIN_SUMMARY_LENGTH,
            do_sample=False
        )
        summaries.append(chunk_summary[0]['summary_text'])

    combined_summary = ' '.join(summaries)
    if len(tokenizer.tokenize(combined_summary)) > max_model_length:
        return summarize_long_text(combined_summary, summarizer, tokenizer)
    return combined_summary