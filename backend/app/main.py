from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, BartTokenizer
import torch
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

MAX_MODEL_LENGTH = 1024
SUMMARY_LENGTH = 150
MIN_SUMMARY_LENGTH = 30
OVERLAP_SIZE = 50


# Load model at startup
@app.on_event("startup")
async def load_model():
    global summarizer, tokenizer
    try:
        logger.info("Starting model loading...")
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        logger.info("Tokenizer loaded")

        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Using device: {'GPU' if device == 0 else 'CPU'}")

        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer=tokenizer,
            device=device
        )
        logger.info("Model successfully loaded")
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise


class TextRequest(BaseModel):
    text: str


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


@app.post("/summarize")
async def summarize(request: TextRequest):
    try:
        text = request.text
        inputs = tokenizer(text, return_tensors="pt", truncation=False)
        input_length = inputs['input_ids'].shape[1]

        if input_length <= MAX_MODEL_LENGTH:
            summary = summarizer(
                text,
                max_length=SUMMARY_LENGTH,
                min_length=MIN_SUMMARY_LENGTH,
                do_sample=False
            )[0]['summary_text']
            return {"summary": summary}

        chunks = split_text(text, tokenizer)
        if not chunks:
            return {"error": "Не удалось разбить текст на части"}

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
        if len(tokenizer.tokenize(combined_summary)) > MAX_MODEL_LENGTH:
            return await summarize(TextRequest(text=combined_summary))

        return {"summary": combined_summary}
    except Exception as e:
        return {"error": str(e)}