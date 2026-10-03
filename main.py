import logging
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Model loading (runs once when the server starts) ---
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small",
    dtype=torch.float16 if device == "cuda" else torch.float32
)
model = model.to(device)
model.eval()

PREFIX = "Summarize the following dialogue:\n\n"


# --- Request and response schemas ---
class SummarizeRequest(BaseModel):
    text: str = Field(description="The dialogue to summarize")
    max_length: int = Field(default=150, description="Maximum summary length in tokens")
    min_length: int = Field(default=15, description="Minimum summary length in tokens")
    model_config = {"extra": "forbid"}


class SummarizeResponse(BaseModel):
    summary: str = Field(description="The generated summary")


# --- API endpoint ---
@app.post("/summarize", response_model=SummarizeResponse)
def summarize_endpoint(request: SummarizeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty.")

    try:
        prompt = PREFIX + request.text
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            summary_ids = model.generate(
                input_ids=inputs["input_ids"],
                max_length=request.max_length,
                min_length=request.min_length,
                num_beams=4,
                early_stopping=True,
            )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return SummarizeResponse(summary=summary)

    except Exception as e:
        logger.exception(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail="Summarization failed. Please try again.")