import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.utils import logging

load_dotenv()

logging.set_verbosity_error()

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small",
    dtype=torch.float16 if device == "cuda" else torch.float32
)
model = model.to(device)
model.eval()

PREFIX = "Summarize the following dialogue:\n\n"


def summarize_safe(text):
    if not isinstance(text, str) or not text.strip():
        return None

    try:
        prompt = PREFIX + text
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            summary_ids = model.generate(
                input_ids=inputs["input_ids"],
                max_length=150,
                min_length=15,
                num_beams=4,
                early_stopping=True,
            )

        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    except Exception as e:
        print(f"Summarization failed: {e}")
        return None


# Test with edge cases and a real ticket
print("Empty string:", summarize_safe(""))
print("None value:", summarize_safe(None))
print()

ticket = "Agent: How can I help?\nCustomer: I need to reset my password.\nAgent: I've sent a reset link to your email.\nCustomer: Got it, thanks."
print("Real ticket:", summarize_safe(ticket))