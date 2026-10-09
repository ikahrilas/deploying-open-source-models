# Loads the saved adapter and runs evaluation without retraining.

import warnings
import logging

import torch
import evaluate
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from peft import PeftModel

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)


MODEL_NAME = "google/flan-t5-small"
ADAPTER_PATH = "./flan-t5-summarizer-adapter"
PREFIX = "Summarize the following dialogue:\n\n"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# Load the fine-tuned adapter on top of the base model
# To get base model behavior, disable the adapter temporarily.
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
peft_model.eval()

# Load test data
full_dataset = load_dataset("knkarthick/dialogsum")
test_data = full_dataset["test"].select(range(20))

# --- Side-by-side comparison ---
print("\n" + "=" * 70)
print("SIDE-BY-SIDE COMPARISON")
print("=" * 70)

# Use spaced indices to get different dialogues
example_indices = [0, 5, 10]

for i in example_indices:
    dialogue = test_data[i]["dialogue"]
    reference = test_data[i]["summary"]

    prompt = PREFIX + dialogue
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        # Disable adapter to get base model predictions
        with peft_model.disable_adapter():
            base_ids = peft_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=64
            )

        # Enable adapter for fine-tuned predictions
        ft_ids = peft_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=64
        )

    base_summary = tokenizer.decode(base_ids[0], skip_special_tokens=True)
    ft_summary = tokenizer.decode(ft_ids[0], skip_special_tokens=True)

    print(f"\n--- Example (index {i}) ---")
    print(f"Dialogue:    {dialogue[:200]}...")
    print(f"Reference:   {reference}")
    print(f"Base model:  {base_summary}")
    print(f"Fine-tuned:  {ft_summary}")

# --- ROUGE evaluation ---
rouge = evaluate.load("rouge")

def generate_summary(text, use_adapter=True):
    prompt = PREFIX + text
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        if use_adapter:
            summary_ids = peft_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=64
            )
        else:
            with peft_model.disable_adapter():
                summary_ids = peft_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=64
                )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

test_docs = test_data["dialogue"]
test_refs = test_data["summary"]

print("\nGenerating summaries for ROUGE evaluation...")
base_preds = [generate_summary(doc, use_adapter=False) for doc in test_docs]
ft_preds = [generate_summary(doc, use_adapter=True) for doc in test_docs]

base_scores = rouge.compute(predictions=base_preds, references=test_refs)
ft_scores = rouge.compute(predictions=ft_preds, references=test_refs)

print("\n" + "=" * 70)
print("ROUGE SCORES")
print("=" * 70)
print(f"\n{'Metric':<12} {'Base Model':>12} {'Fine-tuned':>12} {'Improvement':>12}")
print("-" * 50)

for key in ["rouge1", "rouge2", "rougeL"]:
    diff = ft_scores[key] - base_scores[key]
    print(f"{key:<12} {base_scores[key]:>12.4f} {ft_scores[key]:>12.4f} {diff:>+12.4f}")