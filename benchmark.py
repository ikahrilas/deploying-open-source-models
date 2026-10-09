import os
import time
import warnings
import logging
import torch
import numpy as np
import evaluate
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from openai import OpenAI
from anthropic import Anthropic

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


# --- Load test data ---

dataset = load_dataset("knkarthick/dialogsum")
test_data = dataset["test"].shuffle(seed=42).select(range(50))
test_docs = test_data["dialogue"]
test_refs = test_data["summary"]


# --- Load the self-hosted model ---

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small",
    dtype=torch.float16 if device == "cuda" else torch.float32
)
base_model = base_model.to(device)
base_model.eval()

# If you have a fine-tuned adapter from the previous lesson, load it:
# peft_model = PeftModel.from_pretrained(base_model, "./flan-t5-summarizer-adapter")
# peft_model.eval()
# model = peft_model
model = base_model

# FLAN-T5 uses an instruction prefix because it's an instruction-tuned model.
# The closed APIs use a different, more detailed prompt because they're
# general-purpose chat models that respond better to natural language instructions.
FLAN_PREFIX = "Summarize the following dialogue:\n\n"

CLOSED_API_PROMPT = """Summarize the following conversation in 2-3 sentences. Be concise and specific.

Conversation:
{text}

Summary:"""


# --- Wrap each candidate in the same function signature ---

def summarize_local(text):
    prompt = FLAN_PREFIX + text
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        ids = model.generate(input_ids=inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(ids[0], skip_special_tokens=True)


openai_client = OpenAI()
anthropic_client = Anthropic()


def summarize_luna(text):
    response = openai_client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": CLOSED_API_PROMPT.format(text=text)}],
        max_tokens=150,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()


def summarize_claude_haiku(text):
    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{"role": "user", "content": CLOSED_API_PROMPT.format(text=text)}]
    )
    return response.content[0].text.strip()


# --- Benchmark runner ---

rouge = evaluate.load("rouge")


def run_benchmark(name, summarize_fn, docs, refs):
    predictions = []
    latencies = []

    for doc in docs:
        try:
            start = time.perf_counter()
            summary = summarize_fn(doc)
            elapsed = time.perf_counter() - start
            predictions.append(summary)
            latencies.append(elapsed)
        except Exception as e:
            print(f"  {name} failed on a document, skipping entire model: {e}")
            return None

    scores = rouge.compute(predictions=predictions, references=refs)
    scores["name"] = name
    scores["latency_p50"] = float(np.percentile(latencies, 50))
    scores["latency_p95"] = float(np.percentile(latencies, 95))
    return scores


# --- Run all three ---
# In a production benchmark you'd parallelize the API calls to speed things up.
# Here we run them sequentially for simplicity.

print("Running benchmarks (this may take a few minutes)...\n")

results = []

result = run_benchmark("FLAN-T5 (self-hosted)", summarize_local, test_docs, test_refs)
if result:
    results.append(result)
    print("  FLAN-T5 done.")

result = run_benchmark("GPT-5.6 Luna", summarize_luna, test_docs, test_refs)
if result:
    results.append(result)
    print("  GPT-5.6 Luna done.")
else:
    print("  GPT-5.6 Luna skipped.")

result = run_benchmark("Claude Haiku", summarize_claude_haiku, test_docs, test_refs)
if result:
    results.append(result)
    print("  Claude Haiku done.")
else:
    print("  Claude Haiku skipped.")


# --- Print results ---

print("\n" + "=" * 70)
print(f"{'Model':<25} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'p50 (s)':>8} {'p95 (s)':>8}")
print("=" * 70)

for r in results:
    print(f"{r['name']:<25} {r['rouge1']:>8.4f} {r['rouge2']:>8.4f} {r['rougeL']:>8.4f} {r['latency_p50']:>8.2f} {r['latency_p95']:>8.2f}")