import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.utils import logging

load_dotenv()

# Hide Transformers warnings while keeping actual errors visible
logging.set_verbosity_error()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small",
    dtype=torch.float16 if device == "cuda" else torch.float32
)

model = model.to(device)
model.eval()

PREFIX = "Summarize the following dialogue:\n\n"

tickets = [
    "Agent: Thank you for calling support. How can I help?\nCustomer: I was charged $247.50 twice on my March invoice for the Enterprise plan. Account 4829 under Jane Doe.\nAgent: I see the duplicate charge. I'll initiate a refund for $247.50. It should appear within 5-7 business days.\nCustomer: That works, thank you.",
    "Agent: How can I assist you today?\nCustomer: The CSV export on the analytics dashboard has been returning empty files since Tuesday. I'm John Smith, account 7201.\nAgent: I've reproduced the issue. It affects all Enterprise users. I'm escalating to engineering as a priority bug. Your ticket number is ENG-4412, expect an update within 48 hours.\nCustomer: Okay, thanks for looking into it.",
    "Agent: Hi, how can I help?\nCustomer: I'd like to upgrade from Starter to Professional plan. Account 3156, Maria Garcia.\nAgent: I can do that now. The upgrade takes effect immediately and the prorated difference of $34.50 will appear on your next invoice.\nCustomer: Great. Can I also add two team members?\nAgent: Sure, let me walk you through the team management settings. Both members are now added.\nCustomer: Perfect, thanks for the help.",
]


def summarize_batch(texts, batch_size=8):
    summaries = []

    # Add the summarization instruction to each dialogue
    prompts = [PREFIX + text for text in texts]

    # Process the dialogues in batches
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]

        # Tokenize the batch and move tensors to the selected device
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)

        # Generate summaries without calculating gradients
        with torch.no_grad():
            summary_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,
                min_length=15,
                num_beams=4,
                early_stopping=True,
            )

        # Convert generated token IDs back into text
        decoded = tokenizer.batch_decode(
            summary_ids,
            skip_special_tokens=True
        )

        summaries.extend(decoded)

    return summaries


results = summarize_batch(tickets)

for i, summary in enumerate(results):
    print(f"Ticket {i + 1}: {summary}")
    print()