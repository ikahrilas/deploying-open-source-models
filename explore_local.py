from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.utils import logging

load_dotenv()

# Hide Transformers warnings, but still show errors
logging.set_verbosity_error()

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

ticket = """
Agent: Thank you for calling support, how can I help you today?
Customer: Hi, I'm calling about my March invoice. I was charged $247.50 twice
for the Enterprise plan. I only have one subscription.
Agent: I'm sorry about that. Let me look into your account. Can I have your
account number?
Customer: It's 4829, under Jane Doe.
Agent: Thank you, Jane. I can see the duplicate charge. I'll initiate a refund
for $247.50 right away. It should appear within 5-7 business days.
Customer: That works. Thank you for the quick resolution.
Agent: You're welcome. Is there anything else I can help with?
Customer: No, that's all. Thanks again.
"""

prompt = "Summarize the following dialogue:\n\n" + ticket

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

summary_ids = model.generate(
    input_ids=inputs["input_ids"],
    max_length=150,
    min_length=15,
    num_beams=4,
    early_stopping=True,
)

summary = tokenizer.decode(
    summary_ids[0],
    skip_special_tokens=True
)

print("FLAN-T5 summary:")
print(summary)