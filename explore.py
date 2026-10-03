import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    provider="hf-inference", # use huggingface inference API
    token=os.environ.get("HF_TOKEN")
)

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

result = client.summarization(ticket, model="facebook/bart-large-cnn")
print("BART summary:")
print(result.summary_text)