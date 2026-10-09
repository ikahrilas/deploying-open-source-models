import warnings
import logging
import torch
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM

from peft import LoraConfig, TaskType, get_peft_model

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)


MODEL_NAME = "google/flan-t5-small"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none"
)

peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()

# Show which layers have adapters
print("\nAdapted layers:")
for name, param in peft_model.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.shape}")