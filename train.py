import os
import warnings
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from peft import LoraConfig, TaskType, get_peft_model

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


MODEL_NAME = "google/flan-t5-small"

# Load the DialogSum dataset from the Hub
full_dataset = load_dataset("knkarthick/dialogsum")

# Take a small subset so training remains practical on CPU
train_data = full_dataset["train"].select(range(50))
test_data = full_dataset["test"].select(range(10))

print(f"Training examples: {len(train_data)}")
print(f"Test examples: {len(test_data)}")
print(f"\nSample dialogue:\n{train_data[0]['dialogue']}")
print(f"\nSample summary:\n{train_data[0]['summary']}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

PREFIX = "Summarize the following dialogue:\n\n"

def tokenize(examples):
    prompts = [PREFIX + d for d in examples["dialogue"]]
    model_inputs = tokenizer(
        prompts,
        max_length=128,
        truncation=True
    )
    labels = tokenizer(
        text_target=examples["summary"],
        max_length=64,
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_data.map(
    tokenize,
    batched=True,
    remove_columns=train_data.column_names
)
tokenized_test = test_data.map(
    tokenize,
    batched=True,
    remove_columns=test_data.column_names
)
tokenized_train.set_format("torch")
tokenized_test.set_format("torch")

print(f"\nTokenized training examples: {len(tokenized_train)}")
print(f"Tokenized test examples: {len(tokenized_test)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

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

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

training_args = Seq2SeqTrainingArguments(
    output_dir="./flan-t5-finetuned",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=5,
    learning_rate=3e-4,
    fp16=(device == "cuda"),
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_steps=5,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=peft_model,
    padding=True
)

trainer = Seq2SeqTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    processing_class=tokenizer,
    data_collator=data_collator,
)

print("\nStarting training...")
trainer.train()
print("Training complete!")

# Save the best adapter
peft_model.save_pretrained("./flan-t5-summarizer-adapter")
print("Adapter saved to ./flan-t5-summarizer-adapter/")