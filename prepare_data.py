import warnings
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_NAME = "google/flan-t5-small"

# Load dataset
full_dataset = load_dataset("knkarthick/dialogsum")
train_data = full_dataset["train"].select(range(50))
test_data = full_dataset["test"].select(range(10))

# Tokenize
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

print(f"Tokenized training examples: {len(tokenized_train)}")
print(f"Tokenized test examples: {len(tokenized_test)}")
print(f"\nFeatures: {list(tokenized_train.features.keys())}")
print(f"First input length: {len(tokenized_train[0]['input_ids'])} tokens")
print(f"First label length: {len(tokenized_train[0]['labels'])} tokens")