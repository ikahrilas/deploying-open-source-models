import warnings
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the DialogSum dataset from the Hub
full_dataset = load_dataset("knkarthick/dialogsum")

# Take a small subset so training remains practical on CPU
train_data = full_dataset["train"].select(range(50))
test_data = full_dataset["test"].select(range(10))

print(f"Training examples: {len(train_data)}")
print(f"Test examples: {len(test_data)}")
print(f"\nSample dialogue:\n{train_data[0]['dialogue']}")
print(f"\nSample summary:\n{train_data[0]['summary']}")