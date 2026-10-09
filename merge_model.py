import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
peft_model = PeftModel.from_pretrained(base_model, "./flan-t5-summarizer-adapter")

merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("./flan-t5-summarizer-merged")
tokenizer.save_pretrained("./flan-t5-summarizer-merged")

print("Merged model saved to ./flan-t5-summarizer-merged/")