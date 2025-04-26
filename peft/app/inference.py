print('shshh')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
print('###########################################dd')
# "C:\Users\Shubham Agrahari\Desktop\LORA\app"
# MODEL_PATH = "C:/Users/Shubham Agrahari/Desktop/peft/app/model/saved_model"
# MODEL_PATH = "/app/model/saved_model"
import os
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/saved_model")  # fallback to old path

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
print('s')
def summarize(dialogue: str):
    input_text = "Summarize: " + dialogue
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=128)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
