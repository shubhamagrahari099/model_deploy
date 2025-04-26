from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
# from config import MODEL_NAME, OUTPUT_DIR, NUM_EPOCHS, BATCH_SIZE
import os
import torch



MODEL_NAME = "google/flan-t5-base"
OUTPUT_DIR = "app/model/saved_model"
NUM_EPOCHS = 5
BATCH_SIZE = 3

# Create model directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

peft_config = LoraConfig(
    r=8, lora_alpha=16,
    task_type=TaskType.SEQ_2_SEQ_LM,
    lora_dropout=0.1, bias="none"
)
model = get_peft_model(model, peft_config)

dataset = load_dataset("knkarthick/dialogsum")
train_data, val_data = dataset["train"], dataset["validation"]
train_data = train_data.select(range(500))
val_data = val_data.select(range(50))



def preprocess(batch):
    inputs = tokenizer(
        ["Summarize: " + d for d in batch["dialogue"]],
        max_length=384,
        truncation=True,
        padding="max_length"
    )
    labels = tokenizer(
        batch["summary"],
        max_length=64,
        truncation=True,
        padding="max_length"
    )

    inputs["labels"] = labels["input_ids"]
    return inputs



train_dataset = train_data.map(preprocess, batched=True)
val_dataset = val_data.map(preprocess, batched=True)


data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=NUM_EPOCHS,
    save_total_limit=1,
    logging_dir="./logs",
    fp16=True,
    report_to="none",
    gradient_accumulation_steps=2 
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# model.save_pretrained(OUTPUT_DIR)
# tokenizer.save_pretrained(OUTPUT_DIR)
