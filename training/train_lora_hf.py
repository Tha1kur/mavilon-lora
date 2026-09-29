from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import os

MODEL = "google/flan-t5-small"
TRAIN_FILE = "data/train.jsonl"
VALID_FILE = "data/valid.jsonl"
OUTPUT_DIR = "lora-output"

EPOCHS = 3
BATCH_SIZE = 8
LR = 2e-4
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 128

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q","v"],
    lora_dropout=0.05,
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
ds = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VALID_FILE})

def preprocess(example):
    model_inputs = tokenizer(example["input"], max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")
    labels = tokenizer(example["output"], max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = ds.map(preprocess, remove_columns=ds["train"].column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# FIXED TrainingArguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=True,  
    save_strategy="epoch",
    logging_steps=50,
)

# Trainer with evaluation strategy here
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
    print("Training finished. Adapter saved.")
