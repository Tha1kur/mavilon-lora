from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import os
import torch

os.environ["WANDB_DISABLED"] = "true"

MODEL = "google/flan-t5-small"
TRAIN_FILE = "data/train.jsonl"
VALID_FILE = "data/valid.jsonl"
OUTPUT_DIR = "lora-output"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
model.config.use_cache = False  # critical fix for LoRA training

# LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)

# Dataset
ds = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VALID_FILE})

def preprocess(example):
    model_inputs = tokenizer(
        example["input"], 
        max_length=256, 
        truncation=True, 
        padding="max_length"
    )
    labels = tokenizer(
        example["output"], 
        max_length=128, 
        truncation=True, 
        padding="max_length"
    )
    
    # IMPORTANT: mask pad tokens with -100 so loss works
    labels_ids = labels["input_ids"]
    labels["input_ids"] = [-100 if id == tokenizer.pad_token_id else id for id in labels_ids]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = ds.map(preprocess, remove_columns=ds["train"].column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=4,
    learning_rate=2e-4,
    logging_steps=30,
    save_strategy="epoch",
    report_to="none",
    fp16=False,     # important for MPS
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter_fixed"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter_fixed"))
    print("Training complete and fixed adapter saved.")
