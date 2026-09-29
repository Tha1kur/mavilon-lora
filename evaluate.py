# evaluate.py
import csv
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import os

MODEL = "google/flan-t5-small"
ADAPTER = "lora-output/lora_adapter_fixed"   # or "Mavilon/mavilon-lora"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device)
model = PeftModel.from_pretrained(base, ADAPTER).to(device)

prompts = [
    "What is artificial intelligence?",
    "Explain machine learning to a 10 year old.",
    "Write a short motivational quote.",
    "Summarize: The cat sat on the mat.",
    "Q: What is 12 times 8? A:",
    "Give me three uses of AI in healthcare.",
    "Translate to Hindi: Hello, how are you?",
    "Write a short poem about rain.",
    "Explain the difference between supervised and unsupervised learning.",
    "How do neural networks learn?"
]

out_rows = []
for p in prompts:
    inputs = tokenizer(p, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=120, num_beams=4, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print("\nPROMPT:", p, "\nANSWER:", text)
    out_rows.append({"prompt": p, "answer": text})

# write CSV to current working directory (explicit path)
csv_path = os.path.join(os.getcwd(), "mavilon_eval.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["prompt", "answer"])
    writer.writeheader()
    writer.writerows(out_rows)

print(f"\nSaved outputs to: {csv_path}")
# print a small preview
print("\nPreview (first 3 rows):")
for r in out_rows[:3]:
    print("-", r["prompt"], "=>", r["answer"][:120].replace("\n", " "))