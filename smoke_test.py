import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

MODEL = "google/flan-t5-small"
ADAPTER = "lora-output/lora_adapter_fixed"

# Choose device (MPS for Apple Silicon)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device)

# Load LoRA adapter
model = PeftModel.from_pretrained(base, ADAPTER).to(device)

# Test prompt
prompt = "Explain artificial intelligence in simple words."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate output
outputs = model.generate(
    **inputs,
    max_new_tokens=80,
    num_beams=4,
    do_sample=False
)

print("\n--- MODEL OUTPUT ---")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
