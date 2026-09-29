# save as debug_smoke.py and run: python3 debug_smoke.py
import torch, os, time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

MODEL = "google/flan-t5-small"
ADAPTER_DIR = "lora-output/lora_adapter"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

# Check adapter folder
if not os.path.isdir(ADAPTER_DIR):
    raise SystemExit("Adapter folder not found at: " + ADAPTER_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device)
print("Base model loaded.")

model = PeftModel.from_pretrained(base, ADAPTER_DIR).to(device)
print("Adapter loaded.")

prompt = "Summarize: The cat sat on the mat."
inputs = tokenizer(prompt, return_tensors="pt").to(device)
print("Input IDs:", inputs["input_ids"][0].tolist())

# Generate with detailed outputs
out = model.generate(**inputs,
                     max_new_tokens=64,
                     do_sample=False,
                     num_beams=4,
                     early_stopping=True,
                     return_dict_in_generate=True,
                     output_scores=True)

sequences = out.sequences
scores = out.scores if hasattr(out, 'scores') else None

print("Generated token IDs:", sequences[0].tolist())
if scores:
    print("Number of score tensors:", len(scores))
    
decoded = tokenizer.decode(sequences[0], skip_special_tokens=False)
decoded_clean = tokenizer.decode(sequences[0], skip_special_tokens=True)
print("Decoded (no skip):", repr(decoded))
print("Decoded (skip_special_tokens=True):", repr(decoded_clean))