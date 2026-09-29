import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)
tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)
inp = tok("Summarize: The cat sat on the mat.", return_tensors="pt").to(device)
out = base.generate(**inp, max_new_tokens=40)
print("Base output:", tok.decode(out[0], skip_special_tokens=True))
