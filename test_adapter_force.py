import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

MODEL="google/flan-t5-small"
ADAPTER="lora-output/lora_adapter"

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device)
# ensure config decoder start
print("decoder_start_token_id:", getattr(base.config, "decoder_start_token_id", None))
print("eos_token_id:", getattr(base.config, "eos_token_id", None))
print("pad_token_id:", getattr(base.config, "pad_token_id", None))

model = PeftModel.from_pretrained(base, ADAPTER).to(device)

prompt = "Summarize: The cat sat on the mat."
inputs = tok(prompt, return_tensors="pt").to(device)

out = model.generate(
    **inputs,
    max_new_tokens=40,
    num_beams=4,
    do_sample=False,
    forced_bos_token_id=getattr(base.config, "decoder_start_token_id", None),
    decoder_start_token_id=getattr(base.config, "decoder_start_token_id", None),
    pad_token_id=tok.pad_token_id,
    eos_token_id=tok.eos_token_id,
)
print("Generated IDs:", out[0].tolist())
print("Decoded:", tok.decode(out[0], skip_special_tokens=True))
