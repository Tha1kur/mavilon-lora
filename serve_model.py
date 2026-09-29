# serve_model.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch
import os
from fastapi.responses import FileResponse

app = FastAPI(title="Mavilon API")

# Allow requests from local UI (adjust if hosting elsewhere)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001", "http://127.0.0.1:8001", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = "google/flan-t5-small"
ADAPTER_LOCAL = os.getenv("ADAPTER_LOCAL", "lora-output/lora_adapter_fixed")
EVAL_CSV_PATH = os.getenv("EVAL_CSV_PATH", "/mnt/data/mavilon_eval.csv")  # optional

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
print("Loading base model...")
base = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device)
print("Loading PEFT adapter from:", ADAPTER_LOCAL)
model = PeftModel.from_pretrained(base, ADAPTER_LOCAL).to(device)
model.eval()

class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    do_sample: bool = False
    num_beams: int = 4

@app.post("/generate")
def generate(req: GenRequest):
    if not req.prompt or not req.prompt.strip():
        return {"error": "Empty prompt"}
    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=req.do_sample,
            num_beams=req.num_beams,
            early_stopping=True,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return {"reply": text}

# Optional: download evaluation CSV
@app.get("/download_eval")
def download_eval():
    if os.path.exists(EVAL_CSV_PATH):
        return FileResponse(EVAL_CSV_PATH, filename="mavilon_eval.csv")
    return {"error": "eval file not found"}
