<!-- Auto-generated guidance for AI coding agents working on the Mavilon repo -->
# Mavilon — Copilot Instructions

Purpose: help AI coding agents be productive quickly by documenting architecture, workflows, and repo-specific patterns.

- **Big picture:** This repo fine-tunes a Seq2Seq model (google/flan-t5-small) with LoRA adapters and provides small utilities to train, evaluate, and serve the adapter. Training produces adapter artifacts under `lora-output/` and a packaged adapter in `mavilon-lora/`.

- **Key files & components:**
  - `training/train_lora_hf.py` — uses Hugging Face `Trainer`, `datasets`, and `peft.LoraConfig` to produce adapters in `lora-output/`.
  - `serve_model.py` — FastAPI server that loads base model + PEFT adapter (env var `ADAPTER_LOCAL` can override path) and exposes `/generate` and `/download_eval` endpoints.
  - `evaluate.py` — runs a small set of prompts and writes `mavilon_eval.csv` in CWD (path printed).
  - `scripts/make_dataset.py` — produces `data/train.jsonl` and `data/valid.jsonl` sample files; follow its JSONL shape (`{ "input":..., "output":... }`).
  - `smoke_test.py`, `debug_smoke.py`, `test_*.py` — lightweight verification scripts used to validate model + adapter loading and generation. These are runnable directly with Python.

- **Data & artifacts:**
  - Training data: `data/train.jsonl`, `data/valid.jsonl` (JSONL with `input` and `output` fields).
  - Adapters & checkpoints: `lora-output/checkpoint-*`, `lora-output/lora_adapter`, and `mavilon-lora/` (packaged adapter). Treat these as large artifacts.

- **Environment & runtime notes:**
  - Default model: `google/flan-t5-small` — code uses Transformers + PEFT.
  - Device auto-detection: scripts prefer `mps` (Apple Silicon) when available, else CPU. Check `torch.backends.mps.is_available()` in the scripts.
  - `serve_model.py` reads `ADAPTER_LOCAL` and `EVAL_CSV_PATH` from env vars. Example: `ADAPTER_LOCAL=lora-output/lora_adapter_fixed`.

- **Common commands (examples):**
  - Create dataset: `python scripts/make_dataset.py`
  - Train adapter (quick): `python training/train_lora_hf.py` (tunes LoRA and writes under `lora-output/`)
  - Run smoke test: `python smoke_test.py` or `python debug_smoke.py`
  - Evaluate and save CSV: `python evaluate.py` (writes `mavilon_eval.csv`)
  - Serve the model: `python serve_model.py` then POST to `http://localhost:8000/generate`

- **Testing & CI expectations:**
  - There is no formal pytest config; tests are simple scripts (`test_base.py`, `test_adapter_force.py`). Run them directly or with `python -m pytest` if you add pytest to CI.

- **Project-specific conventions / important patterns:**
  - LoRA adapter workflow: training uses PEFT `get_peft_model` then `model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))`. Consumers call `PeftModel.from_pretrained(base, ADAPTER_DIR)`.
  - Generation parameters: scripts consistently use `max_new_tokens`, `num_beams`, `do_sample=False`, and often `early_stopping=True`.
  - Tokenizer & base model live on HF (`AutoTokenizer.from_pretrained(MODEL)` and `AutoModelForSeq2SeqLM.from_pretrained(MODEL)`); adapters are layered on top.
  - Check `decoder_start_token_id`, `pad_token_id`, and `eos_token_id` in `test_adapter_force.py` — some generation calls set forced/decoder token ids explicitly.

- **Integration & dependencies:**
  - Primary libs: `transformers`, `datasets`, `peft`, `torch`, `fastapi`, `pydantic`.
  - Install the repo dependencies with: `python -m pip install -r requirements.txt` (use a virtualenv).

- **When editing code, be mindful of:**
  - Large-model downloads: tests and training download HF weights; CI or developers should provide network access or cached models.
  - Artifact paths: many scripts assume `lora-output/...` relative paths; use the `ADAPTER_LOCAL` override when changing runtime locations.
  - Device differences: MPS vs CPU vs CUDA — keep device selection code unchanged unless adding explicit CUDA support.

If anything here is unclear or you want broader coverage (example: suggested CI job steps, or adding a `Makefile`), tell me which area to expand and I'll iterate.
