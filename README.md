# Mavilon LoRA

**Status: work in progress.** A small FLAN-T5 LoRA fine-tuning, evaluation, and FastAPI serving experiment.

## Setup

Create a Python virtual environment and install `requirements.txt`. The training scripts additionally import `datasets` and use the Transformers training stack; a complete, tested training dependency set remains to be established.

Serve from the repository root with:

```sh
EVAL_CSV_PATH=mavilon_eval.csv python -m uvicorn serve_model:app --host 127.0.0.1 --port 8000
```

The server loads `google/flan-t5-small` and defaults to `lora-output/lora_adapter_fixed`. Initial model downloads require network access. `ADAPTER_LOCAL` selects a different adapter. Training entry points live in `training/` and may overwrite local artifacts.

## Baseline and limitations

Saved final adapter/tokenizer directories and small training/evaluation data are included. Intermediate checkpoint directories, optimizer state, logs, and the downloaded evaluation file remain local and in the private recovery backup.

The train/validation files share at least one identical example, so validation independence needs review. The requirements are unpinned, model-loading scripts have startup side effects, and the default evaluation path in the server assumes another filesystem; the command above overrides that path.

The packaged adapter's inherited Git LFS attributes were replaced with ordinary binary-file handling because the included files are small enough for normal Git. Model bytes are unchanged.

Source was preserved during the 2026-09-29 storage cleanup. Dependencies and generated caches were removed. Runtime behavior, model accuracy, and end-to-end operation were not revalidated. Any README.pre-cleanup.md is historical documentation; its claims are not fresh test results.

See [ROADMAP.md](ROADMAP.md) for the next improvements.
