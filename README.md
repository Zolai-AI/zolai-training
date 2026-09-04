# zolai-training — LoRA/QLoRA fine-tuning + GGUF export

Fine-tuning small Zolai LLMs (0.5B–3B) via PEFT/TRL, adapter merge + GGUF export.

## What's here
- `scripts/training/` — LoRA (FP16 r=16) / QLoRA (NF4 r=8) sessions
- `kaggle_dataset/` — Kaggle metadata + `train_kaggle_t4x2.py` (heavy `.jsonl` mirrored, not committed)

## Hardware
- LoRA FP16 on T4x2; QLoRA NF4 for 3B; target ~5.1M-pair dataset.

## Note
Models/adapters publish to HF Hub (`peterpausianlian/zolai-*`) — credentials env-only.
