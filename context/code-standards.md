# Code Standards — Zolai Training

- **Languages:** Python (PyTorch/PEFT/TRL), notebooks, bash.
- Notebooks are source of record for experiments; keep outputs deterministic.
- Fine-tuning is LoRA/QLoRA on small models (0.5B–3B) then GGUF export.
- Git-ignore model checkpoints, adapters, and large artifacts.
