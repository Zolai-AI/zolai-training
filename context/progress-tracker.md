# Progress Tracker

## 2026-09-04 — Setup baseline
- Repo connected to `Zolai-AI/zolai-training`.
- Received `kaggle_dataset/` + `notebooks/` from monorepo distribution.
- Training strategy: RAG/embeddings-first; fine-tuning only for generation, GGUF export.

## 2026-09-04
- Fixed ruff lint errors across all scripts (ruff.toml with per-file-ignores)

## 2026-10-26 — SFT dataset exporter (zolai.db -> instruction JSONL)

- NEW `scripts/export_zolai_dataset.py` — importable + CLI exporter reading `zolai.db`
  READ-ONLY (`file:...?mode=ro`; refuses to run if db mtime/size change) and emitting
  `train.jsonl` + `eval.jsonl` + `eval_probes.jsonl` + `dataset_manifest.json`
  (default out dir `../data/exports/zolai-sft-v1/`, git-ignored, never committed).
- Full mix, quality-filtered: qwen3 chat corpus, parallel pairs, translations
  (zo↔en only — `en→my` excluded at SQL level), legacy + error-correction exercises,
  proverbs (both directions), dictionary/vocabulary glosses, wiki lessons. Filters:
  ZVS-2018 orthography (forbidden-form regex on the Zolai side only), length, control
  chars, ratio guard, global dedup, and an eval-leakage guard (eval registered before
  train emission; held-out translations slice sampled first). Every drop counted per
  source in the manifest audit.
- Eval: `eval_cases` mapped per payload variant (qa/gloss/correction/translation)
  + 400/direction held-out `translations` slice; kind `zvs` rows go to
  `eval_probes.jsonl` as prompt-only probes (`output=""`, never scored).
- NEW `tests/test_export_zolai_dataset.py` — 24 hermetic tests on a synthetic sqlite
  fixture in `tmp_path` (never the real db): CLI exit codes, en→my exclusion, ZVS/
  control/length/ratio drops, dedup, leakage, eval mappings, probes, manifest schema +
  checksums, per-source accounting reconciliation, determinism, db-bytes-unchanged,
  `--max-train-rows` cap. `ruff check` clean.
- Docs: `docs/sft-dataset-export.md` (contract + filters + eval design + Kaggle
  publish recipe) and `docs/sft-dataset-manifest.example.json`.
- First full export (2026-10-26): **train 294,817 / eval 994 / probes 52**; `en→my`
  emitted = 0 (150,965 excluded at SQL level, counted); 954 leakage drops; checksums
  present; db mtime+size byte-identical before vs after.
- Consumed by P-Core-System/pcore-kaggle `finetune-qwen3-0.6b-lora.ipynb`
  (resolution order: `$ZOLAI_DATASET_DIR` → `/kaggle/input/*/train.jsonl` →
  `WORK_DIR/zolai-dataset/`; synthetic fallback retained for standalone runs).
- Next (owner): publish the export as Kaggle Dataset `peterpausianlian/zolai-sft-dataset`
  (CC0-1.0), then push the qwen3 kernel and Run All on T4 x2.
