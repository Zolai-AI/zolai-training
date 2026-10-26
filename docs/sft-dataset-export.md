# SFT dataset export contract — `scripts/export_zolai_dataset.py`

Reads `zolai.db` **READ-ONLY** (`file:...?mode=ro` URI — the exporter refuses to run
if the db file's mtime/size change during the run) and emits a quality-filtered
Zolai instruction dataset for supervised fine-tuning.

Used by the pcore-kaggle Qwen3-0.6B LoRA kernel
(`notebooks/finetune-qwen3-0.6b/finetune-qwen3-0.6b-lora.ipynb` in
[P-Core-System/pcore-kaggle](https://github.com/P-Core-System/pcore-kaggle)); the
kernelside contract is documented in that repo's `docs/dataset-format.md`.

## Run

```bash
python3 scripts/export_zolai_dataset.py \
    --db ../data/zolai.db \
    --out-dir ../data/exports/zolai-sft-v1 \
    [--seed 42] [--eval-per-direction 400] [--min-confidence 0.9] \
    [--max-train-rows N]
```

Defaults: db `../data/zolai.db`, out dir `../data/exports/zolai-sft-v1`, seed 42.
Exit codes: `0` success, `2` error (missing db / sqlite error / db changed).
One full export (no cap) ≈ 300k train rows ≈ 90 MB; it serves both local full runs
and capped Kaggle sessions (the notebook caps independently via `TRAIN_MAX_ROWS`).
Generated data files are git-ignored (`data/`, `*.jsonl`) — commit code + docs only.

## Artifacts

| File | Contents |
|---|---|
| `train.jsonl` | instruction rows for SFT |
| `eval.jsonl` | scored held-out rows (`eval_cases` + a dedup-guarded `translations` slice) |
| `eval_probes.jsonl` | prompt-only generation probes (`eval_cases` kind `zvs`, `output=""`) — never scored |
| `dataset_manifest.json` | counts, per-source drop audit, config, sha256 checksums |

### Row schema (JSONL, UTF-8, `ensure_ascii=False`)

```json
{"instruction": "Translate to Zolai: The sun rises.", "input": "", "output": "Ni a suak hi.", "task": "translation_en_zo", "source": "translations:GEN 1:14"}
```

| Field | Rules |
|---|---|
| `instruction` | non-empty after strip, 3–500 chars |
| `input` | may be `""`; ≤ 2,000 chars |
| `output` | non-empty after strip, 3–2,000 chars (`""` only in probes; wiki capped at 1,500 pre-emit) |
| `task` | one of the task labels in the mix table below |
| `source` | `{table}:{row-ref}` provenance |

## Task mix (source → emitted task)

| Source table | `task` label(s) | Instruction shape |
|---|---|---|
| `training_corpus_qwen3_import` | `{task}:{pattern}` e.g. `translation_en_zo:question` | user message → assistant message (system prompt NOT emitted — its rules are enforced by the ZVS filter) |
| `corpus_parallel_pairs` | `translation_zo_en` | `Translate to English: {source}` → `target` |
| `translations` (zo→en) | `translation_zo_en` | `Translate to English: {source}` → `target` |
| `translations` (en→zo) | `translation_en_zo` | `Translate to Zolai: {source}` → `target` |
| `training_exercises` legacy | `exercise_{exercise_type}` | `Translate to English: {zolai}` → `english` (ratio-guarded) |
| `training_exercises` error_correction | `error_correction` | input = corrupted `{english}`, output = clean `{zolai}` |
| `training_exercises` translate_zo_en/vocab | `word_gloss` | `What is the English meaning of the Zolai word '{zolai}'?` → `{english}` |
| `proverbs` | `proverb_translation` + `proverb_translation_rev` | two rows per proverb, both directions |
| `dictionary_en_zo` | `lexical_gloss_en_zo` | `What is the Zolai word for '{headword}'?` → `{translations_clean}` |
| `vocabulary` | `lexical_gloss_zo_en` | `What does the Zolai word '{headword}' mean in English?` → `{english}` |
| `wiki_lessons` | `wiki_{lesson_type}` | `Explain this Zolai {lesson_type} topic in simple terms.` input=`{title}` output=`{content_summary}` (≤1,500 chars) |
| `eval_cases` scored | `eval_qa` / `eval_gloss` / `eval_correction` / `eval_translation` | per payload variant — see Eval design |
| `eval_cases` kind `zvs` | `eval_zvs_probe` | bare Zolai sentence, `output=""` (probes file) |

## Quality filters (every drop counted per source in the manifest)

1. **Direction exclusion (SQL)** — `translations`: only `zo_to_en`/`en_to_zo`;
   the `en→my` Myanmar direction (150,965 rows) never leaves SQLite.
2. **Source junk exclusion (SQL)** — `wiki_lessons`: `source_file NOT LIKE '%site-packages%'`;
   `dictionary_en_zo`/`eval_cases`: active rows only.
3. **Confidence (SQL)** — `translations`: `CAST(confidence AS REAL) >= 0.9`.
4. **Length guard** — every text stripped; instruction 3–500, input ≤ 2,000,
   output 3–2,000; wiki output capped to 1,500 at emit.
5. **Control/non-printable chars** — rows with C0 controls (other than `\n`/`\t`)
   are dropped, not stripped (the `vocabulary` table is polluted).
6. **ZVS-2018 orthography** — word-boundary, case-insensitive regex over the
   **Zolai-side text only** (output for en→zo rows, source text for zo→en rows):
   `\b(pathian|ram|fapa|bawipa|siangpahrang|cu|cun|suah|zalenna|nunnak)\b` — the
   exact forbidden-form list from the corpus system prompt. Violations are
   dropped so bad orthography never reaches training.
7. **Length-ratio guard** — legacy `training_exercises` pairs only: drop if
   `len(zolai)/len(english)` is outside 0.25–4.0.
8. **Dedup** — global over train: sha256 over NFKC-normalized + casefolded +
   whitespace-collapsed `(task, instruction, output)`; `translations.content_hash`
   as secondary key within that source. Eval has its own registry.
9. **Eval leakage guard** — every eval row's normalized `(instruction, output)`
   pair is registered **before** train emission; matching train rows are dropped
   and counted `leakage`. The held-out `translations` slice is sampled first.
10. **Provenance** — every surviving row carries `task` + `source`.

## Eval design

`eval.jsonl` = two streams:

1. **`eval_cases`** (all `is_active=1`), mapped per payload variant:
   - `{question, hyp, answer}` → `eval_qa` (`hyp` is a prior model output — ignored)
   - `{input, expected}` (vocabulary) → `eval_gloss` (`Translate to English: {input}`)
   - `{input, expected_correction}` (zvs-compliance) → `eval_correction`
   - `translation` `{source, ref}` → `eval_translation` (`Translate to Zolai: {source}`)
   - kind `zvs` `{text}` → **not scored** — goes to `eval_probes.jsonl` with
     `output=""` (no reference exists; inventing a task would be dishonest).
2. **Held-out `translations` slice** — 400/direction (default), conf ≥ 0.9,
   seeded sample executed before train registration and excluded from the train
   stream by row id.

## Publishing to Kaggle (owner step)

The pcore-kaggle notebook attaches the exported dir as Kaggle Dataset
`peterpausianlian/zolai-sft-dataset`. To (re)publish, from the export dir:

```bash
kaggle datasets create -p ../data/exports/zolai-sft-dataset \
    --title "Zolai SFT Dataset v1" --cc0-1.0
# or update an existing dataset:
kaggle datasets version -p ../data/exports/zolai-sft-dataset \
    --dir-mode skip
```

with `dataset-metadata.json`:

```json
{"id": "zolai-sft-dataset", "ref": "peterpausianlian/zolai-sft-dataset",
 "title": "Zolai SFT Dataset v1", "licenses": [{"name": "CC0-1.0"}]}
```

CC0-1.0 matches prior Zolai dataset releases. The generated JSONL stays out of git.

## Tests

`tests/test_export_zolai_dataset.py` — hermetic, synthetic sqlite fixture in
`tmp_path` (never the real db). Covers: CLI exit codes, `en→my` exclusion, ZVS
drops, control chars, length guard + wiki cap, ratio guard, dedup, the
eval-leakage guard, all `eval_cases` mappings, probes-only file, manifest schema +
checksums, per-source accounting reconciliation (`read == emitted + sum(dropped)`),
determinism (same seed → identical bytes), db-bytes-unchanged proof, and the
`--max-train-rows` cap.

```bash
python3 -m pytest tests/test_export_zolai_dataset.py -q
ruff check scripts/export_zolai_dataset.py tests/test_export_zolai_dataset.py
```
