#!/usr/bin/env python3
"""Export a quality-filtered Zolai SFT instruction dataset from ``zolai.db``.

The database is opened READ-ONLY (``file:...?mode=ro`` URI) and is never mutated.
The exporter emits four artifacts into an output directory:

- ``train.jsonl``          — instruction rows for supervised fine-tuning
- ``eval.jsonl``           — scored held-out rows (``eval_cases`` + translations slice)
- ``eval_probes.jsonl``    — prompt-only generation probes (``eval_cases`` kind ``zvs``)
- ``dataset_manifest.json`` — counts, per-source drop audit, config, checksums

Row contract (one JSON object per line, ``ensure_ascii=False``, UTF-8)::

    {"instruction": "...", "input": "", "output": "...",
     "task": "translation_en_zo", "source": "translations:GEN 1:14"}

Quality filters (applied in this order; every drop is counted per source in the
manifest so the owner can audit):

1. Direction exclusion (SQL)     — ``translations``: only ``zo_to_en``/``en_to_zo``;
                                   the ``en→my`` Myanmar direction never leaves SQLite.
2. Source junk exclusion (SQL)   — ``wiki_lessons``: exclude ``%site-packages%``;
                                   ``dictionary_en_zo``/``eval_cases``: active rows only.
3. Confidence filter             — ``translations`` only: ``CAST(confidence AS REAL) >= 0.9``.
4. Length guard                  — instruction 3..500, input <= 2000, output 3..2000
                                   (wiki ``content_summary`` capped to 1,500 chars at emit).
5. Control/non-printable chars   — drop (not strip) rows containing C0 controls other
                                   than ``\\n``/``\\t`` (the ``vocabulary`` table is polluted).
6. ZVS-2018 orthography          — word-boundary, case-insensitive regex over the
                                   *Zolai-side text only* (output for en→zo rows, source
                                   text for zo→en rows); violations are dropped so bad
                                   orthography never reaches training.
7. Length-ratio guard            — legacy ``training_exercises`` pairs only: drop if
                                   len(zolai)/len(english) is outside 0.25..4.0.
8. Dedup                         — global over train: sha256 over NFKC-normalized +
                                   casefolded + whitespace-collapsed ``(task, instruction,
                                   output)``; ``translations.content_hash`` as a secondary
                                   key within that source. Eval has its own registry.
9. Eval leakage guard            — every eval row's normalized ``(instruction, output)``
                                   pair is registered BEFORE train emission; matching
                                   train rows are dropped and counted ``leakage``. The
                                   held-out ``translations`` slice is sampled first.
10. Provenance                   — every surviving row carries ``task`` + ``source``.

CLI::

    python3 scripts/export_zolai_dataset.py \
        --db /path/to/zolai.db \
        --out-dir ../data/exports/zolai-sft-v1 \
        [--seed 42] [--eval-per-direction 400] [--min-confidence 0.9] \
        [--max-train-rows N]

Exit codes: 0 = success, 2 = error (missing db / sqlite error / db changed).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
import sqlite3
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
GENERATOR = "zolai-training/scripts/export_zolai_dataset.py"

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = _REPO_ROOT.parent / "data" / "zolai.db"
DEFAULT_OUT_DIR = _REPO_ROOT.parent / "data" / "exports" / "zolai-sft-v1"

#: ZVS-2018 forbidden orthography (from the ``training_corpus_qwen3_import`` system
#: prompt): pathian->pasian, ram->gam, fapa->tapa, bawipa->topa, siangpahrang->kumpipa,
#: cu/cun->tua, suah->chuak, zalenna->suahtakna, nunnak->tuntakna.
ZVS_FORBIDDEN = (
    "pathian",
    "ram",
    "fapa",
    "bawipa",
    "siangpahrang",
    "cu",
    "cun",
    "suah",
    "zalenna",
    "nunnak",
)
ZVS_FORBIDDEN_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(word) for word in ZVS_FORBIDDEN)
    + r")\b",
    re.IGNORECASE,
)

#: The Myanmar direction is excluded at SQL level and never leaves SQLite.
DIRECTIONS_ALLOWED = ("zo_to_en", "en_to_zo")
DIRECTIONS_EXCLUDED_LABEL = "en→my"

#: Legacy ``training_exercises`` rows of these types are aligned zolai<->english pairs.
LEGACY_EXERCISE_TYPES = ("negation", "question", "pronoun", "conditional")
#: Word-level gloss rows in ``training_exercises``.
GLOSS_EXERCISE_TYPES = ("translate_zo_en", "vocab")

#: Ordered drop-reason keys emitted for every source in the manifest.
DROP_REASONS = (
    "length",
    "control_chars",
    "zvs",
    "dedup",
    "ratio",
    "confidence",
    "excluded_direction",
    "junk_source",
    "leakage",
    "heldout",
    "malformed",
)

ROW_FIELDS = ("instruction", "input", "output", "task", "source")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExportConfig:
    """Exporter knobs (mirrored into the manifest ``config`` block)."""

    seed: int = 42
    min_len: int = 3
    max_instruction: int = 500
    max_input: int = 2000
    max_output: int = 2000
    wiki_output_cap: int = 1500
    min_confidence: float = 0.9
    eval_per_direction: int = 400
    max_train_rows: int | None = None
    ratio_min: float = 0.25
    ratio_max: float = 4.0


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _has_control_chars(text: str) -> bool:
    """True if ``text`` contains C0/C1 controls other than ``\\n``/``\\t``."""
    return any(
        (ord(ch) < 32 and ch not in ("\n", "\t")) or ord(ch) == 127 for ch in text
    )


def norm_key(*parts: str) -> str:
    """Normalized dedup/leakage key: NFKC + casefold + whitespace collapse."""
    joined = "\x1f".join(parts)
    normalized = unicodedata.normalize("NFKC", joined).casefold()
    normalized = " ".join(normalized.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_row(
    cfg: ExportConfig,
    task: str,
    source: str,
    instruction: str,
    input_text: str,
    output: str,
    zolai_side: str | None = None,
    probe: bool = False,
) -> tuple[dict[str, str] | None, str | None]:
    """Apply shared guards (length, control chars, ZVS) and build a row.

    Returns ``(row, None)`` on success or ``(None, drop_reason)``. ``zolai_side`` is
    the Zolai text of the pair (checked against the ZVS forbidden-form regex); pass
    ``None`` for eval rows whose Zolai side is deliberately non-clean (e.g. the
    orthography-compliance eval cases) or absent. ``probe=True`` allows ``output=""``
    (prompt-only generation probes).
    """
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    output = (output or "").strip()
    for field_text in (instruction, input_text, output):
        if _has_control_chars(field_text):
            return None, "control_chars"
    if len(instruction) < cfg.min_len or len(instruction) > cfg.max_instruction:
        return None, "length"
    if len(input_text) > cfg.max_input:
        return None, "length"
    if probe:
        if len(output) > cfg.max_output:
            return None, "length"
    elif len(output) < cfg.min_len or len(output) > cfg.max_output:
        return None, "length"
    if zolai_side is not None and ZVS_FORBIDDEN_RE.search(zolai_side.strip()):
        return None, "zvs"
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "task": task,
        "source": source,
    }, None


# ---------------------------------------------------------------------------
# Per-source stats
# ---------------------------------------------------------------------------
class SourceStats:
    """Per-source read/emitted/dropped counters (manifest ``per_source``)."""

    def __init__(self) -> None:
        self._stats: dict[str, dict[str, Any]] = {}

    def slot(self, source: str) -> dict[str, Any]:
        if source not in self._stats:
            self._stats[source] = {
                "read": 0,
                "emitted": 0,
                "dropped": {reason: 0 for reason in DROP_REASONS},
            }
        return self._stats[source]

    def set_read(self, source: str, n: int) -> None:
        self.slot(source)["read"] = int(n)

    def drop(self, source: str, reason: str, n: int = 1) -> None:
        dropped = self.slot(source)["dropped"]
        if reason not in dropped:
            raise KeyError(f"unknown drop reason: {reason!r}")
        dropped[reason] += n

    def emit(self, source: str) -> None:
        self.slot(source)["emitted"] += 1

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "read": data["read"],
                "emitted": data["emitted"],
                "dropped": dict(data["dropped"]),
            }
            for name, data in sorted(self._stats.items())
        }


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------
class Exporter:
    """Builds the dataset from a read-only ``zolai.db`` connection."""

    def __init__(self, con: sqlite3.Connection, cfg: ExportConfig) -> None:
        self.con = con
        self.cfg = cfg
        self.stats = SourceStats()
        self.task_counts: Counter[str] = Counter()
        # Eval is fully built first; its normalized (instruction, output) pairs are
        # registered here BEFORE any train row is emitted (leakage guard).
        self.leakage_keys: set[str] = set()
        self.eval_registry: set[str] = set()
        self.heldout_ids: set[int] = set()
        self.seen_train_keys: set[str] = set()
        self.seen_translation_hashes: set[str] = set()
        self.train_rows: list[dict[str, str]] = []
        self.eval_rows: list[dict[str, str]] = []
        self.probe_rows: list[dict[str, str]] = []

    # -- plumbing -----------------------------------------------------------
    def _fetch_count(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        return int(self.con.execute(sql, params).fetchone()[0])

    def _emit_eval(
        self, bucket: list[dict[str, str]], source: str, row: dict[str, str]
    ) -> None:
        key = norm_key(row["instruction"], row["output"])
        if key in self.eval_registry:
            self.stats.drop(source, "dedup")
            return
        self.eval_registry.add(key)
        self.leakage_keys.add(key)
        self.stats.emit(source)
        bucket.append(row)
        self.task_counts[row["task"]] += 1

    def _emit_train(
        self, source: str, row: dict[str, str], content_hash: str | None = None
    ) -> None:
        if norm_key(row["instruction"], row["output"]) in self.leakage_keys:
            self.stats.drop(source, "leakage")
            return
        key = norm_key(row["task"], row["instruction"], row["output"])
        if key in self.seen_train_keys:
            self.stats.drop(source, "dedup")
            return
        if content_hash is not None:
            if content_hash in self.seen_translation_hashes:
                self.stats.drop(source, "dedup")
                return
            self.seen_translation_hashes.add(content_hash)
        self.seen_train_keys.add(key)
        self.stats.emit(source)
        self.train_rows.append(row)
        self.task_counts[row["task"]] += 1

    def _build(
        self,
        source: str,
        task: str,
        provenance: str,
        instruction: str,
        input_text: str,
        output: str,
        zolai_side: str | None,
        probe: bool = False,
    ) -> dict[str, str] | None:
        row, reason = build_row(
            self.cfg, task, provenance, instruction, input_text, output,
            zolai_side=zolai_side, probe=probe,
        )
        if row is None:
            self.stats.drop(source, reason)  # type: ignore[arg-type]
        return row

    # -- eval streams -------------------------------------------------------
    def build_eval(self) -> None:
        """Build ``eval.jsonl`` + ``eval_probes.jsonl`` and register leakage keys."""
        self._sample_heldout_translations()
        self._build_eval_cases()

    def _sample_heldout_translations(self) -> None:
        source = "translations_heldout"
        rng = random.Random(self.cfg.seed)
        for direction, task in (
            ("en_to_zo", "translation_en_zo"),
            ("zo_to_en", "translation_zo_en"),
        ):
            rows = self.con.execute(
                "SELECT id, source, target, reference FROM translations"
                " WHERE direction = ? AND CAST(confidence AS REAL) >= ?"
                " ORDER BY id",
                (direction, self.cfg.min_confidence),
            ).fetchall()
            self.stats.slot(source)["read"] += len(rows)
            k = min(self.cfg.eval_per_direction, len(rows))
            for rec in rng.sample(rows, k) if k else []:
                self.heldout_ids.add(int(rec["id"]))
                # en->zo: source is English, target is Zolai; zo->en: mirrored.
                if direction == "en_to_zo":
                    instruction, output, zolai = (
                        f"Translate to Zolai: {rec['source']}",
                        rec["target"],
                        rec["target"],
                    )
                else:
                    instruction, output, zolai = (
                        f"Translate to English: {rec['source']}",
                        rec["target"],
                        rec["source"],
                    )
                row = self._build(
                    source, task, f"translations:{rec['reference']}",
                    instruction, "", output, zolai,
                )
                if row is not None:
                    self._emit_eval(self.eval_rows, source, row)
        # Audit: allowed-direction rows below the confidence threshold (SQL-level).
        low_conf = self._fetch_count(
            "SELECT COUNT(*) FROM translations"
            " WHERE direction IN (?, ?) AND CAST(confidence AS REAL) < ?",
            (*DIRECTIONS_ALLOWED, self.cfg.min_confidence),
        )
        if low_conf:
            self.stats.drop("translations", "confidence", low_conf)

    def _build_eval_cases(self) -> None:
        source = "eval_cases"
        rows = self.con.execute(
            "SELECT id, kind, payload FROM eval_cases WHERE is_active = 1 ORDER BY id"
        ).fetchall()
        self.stats.set_read(source, len(rows))
        for rec in rows:
            try:
                payload = json.loads(rec["payload"])
            except (TypeError, json.JSONDecodeError):
                self.stats.drop(source, "malformed")
                continue
            case_id = int(rec["id"])
            kind = rec["kind"]
            if kind == "zvs" and "text" in payload:
                # Prompt-only generation probe: no reference exists, so the prompt is
                # the bare Zolai sentence (inventing a task would be dishonest).
                row = self._build(
                    source, "eval_zvs_probe", f"eval_cases:{case_id}",
                    str(payload["text"]), "", "", None, probe=True,
                )
                if row is not None:
                    self._emit_eval(self.probe_rows, source, row)
            elif kind == "qa" and {"question", "answer"} <= payload.keys():
                row = self._build(
                    source, "eval_qa", f"eval_cases:{case_id}",
                    str(payload["question"]), "", str(payload["answer"]), None,
                )
                if row is not None:
                    self._emit_eval(self.eval_rows, source, row)
            elif kind == "qa" and {"input", "expected"} <= payload.keys():
                row = self._build(
                    source, "eval_gloss", f"eval_cases:{case_id}",
                    f"Translate to English: {payload['input']}", "",
                    str(payload["expected"]), None,
                )
                if row is not None:
                    self._emit_eval(self.eval_rows, source, row)
            elif kind == "qa" and {"input", "expected_correction"} <= payload.keys():
                row = self._build(
                    source, "eval_correction", f"eval_cases:{case_id}",
                    f"Correct this word to ZVS 2018 orthography: {payload['input']}",
                    "", str(payload["expected_correction"]), None,
                )
                if row is not None:
                    self._emit_eval(self.eval_rows, source, row)
            elif kind == "translation" and {"source", "ref"} <= payload.keys():
                row = self._build(
                    source, "eval_translation", f"eval_cases:{case_id}",
                    f"Translate to Zolai: {payload['source']}", "",
                    str(payload["ref"]), None,
                )
                if row is not None:
                    self._emit_eval(self.eval_rows, source, row)
            else:
                self.stats.drop(source, "malformed")

    # -- train streams ------------------------------------------------------
    def build_train(self) -> None:
        """Build ``train.jsonl`` rows from every source table (mix-table order)."""
        self._train_translations()
        self._train_parallel_pairs()
        self._train_qwen3_corpus()
        self._train_exercises()
        self._train_proverbs()
        self._train_dictionary()
        self._train_vocabulary()
        self._train_wiki()

    def _train_translations(self) -> None:
        source = "translations"
        table_count = self._fetch_count("SELECT COUNT(*) FROM translations")
        self.stats.set_read(source, table_count)
        # Audit: the en->my Myanmar direction is excluded at SQL level; counted so
        # the manifest proves the exclusion happened (these rows never get fetched).
        excluded = table_count - self._fetch_count(
            "SELECT COUNT(*) FROM translations WHERE direction IN (?, ?)",
            DIRECTIONS_ALLOWED,
        )
        if excluded:
            self.stats.drop(source, "excluded_direction", excluded)
        rows = self.con.execute(
            "SELECT id, source, target, direction, reference, content_hash"
            " FROM translations"
            " WHERE direction IN (?, ?) AND CAST(confidence AS REAL) >= ?"
            " ORDER BY id",
            (*DIRECTIONS_ALLOWED, self.cfg.min_confidence),
        ).fetchall()
        for rec in rows:
            if int(rec["id"]) in self.heldout_ids:
                self.stats.drop(source, "heldout")
                continue
            if not (rec["source"] or "").strip() or not (rec["target"] or "").strip():
                self.stats.drop(source, "length")
                continue
            if rec["direction"] == "en_to_zo":
                instruction, zolai = (
                    f"Translate to Zolai: {rec['source']}", rec["target"],
                )
            else:
                instruction, zolai = (
                    f"Translate to English: {rec['source']}", rec["source"],
                )
            task = (
                "translation_en_zo"
                if rec["direction"] == "en_to_zo"
                else "translation_zo_en"
            )
            row = self._build(
                source, task, f"translations:{rec['reference']}",
                instruction, "", rec["target"], zolai,
            )
            if row is not None:
                self._emit_train(source, row, content_hash=rec["content_hash"])

    def _train_parallel_pairs(self) -> None:
        source = "corpus_parallel_pairs"
        self.stats.set_read(
            source, self._fetch_count("SELECT COUNT(*) FROM corpus_parallel_pairs")
        )
        rows = self.con.execute(
            "SELECT ref, source, target FROM corpus_parallel_pairs"
            " WHERE source_lang = 'zo' ORDER BY ref, seq"
        ).fetchall()
        for rec in rows:
            if not (rec["source"] or "").strip() or not (rec["target"] or "").strip():
                self.stats.drop(source, "length")
                continue
            row = self._build(
                source, "translation_zo_en", f"corpus_parallel_pairs:{rec['ref']}",
                f"Translate to English: {rec['source']}", "", rec["target"],
                rec["source"],
            )
            if row is not None:
                self._emit_train(source, row)

    def _train_qwen3_corpus(self) -> None:
        source = "training_corpus_qwen3_import"
        self.stats.set_read(
            source,
            self._fetch_count("SELECT COUNT(*) FROM training_corpus_qwen3_import"),
        )
        rows = self.con.execute(
            "SELECT messages, task, pattern, source"
            " FROM training_corpus_qwen3_import ORDER BY rowid"
        ).fetchall()
        for rec in rows:
            try:
                messages = json.loads(rec["messages"])
                turns = [
                    (m["content"], nxt["content"])
                    for m, nxt in itertools.pairwise(messages)
                    if m.get("role") == "user" and nxt.get("role") == "assistant"
                ]
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                self.stats.drop(source, "malformed")
                continue
            if not turns:
                self.stats.drop(source, "malformed")
                continue
            task = f"{rec['task']}:{rec['pattern']}"
            provenance = f"{source}:{rec['source']}"
            for user_text, assistant_text in turns:
                if "en_zo" in str(rec["task"]):
                    zolai = assistant_text  # en->zo: the assistant reply is Zolai
                elif "zo_en" in str(rec["task"]):
                    zolai = user_text  # zo->en: the user prompt is Zolai
                else:
                    zolai = f"{user_text} {assistant_text}"
                row = self._build(
                    source, task, provenance,
                    user_text, "", assistant_text, zolai,
                )
                if row is not None:
                    self._emit_train(source, row)

    def _train_exercises(self) -> None:
        source = "training_exercises"
        self.stats.set_read(
            source, self._fetch_count("SELECT COUNT(*) FROM training_exercises")
        )
        rows = self.con.execute(
            "SELECT id, exercise_type, zolai, english, source"
            " FROM training_exercises ORDER BY id"
        ).fetchall()
        for rec in rows:
            exercise_type = rec["exercise_type"]
            zolai = (rec["zolai"] or "").strip()
            english = (rec["english"] or "").strip()
            provenance = f"{source}:{rec['source']}"
            if exercise_type in LEGACY_EXERCISE_TYPES:
                # Aligned translation pairs; misaligned legacy rows exist, so guard
                # on the length ratio before emitting.
                if english and not (
                    self.cfg.ratio_min
                    <= len(zolai) / len(english)
                    <= self.cfg.ratio_max
                ):
                    self.stats.drop(source, "ratio")
                    continue
                row = self._build(
                    source, f"exercise_{exercise_type}", provenance,
                    f"Translate to English: {zolai}", "", english, zolai,
                )
            elif exercise_type == "error_correction":
                # ``english`` holds the corrupted text (input), ``zolai`` the clean
                # ZVS-2018 target (output) -> only the output is ZVS-checked.
                row = self._build(
                    source, "error_correction", provenance,
                    "Correct the Zolai text to ZVS 2018 orthography.",
                    english, zolai, zolai,
                )
            elif exercise_type in GLOSS_EXERCISE_TYPES:
                if not zolai:
                    self.stats.drop(source, "length")
                    continue
                row = self._build(
                    source, "word_gloss", provenance,
                    f"What is the English meaning of the Zolai word '{zolai}'?",
                    "", english, zolai,
                )
            else:
                self.stats.drop(source, "malformed")
                continue
            if row is not None:
                self._emit_train(source, row)

    def _train_proverbs(self) -> None:
        source = "proverbs"
        self.stats.set_read(source, self._fetch_count("SELECT COUNT(*) FROM proverbs"))
        rows = self.con.execute(
            "SELECT id, zolai, english, source FROM proverbs ORDER BY id"
        ).fetchall()
        for rec in rows:
            zolai = (rec["zolai"] or "").strip()
            english = (rec["english"] or "").strip()
            provenance = f"{source}:{rec['source']}"
            forward = None
            if zolai:
                forward = self._build(
                    source, "proverb_translation", provenance,
                    f"Translate to English: {zolai}", "", english, zolai,
                )
            else:
                self.stats.drop(source, "length")
            if forward is not None:
                self._emit_train(source, forward)
            reverse = None
            if english:
                reverse = self._build(
                    source, "proverb_translation_rev", provenance,
                    f"Translate to Zolai: {english}", "", zolai, zolai,
                )
            else:
                self.stats.drop(source, "length")
            if reverse is not None:
                self._emit_train(source, reverse)

    def _train_dictionary(self) -> None:
        source = "dictionary_en_zo"
        table_count = self._fetch_count("SELECT COUNT(*) FROM dictionary_en_zo")
        self.stats.set_read(source, table_count)
        deleted = self._fetch_count(
            "SELECT COUNT(*) FROM dictionary_en_zo WHERE is_deleted = 1"
        )
        if deleted:
            self.stats.drop(source, "junk_source", deleted)
        rows = self.con.execute(
            "SELECT id, headword, translations_clean FROM dictionary_en_zo"
            " WHERE is_deleted = 0 ORDER BY id"
        ).fetchall()
        for rec in rows:
            headword = (rec["headword"] or "").strip()
            if not headword:
                self.stats.drop(source, "length")
                continue
            clean = (rec["translations_clean"] or "").strip()
            row = self._build(
                source, "lexical_gloss_en_zo", f"{source}:{rec['id']}",
                f"What is the Zolai word for '{headword}'?", "", clean, clean,
            )
            if row is not None:
                self._emit_train(source, row)

    def _train_vocabulary(self) -> None:
        source = "vocabulary"
        self.stats.set_read(
            source, self._fetch_count("SELECT COUNT(*) FROM vocabulary")
        )
        rows = self.con.execute(
            "SELECT id, headword, english FROM vocabulary ORDER BY id"
        ).fetchall()
        for rec in rows:
            headword = (rec["headword"] or "").strip()
            if not headword:
                self.stats.drop(source, "length")
                continue
            row = self._build(
                source, "lexical_gloss_zo_en", f"{source}:{rec['id']}",
                f"What does the Zolai word '{headword}' mean in English?",
                "", rec["english"], headword,
            )
            if row is not None:
                self._emit_train(source, row)

    def _train_wiki(self) -> None:
        source = "wiki_lessons"
        table_count = self._fetch_count("SELECT COUNT(*) FROM wiki_lessons")
        self.stats.set_read(source, table_count)
        junk = self._fetch_count(
            "SELECT COUNT(*) FROM wiki_lessons WHERE source_file LIKE '%site-packages%'"
        )
        if junk:
            self.stats.drop(source, "junk_source", junk)
        rows = self.con.execute(
            "SELECT id, lesson_type, title, content_summary FROM wiki_lessons"
            " WHERE source_file NOT LIKE '%site-packages%' ORDER BY id"
        ).fetchall()
        for rec in rows:
            summary = (rec["content_summary"] or "").strip()[: self.cfg.wiki_output_cap]
            row = self._build(
                source, f"wiki_{rec['lesson_type']}", f"{source}:{rec['id']}",
                f"Explain this Zolai {rec['lesson_type']} topic in simple terms.",
                rec["title"], summary, summary,
            )
            if row is not None:
                self._emit_train(source, row)

    # -- finalize -----------------------------------------------------------
    def finalize(self) -> None:
        """Seeded shuffle + optional cap (one full export serves capped runs)."""
        if self.cfg.max_train_rows is not None and (
            len(self.train_rows) > self.cfg.max_train_rows
        ):
            rng = random.Random(self.cfg.seed + 1)
            self.train_rows = rng.sample(self.train_rows, self.cfg.max_train_rows)

    def run(self) -> None:
        self.build_eval()  # eval first: registers the leakage-exclusion set
        self.build_train()
        self.finalize()


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------
def write_jsonl(path: Path, rows: list[dict[str, str]]) -> str:
    """Write rows as UTF-8 JSONL; return the sha256 of the written bytes."""
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


def export_dataset(
    db_path: Path, out_dir: Path, cfg: ExportConfig | None = None
) -> dict[str, Any]:
    """Export the dataset from a READ-ONLY ``zolai.db``; return the manifest."""
    cfg = cfg or ExportConfig()
    db_path = Path(db_path)
    out_dir = Path(out_dir)
    if not db_path.exists():
        raise FileNotFoundError(f"database not found: {db_path}")
    before = db_path.stat()
    con = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        exporter = Exporter(con, cfg)
        exporter.run()
    finally:
        con.close()
    after_read = db_path.stat()
    if (after_read.st_mtime_ns, after_read.st_size) != (
        before.st_mtime_ns,
        before.st_size,
    ):
        raise RuntimeError("database file changed during export — it must be read-only")

    out_dir.mkdir(parents=True, exist_ok=True)
    checksums = {
        "train.jsonl": write_jsonl(out_dir / "train.jsonl", exporter.train_rows),
        "eval.jsonl": write_jsonl(out_dir / "eval.jsonl", exporter.eval_rows),
        "eval_probes.jsonl": write_jsonl(
            out_dir / "eval_probes.jsonl", exporter.probe_rows
        ),
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": GENERATOR,
        "db": {
            "basename": db_path.name,
            "mtime_utc": datetime.fromtimestamp(
                before.st_mtime, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "size_bytes": before.st_size,
        },
        "config": asdict(cfg),
        "counts": {
            "train": len(exporter.train_rows),
            "eval": len(exporter.eval_rows),
            "probes": len(exporter.probe_rows),
        },
        "per_source": exporter.stats.as_dict(),
        "task_counts": dict(sorted(exporter.task_counts.items())),
        "directions_excluded": [DIRECTIONS_EXCLUDED_LABEL],
        "checksums": checksums,
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a quality-filtered Zolai SFT dataset from zolai.db (READ-ONLY)."
    )
    parser.add_argument("--db", default=str(DEFAULT_DB), help="path to zolai.db")
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_OUT_DIR), help="output directory"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-per-direction", type=int, default=400)
    parser.add_argument("--min-confidence", type=float, default=0.9)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="seeded shuffle + cap of train.jsonl (default: no cap)",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: database not found: {db_path}", file=sys.stderr)
        return 2
    cfg = ExportConfig(
        seed=args.seed,
        eval_per_direction=args.eval_per_direction,
        min_confidence=args.min_confidence,
        max_train_rows=args.max_train_rows,
    )
    started = time.monotonic()
    try:
        manifest = export_dataset(db_path, Path(args.out_dir), cfg)
    except (sqlite3.Error, OSError, RuntimeError) as exc:
        print(f"ERROR: export failed: {exc}", file=sys.stderr)
        return 2
    elapsed = time.monotonic() - started
    counts = manifest["counts"]
    print(f"wrote train.jsonl     ({counts['train']:>7} rows)")
    print(f"wrote eval.jsonl      ({counts['eval']:>7} rows)")
    print(f"wrote eval_probes.jsonl ({counts['probes']:>5} rows)")
    print(f"manifest -> {Path(args.out_dir) / 'dataset_manifest.json'}")
    print(f"db unchanged: {manifest['db']['basename']} "
          f"({manifest['db']['size_bytes']} bytes, mtime {manifest['db']['mtime_utc']})")
    print(f"done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
