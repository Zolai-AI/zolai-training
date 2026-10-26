"""Synthetic-sqlite fixture tests for scripts/export_zolai_dataset.py.

Hermetic by design: every test runs against a SMALL synthetic database created in
``tmp_path`` with the REAL table schemas and a handful of adversarial rows per table.
NEVER the production 2.3 GB zolai.db.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "export_zolai_dataset.py"

_spec = importlib.util.spec_from_file_location("export_zolai_dataset", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
exporter = importlib.util.module_from_spec(_spec)
sys.modules["export_zolai_dataset"] = exporter
_spec.loader.exec_module(exporter)

ExportConfig = exporter.ExportConfig
export_dataset = exporter.export_dataset
main = exporter.main
ZVS_FORBIDDEN_RE = exporter.ZVS_FORBIDDEN_RE

# ---------------------------------------------------------------------------
# Fixture DDL (real schemas, verbatim column names) + adversarial rows
# ---------------------------------------------------------------------------
DDL = [
    """
    CREATE TABLE "training_corpus_qwen3_import" (
      "messages" TEXT,
      "task" TEXT,
      "pattern" TEXT,
      "source" TEXT,
      "import_batch_id" TEXT,
      "source_file" TEXT,
      "version" INTEGER DEFAULT 1,
      "imported_at" TEXT
    )
    """,
    """
    CREATE TABLE corpus_parallel_pairs (
        pair_id TEXT PRIMARY KEY,
        ref TEXT NOT NULL,
        source TEXT NOT NULL,
        target TEXT NOT NULL,
        source_lang TEXT NOT NULL,
        target_lang TEXT NOT NULL,
        pair_type TEXT NOT NULL,
        seq INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE "translations" (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT NOT NULL,
        target TEXT NOT NULL,
        direction TEXT NOT NULL,
        reference TEXT NOT NULL,
        confidence REAL NOT NULL,
        myanmar TEXT,
        version INTEGER DEFAULT 1,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now')),
        content_hash TEXT
    )
    """,
    """
    CREATE TABLE training_exercises (
        id INTEGER NOT NULL,
        exercise_type VARCHAR NOT NULL,
        zolai TEXT NOT NULL,
        english TEXT NOT NULL,
        source VARCHAR NOT NULL,
        difficulty VARCHAR NOT NULL,
        myanmar TEXT,
        import_batch_id TEXT,
        source_file TEXT,
        version INTEGER DEFAULT 1,
        imported_at TEXT,
        PRIMARY KEY (id)
    )
    """,
    """
    CREATE TABLE "proverbs" (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        zolai TEXT NOT NULL,
        english TEXT,
        source TEXT NOT NULL,
        category TEXT,
        version INTEGER DEFAULT 1,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now')),
        content_hash TEXT,
        myanmar TEXT,
        usage_examples TEXT
    )
    """,
    """
    CREATE TABLE "dictionary_en_zo" (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        headword TEXT NOT NULL,
        translations TEXT,
        translations_clean TEXT,
        pos TEXT,
        source TEXT,
        myanmar TEXT,
        entry_version TEXT DEFAULT 'v1.0',
        update_remarks TEXT DEFAULT '',
        update_description TEXT DEFAULT '',
        zvs_compliance_status TEXT DEFAULT 'pending',
        is_deleted INTEGER DEFAULT 0,
        deleted_at TEXT,
        version INTEGER DEFAULT 1,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now')),
        content_hash TEXT
    )
    """,
    """
    CREATE TABLE "vocabulary" (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        headword TEXT NOT NULL,
        english TEXT NOT NULL,
        frequency INTEGER NOT NULL,
        books TEXT NOT NULL,
        examples TEXT NOT NULL,
        myanmar TEXT,
        version INTEGER DEFAULT 1,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now')),
        content_hash TEXT
    )
    """,
    """
    CREATE TABLE eval_cases(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        set_name TEXT NOT NULL,
        kind TEXT NOT NULL CHECK(kind IN ('zvs','qa','translation')),
        payload TEXT NOT NULL,
        source_table TEXT,
        ordinal INTEGER,
        is_active INTEGER DEFAULT 1,
        created_at TEXT
    )
    """,
    """
    CREATE TABLE wiki_lessons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lesson_type TEXT NOT NULL,
        title TEXT NOT NULL,
        content_summary TEXT,
        word_count INTEGER,
        grammar_patterns TEXT,
        vocabulary_list TEXT,
        source_file TEXT,
        entry_version TEXT DEFAULT 'v1.0',
        update_remarks TEXT DEFAULT '',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        "import_batch_id" TEXT,
        "version" INTEGER DEFAULT 1,
        "imported_at" TEXT
    )
    """,
]

def _messages(user: str, assistant: str) -> str:
    return json.dumps(
        [
            {"role": "system", "content": "You are a Zolai language expert."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    )


def build_fixture_db(db_path: Path) -> None:
    """Create the synthetic fixture db with one adversarial case per filter."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    for stmt in DDL:
        cur.execute(stmt)

    # translations: conf-eligible rows -> held-out slice candidates.
    cur.executemany(
        "INSERT INTO translations (id, source, target, direction, reference,"
        " confidence, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            # en->zo, conf>=0.9: the ONLY one -> always held out -> eval.
            (1, "The sun rises in the morning.", "Ni a van a suak hi.",
             "en_to_zo", "T-ENZO-1", 0.95, "hash-enzo-1"),
            # low confidence -> dropped.confidence (counted at SQL level).
            (2, "I believe.", "Kei gup in.", "en_to_zo", "T-ENZO-2", 0.5,
             "hash-enzo-2"),
            # Myanmar direction -> excluded at SQL level, never fetched.
            (3, "Hello", "Mingalaba.", "en→my", "T-MY-1", 0.95, "hash-my-1"),
            # zo->en, conf>=0.9: 1 of these 2 is held out (seeded sample).
            (4, "Na bang ciang a?", "How are you?", "zo_to_en", "T-ZOEN-1",
             0.95, "hash-zoen-1"),
            (5, "Zing sang tha.", "Good morning.", "zo_to_en", "T-ZOEN-2",
             0.95, "hash-zoen-2"),
        ],
    )

    # corpus_parallel_pairs: twins of the translations rows (leakage/dedup)
    # plus one clean pair and one ZVS-violating pair.
    cur.executemany(
        "INSERT INTO corpus_parallel_pairs (pair_id, ref, source, target,"
        " source_lang, target_lang, pair_type, seq, created_at)"
        " VALUES (?, ?, ?, ?, 'zo', 'en', 'verse', 0, '2026-01-01')",
        [
            ("fp-l4", "FIXPAR L4", "Na bang ciang a?", "How are you?"),
            ("fp-l5", "FIXPAR L5", "Zing sang tha.", "Good morning."),
            ("fp-1", "FIXPAR 2:1", "Zai bai pen hiam?",
             "Where is the market?"),
            ("fp-2", "FIXPAR 3:1", "A ram in hong lak a.",
             "He built a ram."),  # ZVS: 'ram' on the Zolai side
        ],
    )

    # training_corpus_qwen3_import: one clean en->zo row, one clean zo->en row,
    # one malformed-messages row. (Note: qwen3-corpus instructions are the BARE
    # user text, so leakage twins of templated eval rows live in proverbs /
    # parallel_pairs below, which share the eval templates.)
    cur.executemany(
        "INSERT INTO training_corpus_qwen3_import (messages, task, pattern,"
        " source) VALUES (?, ?, ?, ?)",
        [
            (_messages("My name is Peter.", "Do ming Peter nga."),
             "translation_en_zo", "question", "FIX 1:1"),
            (_messages("Do ming Peter nga.", "My name is Peter."),
             "translation_zo_en", "bible_original", "FIX 2:1"),
            ("not json at all", "translation_en_zo", "question", "FIX 3:1"),
        ],
    )

    # training_exercises: legacy pair, >4x ratio pair, ZVS pair, exact
    # duplicates, error_correction (clean + bad output), glosses.
    cur.executemany(
        "INSERT INTO training_exercises (id, exercise_type, zolai, english,"
        " source, difficulty) VALUES (?, ?, ?, ?, ?, 'A1')",
        [
            (1, "negation", "Lamek' ta Noah ahi hi.",
             "Noah was the son of Lamech.", "negation_exercises:FIX 1:4"),
            (2, "negation", "Ka.",
             ("This English side is deliberately many times longer than the "
              "tiny Zolai side."), "negation_exercises:FIX 1:5"),
            (3, "negation", "A ram in hong lak hi.", "He built a ram.",
             "negation_exercises:FIX 1:6"),
            (4, "question", "A do hong hi.", "Yes, it is.",
             "question_exercises:FIX 5:1"),
            (5, "question", "A do hong hi.", "Yes, it is.",
             "question_exercises:FIX 5:1"),  # exact duplicate of id 4
            (6, "error_correction",
             "Pasian in vantung le leitung a piangsak hi.",
             "Pathian in vantung le leitung a piangsak hi.",
             "error_correction:FIX-EC-1"),  # input has forbidden form, output clean
            (7, "error_correction", "A pathian hong ci a.", "He wrote something.",
             "error_correction:FIX-EC-2"),  # output has forbidden form -> zvs
            (8, "translate_zo_en", "pasian", "God", "vocab:FIX-V-1"),
            (9, "vocab", "pathian", "old spelling", "vocab:FIX-V-2"),
        ],
    )

    # proverbs: one clean (emits forward + reverse), one with NULL english,
    # one that twins the en->zo HELD-OUT translations row via the shared
    # "Translate to Zolai:" template (its reverse row -> leakage).
    cur.executemany(
        "INSERT INTO proverbs (id, zolai, english, source) VALUES (?, ?, ?, ?)",
        [
            (1, "Kush in Nimrod kici tapa khat nei hi.",
             "And Cush begat Nimrod.", "1CH 1:10"),
            (2, "In khat hi.", None, "FIX P2"),
            (3, "Ni a van a suak hi.", "The sun rises in the morning.",
             "FIX P3"),
        ],
    )

    # dictionary: one clean, one deleted, one empty-clean, one bad-output.
    cur.executemany(
        "INSERT INTO dictionary_en_zo (id, headword, translations_clean,"
        " is_deleted) VALUES (?, ?, ?, ?)",
        [
            (1, "laimal", "good", 0),
            (2, "gong", "old", 1),
            (3, "gawm", "", 0),
            (4, "khat", "pathian", 0),  # Zolai answer is bad orthography
        ],
    )

    # vocabulary: control-char headword, clean word, ZVS headword, empty
    # english, empty headword.
    cur.executemany(
        "INSERT INTO vocabulary (id, headword, english, frequency, books,"
        " examples) VALUES (?, ?, ?, 0, '[]', '[]')",
        [
            (1, "\x13 embed coreldraw", "swastika"),
            (2, "hiam", "yes/no question word"),
            (3, "pathian", "old spelling of pasian"),
            (4, "ok", ""),
            (5, "", "orphan"),
        ],
    )

    # eval_cases: one row per payload variant + one inactive row.
    cur.executemany(
        "INSERT INTO eval_cases (id, set_name, kind, payload, is_active)"
        " VALUES (?, 'fixture', ?, ?, ?)",
        [
            (1, "qa", json.dumps({"question": "Kua in van leh lei a piangsak"
                                              " hiam?", "hyp": "Pasian",
                                  "answer": "Pasian"}), 1),
            (2, "qa", json.dumps({"id": "qa_001", "category": "vocabulary",
                                  "input": "pasian", "expected": "God",
                                  "difficulty": "A1"}), 1),
            (3, "qa", json.dumps({"id": "zvs_001", "category": "zvs_compliance",
                                  "input": "pathian",
                                  "expected_correction": "pasian",
                                  "difficulty": "A1"}), 1),
            (4, "translation", json.dumps({"source": "In the beginning God"
                                                     " created the heavens.",
                                           "hyp": "ignored hypothesis",
                                           "ref": "A kipat cil-in Pasian in"
                                                  " vantung a piangsak hi."}), 1),
            (5, "zvs", json.dumps({"text": "Pasian in van leh lei a piangsak"
                                           " hi."}), 1),
            (6, "qa", json.dumps({"question": "inactive?", "hyp": "x",
                                  "answer": "y"}), 0),
        ],
    )

    # wiki_lessons: site-packages junk, long clean lesson, short clean lesson.
    cur.executemany(
        "INSERT INTO wiki_lessons (id, lesson_type, title, content_summary,"
        " source_file) VALUES (?, ?, ?, ?, ?)",
        [
            (1, "reference",
             "zolai-wiki/.venv/lib/python3.14/site-packages/pip/idna/LICENSE.md",
             "BSD 3-Clause License junk.",
             "zolai-wiki/.venv/lib/python3.14/site-packages/pip/idna/LICENSE.md"),
            (2, "grammar", "SOV word order", "a" * 1600, "notes/sov.md"),
            (3, "vocabulary", "Numbers", "Khat, hni, thum.", "notes/num.md"),
        ],
    )
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def fixture_db(tmp_path: Path) -> Path:
    db = tmp_path / "zolai-fixture.db"
    build_fixture_db(db)
    return db


@pytest.fixture()
def cfg() -> ExportConfig:
    # eval_per_direction=1 -> exactly one zo->en row is held out (deterministic
    # count regardless of which one the seeded sample picks).
    return ExportConfig(eval_per_direction=1)


@pytest.fixture()
def exported(fixture_db: Path, tmp_path: Path, cfg: ExportConfig) -> dict[str, Any]:
    out = tmp_path / "out"
    manifest = export_dataset(fixture_db, out, cfg)
    return {"manifest": manifest, "out": out, "db": fixture_db, "cfg": cfg}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. Importable + CLI
# ---------------------------------------------------------------------------
def test_exporter_importable() -> None:
    for symbol in ("ExportConfig", "Exporter", "export_dataset", "main",
                   "build_row", "write_jsonl", "ZVS_FORBIDDEN_RE"):
        assert hasattr(exporter, symbol), f"missing symbol: {symbol}"


def test_main_missing_db_returns_2(tmp_path: Path) -> None:
    rc = main(["--db", str(tmp_path / "nope.db"),
               "--out-dir", str(tmp_path / "out")])
    assert rc == 2


def test_main_success_returns_0(fixture_db: Path, tmp_path: Path) -> None:
    out = tmp_path / "cli-out"
    rc = main(["--db", str(fixture_db), "--out-dir", str(out),
               "--eval-per-direction", "1"])
    assert rc == 0
    for name in ("train.jsonl", "eval.jsonl", "eval_probes.jsonl",
                 "dataset_manifest.json"):
        assert (out / name).is_file(), f"missing artifact: {name}"


# ---------------------------------------------------------------------------
# 2. Direction exclusion (en->my)
# ---------------------------------------------------------------------------
def test_en_to_my_excluded_everywhere(exported: dict[str, Any]) -> None:
    out = exported["out"]
    blob = "".join(
        (out / name).read_text(encoding="utf-8")
        for name in ("train.jsonl", "eval.jsonl", "eval_probes.jsonl")
    )
    assert "Mingalaba" not in blob
    assert "en→my" not in blob
    drops = exported["manifest"]["per_source"]["translations"]["dropped"]
    assert drops["excluded_direction"] == 1
    assert exported["manifest"]["directions_excluded"] == ["en→my"]
    # myanmar column is never required by the schema the exporter queries.


# ---------------------------------------------------------------------------
# 3. ZVS-2018 orthography filter
# ---------------------------------------------------------------------------
def test_zvs_violations_absent_from_train_and_counted(
    exported: dict[str, Any],
) -> None:
    train = read_jsonl(exported["out"] / "train.jsonl")
    for row in train:
        zolai_sides = [row["output"]]
        if row["instruction"].startswith("Translate to English: "):
            zolai_sides.append(row["instruction"][len("Translate to English: "):])
        if row["task"] == "lexical_gloss_zo_en":
            zolai_sides.append(row["instruction"])
        for text in zolai_sides:
            assert not ZVS_FORBIDDEN_RE.search(text), (
                f"ZVS violation leaked into train: {row}"
            )
    per_source = exported["manifest"]["per_source"]
    assert per_source["vocabulary"]["dropped"]["zvs"] == 1
    assert per_source["training_exercises"]["dropped"]["zvs"] == 3  # legacy + ec + gloss
    assert per_source["corpus_parallel_pairs"]["dropped"]["zvs"] == 1


def test_error_correction_input_is_not_zvs_checked(
    exported: dict[str, Any],
) -> None:
    """The corrupted INPUT legitimately contains 'Pathian'; only output is checked."""
    train = read_jsonl(exported["out"] / "train.jsonl")
    corrections = [
        r for r in train if r["task"] == "error_correction" and "FIX-EC-1" in r["source"]
    ]
    assert len(corrections) == 1
    assert "Pathian" in corrections[0]["input"]
    assert corrections[0]["output"].startswith("Pasian")


# ---------------------------------------------------------------------------
# 4. Length / control-char guards
# ---------------------------------------------------------------------------
def test_control_char_rows_dropped(exported: dict[str, Any]) -> None:
    blob = (exported["out"] / "train.jsonl").read_text(encoding="utf-8")
    assert "coreldraw" not in blob
    assert exported["manifest"]["per_source"]["vocabulary"]["dropped"][
        "control_chars"
    ] == 1


def test_length_guard_and_wiki_cap(exported: dict[str, Any]) -> None:
    per_source = exported["manifest"]["per_source"]
    assert per_source["dictionary_en_zo"]["dropped"]["length"] == 1  # empty clean
    assert per_source["vocabulary"]["dropped"]["length"] == 2  # empty en + headword
    train = read_jsonl(exported["out"] / "train.jsonl")
    wiki_rows = [r for r in train if r["task"].startswith("wiki_")]
    assert len(wiki_rows) == 2
    long_rows = [r for r in wiki_rows if r["input"] == "SOV word order"]
    assert long_rows and len(long_rows[0]["output"]) == 1500  # capped


# ---------------------------------------------------------------------------
# 5. Ratio guard
# ---------------------------------------------------------------------------
def test_ratio_guard_drops_misaligned_pair(exported: dict[str, Any]) -> None:
    train = read_jsonl(exported["out"] / "train.jsonl")
    assert exported["manifest"]["per_source"]["training_exercises"]["dropped"][
        "ratio"
    ] == 1
    kept = [r for r in train if "FIX 1:4" in r["source"]]
    assert kept and kept[0]["output"] == "Noah was the son of Lamech."


# ---------------------------------------------------------------------------
# 6. Dedup
# ---------------------------------------------------------------------------
def test_duplicates_collapse(exported: dict[str, Any]) -> None:
    train = read_jsonl(exported["out"] / "train.jsonl")
    dup = [r for r in train if "question_exercises:FIX 5:1" in r["source"]]
    assert len(dup) == 1
    total_dedup = sum(
        src["dropped"]["dedup"] for src in exported["manifest"]["per_source"].values()
    )
    assert total_dedup == 2  # exercise duplicate + one held-out twin


# ---------------------------------------------------------------------------
# 7. Leakage guard
# ---------------------------------------------------------------------------
def test_eval_leakage_guard(exported: dict[str, Any]) -> None:
    out = exported["out"]
    eval_rows = read_jsonl(out / "eval.jsonl")
    train = read_jsonl(out / "train.jsonl")
    train_pairs = {(r["instruction"], r["output"]) for r in train}
    heldout_translations = [
        r for r in eval_rows if r["task"].startswith("translation_")
    ]
    assert len(heldout_translations) == 2  # one per direction
    for row in heldout_translations:
        assert (row["instruction"], row["output"]) not in train_pairs
    drops = [
        src["dropped"]["leakage"]
        for src in exported["manifest"]["per_source"].values()
    ]
    assert sum(drops) >= 1
    # en->zo held-out row's twin lives in proverbs (shared template);
    # zo->en held-out row's twin lives in corpus_parallel_pairs.
    assert exported["manifest"]["per_source"]["proverbs"]["dropped"][
        "leakage"
    ] == 1
    assert exported["manifest"]["per_source"]["corpus_parallel_pairs"][
        "dropped"
    ]["leakage"] == 1


# ---------------------------------------------------------------------------
# 8. eval_cases mapping (all 4 scored variants + probes)
# ---------------------------------------------------------------------------
def test_eval_cases_variant_mapping(exported: dict[str, Any]) -> None:
    eval_rows = read_jsonl(exported["out"] / "eval.jsonl")
    by_source = {r["source"]: r for r in eval_rows}
    qa = by_source["eval_cases:1"]
    assert qa["task"] == "eval_qa"
    assert qa["instruction"] == "Kua in van leh lei a piangsak hiam?"
    assert qa["output"] == "Pasian"
    gloss = by_source["eval_cases:2"]
    assert gloss["task"] == "eval_gloss"
    assert gloss["instruction"] == "Translate to English: pasian"
    assert gloss["output"] == "God"
    correction = by_source["eval_cases:3"]
    assert correction["task"] == "eval_correction"
    assert correction["instruction"] == (
        "Correct this word to ZVS 2018 orthography: pathian"
    )
    assert correction["output"] == "pasian"
    translation = by_source["eval_cases:4"]
    assert translation["task"] == "eval_translation"
    assert translation["instruction"].startswith("Translate to Zolai: In the"
                                                  " beginning")
    assert translation["output"].startswith("A kipat cil-in")
    # 'hyp' hypothesis fields are ignored everywhere; the inactive case is out.
    assert "eval_cases:6" not in by_source
    assert all("hyp" not in r["instruction"] for r in eval_rows)


def test_probes_file_is_zvs_only(exported: dict[str, Any]) -> None:
    probes = read_jsonl(exported["out"] / "eval_probes.jsonl")
    assert len(probes) == 1
    assert probes[0]["task"] == "eval_zvs_probe"
    assert probes[0]["output"] == ""
    assert probes[0]["instruction"] == "Pasian in van leh lei a piangsak hi."
    assert probes[0]["source"] == "eval_cases:5"


# ---------------------------------------------------------------------------
# 9. Manifest schema, checksums, reconciliation
# ---------------------------------------------------------------------------
def test_manifest_schema_complete(exported: dict[str, Any]) -> None:
    manifest = exported["manifest"]
    for key in ("schema_version", "generated_at", "generator", "db", "config",
                "counts", "per_source", "task_counts", "directions_excluded",
                "checksums"):
        assert key in manifest, f"manifest missing key: {key}"
    assert manifest["schema_version"] == 1
    for key in ("seed", "min_len", "max_instruction", "max_input", "max_output",
                "wiki_output_cap", "min_confidence", "eval_per_direction",
                "max_train_rows"):
        assert key in manifest["config"], f"config missing: {key}"
    expected_sources = {
        "training_corpus_qwen3_import", "corpus_parallel_pairs", "translations",
        "training_exercises", "proverbs", "dictionary_en_zo", "vocabulary",
        "wiki_lessons", "eval_cases", "translations_heldout",
    }
    assert expected_sources <= set(manifest["per_source"])
    for source, data in manifest["per_source"].items():
        assert set(data["dropped"].keys()) == set(exporter.DROP_REASONS), source


def test_manifest_checksums_match_files(exported: dict[str, Any]) -> None:
    out, manifest = exported["out"], exported["manifest"]
    for name in ("train.jsonl", "eval.jsonl", "eval_probes.jsonl"):
        expected = f"sha256:{sha256_file(out / name)}"
        assert manifest["checksums"][name] == expected, name


def test_manifest_counts_match_files(exported: dict[str, Any]) -> None:
    out, manifest = exported["out"], exported["manifest"]
    assert manifest["counts"]["train"] == len(read_jsonl(out / "train.jsonl"))
    assert manifest["counts"]["eval"] == len(read_jsonl(out / "eval.jsonl"))
    assert manifest["counts"]["probes"] == len(
        read_jsonl(out / "eval_probes.jsonl")
    )
    assert manifest["counts"]["train"] > 0
    assert manifest["counts"]["eval"] > 0
    assert manifest["counts"]["probes"] > 0
    for source, data in manifest["per_source"].items():
        assert data["emitted"] > 0, f"source emitted nothing: {source}"


def test_per_source_accounting_reconciles(exported: dict[str, Any]) -> None:
    """read == emitted + sum(dropped) (translations_heldout: non-sampled rows
    continue into the train stream; proverbs emit TWO candidates per row)."""
    for source, data in exported["manifest"]["per_source"].items():
        dropped = sum(data["dropped"].values())
        if source == "translations_heldout":
            continue
        factor = 2 if source == "proverbs" else 1
        assert data["read"] * factor == data["emitted"] + dropped, source


# ---------------------------------------------------------------------------
# 10. Determinism + read-only proof + cap
# ---------------------------------------------------------------------------
def test_determinism_same_seed_identical_bytes(
    fixture_db: Path, tmp_path: Path, cfg: ExportConfig
) -> None:
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    export_dataset(fixture_db, out_a, cfg)
    export_dataset(fixture_db, out_b, cfg)
    for name in ("train.jsonl", "eval.jsonl", "eval_probes.jsonl"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes(), name


def test_db_bytes_unchanged_after_export(
    fixture_db: Path, tmp_path: Path, cfg: ExportConfig
) -> None:
    before = (sha256_file(fixture_db), fixture_db.stat().st_mtime_ns,
              fixture_db.stat().st_size)
    export_dataset(fixture_db, tmp_path / "out", cfg)
    after = (sha256_file(fixture_db), fixture_db.stat().st_mtime_ns,
             fixture_db.stat().st_size)
    assert before == after


def test_max_train_rows_cap(fixture_db: Path, tmp_path: Path) -> None:
    cfg = ExportConfig(eval_per_direction=1, max_train_rows=5)
    out = tmp_path / "capped"
    manifest = export_dataset(fixture_db, out, cfg)
    assert manifest["counts"]["train"] == 5
    assert len(read_jsonl(out / "train.jsonl")) == 5
    # eval is never capped
    assert manifest["counts"]["eval"] > 5


# ---------------------------------------------------------------------------
# 11. Row contract
# ---------------------------------------------------------------------------
def test_row_contract_all_files(exported: dict[str, Any]) -> None:
    out, cfg = exported["out"], exported["cfg"]
    for name, allow_empty_output in (("train.jsonl", False),
                                     ("eval.jsonl", False),
                                     ("eval_probes.jsonl", True)):
        for row in read_jsonl(out / name):
            assert list(row.keys()) == list(exporter.ROW_FIELDS), row
            assert cfg.min_len <= len(row["instruction"]) <= cfg.max_instruction
            assert len(row["input"]) <= cfg.max_input
            if allow_empty_output:
                assert len(row["output"]) <= cfg.max_output
            else:
                assert cfg.min_len <= len(row["output"]) <= cfg.max_output
            assert row["task"] and row["source"]
            assert ":" in row["source"]


def test_task_labels_and_provenance(exported: dict[str, Any]) -> None:
    manifest = exported["manifest"]
    train = read_jsonl(exported["out"] / "train.jsonl")
    tasks = {r["task"] for r in train}
    assert {
        "translation_zo_en", "translation_en_zo:question",
        "exercise_negation", "exercise_question", "error_correction",
        "word_gloss", "proverb_translation", "proverb_translation_rev",
        "lexical_gloss_en_zo", "lexical_gloss_zo_en", "wiki_grammar",
        "wiki_vocabulary", "translation_zo_en:bible_original",
    } <= tasks
    sources = {r["source"] for r in train}
    assert "training_corpus_qwen3_import:FIX 2:1" in sources
    assert "corpus_parallel_pairs:FIXPAR 2:1" in sources
    assert "training_exercises:negation_exercises:FIX 1:4" in sources
    assert any(s.startswith("translations:T-ZOEN-") for s in sources)
    assert "wiki_lessons:2" in sources
    assert set(manifest["task_counts"]) >= {
        r["task"] for r in read_jsonl(exported["out"] / "eval.jsonl")
    } | {"eval_zvs_probe"}


def test_qwen3_corpus_shapes(exported: dict[str, Any]) -> None:
    """The qwen3 corpus row survives with its system prompt NOT emitted."""
    train = read_jsonl(exported["out"] / "train.jsonl")
    row = next(r for r in train if r["source"].endswith("FIX 2:1"))
    assert row["instruction"] == "Do ming Peter nga."
    assert row["output"] == "My name is Peter."
    assert row["task"] == "translation_zo_en:bible_original"
    assert "system" not in json.dumps(row).lower()


def test_zvs_regex_word_boundaries() -> None:
    assert ZVS_FORBIDDEN_RE.search("A pathian in biakinn uh hi.")
    assert ZVS_FORBIDDEN_RE.search("A RAM in hong lak hi.")  # case-insensitive
    assert ZVS_FORBIDDEN_RE.search("tua cu hong hi.")
    assert not ZVS_FORBIDDEN_RE.search("Pasian in van leh lei a piangsak hi.")
    assert not ZVS_FORBIDDEN_RE.search("diagram leh category hong sawm hi.")
    assert not ZVS_FORBIDDEN_RE.search("pasian")  # no partial matches
