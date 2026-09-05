"""Basic validation tests for zolai-training scripts.

Tests are fast, require no GPU, and no actual training data.
"""

from __future__ import annotations

import ast
import py_compile
from pathlib import Path
from typing import ClassVar

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts" / "training"
ALL_PY = sorted(SCRIPTS_DIR.glob("*.py"))


# ---------------------------------------------------------------------------
# 1. All .py scripts compile without syntax errors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("script", ALL_PY, ids=[p.name for p in ALL_PY])
def test_script_compiles(script: Path) -> None:
    """Each training script must be valid, importable Python."""
    assert script.exists(), f"Script missing: {script}"
    py_compile.compile(str(script), doraise=True)


# ---------------------------------------------------------------------------
# 2. All .py scripts parse into valid ASTs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("script", ALL_PY, ids=[p.name for p in ALL_PY])
def test_script_ast_valid(script: Path) -> None:
    """Each script should parse into a complete AST (no syntax issues)."""
    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script))
    assert tree is not None


# ---------------------------------------------------------------------------
# 3. prepare_train.py has expected symbols
# ---------------------------------------------------------------------------
class TestPrepareTrain:
    EXPECTED_FUNCS: ClassVar[tuple[str, ...]] = ("convert_format", "main")
    EXPECTED_CONSTANTS: ClassVar[tuple[str, ...]] = ("TRAIN_FILE", "VAL_FILE", "OUTPUT_DIR", "SPLIT_RATIO")

    def _tree(self) -> ast.Module:
        return ast.parse(
            (SCRIPTS_DIR / "prepare_train.py").read_text(encoding="utf-8"),
            filename="prepare_train.py",
        )

    def test_functions_defined(self) -> None:
        tree = self._tree()
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for fn in self.EXPECTED_FUNCS:
            assert fn in names, f"Missing function: {fn}"

    def test_constants_defined(self) -> None:
        tree = self._tree()
        names = {n.targets[0].id for n in ast.iter_child_nodes(tree) if isinstance(n, ast.Assign)}
        for const in self.EXPECTED_CONSTANTS:
            assert const in names, f"Missing constant: {const}"


# ---------------------------------------------------------------------------
# 4. merge_adapter.py has expected CLI structure
# ---------------------------------------------------------------------------
class TestMergeAdapter:
    EXPECTED_ARGS: ClassVar[tuple[str, ...]] = ("--base", "--adapter", "--output", "--dtype")

    def test_main_function_exists(self) -> None:
        tree = ast.parse(
            (SCRIPTS_DIR / "merge_adapter.py").read_text(encoding="utf-8"),
            filename="merge_adapter.py",
        )
        funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "main" in funcs

    def test_cli_args_defined(self) -> None:
        """Verify expected CLI arguments appear as string literals."""
        source = (SCRIPTS_DIR / "merge_adapter.py").read_text(encoding="utf-8")
        for arg in self.EXPECTED_ARGS:
            assert arg in source, f"CLI argument not found: {arg}"


# ---------------------------------------------------------------------------
# 5. convert_training.py has expected functions
# ---------------------------------------------------------------------------
class TestConvertTraining:
    EXPECTED_FUNCS: ClassVar[tuple[str, ...]] = ("load_jsonl", "extract_text", "convert_to_chatml", "convert_to_qa")

    def test_functions_defined(self) -> None:
        tree = ast.parse(
            (SCRIPTS_DIR / "convert_training.py").read_text(encoding="utf-8"),
            filename="convert_training.py",
        )
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for fn in self.EXPECTED_FUNCS:
            assert fn in names, f"Missing function: {fn}"


# ---------------------------------------------------------------------------
# 6. ruff.toml exists and is valid TOML
# ---------------------------------------------------------------------------
class TestRuffToml:
    def test_exists(self) -> None:
        assert (REPO_ROOT / "ruff.toml").is_file()

    def test_valid_toml(self) -> None:
        """ruff.toml must be parseable as TOML (stdlib or tomllib)."""
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]

        content = (REPO_ROOT / "ruff.toml").read_text(encoding="utf-8")
        data = tomllib.loads(content)
        assert isinstance(data, dict)
