#!/usr/bin/env python3
"""Parse Kaggle training logs and suggest improvements"""

import re
import os
from pathlib import Path
def _data_root() -> Path:
    """Shared workspace ``data/`` dir, matching the workspace Data Layer Standard.

    The old default was an absolute path in a *different* project
    (``Projects/zolai``, not ``Projects/zolai-ai``), so it resolved for nobody.
    Resolution order: ``ZOLAI_DATA_ROOT``, then the repo's own ``data/`` if it
    holds training files, then the workspace ``data/`` beside this repo.
    """
    env = os.environ.get("ZOLAI_DATA_ROOT", "").strip()
    if env:
        return Path(env).expanduser()
    repo_root = Path(__file__).resolve().parents[2]
    local = repo_root / "data" / "training"
    if local.is_dir():
        return repo_root / "data"
    return Path(__file__).resolve().parents[3] / "data"



def parse_logs(log_file):
    """Extract training metrics from logs"""
    with open(log_file) as f:
        logs = f.read()
    
    # Extract loss values
    losses = re.findall(r'"loss":\s*([\d.]+)', logs)
    eval_losses = re.findall(r'"eval_loss":\s*([\d.]+)', logs)
    
    print("=== TRAINING SUMMARY ===\n")
    
    if losses:
        print("Training Loss:")
        print(f"  Start: {losses[0]}")
        print(f"  End: {losses[-1]}")
        print(f"  Improvement: {float(losses[0]) - float(losses[-1]):.4f}")
    
    if eval_losses:
        print("\nValidation Loss:")
        print(f"  Start: {eval_losses[0]}")
        print(f"  End: {eval_losses[-1]}")
    
    # Check for errors
    errors = re.findall(r'(Error|Exception|Traceback).*', logs)
    if errors:
        print(f"\n⚠️ ERRORS FOUND: {len(errors)}")
        for err in errors[:3]:
            print(f"  - {err[:80]}")
    else:
        print("\n✓ No errors detected")
    
    # Suggestions
    print("\n=== IMPROVEMENTS ===")
    if float(losses[-1]) > 2.0:
        print("• Loss still high - increase training epochs")
    if "OOM" in logs or "out of memory" in logs:
        print("• Reduce batch size or max_length")
    if "cuda" not in logs.lower():
        print("• Verify GPU is being used")
    else:
        print("✓ GPU training confirmed")

if __name__ == "__main__":
    # Old default was an absolute home path in a different project; resolve from
    # the workspace instead, with an env override.
    log_file = Path(os.environ.get("ZOLAI_TRAINING_LOG")
                    or _data_root().parent / "training_test.log")
    if log_file.exists():
        parse_logs(log_file)
    else:
        print("No logs found yet. Run training first.")
