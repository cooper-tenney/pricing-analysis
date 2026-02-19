"""
Run utilities: run_id generation and artifacts/latest.txt persistence.
"""

import random
import string
from datetime import datetime
from pathlib import Path


def generate_run_id() -> str:
    """
    Generate run_id = timestamp (YYYYMMDD_HHMM) + "_" + random 4-char suffix.
    Example: 20250217_1432_a7f2
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{ts}_{suffix}"


def write_latest_run_id(artifacts_dir: Path, run_id: str) -> Path:
    """
    Write artifacts/latest.txt containing just the run_id string.
    Returns path to latest.txt.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    latest_path = artifacts_dir / "latest.txt"
    latest_path.write_text(run_id.strip(), encoding="utf-8")
    return latest_path


def read_latest_run_id(artifacts_dir: Path) -> str | None:
    """
    Read run_id from artifacts/latest.txt. Returns None if missing or invalid.
    """
    latest_path = Path(artifacts_dir) / "latest.txt"
    if not latest_path.exists():
        return None
    try:
        return latest_path.read_text(encoding="utf-8").strip() or None
    except Exception:
        return None
