"""Structured, pre-specified robustness-run manifest support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_experiment_manifest(path: str | Path, settings: dict[str, Any], note: str) -> None:
    """Store configurations and purpose before reviewing results to discourage cherry-picking."""
    payload = {"settings": settings, "pre_specified_note": note}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
