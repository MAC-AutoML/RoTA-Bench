"""Shared dataset metadata and hashing helpers for split-building scripts.

The public release ships the frozen 674-task corpus in ``data/tasks_674.jsonl``.
The split builders import this module for the canonical path, checksum, and a
streaming SHA-256 implementation; generation inputs such as Beam/BFS result
files remain optional research artifacts and are not required for evaluation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


SOURCE_RELATIVE_PATH = "data/tasks_674.jsonl"
SOURCE_SHA256 = "6927d49e1a7d6e6283085fc3e962dd2b2f2afdb0e448c6bc6a7b32d9a7988a72"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

