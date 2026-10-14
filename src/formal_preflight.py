from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_formal_run(
    root: Path,
    *,
    formal_config_path: Path,
    tasks_path: Path,
    split_path: Path,
    prompt_protocol: str,
    metric_feedback_mode: str,
    hide_iou_feedback: bool,
    stagnation_patience: int,
    max_tokens: int,
    timeout_seconds: int,
    history_window: int,
    include_history_cot: bool,
    legacy_stop_as_success: bool,
    api_temperature: float,
    api_seed: int,
    display_size: tuple[int, int],
) -> dict[str, Any]:
    config = read_json(formal_config_path)
    protocol = config["protocol"]
    run = config["run"]
    expected = {
        "prompt_protocol": protocol["prompt_protocol"],
        "metric_feedback_mode": protocol["metric_feedback_mode"],
        "hide_iou_feedback": bool(protocol["hide_iou_feedback"]),
        "stagnation_patience": int(run["stagnation_patience"]),
        "max_tokens": int(run["max_tokens"]),
        "timeout_seconds": int(run["timeout_seconds"]),
        "history_window": int(protocol["history_window"]),
        "include_history_cot": bool(protocol["include_history_cot"]),
        "api_temperature": float(run["temperature"]),
        "api_seed": int(run["api_seed"]),
        "display_size": (
            int(config["renderer"]["api_display_width"]),
            int(config["renderer"]["api_display_height"]),
        ),
    }
    actual = {
        "prompt_protocol": prompt_protocol,
        "metric_feedback_mode": metric_feedback_mode,
        "hide_iou_feedback": hide_iou_feedback,
        "stagnation_patience": stagnation_patience,
        "max_tokens": max_tokens,
        "timeout_seconds": timeout_seconds,
        "history_window": history_window,
        "include_history_cot": include_history_cot,
        "api_temperature": float(api_temperature),
        "api_seed": int(api_seed),
        "display_size": tuple(int(value) for value in display_size),
    }
    mismatches = {key: {"expected": expected[key], "actual": actual[key]} for key in expected if expected[key] != actual[key]}
    if mismatches or legacy_stop_as_success:
        raise RuntimeError(f"Formal protocol mismatch: {mismatches or {'legacy_stop_as_success': True}}")

    canonical_expected = str(config["dataset"]["canonical_tasks_sha256"])
    if sha256_file(tasks_path) != canonical_expected:
        raise RuntimeError("Canonical tasks SHA-256 mismatch")
    split = read_json(split_path)
    split_ids = [str(value) for value in split.get("sample_ids", [])]
    if len(split_ids) != len(set(split_ids)) or not split_ids:
        raise RuntimeError("Split must contain unique task IDs")
    canonical_ids = {
        str(json.loads(line)["sample_id"])
        for line in tasks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not set(split_ids).issubset(canonical_ids):
        raise RuntimeError("Split contains IDs outside the canonical tasks file")
    expected_split_hash = str(config["dataset"]["test_ids_sha256"])
    if sha256_file(split_path) != expected_split_hash:
        raise RuntimeError("Formal test split SHA-256 mismatch")
    if str(split.get("split_name", "")) != "test" or len(split_ids) != int(config["dataset"]["test_task_count"]):
        raise RuntimeError("Formal run must use the frozen 100-task test split")

    return {
        "status": "passed",
        "formal_config": str(formal_config_path.relative_to(root)).replace("\\", "/"),
        "split_file": str(split_path.relative_to(root)).replace("\\", "/"),
        "split_name": str(split.get("split_name", "")),
        "task_count": len(split_ids),
        "canonical_tasks_sha256": canonical_expected,
        "split_sha256": sha256_file(split_path),
        "protocol": actual,
    }
