"""Build deterministic proportional test/train splits from hybrid references."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from build_dataset_splits import SOURCE_RELATIVE_PATH, SOURCE_SHA256, sha256_file


CANONICAL_TASK_COUNT = 676
TEST_COUNT = 100
MIN_TEST_STEPS = 4
SELECTION_SEED = "hybrid-proportional-test-v2"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def proportional_quotas(counts: Counter[int], sample_count: int) -> dict[int, int]:
    total = sum(counts.values())
    exact = {step: count * sample_count / total for step, count in counts.items()}
    quotas = {step: int(value) for step, value in exact.items()}
    remaining = sample_count - sum(quotas.values())
    order = sorted(counts, key=lambda step: (-(exact[step] - quotas[step]), step))
    for step in order[:remaining]:
        quotas[step] += 1
    return dict(sorted(quotas.items()))


def stable_rank(sample_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{sample_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build proportional test/train splits")
    parser.add_argument(
        "--source-dir",
        default="dataset_splits/hybrid_beam256_exact30_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset_splits/hybrid_beam256_exact30_proportional_v2",
    )
    args = parser.parse_args()

    root = Path.cwd().resolve()
    source_dir = root / args.source_dir
    output_dir = root / args.output_dir
    tasks_path = root / SOURCE_RELATIVE_PATH
    results_path = source_dir / "hybrid_reference_results.jsonl"
    if sha256_file(tasks_path) != SOURCE_SHA256:
        raise RuntimeError("Canonical task SHA-256 mismatch")

    rows = read_jsonl(results_path)
    ids = [str(row["sample_id"]) for row in rows]
    if len(rows) != CANONICAL_TASK_COUNT or len(set(ids)) != CANONICAL_TASK_COUNT:
        raise RuntimeError("Hybrid references must contain exactly 676 unique tasks")

    by_step: dict[int, list[str]] = defaultdict(list)
    rows_by_id = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        rows_by_id[sample_id] = row
        by_step[int(row["best_known_reference_steps"])].append(sample_id)

    overall = Counter({step: len(step_ids) for step, step_ids in by_step.items()})
    eligible = Counter({step: count for step, count in overall.items() if step >= MIN_TEST_STEPS})
    quotas = proportional_quotas(eligible, TEST_COUNT)
    test = sorted(
        sample_id
        for step, step_ids in by_step.items()
        if step >= MIN_TEST_STEPS
        for sample_id in sorted(step_ids, key=stable_rank)[: quotas[step]]
    )
    train = sorted(set(ids) - set(test))
    splits = {"test": test, "train": train}
    if len(test) != TEST_COUNT or len(train) != CANONICAL_TASK_COUNT - TEST_COUNT:
        raise RuntimeError("Split counts are incorrect")
    if set(test) & set(train) or set(test) | set(train) != set(ids):
        raise RuntimeError("Splits must cover every task exactly once")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_version = "hybrid_beam256_exact30_proportional_v2"
    for name, sample_ids in splits.items():
        write_json(output_dir / f"{name}.json", {
            "split_version": split_version,
            "split_name": name,
            "count": len(sample_ids),
            "sample_ids": sample_ids,
        })

    assignment = {sample_id: name for name, sample_ids in splits.items() for sample_id in sample_ids}
    with (output_dir / "assignments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "split", "best_known_reference_steps", "reference_source"])
        for sample_id in sorted(ids):
            row = rows_by_id[sample_id]
            writer.writerow([
                sample_id, assignment[sample_id], row["best_known_reference_steps"], row["reference_source"]
            ])

    split_distribution = {
        name: dict(sorted(Counter(
            int(rows_by_id[sample_id]["best_known_reference_steps"]) for sample_id in sample_ids
        ).items()))
        for name, sample_ids in splits.items()
    }
    manifest = {
        "split_version": split_version,
        "created_at": date.today().isoformat(),
        "canonical_tasks_file": SOURCE_RELATIVE_PATH,
        "canonical_tasks_sha256": SOURCE_SHA256,
        "canonical_task_count": CANONICAL_TASK_COUNT,
        "reference_results": str(results_path.relative_to(root)).replace("\\", "/"),
        "reference_results_sha256": sha256_file(results_path),
        "difficulty_name": "best_known_reference_steps; not a global optimality proof",
        "selection": {
            "test": "Hamilton largest-remainder proportional allocation across eligible reference-step strata",
            "test_eligibility": f"best_known_reference_steps >= {MIN_TEST_STEPS}",
            "within_stratum": "deterministic SHA-256 rank of selection seed and sample_id",
            "selection_seed": SELECTION_SEED,
            "train": "all remaining tasks",
            "challenge": "not used",
            "near_duplicate_policy": "ignored by design",
        },
        "split_counts": {name: len(sample_ids) for name, sample_ids in splits.items()},
        "overall_step_distribution": dict(sorted(overall.items())),
        "eligible_step_distribution": dict(sorted(eligible.items())),
        "test_step_quotas": quotas,
        "split_step_distribution": split_distribution,
        "files": {},
    }
    for path in sorted(output_dir.iterdir()):
        if path.name != "manifest.json":
            manifest["files"][path.name] = sha256_file(path)
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({
        "split_counts": manifest["split_counts"],
        "overall_step_distribution": manifest["overall_step_distribution"],
        "test_step_quotas": quotas,
        "split_step_distribution": split_distribution,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
