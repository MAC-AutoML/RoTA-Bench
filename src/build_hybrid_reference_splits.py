"""Build traceable hybrid Beam/exact-30 reference results and dataset splits."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import Counter, defaultdict, deque
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from benchmark_core import Pose, _euler_to_matrix, _mat_mul, get_last_render_backend, iou, render_views_fast
from bfs_iou_oracle import _matrix_to_euler_local
from bfs_pose_oracle import (
    ACTIONS,
    INVERSE_ACTION,
    _delta_matrices,
    _state_key,
    matrix_error,
    matrix_from_actions,
)
from build_dataset_splits import SOURCE_RELATIVE_PATH, SOURCE_SHA256, sha256_file
from render_protocol import STANDARD_IMAGE_SIZE, require_cuda_backend


CANONICAL_TASK_COUNT = 676
SPLIT_COUNTS = {"test": 100, "challenge": 100, "train": 476}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_exact_steps(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            str(row["sample_id"]): int(row["bfs_optimal_steps"])
            for row in csv.DictReader(handle)
        }


def build_radius_ball(theta: int, radius: int):
    identity = _euler_to_matrix(0.0, 0.0, 0.0)
    deltas = _delta_matrices(theta)
    matrices = [identity]
    depths = [0]
    parents = [-1]
    parent_actions = [""]
    previous_actions = [""]
    visited = {_state_key(identity): 0}
    queue = deque([0])

    while queue:
        index = queue.popleft()
        depth = depths[index]
        if depth >= radius:
            continue
        matrix = matrices[index]
        previous_action = previous_actions[index]
        for action in ACTIONS:
            if previous_action and action == INVERSE_ACTION[previous_action]:
                continue
            nxt = _mat_mul(deltas[action], matrix)
            key = _state_key(nxt)
            if key in visited:
                continue
            child_index = len(matrices)
            visited[key] = child_index
            matrices.append(nxt)
            depths.append(depth + 1)
            parents.append(index)
            parent_actions.append(action)
            previous_actions.append(action)
            queue.append(child_index)

    matrix_array = np.asarray(matrices, dtype=np.float64)
    depth_array = np.asarray(depths, dtype=np.int16)
    key_array = np.rint(matrix_array.reshape(-1, 9) * 1_000_000_000).astype(np.int64)
    row_dtype = np.dtype((np.void, key_array.dtype.itemsize * key_array.shape[1]))
    key_rows = np.ascontiguousarray(key_array).view(row_dtype).reshape(-1)
    order = np.argsort(key_rows)
    return {
        "matrices": matrix_array,
        "depths": depth_array,
        "parents": parents,
        "parent_actions": parent_actions,
        "sorted_rows": key_rows[order],
        "sorted_indices": order,
        "row_dtype": row_dtype,
    }


def reconstruct_actions(parents: list[int], parent_actions: list[str], index: int) -> list[str]:
    actions: list[str] = []
    while parents[index] >= 0:
        actions.append(parent_actions[index])
        index = parents[index]
    actions.reverse()
    return actions


def reconstruct_exact30_path(task: dict[str, Any], distance: int, ball: dict[str, Any]) -> list[str]:
    target_pose = Pose.from_dict(task["target_pose"])
    target_matrix = _euler_to_matrix(target_pose.roll, target_pose.pitch, target_pose.yaw)
    identity = _euler_to_matrix(0.0, 0.0, 0.0)
    if matrix_error(target_matrix, identity) > 2e-8:
        raise RuntimeError(f"{task['sample_id']}: exact30 source assumes an identity target pose")

    init_pose = Pose.from_dict(task["init_pose"])
    init_matrix = _euler_to_matrix(init_pose.roll, init_pose.pitch, init_pose.yaw)
    generated = matrix_from_actions(task["scramble_actions"], 30)
    if matrix_error(init_matrix, generated) > 2e-8:
        raise RuntimeError(f"{task['sample_id']}: init pose does not match its generated scramble")

    matrices = ball["matrices"]
    depths = ball["depths"]
    source_radius = distance // 2
    target_radius = distance - source_radius
    source_indices = np.flatnonzero(depths <= source_radius)
    sorted_rows = ball["sorted_rows"]
    sorted_indices = ball["sorted_indices"]
    row_dtype = ball["row_dtype"]
    selected_pair: tuple[int, int] | None = None

    for start in range(0, len(source_indices), 50_000):
        indices = source_indices[start : start + 50_000]
        transformed = np.matmul(np.asarray(generated), np.transpose(matrices[indices], (0, 2, 1)))
        transformed_keys = np.rint(transformed.reshape(-1, 9) * 1_000_000_000).astype(np.int64)
        query_rows = np.ascontiguousarray(transformed_keys).view(row_dtype).reshape(-1)
        positions = np.searchsorted(sorted_rows, query_rows)
        valid = positions < len(sorted_rows)
        safe_positions = np.minimum(positions, len(sorted_rows) - 1)
        valid &= sorted_rows[safe_positions] == query_rows
        target_indices = sorted_indices[safe_positions]
        valid &= depths[target_indices] <= target_radius
        valid &= depths[indices] + depths[target_indices] == distance
        matches = np.flatnonzero(valid)
        if len(matches):
            match = int(matches[0])
            selected_pair = int(indices[match]), int(target_indices[match])
            break

    if selected_pair is None:
        raise RuntimeError(f"{task['sample_id']}: could not reconstruct exact30 path at depth {distance}")
    prefix = reconstruct_actions(ball["parents"], ball["parent_actions"], selected_pair[0])
    suffix = reconstruct_actions(ball["parents"], ball["parent_actions"], selected_pair[1])
    identity_to_init = prefix + suffix
    if len(identity_to_init) != distance:
        raise RuntimeError(f"{task['sample_id']}: reconstructed path length mismatch")
    if matrix_error(matrix_from_actions(identity_to_init, 30), init_matrix) > 2e-8:
        raise RuntimeError(f"{task['sample_id']}: reconstructed identity-to-init path is invalid")
    return [INVERSE_ACTION[action] for action in reversed(identity_to_init)]


def replay_exact30(task: dict[str, Any], actions: Iterable[str]) -> tuple[float, float, str]:
    target_pose = Pose.from_dict(task["target_pose"])
    init_pose = Pose.from_dict(task["init_pose"])
    current = _euler_to_matrix(init_pose.roll, init_pose.pitch, init_pose.yaw)
    for action in actions:
        current = _mat_mul(_delta_matrices(30)[action], current)
    target_matrix = _euler_to_matrix(target_pose.roll, target_pose.pitch, target_pose.yaw)
    pose_error = matrix_error(current, target_matrix)
    roll, pitch, yaw = _matrix_to_euler_local(current)

    task["display_size"] = {"width": STANDARD_IMAGE_SIZE, "height": STANDARD_IMAGE_SIZE}
    task["eval_size"] = STANDARD_IMAGE_SIZE
    task["cuda_point_batch_size"] = 100_000
    _, target_eval, _ = render_views_fast(
        task,
        target_pose,
        display_size=(STANDARD_IMAGE_SIZE, STANDARD_IMAGE_SIZE),
        eval_size=STANDARD_IMAGE_SIZE,
        need_oblique=False,
    )
    require_cuda_backend(get_last_render_backend(), context="exact30 target replay")
    _, replay_eval, _ = render_views_fast(
        task,
        Pose(roll, pitch, yaw),
        display_size=(STANDARD_IMAGE_SIZE, STANDARD_IMAGE_SIZE),
        eval_size=STANDARD_IMAGE_SIZE,
        need_oblique=False,
    )
    backend = get_last_render_backend()
    require_cuda_backend(backend, context="exact30 path replay")
    return float(iou(target_eval, replay_eval)), pose_error, backend


def select_uniform_test(task_ids: list[str], steps: dict[str, int], count: int) -> list[str]:
    by_depth: dict[int, deque[str]] = defaultdict(deque)
    for task_id in sorted(task_ids):
        if steps[task_id] >= 4:
            by_depth[steps[task_id]].append(task_id)
    depths = sorted(by_depth)
    if sum(len(by_depth[depth]) for depth in depths) < count:
        raise RuntimeError("Not enough tasks with best-known reference steps >= 4")
    selected: list[str] = []
    while len(selected) < count:
        made_progress = False
        for depth in depths:
            if by_depth[depth] and len(selected) < count:
                selected.append(by_depth[depth].popleft())
                made_progress = True
        if not made_progress:
            raise RuntimeError("Could not fill uniform test split")
    return sorted(selected)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hybrid Beam/exact30 best-known references and splits")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--beam-results",
        default="artifacts/guided_iou_beam_256_320_server_v2/beam_256.jsonl",
    )
    parser.add_argument(
        "--exact30-csv",
        default="dataset_splits/bfs30_exact_pose_no_dedup_v5/bfs_optimal_steps.csv",
    )
    parser.add_argument("--output-dir", default="dataset_splits/hybrid_beam256_exact30_v1")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    tasks_path = root / SOURCE_RELATIVE_PATH
    beam_path = root / args.beam_results
    exact_path = root / args.exact30_csv
    output_dir = root / args.output_dir
    if sha256_file(tasks_path) != SOURCE_SHA256:
        raise RuntimeError("Canonical task SHA-256 mismatch")

    beam_rows = read_jsonl(beam_path)
    beam_by_id = {str(row["sample_id"]): row for row in beam_rows}
    exact_steps = read_exact_steps(exact_path)
    if len(beam_rows) != CANONICAL_TASK_COUNT or len(beam_by_id) != CANONICAL_TASK_COUNT:
        raise RuntimeError("Beam results must contain exactly 676 unique tasks")
    if len(exact_steps) != CANONICAL_TASK_COUNT:
        raise RuntimeError("Exact30 results must contain exactly 676 unique tasks")

    ball = build_radius_ball(theta=30, radius=8)
    hybrid_rows: list[dict[str, Any]] = []
    canonical_ids: list[str] = []
    with tasks_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            task = json.loads(line)
            sample_id = str(task["sample_id"])
            canonical_ids.append(sample_id)
            if sample_id not in beam_by_id or sample_id not in exact_steps:
                raise RuntimeError(f"{sample_id}: missing source result")

            exact_actions = reconstruct_exact30_path(task, exact_steps[sample_id], ball)
            replay_iou, pose_error, replay_backend = replay_exact30(task, exact_actions)
            threshold = float(task.get("tau_success", 0.9))
            if replay_iou + 1e-12 < threshold or pose_error > 2e-8:
                raise RuntimeError(f"{sample_id}: exact30 replay failed")

            beam = beam_by_id[sample_id]
            beam_success = beam.get("search_status") == "success"
            beam_steps = int(beam["reference_steps"]) if beam_success else None
            exact_value = exact_steps[sample_id]
            if not beam_success or exact_value < beam_steps:
                source = "exact30_bfs"
                best_steps = exact_value
            elif beam_steps < exact_value:
                source = "beam256"
                best_steps = beam_steps
            else:
                source = "tie"
                best_steps = beam_steps

            hybrid_rows.append({
                "sample_id": sample_id,
                "best_known_reference_steps": best_steps,
                "reference_source": source,
                "beam": {
                    "search_status": beam.get("search_status"),
                    "steps": beam_steps,
                    "success_iou": beam.get("success_iou"),
                    "path": beam.get("path"),
                    "best_iou": beam.get("best_iou"),
                    "best_depth": beam.get("best_depth"),
                    "expanded_states": beam.get("expanded_states"),
                    "visited_states": beam.get("visited_states"),
                    "elapsed_seconds": beam.get("elapsed_seconds"),
                    "render_backend": beam.get("render_backend"),
                    "scoring_backend": beam.get("scoring_backend"),
                },
                "exact30": {
                    "search_status": "success",
                    "steps": exact_value,
                    "path": [{"action": action, "angle": 30} for action in exact_actions],
                    "replay_iou": round(replay_iou, 9),
                    "final_pose_matrix_error": pose_error,
                    "render_backend": replay_backend,
                },
            })

    if len(canonical_ids) != CANONICAL_TASK_COUNT or len(set(canonical_ids)) != CANONICAL_TASK_COUNT:
        raise RuntimeError("Canonical source must contain exactly 676 unique tasks")
    if set(canonical_ids) != set(beam_by_id) or set(canonical_ids) != set(exact_steps):
        raise RuntimeError("Source result IDs do not exactly match canonical IDs")

    best_steps = {row["sample_id"]: int(row["best_known_reference_steps"]) for row in hybrid_rows}
    test = select_uniform_test(canonical_ids, best_steps, SPLIT_COUNTS["test"])
    remaining = set(canonical_ids) - set(test)
    challenge = sorted(sorted(remaining, key=lambda task_id: (-best_steps[task_id], task_id))[:100])
    train = sorted(remaining - set(challenge))
    splits = {"test": test, "challenge": challenge, "train": train}
    if {name: len(ids) for name, ids in splits.items()} != SPLIT_COUNTS:
        raise RuntimeError("Split counts are incorrect")
    assignment = {task_id: split for split, ids in splits.items() for task_id in ids}
    if len(assignment) != CANONICAL_TASK_COUNT or set(assignment) != set(canonical_ids):
        raise RuntimeError("Splits must cover every canonical task exactly once")

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "hybrid_reference_results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in hybrid_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    split_version = "hybrid_beam256_exact30_v1"
    for name, ids in splits.items():
        write_json(output_dir / f"{name}.json", {
            "split_version": split_version,
            "split_name": name,
            "count": len(ids),
            "sample_ids": ids,
        })

    rows_by_id = {row["sample_id"]: row for row in hybrid_rows}
    with (output_dir / "assignments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sample_id", "split", "best_known_reference_steps", "reference_source",
            "beam_status", "beam_steps", "beam_success_iou", "exact30_steps", "exact30_replay_iou",
        ])
        for task_id in sorted(canonical_ids):
            row = rows_by_id[task_id]
            writer.writerow([
                task_id,
                assignment[task_id],
                row["best_known_reference_steps"],
                row["reference_source"],
                row["beam"]["search_status"],
                row["beam"]["steps"],
                row["beam"]["success_iou"],
                row["exact30"]["steps"],
                row["exact30"]["replay_iou"],
            ])

    source_distribution = dict(sorted(Counter(row["reference_source"] for row in hybrid_rows).items()))
    split_distribution = {
        name: dict(sorted(Counter(best_steps[task_id] for task_id in ids).items()))
        for name, ids in splits.items()
    }
    overall_distribution = dict(sorted(Counter(best_steps.values()).items()))
    manifest = {
        "split_version": split_version,
        "created_at": date.today().isoformat(),
        "canonical_tasks_file": SOURCE_RELATIVE_PATH,
        "canonical_tasks_sha256": SOURCE_SHA256,
        "canonical_task_count": CANONICAL_TASK_COUNT,
        "difficulty_definition": "minimum valid step count from Beam-256 and exact fixed-30-degree pose BFS",
        "difficulty_name": "best_known_reference_steps; not a global optimality proof",
        "beam_results": args.beam_results,
        "beam_results_sha256": sha256_file(beam_path),
        "exact30_results": args.exact30_csv,
        "exact30_results_sha256": sha256_file(exact_path),
        "exact30_replay": {
            "image_size": [STANDARD_IMAGE_SIZE, STANDARD_IMAGE_SIZE],
            "success_rule": "IoU >= task tau_success (0.9 for canonical tasks)",
            "required_backend": "cuda",
            "path_rule": "deterministically reconstructed exact shortest path and replayed from init pose",
        },
        "selection": {
            "test": "100 deterministic round-robin tasks across all best-known step strata >= 4",
            "challenge": "100 greatest best-known reference steps after removing test; sample_id breaks ties",
            "train": "all remaining 476 tasks",
            "near_duplicate_policy": "ignored by design",
        },
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "reference_source_distribution": source_distribution,
        "overall_step_distribution": overall_distribution,
        "split_step_distribution": split_distribution,
        "environment": {"platform": platform.platform(), "python": sys.version, "opencv": cv2.__version__},
        "files": {},
    }
    for path in sorted(output_dir.iterdir()):
        if path.name != "manifest.json":
            manifest["files"][path.name] = sha256_file(path)
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({
        "split_counts": manifest["split_counts"],
        "reference_source_distribution": source_distribution,
        "overall_step_distribution": overall_distribution,
        "split_step_distribution": split_distribution,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
