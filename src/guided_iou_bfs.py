"""Layered IoU search using one or two pose-guided angle choices per axis."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from benchmark_core import (
    Pose,
    _euler_to_matrix,
    _mat_mul,
    get_last_render_backend,
    iou,
    render_views_fast,
)
from bfs_iou_oracle import (
    ACTION_DIRECTIONS,
    INVERSE_AXIS,
    _matrix_to_euler_local,
    matrix_key,
    resolve_size,
    score_batch_cuda,
)

try:
    import torch
except Exception:
    torch = None

ALLOWED_ANGLES = (30, 10, 5, 1)
CANONICAL_TASK_COUNT = 676
CANONICAL_TASKS_SHA256 = "c677e3cfdce7ddf3b6e2e9e932146e94e0ce66c52cd70940ebe0d310207896d9"
DEFAULT_OUTPUT = "artifacts/guided_iou_nearest1_676_v1/results.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transpose(matrix):
    return tuple(tuple(matrix[column][row] for column in range(3)) for row in range(3))


def guided_actions(matrix, target_matrix, nearest_count: int):
    relative = _mat_mul(target_matrix, transpose(matrix))
    residual_x, residual_y, residual_z = _matrix_to_euler_local(relative)
    residuals = {
        "PITCH": abs(residual_x),
        "YAW": abs(residual_y),
        "ROLL": abs(residual_z),
    }
    actions = []
    for axis, rotation_fn in ACTION_DIRECTIONS:
        axis_name = axis.split("_", 1)[0]
        angles = sorted(ALLOWED_ANGLES, key=lambda value: (abs(value - residuals[axis_name]), -value))[:nearest_count]
        for angle in angles:
            actions.append((axis, angle, rotation_fn(angle)))
    return actions


def reconstruct_path(parents, key):
    path = []
    while parents[key] is not None:
        parent_key, axis, angle = parents[key]
        path.append({"action": axis, "angle": angle})
        key = parent_key
    path.reverse()
    return path


def search_task(
    task: dict[str, Any],
    *,
    nearest_count: int,
    max_depth: int,
    max_states: int,
    score_chunk_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    target_pose = Pose.from_dict(task["target_pose"])
    initial_pose = Pose.from_dict(task["init_pose"])
    display_size = resolve_size(task.get("display_size"))
    eval_size = int(task.get("eval_size", 320))
    threshold = float(task.get("tau_success", 0.9))
    _, target_eval, _ = render_views_fast(
        task, target_pose, display_size=display_size, eval_size=eval_size, need_oblique=False
    )
    target_matrix = _euler_to_matrix(target_pose.roll, target_pose.pitch, target_pose.yaw)
    initial_matrix = _euler_to_matrix(initial_pose.roll, initial_pose.pitch, initial_pose.yaw)
    initial_key = matrix_key(initial_matrix)
    frontier = [(initial_matrix, initial_key, None)]
    visited = {initial_key}
    parents = {initial_key: None}
    layer_stats = []
    best_iou = -1.0
    best_depth = 0

    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for depth in range(max_depth + 1):
        layer_started = time.perf_counter()
        success_index = None
        success_iou = None
        layer_best_iou = -1.0
        scored_states = 0
        for chunk_start in range(0, len(frontier), score_chunk_size):
            chunk = frontier[chunk_start : chunk_start + score_chunk_size]
            matrices = [item[0] for item in chunk]
            scores = score_batch_cuda(task, matrices, target_eval)
            if scores is None:
                scores = []
                for matrix in matrices:
                    roll, pitch, yaw = _matrix_to_euler_local(matrix)
                    _, current_eval, _ = render_views_fast(
                        task, Pose(roll, pitch, yaw), display_size=display_size, eval_size=eval_size, need_oblique=False
                    )
                    scores.append(float(iou(target_eval, current_eval)))
            scored_states += len(scores)
            layer_best_iou = max(layer_best_iou, max(scores))
            for chunk_index, value in enumerate(scores):
                if value >= threshold:
                    success_index = chunk_start + chunk_index
                    success_iou = value
                    break
            if success_index is not None:
                break
        score_seconds = time.perf_counter() - layer_started
        if layer_best_iou > best_iou:
            best_iou = layer_best_iou
            best_depth = depth
        layer_stats.append({
            "depth": depth,
            "frontier_states": len(frontier),
            "scored_states": scored_states,
            "score_seconds": round(score_seconds, 6),
            "best_iou": round(layer_best_iou, 9),
        })
        if success_index is not None:
            success_key = frontier[success_index][1]
            elapsed = time.perf_counter() - started
            return {
                "sample_id": task["sample_id"],
                "nearest_angle_count": nearest_count,
                "search_status": "success",
                "optimal_steps_under_guided_actions": depth,
                "success_iou": round(float(success_iou), 9),
                "path": reconstruct_path(parents, success_key),
                "expanded_states": sum(row["scored_states"] for row in layer_stats),
                "visited_states": len(visited),
                "elapsed_seconds": round(elapsed, 6),
                "seconds_per_search_depth": round(elapsed / max(1, depth), 6),
                "layer_stats": layer_stats,
                "gpu_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 6)
                if torch is not None and torch.cuda.is_available() else 0.0,
            }
        if depth >= max_depth:
            break

        generation_started = time.perf_counter()
        next_frontier = []
        for matrix, key, previous_action in frontier:
            for axis, angle, delta in guided_actions(matrix, target_matrix, nearest_count):
                if previous_action == (INVERSE_AXIS[axis], angle):
                    continue
                next_matrix = _mat_mul(delta, matrix)
                next_key = matrix_key(next_matrix)
                if next_key in visited:
                    continue
                visited.add(next_key)
                parents[next_key] = (key, axis, angle)
                next_frontier.append((next_matrix, next_key, (axis, angle)))
                if len(visited) >= max_states:
                    elapsed = time.perf_counter() - started
                    return {
                        "sample_id": task["sample_id"],
                        "nearest_angle_count": nearest_count,
                        "search_status": "state_limit_reached",
                        "optimal_steps_under_guided_actions": None,
                        "best_iou": round(best_iou, 9),
                        "best_depth": best_depth,
                        "expanded_states": sum(row["scored_states"] for row in layer_stats),
                        "visited_states": len(visited),
                        "elapsed_seconds": round(elapsed, 6),
                        "layer_stats": layer_stats,
                    }
        layer_stats[-1]["generation_seconds"] = round(time.perf_counter() - generation_started, 6)
        layer_stats[-1]["next_frontier_states"] = len(next_frontier)
        frontier = next_frontier
        if not frontier:
            break

    elapsed = time.perf_counter() - started
    return {
        "sample_id": task["sample_id"],
        "nearest_angle_count": nearest_count,
        "search_status": "not_found_within_depth",
        "optimal_steps_under_guided_actions": None,
        "best_iou": round(best_iou, 9),
        "best_depth": best_depth,
        "expanded_states": sum(row["scored_states"] for row in layer_stats),
        "visited_states": len(visited),
        "elapsed_seconds": round(elapsed, 6),
        "layer_stats": layer_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-file", default="artifacts/ablation/holes_20260316_154921/attempts_0/tasks/tasks.jsonl")
    parser.add_argument("--ids", default="", help="Comma-separated subset; empty means all canonical tasks")
    parser.add_argument("--nearest-count", type=int, choices=(1, 2), default=1)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--max-states", type=int, default=2_000_000)
    parser.add_argument("--candidate-batch-size", type=int, default=32)
    parser.add_argument("--point-batch-size", type=int, default=100_000)
    parser.add_argument("--score-chunk-size", type=int, default=4096)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.48)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if torch is not None and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
    tasks_path = Path(args.tasks_file)
    tasks_sha256 = sha256_file(tasks_path)
    if tasks_sha256 != CANONICAL_TASKS_SHA256:
        raise RuntimeError("Canonical tasks SHA-256 mismatch")
    selected_ids = {value.strip() for value in args.ids.split(",") if value.strip()}
    tasks = []
    source_ids = []
    with tasks_path.open(encoding="utf-8") as handle:
        for line in handle:
            task = json.loads(line)
            source_ids.append(task["sample_id"])
            if not selected_ids or task["sample_id"] in selected_ids:
                task["bfs_candidate_batch_size"] = args.candidate_batch_size
                task["cuda_point_batch_size"] = args.point_batch_size
                tasks.append(task)
    if len(source_ids) != CANONICAL_TASK_COUNT or len(set(source_ids)) != CANONICAL_TASK_COUNT:
        raise RuntimeError("Canonical source must contain exactly 676 unique tasks")
    tasks.sort(key=lambda task: task["sample_id"])
    if selected_ids and {task["sample_id"] for task in tasks} != selected_ids:
        raise RuntimeError("Requested task IDs were not found")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol_version": "guided_iou_nearest1_v1" if args.nearest_count == 1 else "guided_iou_nearest2_comparison_v1",
        "tasks_file": str(tasks_path).replace("\\", "/"),
        "tasks_sha256": tasks_sha256,
        "canonical_task_count": CANONICAL_TASK_COUNT,
        "selected_task_count": len(tasks),
        "selected_ids": [task["sample_id"] for task in tasks] if selected_ids else "all_canonical_tasks",
        "search": {
            "directions": [axis for axis, _ in ACTION_DIRECTIONS],
            "allowed_angles_degrees": list(ALLOWED_ANGLES),
            "nearest_angle_count_per_axis": args.nearest_count,
            "relative_pose_method": "rotation_matrix",
            "success_condition": "IoU >= task tau_success (canonical value 0.9)",
            "algorithm": "strict layer-by-layer BFS; first successful layer is optimal under guided actions",
            "max_depth": args.max_depth,
            "max_states": args.max_states,
        },
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": getattr(torch, "__version__", None),
            "torch_cuda": getattr(getattr(torch, "version", None), "cuda", None),
            "cuda_available": bool(torch is not None and torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch is not None and torch.cuda.is_available() else None,
            "gpu_memory_fraction": args.gpu_memory_fraction,
            "candidate_batch_size": args.candidate_batch_size,
            "point_batch_size": args.point_batch_size,
            "score_chunk_size": args.score_chunk_size,
        },
        "scope_note": "Pose-guided constrained shortest path, not the global optimum over the full 24-action space.",
    }
    manifest_path = output.with_suffix(".manifest.json")
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise RuntimeError(f"Run manifest mismatch: {manifest_path}")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    completed = set()
    if output.exists():
        existing_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(row.get("nearest_angle_count") != args.nearest_count for row in existing_rows):
            raise RuntimeError("Existing results use a different nearest-angle count")
        completed = {row["sample_id"] for row in existing_rows}
    with output.open("a", encoding="utf-8") as handle:
        for task in tasks:
            if task["sample_id"] in completed:
                continue
            print(json.dumps({"event": "start", "sample_id": task["sample_id"], "nearest_count": args.nearest_count}), flush=True)
            result = search_task(
                task,
                nearest_count=args.nearest_count,
                max_depth=args.max_depth,
                max_states=args.max_states,
                score_chunk_size=args.score_chunk_size,
            )
            result["render_backend"] = get_last_render_backend()
            result["scoring_backend"] = (
                "cuda_batch" if torch is not None and torch.cuda.is_available() else "renderer_per_state"
            )
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
