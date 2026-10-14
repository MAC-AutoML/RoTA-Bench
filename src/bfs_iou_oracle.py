"""Exact BFS difficulty under the benchmark action set and IoU success rule."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from benchmark_core import (
    Pose,
    _euler_to_matrix,
    _mat_mul,
    _rot_x,
    _rot_y,
    _rot_z,
    iou,
    render_views_fast,
)
try:
    import cv2
    import numpy as np
    import torch
    from benchmark_core import _prepare_spherical_cuda_geometry, _cuda_disk_offsets
except Exception:
    cv2 = None
    np = None
    torch = None
    _prepare_spherical_cuda_geometry = None
    _cuda_disk_offsets = None

ANGLES = (30,)
ACTION_DIRECTIONS = (
    ("PITCH_POS", _rot_x),
    ("PITCH_NEG", lambda x: _rot_x(-x)),
    ("YAW_POS", _rot_y),
    ("YAW_NEG", lambda x: _rot_y(-x)),
    ("ROLL_POS", _rot_z),
    ("ROLL_NEG", lambda x: _rot_z(-x)),
)
INVERSE_AXIS = {
    "PITCH_POS": "PITCH_NEG", "PITCH_NEG": "PITCH_POS",
    "YAW_POS": "YAW_NEG", "YAW_NEG": "YAW_POS",
    "ROLL_POS": "ROLL_NEG", "ROLL_NEG": "ROLL_POS",
}


def matrix_key(matrix) -> tuple[int, ...]:
    return tuple(round(value * 1_000_000_000) for row in matrix for value in row)


def apply_matrix(matrix, delta):
    return _mat_mul(delta, matrix)


def action_deltas() -> tuple[tuple[str, int, tuple[tuple[float, ...], ...]], ...]:
    return tuple((axis, angle, fn(angle)) for axis, fn in ACTION_DIRECTIONS for angle in ANGLES)


DELTAS = action_deltas()


def resolve_size(value: Any) -> tuple[int, int]:
    if isinstance(value, dict):
        return int(value.get("width", 640)), int(value.get("height", 360))
    if isinstance(value, (list, tuple)):
        return int(value[0]), int(value[1])
    return int(value or 640), int(value or 640)


def score_batch_cuda(task: dict[str, Any], matrices: list, target_eval) -> list[float] | None:
    if torch is None or cv2 is None or np is None or not torch.cuda.is_available():
        return None
    geometry = _prepare_spherical_cuda_geometry(task)
    if geometry is None:
        return None
    eval_size = int(task.get("eval_size", 320))
    ew = eh = eval_size
    scale = min(ew, eh) / 64.0
    cx, cy = ew / 2.0, eh / 2.0
    radius = max(1, int(round(scale * (0.36 if task.get("source_dataset") == "game_icons" else 0.2))))
    ox, oy, _ = _cuda_disk_offsets(radius)
    target = target_eval > 0
    outputs = []
    candidate_batch = max(1, int(task.get("bfs_candidate_batch_size", 4)))
    point_batch = max(20_000, int(task.get("cuda_point_batch_size", 800_000)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for candidate_start in range(0, len(matrices), candidate_batch):
        candidate_matrices = matrices[candidate_start : candidate_start + candidate_batch]
        candidate_count = len(candidate_matrices)
        pose_tensor = torch.as_tensor(np.asarray(candidate_matrices), dtype=torch.float32, device="cuda")
        masks = torch.zeros((candidate_count, ew * eh), dtype=torch.uint8, device="cuda")
        for start in range(0, int(geometry.shape[0]), point_batch):
            points = geometry[start : start + point_batch]
            coords = torch.matmul(points[None, :, :], pose_tensor.transpose(1, 2))
            xi = torch.round(cx + coords[:, :, 0] * scale).to(torch.int64)
            yi = torch.round(cy + coords[:, :, 1] * scale).to(torch.int64)
            for offset_x, offset_y in zip(ox, oy):
                xx = xi + offset_x
                yy = yi + offset_y
                valid = (xx >= 0) & (xx < ew) & (yy >= 0) & (yy < eh)
                flat = yy.clamp(0, eh - 1) * ew + xx.clamp(0, ew - 1)
                values = valid.to(dtype=torch.uint8) * 255
                masks.scatter_reduce_(1, flat, values, reduce="amax", include_self=True)
        masks = masks.reshape(candidate_count, eh, ew).cpu().numpy()
        for mask in masks:
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            a = closed > 0
            b = target
            union = np.count_nonzero(a | b)
            outputs.append(float(np.count_nonzero(a & b) / union) if union else 1.0)
    return outputs


def search_task(task: dict[str, Any], max_depth: int = 16) -> dict[str, Any]:
    started = time.perf_counter()
    target = Pose.from_dict(task["target_pose"])
    initial = Pose.from_dict(task["init_pose"])
    display_size = resolve_size(task.get("display_size"))
    eval_size = int(task.get("eval_size", 320))
    threshold = float(task.get("tau_success", 0.9))
    _, target_eval, _ = render_views_fast(
        task, target, display_size=display_size, eval_size=eval_size, need_oblique=False
    )

    initial_matrix = _euler_to_matrix(initial.roll, initial.pitch, initial.yaw)
    frontier = [(initial_matrix, None)]
    visited = {matrix_key(initial_matrix)}
    expanded = 0
    best_iou = -1.0
    best_depth = None
    for depth in range(max_depth + 1):
        if not frontier:
            break
        matrices = [item[0] for item in frontier]
        scores = score_batch_cuda(task, matrices, target_eval)
        if scores is None:
            scores = []
            for matrix in matrices:
                roll, pitch, yaw = _matrix_to_euler_local(matrix)
                _, current_eval, _ = render_views_fast(
                    task, Pose(roll, pitch, yaw), display_size=display_size, eval_size=eval_size, need_oblique=False
                )
                scores.append(float(iou(target_eval, current_eval)))
        expanded += len(frontier)
        for current_iou in scores:
            if current_iou > best_iou:
                best_iou = current_iou
                best_depth = depth
        success_indices = [index for index, value in enumerate(scores) if value >= threshold]
        if success_indices:
            index = success_indices[0]
            return {
                "sample_id": str(task["sample_id"]),
                "bfs_iou_optimal_steps": depth,
                "success_iou": round(scores[index], 9),
                "expanded_states": expanded,
                "visited_states": len(visited),
                "search_status": "success",
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            }
        if depth >= max_depth:
            continue
        next_frontier = []
        for matrix, previous_action in frontier:
            for axis, angle, delta in DELTAS:
                if previous_action and (axis, angle) == (INVERSE_AXIS[previous_action[0]], previous_action[1]):
                    continue
                next_matrix = apply_matrix(matrix, delta)
                key = matrix_key(next_matrix)
                if key in visited:
                    continue
                visited.add(key)
                next_frontier.append((next_matrix, (axis, angle)))
        frontier = next_frontier
    return {
        "sample_id": str(task["sample_id"]),
        "bfs_iou_optimal_steps": None,
        "success_iou": round(best_iou, 9),
        "best_depth": best_depth,
        "expanded_states": expanded,
        "visited_states": len(visited),
        "search_status": "not_found_within_depth",
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def _matrix_to_euler_local(matrix):
    value = max(-1.0, min(1.0, -matrix[2][0]))
    import math
    pitch = math.degrees(math.asin(value))
    cosine = math.cos(math.radians(pitch))
    if abs(cosine) < 1e-8:
        return 0.0, pitch, math.degrees(math.atan2(-matrix[0][1], matrix[1][1]))
    roll = math.degrees(math.atan2(matrix[2][1], matrix[2][2]))
    yaw = math.degrees(math.atan2(matrix[1][0], matrix[0][0]))
    return roll, pitch, yaw


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-file", default="artifacts/ablation/holes_20260316_154921/attempts_0/tasks/tasks.jsonl")
    parser.add_argument("--output", default="artifacts/bfs_iou_v1/results.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids-file", default="", help="JSON split file containing sample_ids")
    parser.add_argument("--upper-bounds-csv", default="", help="CSV containing sample_id and bfs_optimal_steps")
    parser.add_argument("--max-depth", type=int, default=16)
    args = parser.parse_args()
    tasks_path = Path(args.tasks_file)
    selected_ids = None
    if args.ids_file:
        split_payload = json.loads(Path(args.ids_file).read_text(encoding="utf-8"))
        selected_ids = {str(value) for value in split_payload["sample_ids"]}
    upper_bounds = {}
    if args.upper_bounds_csv:
        with Path(args.upper_bounds_csv).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                upper_bounds[str(row["sample_id"])] = int(row["bfs_optimal_steps"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    completed_ids = set()
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            completed_ids = {str(json.loads(line)["sample_id"]) for line in handle if line.strip()}
    mode = "a" if output.exists() else "w"
    selected_tasks = []
    with tasks_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            task = json.loads(line)
            if selected_ids is not None and str(task["sample_id"]) not in selected_ids:
                continue
            if str(task["sample_id"]) in completed_ids:
                continue
            selected_tasks.append(task)
    selected_tasks.sort(key=lambda task: (upper_bounds.get(str(task["sample_id"]), args.max_depth), str(task["sample_id"])))
    with output.open(mode, encoding="utf-8") as dest:
        for task in selected_tasks:
            if args.limit and count >= args.limit:
                break
            task_max_depth = min(args.max_depth, upper_bounds.get(str(task["sample_id"]), args.max_depth))
            print(json.dumps({"event": "start_task", "sample_id": task["sample_id"], "max_depth": task_max_depth}), flush=True)
            result = search_task(task, task_max_depth)
            dest.write(json.dumps(result, ensure_ascii=False) + "\n")
            dest.flush()
            count += 1
            print(json.dumps(result, ensure_ascii=False), flush=True)
    print(json.dumps({"processed": count, "angles": list(ANGLES), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
