"""Deterministic pose-guided beam search for reference task difficulty."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from benchmark_core import Pose, _euler_to_matrix, _mat_mul, get_last_render_backend, iou, render_views_fast
from bfs_iou_oracle import ACTION_DIRECTIONS, INVERSE_AXIS, _matrix_to_euler_local, matrix_key, resolve_size, score_batch_cuda
from guided_iou_bfs import (
    CANONICAL_TASK_COUNT,
    CANONICAL_TASKS_SHA256,
    guided_actions,
    reconstruct_path,
    sha256_file,
    transpose,
)
from render_protocol import (
    RENDER_PARAMETER_SCHEMA,
    STANDARD_CANDIDATE_BATCH_SIZE,
    STANDARD_DISPLAY_SIZE,
    STANDARD_IMAGE_SIZE,
    require_cuda_backend,
    STANDARD_POINT_BATCH_SIZE,
)

try:
    import torch
except Exception:
    torch = None


def residual_bucket(matrix, target_matrix, bucket_degrees: int) -> tuple[int, int, int]:
    relative = _mat_mul(target_matrix, transpose(matrix))
    residual = _matrix_to_euler_local(relative)
    return tuple(round(value / bucket_degrees) for value in residual)


def residual_magnitude(matrix, target_matrix) -> float:
    relative = _mat_mul(target_matrix, transpose(matrix))
    return sum(abs(value) for value in _matrix_to_euler_local(relative))


def prefilter_candidates(candidates, limit: int):
    if len(candidates) <= limit:
        return candidates
    by_parent = defaultdict(list)
    parent_scores = {}
    for candidate in candidates:
        parent_key, parent_score = candidate[3], candidate[4]
        by_parent[parent_key].append(candidate)
        parent_scores[parent_key] = parent_score
    parent_keys = sorted(by_parent, key=lambda key: (-parent_scores[key], key))
    for values in by_parent.values():
        values.sort(key=lambda item: (item[5], item[1]))
    selected = []
    child_rank = 0
    while len(selected) < limit:
        added = False
        for parent_key in parent_keys:
            values = by_parent[parent_key]
            if child_rank < len(values):
                selected.append(values[child_rank])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        child_rank += 1
    return selected


def select_beam(frontier, scores, target_matrix, beam_width: int, bucket_degrees: int, max_per_bucket: int):
    ranked = sorted(range(len(frontier)), key=lambda index: (-scores[index], frontier[index][1]))
    selected = []
    selected_set = set()
    bucket_counts = defaultdict(int)
    for index in ranked:
        bucket = residual_bucket(frontier[index][0], target_matrix, bucket_degrees)
        if bucket_counts[bucket] >= max_per_bucket:
            continue
        selected.append(index)
        selected_set.add(index)
        bucket_counts[bucket] += 1
        if len(selected) == beam_width:
            break
    if len(selected) < beam_width:
        for index in ranked:
            if index not in selected_set:
                selected.append(index)
                if len(selected) == beam_width:
                    break
    return selected


def score_frontier(task, frontier, target_eval, display_size, eval_size, score_chunk_size, threshold):
    all_scores = []
    success_index = None
    success_iou = None
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
        all_scores.extend(scores)
        for chunk_index, value in enumerate(scores):
            if value >= threshold:
                success_index = chunk_start + chunk_index
                success_iou = value
                break
        if success_index is not None:
            break
    return all_scores, success_index, success_iou


def search_task(
    task: dict[str, Any],
    *,
    beam_width: int,
    max_scored_per_depth: int,
    max_depth: int,
    max_states: int,
    score_chunk_size: int,
    bucket_degrees: int,
    max_per_bucket: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    target_pose = Pose.from_dict(task["target_pose"])
    initial_pose = Pose.from_dict(task["init_pose"])
    display_size = STANDARD_DISPLAY_SIZE
    eval_size = STANDARD_IMAGE_SIZE
    threshold = float(task.get("tau_success", 0.9))
    _, target_eval, _ = render_views_fast(
        task, target_pose, display_size=display_size, eval_size=eval_size, need_oblique=False
    )
    require_cuda_backend(get_last_render_backend(), context="Beam target renderer")
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
        score_started = time.perf_counter()
        scores, success_index, success_iou = score_frontier(
            task, frontier, target_eval, display_size, eval_size, score_chunk_size, threshold
        )
        layer_best = max(scores)
        if layer_best > best_iou:
            best_iou, best_depth = layer_best, depth
        layer = {
            "depth": depth,
            "candidate_states": len(frontier),
            "scored_states": len(scores),
            "score_seconds": round(time.perf_counter() - score_started, 6),
            "best_iou": round(layer_best, 9),
        }
        layer_stats.append(layer)
        if success_index is not None:
            elapsed = time.perf_counter() - started
            return {
                "sample_id": task["sample_id"],
                "beam_width": beam_width,
                "search_status": "success",
                "reference_steps": depth,
                "success_iou": round(float(success_iou), 9),
                "path": reconstruct_path(parents, frontier[success_index][1]),
                "expanded_states": sum(row["scored_states"] for row in layer_stats),
                "visited_states": len(visited),
                "elapsed_seconds": round(elapsed, 6),
                "layer_stats": layer_stats,
                "gpu_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 6)
                if torch is not None and torch.cuda.is_available() else 0.0,
            }
        if depth >= max_depth:
            break

        selected_indices = select_beam(
            frontier, scores, target_matrix, beam_width, bucket_degrees, max_per_bucket
        )
        selected = [(frontier[index], scores[index]) for index in selected_indices]
        layer["retained_states"] = len(selected)
        generation_started = time.perf_counter()
        candidates = []
        for (matrix, key, previous_action), parent_score in selected:
            for axis, angle, delta in guided_actions(matrix, target_matrix, 1):
                if previous_action == (INVERSE_AXIS[axis], angle):
                    continue
                next_matrix = _mat_mul(delta, matrix)
                next_key = matrix_key(next_matrix)
                if next_key in visited:
                    continue
                visited.add(next_key)
                parents[next_key] = (key, axis, angle)
                candidates.append((
                    next_matrix,
                    next_key,
                    (axis, angle),
                    key,
                    parent_score,
                    residual_magnitude(next_matrix, target_matrix),
                ))
                if len(visited) >= max_states:
                    return {
                        "sample_id": task["sample_id"],
                        "beam_width": beam_width,
                        "search_status": "state_limit_reached",
                        "reference_steps": None,
                        "best_iou": round(best_iou, 9),
                        "best_depth": best_depth,
                        "expanded_states": sum(row["scored_states"] for row in layer_stats),
                        "visited_states": len(visited),
                        "elapsed_seconds": round(time.perf_counter() - started, 6),
                        "layer_stats": layer_stats,
                    }
        layer["generated_states"] = len(candidates)
        candidates = prefilter_candidates(candidates, max_scored_per_depth)
        layer["candidates_after_prefilter"] = len(candidates)
        layer["generation_seconds"] = round(time.perf_counter() - generation_started, 6)
        frontier = [(item[0], item[1], item[2]) for item in candidates]
        if not frontier:
            break

    return {
        "sample_id": task["sample_id"],
        "beam_width": beam_width,
        "search_status": "not_found_within_depth",
        "reference_steps": None,
        "best_iou": round(best_iou, 9),
        "best_depth": best_depth,
        "expanded_states": sum(row["scored_states"] for row in layer_stats),
        "visited_states": len(visited),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "layer_stats": layer_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic guided IoU beam calibration")
    parser.add_argument("--tasks-file", default="artifacts/ablation/holes_20260316_154921/attempts_0/tasks/tasks.jsonl")
    parser.add_argument("--ids", default="", help="Comma-separated IDs; empty or 'all' selects all canonical tasks")
    parser.add_argument("--beam-widths", default="256")
    parser.add_argument("--max-scored-per-depth", type=int, default=5000)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--max-states", type=int, default=1_000_000)
    parser.add_argument("--candidate-batch-size", type=int, default=STANDARD_CANDIDATE_BATCH_SIZE)
    parser.add_argument("--point-batch-size", type=int, default=STANDARD_POINT_BATCH_SIZE)
    parser.add_argument("--score-chunk-size", type=int, default=4096)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.48)
    parser.add_argument("--bucket-degrees", type=int, default=10)
    parser.add_argument("--max-per-bucket", type=int, default=4)
    parser.add_argument("--output-dir", default="artifacts/guided_iou_beam_256_320_v2")
    args = parser.parse_args()

    if torch is not None and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
    tasks_path = Path(args.tasks_file)
    if sha256_file(tasks_path) != CANONICAL_TASKS_SHA256:
        raise RuntimeError("Canonical tasks SHA-256 mismatch")
    requested_ids = {value.strip() for value in args.ids.split(",") if value.strip()}
    tasks = []
    source_ids = []
    with tasks_path.open(encoding="utf-8") as handle:
        for line in handle:
            task = json.loads(line)
            source_ids.append(task["sample_id"])
            if not requested_ids or "all" in requested_ids or task["sample_id"] in requested_ids:
                task["display_size"] = {"width": STANDARD_IMAGE_SIZE, "height": STANDARD_IMAGE_SIZE}
                task["eval_size"] = STANDARD_IMAGE_SIZE
                task["bfs_candidate_batch_size"] = args.candidate_batch_size
                task["cuda_point_batch_size"] = args.point_batch_size
                tasks.append(task)
    if len(source_ids) != CANONICAL_TASK_COUNT or len(set(source_ids)) != CANONICAL_TASK_COUNT:
        raise RuntimeError("Canonical source must contain exactly 676 unique tasks")
    selected_ids = {task["sample_id"] for task in tasks}
    if requested_ids and "all" not in requested_ids and selected_ids != requested_ids:
        raise RuntimeError("Requested task IDs were not found")
    tasks.sort(key=lambda task: task["sample_id"])
    beam_widths = [int(value) for value in args.beam_widths.split(",")]
    if any(width <= 0 for width in beam_widths):
        raise ValueError("Beam widths must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol_version": "beam256_cuda320_difficulty_v2",
        "task_ids": [task["sample_id"] for task in tasks],
        "beam_widths": beam_widths,
        "display_size": [STANDARD_IMAGE_SIZE, STANDARD_IMAGE_SIZE],
        "eval_size": STANDARD_IMAGE_SIZE,
        "candidate_batch_size": args.candidate_batch_size,
        "point_batch_size": args.point_batch_size,
        "render_parameters": RENDER_PARAMETER_SCHEMA,
        "max_scored_per_depth": args.max_scored_per_depth,
        "max_depth": args.max_depth,
        "nearest_angle_count_per_axis": 1,
        "directions_per_state": 6,
        "bucket_degrees": args.bucket_degrees,
        "max_per_bucket": args.max_per_bucket,
        "scope": "deterministic reference search; not a global optimality proof",
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("Calibration manifest mismatch")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    completed_by_width = {}
    for beam_width in beam_widths:
        output = output_dir / f"beam_{beam_width}.jsonl"
        completed = set()
        if output.exists():
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            if any(row.get("beam_width") != beam_width for row in rows):
                raise RuntimeError(f"Existing results have the wrong width: {output}")
            completed = {row["sample_id"] for row in rows}
        completed_by_width[beam_width] = completed

    progress = tqdm(
        total=len(beam_widths) * len(tasks),
        initial=sum(len(values) for values in completed_by_width.values()),
        desc="beam calibration",
        unit="run",
        mininterval=1.0,
    )
    for beam_width in beam_widths:
        output = output_dir / f"beam_{beam_width}.jsonl"
        completed = completed_by_width[beam_width]
        with output.open("a", encoding="utf-8") as handle:
            for task in tasks:
                if task["sample_id"] in completed:
                    continue
                progress.set_postfix(width=beam_width, task=task["sample_id"])
                print(json.dumps({"event": "start", "sample_id": task["sample_id"], "beam_width": beam_width}), flush=True)
                result = search_task(
                    task,
                    beam_width=beam_width,
                    max_scored_per_depth=args.max_scored_per_depth,
                    max_depth=args.max_depth,
                    max_states=args.max_states,
                    score_chunk_size=args.score_chunk_size,
                    bucket_degrees=args.bucket_degrees,
                    max_per_bucket=args.max_per_bucket,
                )
                result["render_backend"] = get_last_render_backend()
                result["scoring_backend"] = "cuda_batch" if torch is not None and torch.cuda.is_available() else "renderer_per_state"
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps(result, ensure_ascii=False), flush=True)
                progress.update(1)
                progress.set_postfix(
                    width=beam_width,
                    task=task["sample_id"],
                    last_seconds=result["elapsed_seconds"],
                )
    progress.close()


if __name__ == "__main__":
    main()
