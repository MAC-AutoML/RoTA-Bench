"""Compute exact shortest 30-degree action paths for generated task poses."""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Iterable

import numpy as np

from benchmark_core import Pose, _euler_to_matrix, _mat_mul, _rot_x, _rot_y, _rot_z


ACTIONS = (
    "PITCH_POS",
    "PITCH_NEG",
    "YAW_POS",
    "YAW_NEG",
    "ROLL_POS",
    "ROLL_NEG",
)
INVERSE_ACTION = {
    "PITCH_POS": "PITCH_NEG",
    "PITCH_NEG": "PITCH_POS",
    "YAW_POS": "YAW_NEG",
    "YAW_NEG": "YAW_POS",
    "ROLL_POS": "ROLL_NEG",
    "ROLL_NEG": "ROLL_POS",
}


def _flatten(matrix: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    return tuple(value for row in matrix for value in row)


def _state_key(matrix: tuple[tuple[float, ...], ...]) -> tuple[int, ...]:
    return tuple(round(value * 1_000_000_000) for value in _flatten(matrix))


def _delta_matrices(theta: int) -> dict[str, tuple[tuple[float, ...], ...]]:
    return {
        "PITCH_POS": _rot_x(theta),
        "PITCH_NEG": _rot_x(-theta),
        "YAW_POS": _rot_y(theta),
        "YAW_NEG": _rot_y(-theta),
        "ROLL_POS": _rot_z(theta),
        "ROLL_NEG": _rot_z(-theta),
    }


def matrix_from_actions(actions: Iterable[str], theta: int = 30):
    matrix = _euler_to_matrix(0.0, 0.0, 0.0)
    deltas = _delta_matrices(theta)
    for action in actions:
        matrix = _mat_mul(deltas[action], matrix)
    return matrix


def matrix_error(left, right) -> float:
    return max(abs(a - b) for a, b in zip(_flatten(left), _flatten(right)))


def bfs_distances_for_tasks(
    tasks: list[dict[str, Any]], theta: int = 30
) -> tuple[dict[str, int], dict[str, Any]]:
    """Find exact distances by intersecting two radius-eight BFS balls."""
    targets: dict[str, tuple[tuple[float, ...], ...]] = {}
    upper_bounds: dict[str, int] = {}
    pose_errors: dict[str, float] = {}
    max_scramble_len = 0
    for task in tasks:
        sample_id = str(task["sample_id"])
        actions = [str(action) for action in task["scramble_actions"]]
        if int(task["theta"]) != theta:
            raise ValueError(f"{sample_id}: expected theta={theta}")
        generated = matrix_from_actions(actions, theta)
        stored = _euler_to_matrix(**{
            "roll_deg": float(task["init_pose"]["roll"]),
            "pitch_deg": float(task["init_pose"]["pitch"]),
            "yaw_deg": float(task["init_pose"]["yaw"]),
        })
        error = matrix_error(generated, stored)
        if error > 2e-8:
            raise ValueError(f"{sample_id}: stored pose differs from scramble path by {error}")
        pose_errors[sample_id] = error
        targets[sample_id] = generated
        upper_bounds[sample_id] = len(actions)
        max_scramble_len = max(max_scramble_len, upper_bounds[sample_id])

    identity = _euler_to_matrix(0.0, 0.0, 0.0)
    identity_key = _state_key(identity)
    queue = deque([(identity, 0, "")])
    visited = {identity_key: 0}
    matrices = [identity]
    depths = [0]
    depth_counts: dict[int, int] = {}
    deltas = _delta_matrices(theta)
    radius = math.ceil(max_scramble_len / 2)

    while queue:
        matrix, depth, previous_action = queue.popleft()
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
        if depth >= radius:
            continue
        for action in ACTIONS:
            if previous_action and action == INVERSE_ACTION[previous_action]:
                continue
            nxt = _mat_mul(deltas[action], matrix)
            nxt_key = _state_key(nxt)
            if nxt_key in visited:
                continue
            visited[nxt_key] = depth + 1
            matrices.append(nxt)
            depths.append(depth + 1)
            queue.append((nxt, depth + 1, action))

    matrix_array = np.asarray(matrices, dtype=np.float64)
    depth_array = np.asarray(depths, dtype=np.int16)
    key_array = np.rint(matrix_array.reshape(-1, 9) * 1_000_000_000).astype(np.int64)
    row_dtype = np.dtype((np.void, key_array.dtype.itemsize * key_array.shape[1]))
    key_rows = np.ascontiguousarray(key_array).view(row_dtype).reshape(-1)
    order = np.argsort(key_rows)
    sorted_rows = key_rows[order]
    sorted_depths = depth_array[order]

    distances: dict[str, int] = {}
    for sample_id in sorted(targets):
        upper = upper_bounds[sample_id]
        source_radius = upper // 2
        target_radius = upper - source_radius
        source_indices = np.flatnonzero(depth_array <= source_radius)
        target = np.asarray(targets[sample_id], dtype=np.float64)
        best = upper + 1
        for start in range(0, len(source_indices), 50_000):
            indices = source_indices[start : start + 50_000]
            transformed = np.matmul(target, np.transpose(matrix_array[indices], (0, 2, 1)))
            transformed_keys = np.rint(transformed.reshape(-1, 9) * 1_000_000_000).astype(np.int64)
            query_rows = np.ascontiguousarray(transformed_keys).view(row_dtype).reshape(-1)
            positions = np.searchsorted(sorted_rows, query_rows)
            valid = positions < len(sorted_rows)
            safe_positions = np.minimum(positions, len(sorted_rows) - 1)
            valid &= sorted_rows[safe_positions] == query_rows
            valid &= sorted_depths[safe_positions] <= target_radius
            if np.any(valid):
                totals = depth_array[indices[valid]] + sorted_depths[safe_positions[valid]]
                best = min(best, int(np.min(totals)))
        if best > upper:
            raise RuntimeError(f"No bidirectional BFS intersection for {sample_id} within {upper} steps")
        distances[sample_id] = best

    metadata = {
        "algorithm": "exact bidirectional breadth-first search",
        "state": "3x3 observer-frame rotation matrix quantized to 1e-9",
        "target": "exact target orientation (identity for canonical tasks)",
        "theta_degrees": theta,
        "actions": list(ACTIONS),
        "inverse-action_pruning": True,
        "search_depth_limit_per_side": radius,
        "task_upper_bound": "length of verified generated scramble path (6-16)",
        "visited_state_count": len(visited),
        "expanded_state_count_by_depth": {str(k): v for k, v in sorted(depth_counts.items())},
        "max_stored_pose_matrix_error": max(pose_errors.values()),
    }
    return distances, metadata
