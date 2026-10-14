#!/usr/bin/env python3
"""Step 02: test models with interactive step-loop (simulated or Gemini API)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import base64
import csv
import hashlib
import json
import math
import os
import pickle
import random
import re
import subprocess
import sys
import time
import platform
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable
try:
    import numpy as np
except Exception:
    np = None
try:
    import cv2
except Exception:
    cv2 = None

from llm_client import API_KEY, BASE_URL, DEFAULT_MODEL, chat_completions
from benchmark_core import (
    ACTIONS,
    Pose,
    apply_action,
    get_last_render_backend,
    iou,
    now_ms,
    read_json,
    render_views_fast,
    resolve_paths,
    write_json,
    write_png_gray,
)
from render_protocol import (
    RENDER_PARAMETER_SCHEMA,
    STANDARD_DISPLAY_SIZE,
    STANDARD_IMAGE_SIZE,
    STANDARD_POINT_BATCH_SIZE,
    STANDARD_PANEL_SEPARATOR,
    STANDARD_PANEL_SEPARATOR_VALUE,
    require_standard_square_size,
    require_cuda_backend,
)

BASE_ACTIONS = [a for a in ACTIONS if a != "STOP"]
ALL_ACTIONS = BASE_ACTIONS + ["STOP"]
VALID_STEP_ANGLES = [30, 10, 5, 1]
PROMPT_PROFILES = ["baseline", "geometry_audit", "projection_first", "game_manual_plus"]
METRIC_FEEDBACK_MODES = ["full", "threshold", "none"]
METRIC_BLIND_SENTENCE = (
    "The task ends automatically when the requirement is met. "
    "If it continues, the current result has not yet met the requirement."
)
RENDER_SEMAPHORE_ENV = "PROJECTION_RENDER_SEMAPHORE"
RENDER_SEMAPHORE_COUNT_ENV = "PROJECTION_RENDER_SEMAPHORE_COUNT"


def render_views_limited(*args, **kwargs):
    """Serialize CUDA rendering across parallel task subprocesses on Windows."""
    semaphore_name = os.environ.get(RENDER_SEMAPHORE_ENV, "").strip()
    if not semaphore_name or os.name != "nt":
        result = render_views_fast(*args, **kwargs)
        require_cuda_backend(get_last_render_backend(), context="LLM benchmark")
        return result

    import ctypes

    count = max(1, int(os.environ.get(RENDER_SEMAPHORE_COUNT_ENV, "1")))
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateSemaphoreW.restype = ctypes.c_void_p
    kernel32.CreateSemaphoreW.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_long, ctypes.c_wchar_p]
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel32.ReleaseSemaphore.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.CreateSemaphoreW(None, count, count, semaphore_name)
    if not handle:
        raise OSError(ctypes.get_last_error(), "CreateSemaphoreW failed")
    try:
        wait_result = kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
        if wait_result != 0:
            raise OSError(ctypes.get_last_error(), f"WaitForSingleObject failed: {wait_result}")
        try:
            result = render_views_fast(*args, **kwargs)
            require_cuda_backend(get_last_render_backend(), context="LLM benchmark")
            return result
        finally:
            if not kernel32.ReleaseSemaphore(handle, 1, None):
                raise OSError(ctypes.get_last_error(), "ReleaseSemaphore failed")
    finally:
        kernel32.CloseHandle(handle)


def read_selected_tasks_and_ids(
    path: Path, max_tasks: int, split_path: Optional[Path] = None
) -> tuple[List[Dict], List[str]]:
    tasks_by_id: Dict[str, Dict] = {}
    all_task_ids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            task = json.loads(s)
            sample_id = str(task.get("sample_id", "")).strip()
            if sample_id:
                all_task_ids.append(sample_id)
            if sample_id:
                if sample_id in tasks_by_id:
                    raise RuntimeError(f"Duplicate task ID in tasks file: {sample_id}")
                tasks_by_id[sample_id] = task
    if split_path is not None:
        split = read_json(split_path)
        split_ids = [str(value) for value in split.get("sample_ids", [])]
        if len(split_ids) != len(set(split_ids)):
            raise RuntimeError(f"Duplicate task ID in split file: {split_path}")
        missing = [sample_id for sample_id in split_ids if sample_id not in tasks_by_id]
        if missing:
            raise RuntimeError(f"Split IDs missing from tasks file: {missing[:5]}")
        selected = [tasks_by_id[sample_id] for sample_id in split_ids]
    else:
        selected = [tasks_by_id[sample_id] for sample_id in all_task_ids]
    if max_tasks > 0:
        selected = selected[:max_tasks]
    return selected, all_task_ids


def resolve_optional_path(root: Path, value: str) -> Optional[Path]:
    if not str(value or "").strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def resolve_stop_outcome(score: float, tau_success: float) -> tuple[bool, str]:
    if score >= tau_success:
        return True, "threshold_reached"
    return False, "stop_no_further_idea"


def select_history_slice(items: List, history_window: int) -> List:
    if history_window < 0:
        return list(items)
    if history_window == 0:
        return []
    return list(items[-history_window:])


def resolve_metric_feedback_mode(mode: str, hide_iou_feedback: bool = False) -> str:
    if hide_iou_feedback:
        return "none"
    value = str(mode or "full").strip().lower()
    if value not in METRIC_FEEDBACK_MODES:
        raise ValueError(f"Unsupported metric feedback mode: {mode}")
    return value


def build_metric_guidance(mode: str, current_iou: float, tau_success: float) -> str:
    mode = resolve_metric_feedback_mode(mode)
    if mode == "full":
        return f"Current IoU before the next action: {current_iou:.6f}. Success threshold: {tau_success:.6f}."
    if mode == "threshold":
        return "The success condition is IoU >= 0.9."
    return METRIC_BLIND_SENTENCE


def metric_feedback_result_dir(mode: str) -> str:
    return {
        "none": "no_iou",
        "threshold": "iou_threshold_0.9",
        "full": "iou_feedback",
    }[resolve_metric_feedback_mode(mode)]


def render_environment_metadata() -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "renderer": "benchmark_core.render_views_fast",
        "render_parameters": RENDER_PARAMETER_SCHEMA,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": getattr(np, "__version__", "unavailable"),
        "opencv_version": getattr(cv2, "__version__", "unavailable"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
    }
    try:
        import torch

        metadata["torch_version"] = str(torch.__version__)
        metadata["torch_cuda_version"] = str(torch.version.cuda or "")
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        metadata["cuda_device_count"] = int(torch.cuda.device_count())
        metadata["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        metadata["torch_import_error"] = ""
    except Exception as exc:
        metadata["torch_version"] = "unavailable"
        metadata["torch_cuda_version"] = ""
        metadata["cuda_available"] = False
        metadata["cuda_device_count"] = 0
        metadata["cuda_device"] = ""
        metadata["torch_import_error"] = f"{type(exc).__name__}: {exc}"
    return metadata


def mask_sha256(mask) -> str:
    if np is not None and isinstance(mask, np.ndarray):
        return hashlib.sha256(mask.tobytes(order="C")).hexdigest()
    return hashlib.sha256(json.dumps(mask, separators=(",", ":")).encode("utf-8")).hexdigest()


def parse_allowed_angles(text: str) -> List[int]:
    vals: List[int] = []
    for p in str(text or "").split(","):
        s = p.strip()
        if not s:
            continue
        try:
            v = int(s)
        except Exception:
            continue
        if v in VALID_STEP_ANGLES and v not in vals:
            vals.append(v)
    return vals if vals else list(VALID_STEP_ANGLES)


def build_prompt_text_v1(
    *,
    effective_mode: str,
    prompt_profile: str,
    token_set_text: str,
    angle_set_text: str,
    step: int,
    step_limit: int,
    current_iou: float,
    tau_success: float,
    last_action: str,
    history_actions: List[str],
    history_window: int,
    recent_state_summaries: List[str],
    stop_caution: bool = False,
    hide_iou_feedback: bool = False,
    legacy_stop_as_success: bool = False,
) -> str:
    history_selected_actions = select_history_slice(history_actions, int(history_window))
    history_selected_summaries = select_history_slice(recent_state_summaries, int(history_window))
    history_label = "all previous states" if int(history_window) < 0 else f"up to {max(0, int(history_window))} previous states"
    observation_text = "images" if hide_iou_feedback else "images and feedback"
    task_intro = (
        "Task: interactive 3D-to-2D alignment game.\n"
        "Goal: rotate the current 3D object so that CURRENT projection matches TARGET projection as closely as possible.\n"
        f"This is a turn-based game: in each turn you observe the {observation_text}, then choose exactly one rotation action.\n"
        "After your action, the environment updates the object pose and gives you a new observation on the next turn.\n"
        "Your objective is to maximize alignment quality within the available turn budget.\n"
    )
    stop_protocol = (
        "Termination protocol:\n"
        "- The environment automatically ends the task successfully as soon as the hidden IoU reaches 0.9.\n"
        "- Do not use STOP to indicate that the projections appear aligned.\n"
        "- Use STOP only when you have no remaining actionable idea and believe further legal rotations cannot improve alignment.\n"
        if not legacy_stop_as_success
        else ""
    )

    input_definition = (
        "Image meaning:\n"
        "- TARGET projection: desired silhouette to match.\n"
        "- CURRENT oblique view: current 3D appearance from an oblique camera.\n"
        "- CURRENT projection: current silhouette that should align to TARGET.\n"
        "- The fixed panel order is always TARGET | CURRENT oblique | CURRENT projection.\n"
    )

    if hide_iou_feedback:
        feedback_definition = (
            "Turn budget before you choose the next action:\n"
            f"- Used steps: {max(0, step - 1)} / {step_limit}\n"
            f"- Remaining steps: {max(0, step_limit - step + 1)}\n"
            + ("- STOP means no remaining actionable idea, not visual success.\n" if not legacy_stop_as_success else "- Stop only when the silhouettes are visually already aligned.\n")
        )
    else:
        feedback_definition = (
            "Environment feedback before you choose the next action:\n"
            f"- Current IoU: {current_iou:.6f}\n"
            f"- Success threshold: {tau_success:.6f}\n"
            f"- Used steps: {max(0, step - 1)} / {step_limit}\n"
            f"- Remaining steps: {max(0, step_limit - step + 1)}\n"
            + ("- Higher IoU is better. The environment automatically ends the task successfully at IoU 0.9; STOP means no remaining actionable idea.\n" if not legacy_stop_as_success else "- Higher IoU is better. Stop only when current IoU is already at or above the success threshold or the silhouettes are visually already aligned.\n")
            + (
                "- Before choosing STOP, compare Current IoU with the Success threshold. If Current IoU is far below the threshold, do not assume success from visual impression; choose a non-STOP rotation to keep searching.\n"
                if stop_caution
                else ""
            )
            + "- The IoU shown here is the score of the CURRENT state before you choose the next action.\n"
        )

    score_rules = "" if hide_iou_feedback else (
        "- IoU is the official score used by the environment. Larger IoU means CURRENT projection overlaps TARGET projection better.\n"
        "- You may use both the images and the score feedback to decide the next move.\n"
        "- Success is determined only by the environment score threshold, not by language confidence.\n"
    )
    rule_definition = (
        "Game rules:\n"
        "- You control only the CURRENT object, never the TARGET.\n"
        "- Each turn you may output exactly one action token and one allowed angle.\n"
        "- The camera is fixed. Rotation tokens are defined in the observer/camera coordinate system.\n"
        "- The environment is deterministic: the next observation depends only on the current pose and your chosen action.\n"
        + score_rules
    )

    action_definition = "".join(
        [
            "Action tokens:\n",
            "- PITCH_POS / PITCH_NEG: observer-frame pitch rotation by +angle / -angle around the camera-fixed horizontal axis.\n",
            "- Intuition: PITCH_POS pitches the CURRENT object upward relative to the observer; PITCH_NEG pitches it downward relative to the observer.\n",
            "- YAW_POS / YAW_NEG: observer-frame yaw rotation by +angle / -angle around the camera-fixed vertical axis.\n",
            "- Intuition: YAW_POS turns the CURRENT object toward the observer's right; YAW_NEG turns it toward the observer's left.\n",
            "- ROLL_POS / ROLL_NEG: observer-frame roll rotation by +angle / -angle around the viewing axis.\n",
            "- In the image plane, ROLL_POS is clockwise and ROLL_NEG is counter-clockwise.\n",
            "- STOP: choose this only when you have no remaining actionable idea and believe further legal rotations cannot improve alignment.\n"
            if not legacy_stop_as_success
            else "- STOP: choose this only when you believe CURRENT projection is already aligned with TARGET.\n",
            (
                "- STOP caution: if the reported IoU is still far below the success threshold, treat STOP as premature and prefer another legal rotation.\n"
                if stop_caution and not hide_iou_feedback
                else ""
            ),
        ]
    )

    score_guidance = "" if hide_iou_feedback else "- Use the IoU feedback as evidence about how well the current state matches the target.\n"
    guidance_definition = (
        "Decision goal:\n"
        "- Your job is not to describe the image, but to choose the single next move that you believe will improve alignment.\n"
        "- Use the TARGET vs CURRENT projection comparison to judge mismatch.\n"
        "- Use the oblique view to understand the 3D pose and how a rotation might change the projection.\n"
        + score_guidance
        + "- Treat each turn as a fresh decision based on the latest observation and feedback.\n"
        + "- Think freely about strategy, but obey the action set and angle set exactly.\n"
    )

    history_definition = (
        "History semantics:\n"
        "- The history is a temporal sequence, not a bag of examples.\n"
        "- The order is always: previous state image -> action taken at that state -> next state image -> next action -> next state image ...\n"
        "- Therefore each historical panel shows a real past state before the following recorded action in the history list.\n"
        "- Read recent panels from oldest to newest. They are consecutive earlier turns leading up to the current state.\n"
        "- If you receive N recent panels, then the actions listed alongside them describe the transitions between those past states and the later states.\n"
        "- Use history to infer how the environment responds to actions, but act for the current final panel only.\n"
    )

    output_schema = (
        "Output must be JSON object only: {\"cot\":\"<your brief reasoning>\",\"type\":\"<TOKEN>\",\"angle\":<ANGLE>}\n"
        "Requirements:\n"
        f"- type must be one of: {token_set_text}.\n"
        f"- angle must be one of: {angle_set_text} (even when type is STOP).\n"
        "- cot should briefly explain why this action is chosen for the current step.\n"
        "- Any token outside this set is INVALID and will be treated as a failed answer.\n"
        "Valid examples:\n"
        "{\"cot\":\"Current view is tilted; adjust pitch.\",\"type\":\"PITCH_NEG\",\"angle\":30}\n"
        "{\"cot\":\"Need fine correction.\",\"type\":\"YAW_POS\",\"angle\":5}\n"
        "{\"cot\":\"Current and target silhouettes are nearly identical.\",\"type\":\"STOP\",\"angle\":10}\n"
        "Invalid example (DO NOT USE):\n"
        "{\"cot\":\"...\",\"type\":\"R_X+\",\"angle\":7}\n"
    )

    if effective_mode == "panel":
        layout_line = "Single panel layout: left=TARGET projection, middle=CURRENT oblique view, right=CURRENT projection.\n"
    else:
        layout_line = (
            "Image 1: target projection silhouette.\n"
            "Image 2: current oblique RGB-like view.\n"
            "Image 3: current projection silhouette.\n"
        )

    baseline_first_line = (
        "1. The right CURRENT projection is the matching view that must match the left TARGET projection.\n"
        if hide_iou_feedback
        else "1. The right CURRENT projection is the scoring view that must match the left TARGET projection.\n"
    )
    game_manual_plus = [
        "Interpretation notes:\n",
        "1. TARGET is fixed; only CURRENT can change from turn to turn.\n",
        "2. The oblique image is not the matching view; it is an auxiliary 3D cue for understanding pose.\n",
        "3. The objective is to match TARGET projection and CURRENT projection.\n",
    ] if hide_iou_feedback else [
        "Interpretation notes:\n",
        "1. TARGET is fixed; only CURRENT can change from turn to turn.\n",
        "2. The oblique image is not the scoring view; it is an auxiliary 3D cue for understanding pose.\n",
        "3. The scoring objective is overlap between TARGET projection and CURRENT projection.\n",
        "4. The reported IoU belongs to the present state before your next action, so use it together with the current images.\n",
        "5. Recent history is context, but the current images and current score are the authoritative game state.\n",
    ]
    profile_lines = {
        "baseline": [
            "Baseline note:\n",
            baseline_first_line,
            "2. The middle oblique image is only auxiliary 3D pose information for choosing the next move.\n",
        ],
        "geometry_audit": [
            "Decision procedure:\n",
            "1. Compare CURRENT projection against TARGET and identify the dominant mismatch.\n",
            "2. Use CURRENT oblique view only to infer which axis/direction can reduce that mismatch.\n",
            "3. Use 30 for obvious large mismatch, 10/5/1 for refinement.\n",
            "4. Use STOP only when silhouettes are nearly identical.\n",
        ],
        "projection_first": [
            "Priority rule:\n",
            "1. First trust the two silhouettes (TARGET vs CURRENT projection).\n",
            "2. Use the oblique view only as a tie-breaker for axis sign or depth ambiguity.\n",
            "3. If unsure, prefer a conservative corrective move over STOP.\n",
        ],
        "game_manual_plus": game_manual_plus,
    }
    return "".join(
        [
            task_intro,
            stop_protocol,
            output_schema,
            layout_line,
            input_definition,
            feedback_definition,
            rule_definition,
            action_definition,
            guidance_definition,
            history_definition,
            "".join(profile_lines.get(prompt_profile, [])),
            f"Current step: {step}.\n",
            f"Step limit: {step_limit}.\n",
            f"Last action: {last_action if last_action else 'NONE'}.\n",
            f"History window requested: {history_label}.\n",
            f"Recent actions in temporal order: {', '.join(history_selected_actions) if history_selected_actions else 'NONE'}.\n",
            "Recent step summaries:\n",
            "".join(f"- {line}\n" for line in history_selected_summaries) if history_selected_summaries else "- NONE\n",
        ]
    )


def build_system_prompt_v1(token_set_text: str, angle_set_text: str, history_window: int, legacy_stop_as_success: bool = False) -> str:
    history_label = "all previous states" if int(history_window) < 0 else f"up to {history_window} previous states"
    return (
        "You are playing a turn-based 3D alignment game. "
        "Return one JSON object only: {\"cot\":\"...\",\"type\":\"TOKEN\",\"angle\":30}. "
        f"TOKEN must be exactly one of: {token_set_text}. "
        f"ANGLE must be exactly one of: {angle_set_text}. "
        "The CURRENT object is controlled in observer coordinates: "
        "YAW_POS=turn toward observer right, YAW_NEG=turn toward observer left, "
        "PITCH_POS=pitch upward, PITCH_NEG=pitch downward, "
        "ROLL_POS=clockwise in the image plane, ROLL_NEG=counter-clockwise in the image plane. "
        f"If history is provided, it is a chronological sequence of {history_label}. "
        "Interpret it as state -> action -> next state -> action -> next state, with panels shown oldest to newest before the current state. "
        + ("The environment automatically terminates successfully when hidden IoU reaches 0.9. Do not use STOP to signal visual alignment; use STOP only when no actionable improvement idea remains. " if not legacy_stop_as_success else "")
        + "Keep cot concise so your entire output completion stays under 4000 output tokens. "
        "No extra text."
    )


def build_system_prompt_v2(
    token_set_text: str,
    angle_set_text: str,
    *,
    hide_iou_feedback: bool = False,
    legacy_stop_as_success: bool = False,
) -> str:
    stop_text = (
        "The environment ends automatically when the requirement is met. Use STOP only when no actionable improvement idea remains; "
        "do not use STOP merely because the silhouettes appear aligned."
        if not legacy_stop_as_success
        else "Use STOP only when CURRENT projection appears aligned with TARGET."
    )
    return (
        "You are playing a deterministic turn-based 3D-to-2D alignment game. "
        "Rotate only the CURRENT object so its projection matches TARGET within the step budget. "
        "Each observation provides images and the current task feedback. "
        "Each panel is one image ordered left-to-right as TARGET projection | CURRENT oblique view | CURRENT projection. "
        "The oblique view is an auxiliary 3D pose cue; compare TARGET with CURRENT projection for alignment. "
        "Rotations use observer/camera coordinates: PITCH_POS=pitch upward, PITCH_NEG=pitch downward, "
        "YAW_POS=turn toward observer right, YAW_NEG=turn toward observer left, "
        "ROLL_POS=clockwise in the image plane, ROLL_NEG=counter-clockwise. "
        f"Allowed action tokens: {token_set_text}. Allowed angles: {angle_set_text}. "
        "History is chronological. Each historical panel is the state before the action written immediately beside it. "
        f"{stop_text} "
        "Return exactly one JSON object and no extra text: "
        '{"cot":"brief current-step reasoning","type":"TOKEN","angle":ANGLE}. '
        "The type and angle must come from the allowed sets."
    )


def build_prompt_text_v2(
    *,
    step: int,
    step_limit: int,
    current_iou: float,
    tau_success: float,
    metric_guidance: str,
    hide_iou_feedback: bool = False,
) -> str:
    lines = [
        f"Current turn: {step}/{step_limit}.",
        f"Used steps: {max(0, step - 1)}.",
        f"Remaining steps: {max(0, step_limit - step + 1)}.",
    ]
    lines.append(metric_guidance)
    return "\n".join(lines) + "\n"


def resolve_effective_allowed_angles(task_angles, global_allowed_angles: List[int]) -> List[int]:
    task_vals: List[int] = []
    for x in task_angles or []:
        try:
            v = int(x)
        except Exception:
            continue
        if v in VALID_STEP_ANGLES and v not in task_vals:
            task_vals.append(v)
    effective = [v for v in global_allowed_angles if v in VALID_STEP_ANGLES]
    if not effective:
        effective = list(VALID_STEP_ANGLES)
    for v in task_vals:
        if v not in effective:
            effective.append(v)
    return effective


def resolve_effective_tau(task: Dict, cfg: Dict) -> float:
    if "tau_success" in cfg:
        try:
            return float(cfg["tau_success"])
        except Exception:
            pass
    return float(task.get("tau_success", 0.9))


def _inverse_action(action: str) -> str:
    mapping = {
        "PITCH_POS": "PITCH_NEG",
        "PITCH_NEG": "PITCH_POS",
        "YAW_POS": "YAW_NEG",
        "YAW_NEG": "YAW_POS",
        "ROLL_POS": "ROLL_NEG",
        "ROLL_NEG": "ROLL_POS",
    }
    return mapping.get(action, "")


def normalize_action_and_angle(action: str, angle: int, allowed_angles: List[int]) -> tuple[str, int]:
    a = str(action or "").strip().upper()
    if a.endswith("_SMALL"):
        a = a.replace("_SMALL", "")
        angle = 10
    if a not in ALL_ACTIONS:
        return "", 0
    try:
        ang = int(angle)
    except Exception:
        ang = allowed_angles[0]
    if ang not in allowed_angles:
        return "", 0
    return a, ang


def apply_action_with_angle(current: Pose, action: str, angle: int) -> Pose:
    return apply_action(current, action, int(angle))


def choose_angle_simulated(delta_abs: float, allowed_angles: List[int]) -> int:
    if 30 in allowed_angles and delta_abs >= 20:
        return 30
    if 10 in allowed_angles and delta_abs >= 8:
        return 10
    if 5 in allowed_angles:
        return 5
    if 1 in allowed_angles:
        return 1
    return allowed_angles[0]


def choose_action_simulated(
    rng: random.Random,
    current: Pose,
    target: Pose,
    stream_type: str,
    recent_actions: List[str],
    allowed_actions: List[str],
    allowed_angles: List[int],
) -> tuple[str, int]:
    follow_prob = 0.70
    oscillation_prob = 0.14
    if stream_type not in ("single", ""):
        follow_prob = 0.62
        oscillation_prob = 0.18

    if recent_actions:
        last = recent_actions[-1]
        opposite = _inverse_action(last)
        if opposite and rng.random() < oscillation_prob:
            return opposite, choose_angle_simulated(999.0, allowed_angles)

    deltas = {
        "roll": ((target.roll - current.roll + 180) % 360) - 180,
        "pitch": ((target.pitch - current.pitch + 180) % 360) - 180,
        "yaw": ((target.yaw - current.yaw + 180) % 360) - 180,
    }
    best_axis = max(deltas, key=lambda k: abs(deltas[k]))
    if abs(deltas[best_axis]) <= 0:
        return "STOP", allowed_angles[0]

    axis_to_actions = {
        "roll": ("ROLL_POS", "ROLL_NEG"),
        "pitch": ("PITCH_POS", "PITCH_NEG"),
        "yaw": ("YAW_POS", "YAW_NEG"),
    }
    pos_action, neg_action = axis_to_actions[best_axis]
    preferred_main = pos_action if deltas[best_axis] > 0 else neg_action
    if preferred_main not in allowed_actions:
        non_stop_actions = [a for a in allowed_actions if a != "STOP"]
        if not non_stop_actions:
            return "STOP", allowed_angles[0]
        fallback = rng.choice(non_stop_actions)
        return fallback, choose_angle_simulated(abs(deltas[best_axis]), allowed_angles)
    preferred = preferred_main
    preferred_angle = choose_angle_simulated(abs(deltas[best_axis]), allowed_angles)

    if rng.random() < follow_prob:
        return preferred, preferred_angle
    non_stop_actions = [a for a in allowed_actions if a != "STOP"]
    if not non_stop_actions:
        return "STOP", allowed_angles[0]
    return rng.choice(non_stop_actions), preferred_angle


def parse_action_token(text: str) -> str:
    if not text:
        return ""
    legacy_small = [
        "PITCH_POS_SMALL",
        "PITCH_NEG_SMALL",
        "YAW_POS_SMALL",
        "YAW_NEG_SMALL",
        "ROLL_POS_SMALL",
        "ROLL_NEG_SMALL",
    ]
    pattern = r"\b(" + "|".join(re.escape(t) for t in (ALL_ACTIONS + legacy_small)) + r")\b"
    tokens = re.findall(pattern, text.upper())
    return tokens[-1] if tokens else ""


def _extract_action_and_cot_from_obj(obj: Dict, allowed_actions: List[str], allowed_angles: List[int]) -> tuple[str, str, int]:
    token = ""
    cot = ""
    angle = allowed_angles[0]
    for k in ("type", "action", "token"):
        if k in obj:
            token = parse_action_token(str(obj.get(k, "")))
            if token:
                break
    if "angle" in obj:
        try:
            angle = int(obj.get("angle"))
        except Exception:
            angle = allowed_angles[0]
    token, angle = normalize_action_and_angle(token, angle, allowed_angles)
    if token and token not in allowed_actions:
        token = ""
    for k in ("cot", "reasoning", "analysis", "thought", "why"):
        if k in obj:
            cot = str(obj.get(k, "")).strip()
            if cot:
                break
    return token, cot, angle


def parse_action_response(text: str, allowed_actions: List[str], allowed_angles: List[int]) -> tuple[str, str, int]:
    """Parse action token/cot/angle from JSON first, then fallback to token regex."""
    if not text:
        return "", "", allowed_angles[0]

    content = text.strip()
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            token, cot, angle = _extract_action_and_cot_from_obj(obj, allowed_actions, allowed_angles)
            if token:
                if token in allowed_actions:
                    return token, cot, angle
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content, flags=re.IGNORECASE)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, dict):
                token, cot, angle = _extract_action_and_cot_from_obj(obj, allowed_actions, allowed_angles)
                if token:
                    if token in allowed_actions:
                        return token, cot, angle
        except Exception:
            pass

    token = parse_action_token(content)
    if token.endswith("_SMALL"):
        token = token.replace("_SMALL", "")
        angle = 10
    else:
        angle = allowed_angles[0]
    token, angle = normalize_action_and_angle(token, angle, allowed_angles)
    if token in allowed_actions:
        return token, "", angle
    return "", "", allowed_angles[0]


def image_to_data_url(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def image_array_to_data_url(mask) -> str:
    if np is not None and cv2 is not None and isinstance(mask, np.ndarray):
        arr = mask
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".png", arr)
        if ok:
            return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    h = len(mask)
    w = len(mask[0]) if h else 0
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        row = mask[y]
        for x in range(w):
            v = row[x]
            iv = int(v)
            if iv in (0, 1):
                iv *= 255
            raw.append(max(0, min(255, iv)))
    compressed = zlib.compress(bytes(raw))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            len(data).to_bytes(4, "big")
            + tag
            + data
            + (zlib.crc32(tag + data) & 0xFFFFFFFF).to_bytes(4, "big")
        )

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    ihdr = (
        w.to_bytes(4, "big")
        + h.to_bytes(4, "big")
        + b"\x08"
        + b"\x00"
        + b"\x00"
        + b"\x00"
        + b"\x00"
    )
    png.extend(chunk(b"IHDR", ihdr))
    png.extend(chunk(b"IDAT", compressed))
    png.extend(chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(bytes(png)).decode("ascii")


def save_data_url_image(path: Path, data_url: str) -> bool:
    try:
        prefix, b64 = str(data_url).split(",", 1)
    except ValueError:
        return False
    if ";base64" not in prefix:
        return False
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return True


def save_llm_debug_bundle(
    debug_dir: Path,
    *,
    payload: Dict[str, Any],
    messages: List[Dict[str, Any]],
    raw_response_text: str = "",
    raw_response_json: Optional[Dict[str, Any]] = None,
    error_text: str = "",
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)

    sanitized_messages: List[Dict[str, Any]] = []
    image_counter = 0
    for msg_idx, message in enumerate(messages, start=1):
        role = str(message.get("role", "unknown"))
        content = message.get("content", "")
        if isinstance(content, str):
            text_path = debug_dir / f"message_{msg_idx:02d}_{role}.txt"
            text_path.write_text(content, encoding="utf-8")
            sanitized_messages.append(
                {
                    "role": role,
                    "content_type": "text",
                    "text_file": text_path.name,
                }
            )
            continue

        if not isinstance(content, list):
            fallback_path = debug_dir / f"message_{msg_idx:02d}_{role}.txt"
            fallback_path.write_text(str(content), encoding="utf-8")
            sanitized_messages.append(
                {
                    "role": role,
                    "content_type": "unknown",
                    "text_file": fallback_path.name,
                }
            )
            continue

        sanitized_items: List[Dict[str, Any]] = []
        for item_idx, item in enumerate(content, start=1):
            if not isinstance(item, dict):
                item_path = debug_dir / f"message_{msg_idx:02d}_{role}_item_{item_idx:02d}.txt"
                item_path.write_text(str(item), encoding="utf-8")
                sanitized_items.append({"type": "unknown", "text_file": item_path.name})
                continue
            item_type = str(item.get("type", ""))
            if item_type == "text":
                item_path = debug_dir / f"message_{msg_idx:02d}_{role}_item_{item_idx:02d}.txt"
                item_path.write_text(str(item.get("text", "")), encoding="utf-8")
                sanitized_items.append({"type": "text", "text_file": item_path.name})
            elif item_type == "image_url":
                image_counter += 1
                image_name = f"image_{image_counter:02d}.png"
                image_path = debug_dir / image_name
                image_url = item.get("image_url", {})
                image_saved = False
                if isinstance(image_url, dict):
                    image_saved = save_data_url_image(image_path, str(image_url.get("url", "")))
                sanitized_items.append(
                    {
                        "type": "image_url",
                        "image_file": image_name if image_saved else "",
                        "image_saved": bool(image_saved),
                    }
                )
            else:
                item_path = debug_dir / f"message_{msg_idx:02d}_{role}_item_{item_idx:02d}.json"
                item_path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
                sanitized_items.append({"type": item_type or "unknown", "raw_file": item_path.name})
        sanitized_messages.append({"role": role, "content_type": "list", "items": sanitized_items})

    request_summary = {
        "model": payload.get("model", ""),
        "temperature": payload.get("temperature", 0.0),
        "seed": payload.get("seed", None),
        "max_tokens": payload.get("max_tokens", 0),
        "response_format": payload.get("response_format", {}),
        "messages": sanitized_messages,
    }
    (debug_dir / "request_summary.json").write_text(
        json.dumps(request_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if raw_response_text:
        (debug_dir / "response.txt").write_text(str(raw_response_text), encoding="utf-8")
    if raw_response_json is not None:
        (debug_dir / "response.json").write_text(
            json.dumps(raw_response_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if error_text:
        (debug_dir / "error.txt").write_text(str(error_text), encoding="utf-8")


def _to_uint8_image(mask):
    if np is not None and isinstance(mask, np.ndarray):
        arr = mask
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    if np is None:
        return None
    h = len(mask)
    w = len(mask[0]) if h else 0
    arr = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        row = mask[y]
        for x in range(w):
            iv = int(row[x])
            if iv in (0, 1):
                iv *= 255
            arr[y, x] = max(0, min(255, iv))
    return arr


def compose_triptych_data_url(target_img, oblique_img, proj_img) -> str:
    if np is None or cv2 is None:
        raise RuntimeError("compose_triptych_data_url requires numpy and opencv.")
    panel = compose_triptych_image(target_img, oblique_img, proj_img)
    ok, buf = cv2.imencode(".png", panel)
    if not ok:
        raise RuntimeError("Failed to encode panel PNG.")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def compose_triptych_image(target_img, oblique_img, proj_img):
    if np is None or cv2 is None:
        raise RuntimeError("compose_triptych_image requires numpy and opencv.")
    t = _to_uint8_image(target_img)
    o = _to_uint8_image(oblique_img)
    p = _to_uint8_image(proj_img)
    if t is None or o is None or p is None:
        raise RuntimeError("Failed to prepare panel images.")
    if t.ndim == 2:
        t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
    if o.ndim == 2:
        o = cv2.cvtColor(o, cv2.COLOR_GRAY2BGR)
    if p.ndim == 2:
        p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
    h = max(t.shape[0], o.shape[0], p.shape[0])
    w = max(t.shape[1], o.shape[1], p.shape[1])
    def fit(img):
        if img.shape[0] != h or img.shape[1] != w:
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        return img
    t, o, p = fit(t), fit(o), fit(p)
    sep = np.full(
        (h, STANDARD_PANEL_SEPARATOR, 3),
        STANDARD_PANEL_SEPARATOR_VALUE,
        dtype=np.uint8,
    )
    return np.concatenate([t, sep, o, sep, p], axis=1)


def save_triptych_panel(
    path: Path,
    panel_img,
) -> None:
    if np is None or cv2 is None:
        return
    ensure_dir = path.parent.mkdir
    ensure_dir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel_img)


def is_zero_temperature_unsupported(error_text: str) -> bool:
    text = str(error_text or "").lower()
    if "temperature" not in text:
        return False
    unsupported_markers = (
        "unsupported",
        "not support",
        "does not support",
        "must be greater than 0",
        "minimum is",
        "minimum value",
        "invalid value",
        "only the default",
    )
    return any(marker in text for marker in unsupported_markers)


def call_openai_compatible_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    image_mode: str,
    target_image_data_url: str,
    current_oblique_data_url: str,
    current_image_data_url: str,
    panel_image_data_url: str,
    history_actions: List[str],
    recent_panel_data_urls: List[str],
    recent_state_summaries: List[str],
    history_window: int,
    step: int,
    step_limit: int,
    current_iou: float,
    tau_success: float,
    last_action: str,
    allowed_actions: List[str],
    allowed_angles: List[int],
    timeout_s: int,
    prompt_profile: str = "baseline",
    prompt_protocol: str = "v2",
    metric_feedback_mode: str = "full",
    api_temperature: float = 0.0,
    api_seed: int = 42,
    api_max_tokens: int = 8192,
    api_response_format: str = "json",
    api_orchestration: str = "direct",
    agent_tool_mode: str = "none",
    agent_workspace=None,
    magentic_max_rounds: int = 4,
    magentic_max_resets: int = 1,
    stop_caution: bool = False,
    hide_iou_feedback: bool = False,
    legacy_stop_as_success: bool = False,
    debug_dir: Optional[Path] = None,
    ledger_path: str = "",
) -> Dict[str, str]:
    metric_feedback_mode = resolve_metric_feedback_mode(metric_feedback_mode, hide_iou_feedback)
    if api_orchestration in {"magentic", "react"} and metric_feedback_mode != "none":
        raise ValueError("Agent orchestrations are black-box protocols and require metric_feedback_mode=none")
    if api_orchestration in {"magentic", "react"} and agent_tool_mode == "python":
        if agent_workspace is None:
            raise ValueError("Agent Python mode requires a task-local workspace")
        if image_mode != "panel" or not panel_image_data_url:
            raise ValueError("Agent Python mode requires the single 320x320 panel observation")
    metric_guidance = build_metric_guidance(metric_feedback_mode, current_iou, tau_success)
    effective_mode = image_mode
    history_panels = select_history_slice(recent_panel_data_urls, int(history_window))
    if effective_mode == "panel" and not panel_image_data_url:
        effective_mode = "triple"
    token_set_text = ", ".join(allowed_actions)
    angle_set_text = ", ".join(str(x) for x in allowed_angles)
    if prompt_protocol == "v1":
        prompt_text = build_prompt_text_v1(
            effective_mode=effective_mode,
            prompt_profile=prompt_profile,
            token_set_text=token_set_text,
            angle_set_text=angle_set_text,
            step=step,
            step_limit=step_limit,
            current_iou=current_iou,
            tau_success=tau_success,
            last_action=last_action,
            history_actions=history_actions,
            history_window=history_window,
            recent_state_summaries=recent_state_summaries,
            stop_caution=stop_caution,
            hide_iou_feedback=hide_iou_feedback,
            legacy_stop_as_success=legacy_stop_as_success,
        )
        system_prompt = build_system_prompt_v1(
            token_set_text,
            angle_set_text,
            history_window,
            legacy_stop_as_success,
        )
    else:
        prompt_text = build_prompt_text_v2(
            step=step,
            step_limit=step_limit,
            current_iou=current_iou,
            tau_success=tau_success,
            metric_guidance=metric_guidance,
            hide_iou_feedback=hide_iou_feedback,
        )
        system_prompt = build_system_prompt_v2(
            token_set_text,
            angle_set_text,
            hide_iou_feedback=hide_iou_feedback,
            legacy_stop_as_success=legacy_stop_as_success,
        )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]
    user_content = [{"type": "text", "text": prompt_text}]
    if history_panels and prompt_protocol == "v1":
        user_content.append(
            {
                "type": "text",
                "text": "Recent panels are previous states in chronological order from oldest to newest before the current state. Each label such as 'N steps ago' is measured relative to the current state. Read them as a real timeline: state, then the recorded action, then the next state. The final image block below is the current state to act on now.",
            }
        )
        for idx, url in enumerate(history_panels, start=1):
            steps_ago = len(history_panels) - idx + 1
            user_content.append({"type": "text", "text": f"{steps_ago} steps ago (older -> newer history state before the recorded action):"})
            user_content.append({"type": "image_url", "image_url": {"url": url}})
    elif history_panels:
        history_summaries = select_history_slice(recent_state_summaries, int(history_window))
        summary_offset = len(history_summaries) - len(history_panels)
        for idx, url in enumerate(history_panels):
            summary_idx = idx + summary_offset
            action_summary = history_summaries[summary_idx] if summary_idx >= 0 else "action=UNKNOWN angle=UNKNOWN"
            user_content.append(
                {
                    "type": "text",
                    "text": f"Historical state {idx + 1}; action after this state: {action_summary}",
                }
            )
            user_content.append({"type": "image_url", "image_url": {"url": url}})
    user_content.append({"type": "text", "text": "Current state to act on now:"})
    if effective_mode == "panel":
        user_content.append({"type": "image_url", "image_url": {"url": panel_image_data_url}})
    else:
        user_content.append({"type": "image_url", "image_url": {"url": target_image_data_url}})
        user_content.append({"type": "image_url", "image_url": {"url": current_oblique_data_url}})
        user_content.append({"type": "image_url", "image_url": {"url": current_image_data_url}})
    messages.append({"role": "user", "content": user_content})
    payload = {
        "model": model,
        "temperature": float(api_temperature),
        "messages": messages,
    }
    if api_seed > 0:
        payload["seed"] = int(api_seed)
    if str(api_response_format or "json").lower() != "none":
        payload["response_format"] = {"type": "json_object"}
    if api_max_tokens > 0:
        payload["max_tokens"] = int(api_max_tokens)
    temperature_fallback_used = False
    response_format_fallback_used = False
    agent_mode = api_orchestration in {"magentic", "react"}
    for _ in range(3):
        try:
            if api_orchestration in {"magentic", "react"}:
                from agent_framework_magentic import run_magentic_chat, run_react_chat

                run_agent_chat = run_magentic_chat if api_orchestration == "magentic" else run_react_chat
                agent_kwargs = {
                    "messages": messages,
                    "model": model,
                    "api_key": api_key,
                    "base_url": base_url,
                    "temperature": float(payload.get("temperature", 0.0)),
                    "max_tokens": int(payload.get("max_tokens", 0) or 0),
                    "seed": int(payload.get("seed", 0) or 0),
                    "response_format": "json" if "response_format" in payload else "none",
                    "timeout_s": int(timeout_s),
                    "python_tool": agent_workspace.execute_python if agent_tool_mode == "python" else None,
                    "image_tool": agent_workspace.view_image if agent_tool_mode == "python" else None,
                    "workspace": agent_workspace if agent_tool_mode == "python" else None,
                    "ledger_path": ledger_path,
                }
                if api_orchestration == "magentic":
                    agent_kwargs["max_rounds"] = int(magentic_max_rounds)
                    agent_kwargs["max_resets"] = int(magentic_max_resets)
                agent_result = run_agent_chat(
                    **agent_kwargs,
                )
                data = {
                    "choices": [{"message": {"content": agent_result["raw_text"]}}],
                    "usage": agent_result.get("usage", {}),
                    "orchestration_trace": agent_result.get("trace", []),
                    "input_transport": agent_result.get("input_transport", ""),
                    "tools_registered": bool(agent_result.get("tools_registered", False)),
                    "tool_protocol": str(agent_result.get("tool_protocol", "")),
                }
            else:
                data = chat_completions(payload, timeout=timeout_s, api_key=api_key, base_url=base_url)
            break
        except Exception as e:
            error_text = str(e)
            error_lower = error_text.lower()
            if agent_mode and api_orchestration == "magentic" and _ < 2:
                messages.append({
                    "role": "user",
                    "content": (
                        "The previous orchestration attempt failed before producing a valid action. "
                        "Recover the task context and return exactly one valid action JSON object. "
                        "Do not repeat the failed tool call. Framework diagnostic: "
                        + error_text[:2000]
                    ),
                })
                payload["messages"] = messages
                continue
            if (
                not temperature_fallback_used
                and not agent_mode
                and float(payload.get("temperature", 0.0)) == 0.0
                and is_zero_temperature_unsupported(error_text)
            ):
                payload["temperature"] = 1e-7
                temperature_fallback_used = True
                continue
            if (
                not response_format_fallback_used
                and "response_format" in payload
                and ("response_format" in error_lower or "json_object" in error_lower)
            ):
                payload.pop("response_format", None)
                response_format_fallback_used = True
                continue
            if debug_dir is not None:
                save_llm_debug_bundle(
                    debug_dir,
                    payload=payload,
                    messages=messages,
                    error_text=error_text,
                )
            raise
    choices = data.get("choices", [])
    if not choices:
        if debug_dir is not None:
            save_llm_debug_bundle(
                debug_dir,
                payload=payload,
                messages=messages,
                raw_response_json=data if isinstance(data, dict) else None,
                error_text=f"No choices in response: {json.dumps(data, ensure_ascii=False)[:300]}",
            )
        raise RuntimeError(f"No choices in response: {json.dumps(data, ensure_ascii=False)[:300]}")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                text_parts.append(c.get("text", ""))
        content = "\n".join(text_parts)

    content_text = str(content).strip()
    if not content_text:
        reasoning_content = str(message.get("reasoning_content", "") or "").strip()
        if reasoning_content:
            content_text = reasoning_content
    if debug_dir is not None:
        save_llm_debug_bundle(
            debug_dir,
            payload=payload,
            messages=messages,
            raw_response_text=content_text,
            raw_response_json=data if isinstance(data, dict) else None,
        )
    if not content_text:
        error_text = "Empty model response content"
        if debug_dir is not None:
            save_llm_debug_bundle(
                debug_dir,
                payload=payload,
                messages=messages,
                raw_response_text=content_text,
                raw_response_json=data if isinstance(data, dict) else None,
                error_text=error_text,
            )
        raise RuntimeError(error_text)
    token, cot, angle = parse_action_response(content_text, allowed_actions, allowed_angles)
    if not token and api_orchestration == "magentic":
        previous_usage = data.get("usage", {}) if isinstance(data, dict) else {}
        previous_trace = data.get("orchestration_trace", []) if isinstance(data, dict) else []
        messages.append({
            "role": "user",
            "content": (
                "Your previous response was not a valid environment action. Return exactly one JSON "
                "object with cot, type, and angle. Do not include Markdown, tool-call JSON, or any "
                "additional object. Invalid response: " + content_text[:2000]
            ),
        })
        agent_kwargs["messages"] = messages
        corrected = run_agent_chat(**agent_kwargs)
        corrected_usage = corrected.get("usage", {})
        usage = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] = int(previous_usage.get(key, 0) or 0) + int(corrected_usage.get(key, 0) or 0)
        content_text = str(corrected.get("raw_text", "") or "").strip()
        data = {
            "choices": [{"message": {"content": content_text}}],
            "usage": usage,
            "orchestration_trace": previous_trace + corrected.get("trace", []),
            "input_transport": corrected.get("input_transport", ""),
            "tools_registered": bool(corrected.get("tools_registered", False)),
            "tool_protocol": str(corrected.get("tool_protocol", "")),
        }
        if debug_dir is not None:
            save_llm_debug_bundle(
                debug_dir,
                payload=payload,
                messages=messages,
                raw_response_text=content_text,
                raw_response_json=data,
            )
        token, cot, angle = parse_action_response(content_text, allowed_actions, allowed_angles)
    if not token:
        error_text = f"Cannot parse action token from content: {content_text}"
        if debug_dir is not None:
            save_llm_debug_bundle(
                debug_dir,
                payload=payload,
                messages=messages,
                raw_response_text=content_text,
                raw_response_json=data if isinstance(data, dict) else None,
                error_text=error_text,
            )
        raise RuntimeError(f"Cannot parse action token from content: {content_text}")
    delivered_text = content_text
    if agent_mode:
        delivered_text = json.dumps(
            {"cot": cot, "type": token, "angle": int(angle)},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return {
        "action": token,
        "angle": int(angle),
        "raw_text": delivered_text,
        "cot": cot,
        "system_prompt": system_prompt,
        "prompt_text": prompt_text,
        "usage": data.get("usage", {}) if isinstance(data, dict) else {},
        "tools_registered": bool(data.get("tools_registered", False)) if isinstance(data, dict) else False,
        "tool_protocol": str(data.get("tool_protocol", "")) if isinstance(data, dict) else "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 02 test models")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--config", type=str, default="artifacts/config/run_config.json")
    parser.add_argument("--formal-config", type=str, default="final_benchmark_config.json")
    parser.add_argument("--tasks-file", type=str, default="artifacts/ablation/holes_20260316_154921/attempts_0/tasks/tasks.jsonl")
    parser.add_argument("--split-file", type=str, default="dataset_splits/hybrid_beam256_exact30_proportional_v2/test.json", help="JSON split file; test.json and train.json are supported")
    parser.add_argument("--output-dir", type=str, default="", help="Default: result/<model name>/<metric feedback protocol>")
    parser.add_argument("--attempt-tag", type=str, default="", help="Optional tag for current run")
    parser.add_argument("--attempts", type=str, default="", help="Batch mode, e.g. 0,16,32,64,128,256")
    parser.add_argument("--ablation-root", type=str, default="", help="Root dir containing attempts_xx folders")
    parser.add_argument("--seed", type=int, default=0, help="Override random seed (0 means use config seed)")
    parser.add_argument("--model-name", type=str, default="", help="Display model name in report")
    parser.add_argument("--checkpoint-path", type=str, default="", help="Optional model checkpoint path or identifier for report metadata")

    parser.add_argument("--backend", type=str, default="simulated", choices=["simulated", "gemini_api"])
    parser.add_argument("--api-key", type=str, default=os.environ.get("PROJECTION_API_KEY", ""))
    parser.add_argument("--base-url", type=str, default=BASE_URL)
    parser.add_argument("--api-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--api-image-mode", type=str, default="panel", choices=["panel", "triple"], help="panel=single triptych image, triple=three separate images")
    parser.add_argument("--api-display-width", type=int, default=STANDARD_IMAGE_SIZE, help="Square API-visible render width")
    parser.add_argument("--api-display-height", type=int, default=STANDARD_IMAGE_SIZE, help="Square API-visible render height")
    parser.add_argument("--prompt-profile", type=str, default="baseline", choices=PROMPT_PROFILES, help="Prompt template profile for API-based evaluation")
    parser.add_argument("--prompt-protocol", type=str, default="v2", choices=["v1", "v2"], help="v1=archived verbose prompt, v2=compact formal prompt")
    parser.add_argument("--history-window", type=int, default=-1, help="How many previous states/panels to include as chronological history context; use -1 for full history")
    parser.add_argument("--include-history-cot", dest="include_history_cot", action="store_true", help="Include each prior model cot with its historical action (formal default)")
    parser.add_argument("--no-history-cot", dest="include_history_cot", action="store_false", help="Omit prior model cot for ablation runs")
    parser.add_argument("--api-temperature", type=float, default=0.0, help="Sampling temperature for API backend")
    parser.add_argument("--api-seed", type=int, default=42, help="Best-effort model sampling seed passed to the API; 0 omits it")
    parser.add_argument("--api-max-tokens", type=int, default=8192, help="Output/completion max_tokens passed to the API backend")
    parser.add_argument("--api-response-format", type=str, default="json", choices=["json", "none"], help="Use response_format=json_object or omit it for providers/models that do not support JSON mode")
    parser.add_argument("--api-orchestration", type=str, default="direct", choices=["direct", "magentic", "react"], help="API orchestration over the same benchmark observations")
    parser.add_argument("--agent-tool-mode", type=str, default="python", choices=["none", "python"], help="Task-local analysis tool for agent orchestrations; direct mode ignores it")
    parser.add_argument(
        "--magentic-max-rounds",
        type=int,
        default=4,
        help="Maximum Magentic coordination rounds per environment step",
    )
    parser.add_argument(
        "--magentic-max-resets",
        type=int,
        default=1,
        help="Maximum Magentic replanning resets per environment step",
    )
    parser.add_argument("--stop-caution", action="store_true", help="Add an explicit warning against STOP when IoU is far below threshold")
    parser.add_argument("--hide-iou-feedback", action="store_true", help="Compatibility alias for --metric-feedback-mode none")
    parser.add_argument("--metric-feedback-mode", type=str, default="none", choices=METRIC_FEEDBACK_MODES, help="Formal default is none; full and threshold are auxiliary protocols only")
    parser.add_argument("--legacy-stop-as-success", action="store_true", help="Use the legacy ambiguous STOP-as-visual-success prompt")
    parser.add_argument("--stagnation-patience", type=int, default=12, help="Stop after this many executed rotations without a strict new historical-best IoU; 0 disables")
    parser.add_argument("--min-best-iou-gain", type=float, default=0.0, help="Strict gain required to reset stagnation patience")
    parser.add_argument("--timeout", type=int, default=1500)
    parser.add_argument("--api-retries", type=int, default=2, help="Retry count on API/network failure")
    parser.add_argument("--api-retry-backoff", type=float, default=1.6, help="Exponential backoff base seconds")
    parser.add_argument("--max-tasks", type=int, default=0, help="0 means all tasks")
    parser.add_argument("--max-step-factor", type=float, default=5.0, help="Deprecated compatibility flag; no longer used for step budget")
    parser.add_argument("--step-limit-multiplier", type=float, default=3.0, help="Multiply task T_max to get step budget (formal benchmark: 3x = 48 steps)")
    parser.add_argument("--max-step-cap", type=int, default=0, help="Optional hard cap; 0 means no cap")
    parser.add_argument("--min-step-limit", type=int, default=1, help="Minimum max-steps budget per task")
    parser.add_argument("--allowed-angles", type=str, default="30,10,5,1", help="Allowed step angles, subset of 30,10,5,1")
    parser.add_argument("--save-frames", action="store_true", help="Save per-step PNG frames to disk")
    parser.add_argument("--save-llm-debug", action="store_true", help="Save full per-step LLM request/response debug bundle")
    parser.add_argument("--debug-timing", action="store_true", help="Print detailed per-stage and per-step timings")
    parser.add_argument("--task-workers", type=int, default=2, help="Number of independent task subprocesses")
    parser.add_argument("--render-workers", type=int, default=1, help="Maximum concurrent CUDA render calls across task subprocesses")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume mode and rerun all tasks")
    parser.set_defaults(resume=True, include_history_cot=True)
    args = parser.parse_args()
    require_standard_square_size(
        (args.api_display_width, args.api_display_height),
        name="--api-display-size",
    )
    return args


def _without_cli_options(argv: List[str], options_with_values: set[str]) -> List[str]:
    cleaned: List[str] = []
    skip_value = False
    for item in argv:
        if skip_value:
            skip_value = False
            continue
        if item in options_with_values:
            skip_value = True
            continue
        if any(item.startswith(f"{option}=") for option in options_with_values):
            continue
        cleaned.append(item)
    return cleaned


def _run_parallel_tasks(args: argparse.Namespace, root: Path) -> None:
    if args.task_workers < 1 or args.render_workers < 1:
        raise ValueError("--task-workers and --render-workers must be positive integers")
    tasks_path = Path(args.tasks_file) if args.tasks_file else resolve_paths(root)["tasks"]
    if not tasks_path.is_absolute():
        tasks_path = root / tasks_path
    split_path = resolve_optional_path(root, args.split_file)
    tasks, _ = read_selected_tasks_and_ids(tasks_path, args.max_tasks, split_path)
    if not tasks:
        raise RuntimeError("No tasks selected")

    safe_model_name = re.sub(r"[^A-Za-z0-9._-]+", "_", args.api_model if args.backend == "gemini_api" else args.model_name).strip("._") or "simulated"
    default_output_root = root / "result" / safe_model_name
    if args.backend == "gemini_api":
        default_output_root /= metric_feedback_result_dir(
            resolve_metric_feedback_mode(args.metric_feedback_mode, args.hide_iou_feedback)
        )
    output_root = Path(args.output_dir) if args.output_dir else default_output_root
    if not output_root.is_absolute():
        output_root = root / output_root
    shard_root = output_root / "parallel_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    worker_count = min(args.task_workers, len(tasks))
    shards: List[List[Dict]] = [[] for _ in range(worker_count)]
    for index, task in enumerate(tasks):
        shards[index % worker_count].append(task)

    removable = {
        "--tasks-file", "--output-dir", "--max-tasks",
        "--task-workers", "--render-workers", "--split-file",
    }
    base_argv = _without_cli_options(sys.argv[1:], removable)
    semaphore_name = f"Local\\projection-render-{os.getpid()}"

    def run_shard(index: int, shard_tasks: List[Dict]) -> tuple[int, Path]:
        shard_dir = shard_root / f"worker_{index:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_tasks_path = shard_dir / "tasks.jsonl"
        with shard_tasks_path.open("w", encoding="utf-8") as f:
            for task in shard_tasks:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")
        cmd = [
            sys.executable, str(Path(__file__).resolve()), *base_argv,
            "--tasks-file", str(shard_tasks_path),
            "--split-file", "",
            "--output-dir", str(shard_dir),
            "--max-tasks", "0",
            "--task-workers", "1",
            "--render-workers", "1",
        ]
        if not args.resume:
            cmd.append("--no-resume")
        env = os.environ.copy()
        env["PROJECTION_PREFLIGHT_DONE"] = "1"
        env[RENDER_SEMAPHORE_ENV] = semaphore_name
        env[RENDER_SEMAPHORE_COUNT_ENV] = str(args.render_workers)
        stdout_path = shard_dir / "run.stdout.log"
        stderr_path = shard_dir / "run.stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
            completed = subprocess.run(cmd, cwd=str(root), env=env, stdout=stdout_f, stderr=stderr_f)
        return completed.returncode, shard_dir

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(lambda pair: run_shard(*pair), enumerate(shards)))
    failures = [(code, path) for code, path in results if code != 0]
    if failures:
        details = ", ".join(f"{path.name}: exit {code}" for code, path in failures)
        raise RuntimeError(f"Parallel task worker failed: {details}")

    rows: List[Dict] = []
    for _, shard_dir in results:
        csv_path = shard_dir / "reports" / "model_raw.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))
    order = {str(task["sample_id"]): index for index, task in enumerate(tasks)}
    rows.sort(key=lambda row: order.get(str(row.get("sample_id", "")), len(order)))
    reports_dir = output_root / "reports"
    logs_dir = output_root / "logs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(reports_dir / "model_raw.csv", rows)

    prompt_tokens = sum(int(row.get("prompt_tokens", 0) or 0) for row in rows)
    completion_tokens = sum(int(row.get("completion_tokens", 0) or 0) for row in rows)
    summary = {
        "model_name": args.model_name.strip() or args.api_model,
        "model_version": args.api_model,
        "backend": args.backend,
        "samples": len(rows),
        "success_rate": round(sum(int(row.get("success", 0) or 0) for row in rows) / len(rows), 6),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "output": _rel_path(reports_dir / "model_raw.csv", root),
        "prompt_protocol": args.prompt_protocol,
        "metric_feedback_mode": resolve_metric_feedback_mode(args.metric_feedback_mode, args.hide_iou_feedback),
        "hide_iou_feedback": bool(args.hide_iou_feedback),
        "history_window": int(args.history_window),
        "api_temperature": float(args.api_temperature),
        "api_seed": int(args.api_seed),
        "api_max_tokens": int(args.api_max_tokens),
        "api_stream": False,
        "task_workers": int(args.task_workers),
        "render_workers": int(args.render_workers),
        "render_environment": render_environment_metadata(),
        "worker_outputs": [_rel_path(path, root) for _, path in results],
    }
    write_json(reports_dir / "step_02_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def compute_api_retry_sleep(
    error_text: str,
    attempt_idx: int,
    backoff: float,
    *,
    rng: Optional[random.Random] = None,
) -> tuple[float, str]:
    """Use slower retries for gateway failures that may already be billable."""
    text = error_text.lower()
    jitter_rng = rng if rng is not None else random
    jitter = 0.8 + jitter_rng.random() * 0.4
    base_sleep = max(0.1, backoff) * (2 ** attempt_idx) * jitter
    if "429" in text or "上游负载已饱和" in error_text or "rate limit" in text:
        return max(65.0, min(300.0, base_sleep * 8.0)), "gateway_busy"
    if "remote end closed" in text or "without response" in text or "connection reset" in text:
        return max(65.0, min(180.0, base_sleep * 6.0)), "connection_closed"
    return min(120.0, base_sleep), "api_error"


def encode_rng_state(state) -> str:
    return base64.b64encode(zlib.compress(pickle.dumps(state))).decode("ascii")


def decode_rng_state(text: str):
    return pickle.loads(zlib.decompress(base64.b64decode(text.encode("ascii"))))


def load_checkpoint(path: Path) -> Dict:
    if not path.exists():
        return {}
    return read_json(path)


def write_csv_rows(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        if not rows:
            return
        fieldnames: List[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def step2_config_fingerprint(
    cfg: Dict,
    args: argparse.Namespace,
    root: Path,
    tasks_path: Path,
    model_name: str,
    all_task_ids: List[str],
    global_allowed_angles: List[int],
) -> str:
    payload = {
        "backend": args.backend,
        "model_name": model_name,
        "checkpoint_path": str(args.checkpoint_path),
        "api_model": args.api_model if args.backend == "gemini_api" else "",
        "base_url": args.base_url if args.backend == "gemini_api" else "",
        "api_image_mode": args.api_image_mode if args.backend == "gemini_api" else "",
        "api_display_width": int(args.api_display_width) if args.backend == "gemini_api" else 0,
        "api_display_height": int(args.api_display_height) if args.backend == "gemini_api" else 0,
        "prompt_profile": args.prompt_profile if args.backend == "gemini_api" else "",
        "prompt_protocol": args.prompt_protocol if args.backend == "gemini_api" else "",
        "metric_feedback_mode": resolve_metric_feedback_mode(args.metric_feedback_mode, args.hide_iou_feedback) if args.backend == "gemini_api" else "",
        "history_window": int(args.history_window) if args.backend == "gemini_api" else 0,
        "include_history_cot": bool(args.include_history_cot) if args.backend == "gemini_api" else False,
        "api_temperature": float(args.api_temperature) if args.backend == "gemini_api" else 0.0,
        "api_seed": int(args.api_seed) if args.backend == "gemini_api" else 0,
        "api_max_tokens": int(args.api_max_tokens) if args.backend == "gemini_api" else 0,
        "api_response_format": str(args.api_response_format) if args.backend == "gemini_api" else "",
        "api_orchestration": str(args.api_orchestration) if args.backend == "gemini_api" else "",
        "magentic_max_rounds": int(args.magentic_max_rounds) if args.backend == "gemini_api" else 0,
        "magentic_max_resets": int(args.magentic_max_resets) if args.backend == "gemini_api" else 0,
        "stop_caution": bool(args.stop_caution) if args.backend == "gemini_api" else False,
        "hide_iou_feedback": bool(args.hide_iou_feedback) if args.backend == "gemini_api" else False,
        "stop_protocol": "legacy_stop_as_visual_success" if args.legacy_stop_as_success else "auto_success_stop_no_further_idea_v1",
        "stagnation_patience": int(args.stagnation_patience),
        "min_best_iou_gain": float(args.min_best_iou_gain),
        "attempt_tag": args.attempt_tag,
        "seed": int(args.seed) if args.seed else int(cfg["random_seed"]),
        "tasks_file": _rel_path(tasks_path, root),
        "all_task_ids": list(all_task_ids),
        "allowed_angles": list(global_allowed_angles),
        "step_limit_multiplier": float(args.step_limit_multiplier),
        "max_step_cap": int(args.max_step_cap),
        "min_step_limit": int(args.min_step_limit),
        "api_display_width": int(args.api_display_width) if args.backend == "gemini_api" else 0,
        "api_display_height": int(args.api_display_height) if args.backend == "gemini_api" else 0,
        "save_frames": bool(args.save_frames),
        "save_llm_debug": bool(args.save_llm_debug) if args.backend == "gemini_api" else False,
        "tau_success": cfg.get("tau_success", None),
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_attempts(text: str) -> List[int]:
    vals: List[int] = []
    for x in text.split(","):
        s = x.strip()
        if not s:
            continue
        vals.append(int(s))
    return vals


def _find_latest_ablation_root(root: Path) -> Optional[Path]:
    base = root / "artifacts" / "ablation"
    if not base.exists():
        return None
    dirs = [p for p in base.glob("holes_*") if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def _run_attempt_batch(args: argparse.Namespace, project_root: Path) -> None:
    attempts = _parse_attempts(args.attempts)
    if not attempts:
        raise RuntimeError("attempts is empty")
    ablation_root = Path(args.ablation_root).resolve() if args.ablation_root else _find_latest_ablation_root(project_root)
    if ablation_root is None or not ablation_root.exists():
        raise RuntimeError("Cannot find ablation root. Please provide --ablation-root.")

    run_summaries: List[Dict] = []
    script_path = (project_root / "step_02_test_models.py").resolve()
    base_cfg_abs = (project_root / args.config).resolve()

    for n in attempts:
        attempt_root = ablation_root / f"attempts_{n}"
        tasks_file = attempt_root / "tasks" / "tasks.jsonl"
        if not tasks_file.exists():
            print(f"[step02][attempts] skip attempts_{n}: tasks file not found: {tasks_file}")
            continue
        out_dir = attempt_root / "step2" / args.backend
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(script_path),
            "--root",
            str(attempt_root),
            "--config",
            str(base_cfg_abs),
            "--tasks-file",
            str(tasks_file),
            "--output-dir",
            str(out_dir),
            "--attempt-tag",
            f"attempts_{n}",
            "--backend",
            args.backend,
            "--seed",
            str(args.seed),
            "--model-name",
            args.model_name,
            "--checkpoint-path",
            args.checkpoint_path,
            "--base-url",
            args.base_url,
            "--api-model",
            args.api_model,
            "--api-image-mode",
            args.api_image_mode,
            "--api-display-width",
            str(args.api_display_width),
            "--api-display-height",
            str(args.api_display_height),
            "--prompt-profile",
            args.prompt_profile,
            "--prompt-protocol",
            args.prompt_protocol,
            "--history-window",
            str(args.history_window),
            "--api-temperature",
            str(args.api_temperature),
            "--api-seed",
            str(args.api_seed),
            "--api-max-tokens",
            str(args.api_max_tokens),
            "--api-response-format",
            str(args.api_response_format),
            "--api-orchestration",
            str(args.api_orchestration),
            "--agent-tool-mode",
            str(args.agent_tool_mode),
            "--metric-feedback-mode",
            str(args.metric_feedback_mode),
            "--magentic-max-rounds",
            str(args.magentic_max_rounds),
            "--magentic-max-resets",
            str(args.magentic_max_resets),
            "--timeout",
            str(args.timeout),
            "--api-retries",
            str(args.api_retries),
            "--api-retry-backoff",
            str(args.api_retry_backoff),
            "--max-tasks",
            str(args.max_tasks),
            "--max-step-factor",
            str(args.max_step_factor),
            "--step-limit-multiplier",
            str(args.step_limit_multiplier),
            "--max-step-cap",
            str(args.max_step_cap),
            "--min-step-limit",
            str(args.min_step_limit),
            "--allowed-angles",
            str(args.allowed_angles),
            "--stagnation-patience",
            str(args.stagnation_patience),
            "--min-best-iou-gain",
            str(args.min_best_iou_gain),
        ]
        if bool(args.stop_caution):
            cmd.append("--stop-caution")
        if bool(args.hide_iou_feedback):
            cmd.append("--hide-iou-feedback")
        if bool(args.legacy_stop_as_success):
            cmd.append("--legacy-stop-as-success")
        if args.api_key:
            cmd.extend(["--api-key", args.api_key])
        if args.save_frames:
            cmd.append("--save-frames")
        if args.save_llm_debug:
            cmd.append("--save-llm-debug")
        if args.debug_timing:
            cmd.append("--debug-timing")
        if not args.resume:
            cmd.append("--no-resume")

        print(f"[step02][attempts] run attempts_{n} backend={args.backend}")
        cp = subprocess.run(cmd, cwd=str(project_root))
        summary_path = out_dir / "reports" / "step_02_summary.json"
        entry = {
            "attempts": n,
            "status_code": int(cp.returncode),
            "summary_path": _rel_path(summary_path, project_root),
        }
        if summary_path.exists():
            try:
                data = read_json(summary_path)
                entry["success_rate"] = data.get("success_rate", 0.0)
                entry["samples"] = data.get("samples", 0)
                entry["output"] = data.get("output", "")
            except Exception:
                pass
        run_summaries.append(entry)

    batch_summary = {
        "backend": args.backend,
        "attempts_requested": attempts,
        "ablation_root": _rel_path(ablation_root, project_root),
        "runs": run_summaries,
    }
    batch_summary_path = ablation_root / f"step2_{args.backend}_batch_summary.json"
    write_json(batch_summary_path, batch_summary)
    print(json.dumps(batch_summary, ensure_ascii=False, indent=2))
    print(f"[step02][attempts] summary: {batch_summary_path}")


def main() -> None:
    args = parse_args()
    metric_feedback_mode = resolve_metric_feedback_mode(args.metric_feedback_mode, args.hide_iou_feedback)
    if metric_feedback_mode == "none":
        args.hide_iou_feedback = True
    root = Path(args.root).resolve()
    if args.backend == "gemini_api" and metric_feedback_mode == "none" and os.environ.get("PROJECTION_PREFLIGHT_DONE") != "1":
        from formal_preflight import validate_formal_run

        tasks_path_for_preflight = resolve_optional_path(root, args.tasks_file)
        split_path_for_preflight = resolve_optional_path(root, args.split_file)
        formal_config_path = resolve_optional_path(root, args.formal_config)
        if tasks_path_for_preflight is None or split_path_for_preflight is None or formal_config_path is None:
            raise RuntimeError("Formal API runs require --formal-config, --tasks-file, and --split-file")
        preflight = validate_formal_run(
            root,
            formal_config_path=formal_config_path,
            tasks_path=tasks_path_for_preflight,
            split_path=split_path_for_preflight,
            prompt_protocol=args.prompt_protocol,
            metric_feedback_mode=metric_feedback_mode,
            hide_iou_feedback=bool(args.hide_iou_feedback),
            stagnation_patience=int(args.stagnation_patience),
            max_tokens=int(args.api_max_tokens),
            timeout_seconds=int(args.timeout),
            history_window=int(args.history_window),
            include_history_cot=bool(args.include_history_cot),
            legacy_stop_as_success=bool(args.legacy_stop_as_success),
            api_temperature=float(args.api_temperature),
            api_seed=int(args.api_seed),
            display_size=(int(args.api_display_width), int(args.api_display_height)),
        )
        print(json.dumps({"formal_preflight": preflight}, ensure_ascii=False), flush=True)
    if args.task_workers < 1 or args.render_workers < 1:
        raise ValueError("--task-workers and --render-workers must be positive integers")
    if args.task_workers > 1:
        if args.attempts.strip():
            raise ValueError("--task-workers cannot be combined with --attempts batch mode")
        _run_parallel_tasks(args, root)
        return
    if args.attempts.strip():
        _run_attempt_batch(args, root)
        return
    paths = resolve_paths(root)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = read_json(cfg_path)
    debug_timing = bool(args.debug_timing or cfg.get("debug_timing", False))
    seed = int(args.seed) if args.seed else int(cfg["random_seed"])
    rng = random.Random(seed)
    render_metadata = render_environment_metadata()

    if args.backend == "gemini_api":
        model_name = args.model_name.strip() or args.api_model
    else:
        model_name = args.model_name.strip() or cfg.get("model_name", "simulated_mllm_v1")
    allowed_actions = list(ALL_ACTIONS)
    global_allowed_angles = parse_allowed_angles(args.allowed_angles)

    tasks_path = Path(args.tasks_file) if args.tasks_file else paths["tasks"]
    if not tasks_path.is_absolute():
        tasks_path = root / tasks_path

    default_eval_size = STANDARD_IMAGE_SIZE
    default_display = STANDARD_DISPLAY_SIZE
    safe_model_name = re.sub(r"[^A-Za-z0-9._-]+", "_", args.api_model if args.backend == "gemini_api" else args.model_name).strip("._") or "simulated"
    default_output_root = root / "result" / safe_model_name
    if args.backend == "gemini_api":
        default_output_root /= metric_feedback_result_dir(metric_feedback_mode)
    output_root = Path(args.output_dir) if args.output_dir else default_output_root
    if not output_root.is_absolute():
        output_root = root / output_root
    output_logs = output_root / "logs"
    output_reports = output_root / "reports"
    out_csv = output_reports / "model_raw.csv"

    split_path = resolve_optional_path(root, args.split_file)
    tasks, all_task_ids = read_selected_tasks_and_ids(tasks_path, args.max_tasks, split_path)
    selected_task_ids = [str(t.get("sample_id", "")).strip() for t in tasks if str(t.get("sample_id", "")).strip()]

    checkpoint_path = output_reports / "step_02_checkpoint.json"
    cfg_fp = step2_config_fingerprint(
        cfg=cfg,
        args=args,
        root=root,
        tasks_path=tasks_path,
        model_name=model_name,
        all_task_ids=all_task_ids,
        global_allowed_angles=global_allowed_angles,
    )
    checkpoint = load_checkpoint(checkpoint_path) if args.resume else {}
    if args.resume and checkpoint:
        if checkpoint.get("config_fingerprint", "") != cfg_fp:
            raise RuntimeError(
                "Step02 checkpoint configuration mismatch. "
                "Delete reports/step_02_checkpoint.json or rerun with --no-resume."
            )
        if str(checkpoint.get("tasks_file", "")) != _rel_path(tasks_path, root):
            raise RuntimeError(
                "Step02 checkpoint tasks file mismatch. "
                "Delete reports/step_02_checkpoint.json or rerun with --no-resume."
            )

    existing_rows: List[Dict] = []
    csv_completed_ids = set()
    if args.resume and out_csv.exists():
        try:
            with out_csv.open("r", encoding="utf-8") as f:
                existing_rows = list(csv.DictReader(f))
            csv_completed_ids = {
                str(r.get("sample_id", "")).strip()
                for r in existing_rows
                if str(r.get("sample_id", "")).strip()
            }
        except Exception:
            existing_rows = []
            csv_completed_ids = set()

    checkpoint_completed_ids = {
        str(x).strip()
        for x in checkpoint.get("completed_sample_ids", [])
        if str(x).strip()
    }
    completed_ids = set(csv_completed_ids)

    if completed_ids:
        tasks = [t for t in tasks if str(t.get("sample_id", "")).strip() not in completed_ids]
    print(
        f"[step02][resume] enabled={bool(args.resume)} "
        f"existing_csv={len(existing_rows)} checkpoint_done={len(checkpoint_completed_ids)} "
        f"pending={len(tasks)} skipped={len(completed_ids)}"
    )

    traj_root = output_logs / "model_traj"
    traj_root.mkdir(parents=True, exist_ok=True)
    frame_root = output_logs / "model_frames"
    if args.save_frames:
        frame_root.mkdir(parents=True, exist_ok=True)
    llm_debug_root = output_logs / "llm_debug"
    if args.save_llm_debug and args.backend == "gemini_api":
        llm_debug_root.mkdir(parents=True, exist_ok=True)
    agent_active_root = output_logs / "agent_workspaces_active"
    agent_archive_root = output_logs / "agent_workspaces"

    rows: List[Dict] = list(existing_rows)
    completed_ids_runtime = set(csv_completed_ids)
    if args.resume and checkpoint and checkpoint.get("rng_state"):
        rng.setstate(decode_rng_state(str(checkpoint.get("rng_state", ""))))
    timing_report: Dict[str, object] = {
        "debug_timing": bool(debug_timing),
        "backend": args.backend,
        "tasks": [],
        "aggregate": {
            "task_setup_s": 0.0,
            "target_render_s": 0.0,
            "target_encode_s": 0.0,
            "step_render_s": 0.0,
            "step_iou_s": 0.0,
            "step_frame_io_s": 0.0,
            "step_encode_s": 0.0,
            "step_policy_api_s": 0.0,
            "step_policy_sim_s": 0.0,
            "step_event_write_s": 0.0,
            "step_apply_action_s": 0.0,
            "step_total_s": 0.0,
        },
    }
    checkpoint = {
        "version": 1,
        "config_fingerprint": cfg_fp,
        "backend": args.backend,
            "model_name": model_name,
            "checkpoint_path": str(args.checkpoint_path),
        "tasks_file": _rel_path(tasks_path, root),
        "output_csv": _rel_path(out_csv, root),
        "raw_events_file": _rel_path(output_logs / "model_events.jsonl", root),
        "trajectory_dir": _rel_path(output_logs / "model_traj", root),
        "attempt_tag": args.attempt_tag,
        "seed": seed,
        "selected_task_count": len(selected_task_ids),
        "selected_task_ids": list(selected_task_ids),
        "completed_count": len(completed_ids_runtime),
        "completed_sample_ids": sorted(completed_ids_runtime),
        "existing_rows_count": len(rows),
        "last_completed_sample_id": checkpoint.get("last_completed_sample_id", "") if checkpoint else "",
        "allowed_angles": list(global_allowed_angles),
        "prompt_profile": args.prompt_profile if args.backend == "gemini_api" else "",
        "prompt_protocol": args.prompt_protocol if args.backend == "gemini_api" else "",
        "hide_iou_feedback": bool(args.hide_iou_feedback) if args.backend == "gemini_api" else False,
        "stop_protocol": "legacy_stop_as_visual_success" if args.legacy_stop_as_success else "auto_success_stop_no_further_idea_v1",
        "stagnation_patience": int(args.stagnation_patience),
        "min_best_iou_gain": float(args.min_best_iou_gain),
        "step_limit_multiplier": float(args.step_limit_multiplier),
        "max_step_cap": int(args.max_step_cap),
        "min_step_limit": int(args.min_step_limit),
        "api_display_width": int(args.api_display_width) if args.backend == "gemini_api" else 0,
        "api_display_height": int(args.api_display_height) if args.backend == "gemini_api" else 0,
        "save_frames": bool(args.save_frames),
        "save_llm_debug": bool(args.save_llm_debug) if args.backend == "gemini_api" else False,
        "llm_debug_dir": _rel_path(llm_debug_root, root) if (args.save_llm_debug and args.backend == "gemini_api") else "",
        "rng_state": encode_rng_state(rng.getstate()),
    }
    write_json(checkpoint_path, checkpoint)
    raw_events_path = output_logs / "model_events.jsonl"
    raw_events_path.parent.mkdir(parents=True, exist_ok=True)
    events_mode = "a" if (args.resume and raw_events_path.exists()) else "w"
    with raw_events_path.open(events_mode, encoding="utf-8") as events_f:
        for task in tqdm(tasks, desc=f"Step02 {args.backend}", unit="task"):
            t_task_start = perf_counter()
            t_setup = perf_counter()
            sample_id = task["sample_id"]
            agent_workspace = None
            if args.backend == "gemini_api" and args.api_orchestration in {"magentic", "react"} and args.agent_tool_mode == "python":
                from agent_python_workspace import AgentPythonWorkspace

                if args.api_image_mode != "panel":
                    raise ValueError("metric_blind_agent_python_v2 requires --api-image-mode panel")
                agent_workspace = AgentPythonWorkspace(agent_active_root, agent_archive_root, sample_id)
            target_pose = Pose.from_dict(task["target_pose"])
            current = Pose.from_dict(task["init_pose"])
            task["cuda_point_batch_size"] = STANDARD_POINT_BATCH_SIZE
            theta = int(task["theta"])
            task_allowed_angles = resolve_effective_allowed_angles(task.get("allowed_angles", []), global_allowed_angles)
            tau = resolve_effective_tau(task, cfg)
            eval_size = STANDARD_IMAGE_SIZE
            display_size = STANDARD_DISPLAY_SIZE
            base_tmax = max(1, int(task.get("T_max", 1)))
            step_limit = max(args.min_step_limit, int(math.ceil(base_tmax * max(1.0, float(args.step_limit_multiplier)))))
            if args.max_step_cap > 0:
                step_limit = min(step_limit, args.max_step_cap)
            dt_setup = perf_counter() - t_setup
            timing_report["aggregate"]["task_setup_s"] += dt_setup

            t_target_render = perf_counter()
            target_mask_disp, target_mask_eval, _ = render_views_limited(
                task,
                target_pose,
                display_size=display_size,
                eval_size=eval_size,
                need_oblique=False,
            )
            render_backends = {get_last_render_backend()}
            target_eval_sha256 = mask_sha256(target_mask_eval)
            initial_eval_sha256 = ""
            initial_iou_full_precision: Optional[float] = None
            dt_target_render = perf_counter() - t_target_render
            timing_report["aggregate"]["target_render_s"] += dt_target_render
            recent_actions: List[str] = []
            recent_panel_data_urls: List[str] = []
            recent_state_summaries: List[str] = []
            trajectory: List[Dict] = []

            t_target_encode = perf_counter()
            target_img_arr = _to_uint8_image(target_mask_disp)
            target_data_url = image_array_to_data_url(target_mask_disp) if args.backend == "gemini_api" else ""
            dt_target_encode = perf_counter() - t_target_encode
            timing_report["aggregate"]["target_encode_s"] += dt_target_encode
            sample_frame_dir = frame_root / sample_id if args.save_frames else None
            if sample_frame_dir is not None:
                sample_frame_dir.mkdir(parents=True, exist_ok=True)
            sample_llm_debug_dir = llm_debug_root / sample_id if (args.save_llm_debug and args.backend == "gemini_api") else None
            if sample_llm_debug_dir is not None:
                sample_llm_debug_dir.mkdir(parents=True, exist_ok=True)

            terminated_reason = "max_steps"
            success = False
            final_iou = 0.0
            previous_iou: Optional[float] = None
            best_iou = float("-inf")
            nonimproving_rotations = 0
            budget_exhaustion_mode = ""
            last_action = ""
            api_fallback_count = 0
            api_error_count = 0
            api_error_last = ""
            sample_prompt_tokens = 0
            sample_completion_tokens = 0
            sample_total_tokens = 0
            task_timing: Dict[str, object] = {
                "sample_id": sample_id,
                "setup_s": round(dt_setup, 6),
                "target_render_s": round(dt_target_render, 6),
                "target_encode_s": round(dt_target_encode, 6),
                "step_limit": step_limit,
                "steps": [],
            }

            start_ms = now_ms()
            for step in range(1, step_limit + 1):
                t_step_total = perf_counter()
                t_step_render = perf_counter()
                current_mask_disp, current_mask_eval, current_oblique = render_views_limited(
                    task,
                    current,
                    display_size=display_size,
                    eval_size=eval_size,
                    need_oblique=True,
                )
                render_backends.add(get_last_render_backend())
                dt_step_render = perf_counter() - t_step_render
                timing_report["aggregate"]["step_render_s"] += dt_step_render
                t_step_iou = perf_counter()
                score = iou(target_mask_eval, current_mask_eval)
                if step == 1:
                    initial_eval_sha256 = mask_sha256(current_mask_eval)
                    initial_iou_full_precision = float(score)
                dt_step_iou = perf_counter() - t_step_iou
                timing_report["aggregate"]["step_iou_s"] += dt_step_iou
                final_iou = score
                if score > best_iou + float(args.min_best_iou_gain):
                    best_iou = score
                    nonimproving_rotations = 0
                else:
                    nonimproving_rotations += 1
                if (
                    score < tau
                    and int(args.stagnation_patience) > 0
                    and nonimproving_rotations >= int(args.stagnation_patience)
                ):
                    budget_exhaustion_mode = "early_stop"
                    terminated_reason = "early_stop"
                    break

                panel_frame_rel = ""
                panel_meta_rel = ""
                panel_text_rel = ""
                current_proj_data_url = ""
                current_oblique_data_url = ""
                panel_data_url = ""
                panel_img = None
                dt_step_frame_io = 0.0
                dt_step_encode = 0.0
                t_step_io = perf_counter()
                if args.save_frames and np is not None and cv2 is not None and target_img_arr is not None:
                    panel_img = compose_triptych_image(target_img_arr, current_oblique, current_mask_disp)
                if args.backend == "gemini_api":
                    if args.api_image_mode == "panel" and target_img_arr is not None and np is not None and cv2 is not None:
                        panel_data_url = compose_triptych_data_url(target_img_arr, current_oblique, current_mask_disp)
                    else:
                        current_proj_data_url = image_array_to_data_url(current_mask_disp)
                        current_oblique_data_url = image_array_to_data_url(current_oblique)
                if agent_workspace is not None and score < tau:
                    agent_workspace.prepare_step(panel_data_url, step)
                dt_step_io = perf_counter() - t_step_io
                if args.save_frames:
                    dt_step_frame_io = dt_step_io
                elif args.backend == "gemini_api":
                    dt_step_encode = dt_step_io
                timing_report["aggregate"]["step_frame_io_s"] += dt_step_frame_io
                timing_report["aggregate"]["step_encode_s"] += dt_step_encode

                dt_policy_api = 0.0
                dt_policy_sim = 0.0
                model_response_text = ""
                model_cot_text = ""
                model_system_prompt = ""
                model_prompt_text = ""
                step_prompt_tokens = 0
                step_completion_tokens = 0
                step_total_tokens = 0
                step_llm_debug_dir_rel = ""
                action_angle = task_allowed_angles[0]
                if score >= tau:
                    action = "STOP"
                    model_response_text = "AUTO_STOP_THRESHOLD_REACHED"
                    model_cot_text = "AUTO_STOP_THRESHOLD_REACHED"
                    model_prompt_text = "AUTO_STOP_THRESHOLD_REACHED"
                    success = True
                    terminated_reason = "threshold_reached"
                else:
                    if args.backend == "gemini_api":
                        api_key = args.api_key.strip() or API_KEY.strip()
                        t_policy = perf_counter()
                        retries = max(0, int(args.api_retries))
                        backoff = max(0.1, float(args.api_retry_backoff))
                        action = "STOP"
                        for attempt_idx in range(retries + 1):
                            try:
                                debug_dir_this_attempt = None
                                if sample_llm_debug_dir is not None:
                                    debug_dir_this_attempt = sample_llm_debug_dir / f"step_{step:03d}" / f"attempt_{attempt_idx + 1:02d}"
                                api_result = call_openai_compatible_chat(
                                    base_url=args.base_url,
                                    api_key=api_key,
                                    model=args.api_model,
                                    image_mode=args.api_image_mode,
                                    target_image_data_url=target_data_url,
                                    current_oblique_data_url=current_oblique_data_url,
                                    current_image_data_url=current_proj_data_url,
                                    panel_image_data_url=panel_data_url,
                                    history_actions=recent_actions,
                                    recent_panel_data_urls=recent_panel_data_urls,
                                    recent_state_summaries=recent_state_summaries,
                                    history_window=int(args.history_window),
                                    step=step,
                                    step_limit=step_limit,
                                    current_iou=score,
                                    tau_success=tau,
                                    last_action=last_action,
                                    allowed_actions=allowed_actions,
                                    allowed_angles=task_allowed_angles,
                                    timeout_s=args.timeout,
                                    prompt_profile=args.prompt_profile,
                                    prompt_protocol=args.prompt_protocol,
                                    metric_feedback_mode=metric_feedback_mode,
                                    api_temperature=args.api_temperature,
                                    api_seed=args.api_seed,
                                    api_max_tokens=args.api_max_tokens,
                                    api_response_format=args.api_response_format,
                                    api_orchestration=args.api_orchestration,
                                    agent_tool_mode=args.agent_tool_mode,
                                    agent_workspace=agent_workspace,
                                    magentic_max_rounds=args.magentic_max_rounds,
                                    magentic_max_resets=args.magentic_max_resets,
                                    stop_caution=bool(args.stop_caution),
                                    hide_iou_feedback=bool(args.hide_iou_feedback),
                                    legacy_stop_as_success=bool(args.legacy_stop_as_success),
                                    debug_dir=debug_dir_this_attempt,
                                    ledger_path=str(sample_llm_debug_dir / "api_calls.jsonl") if sample_llm_debug_dir is not None else "",
                                )
                                action = api_result["action"]
                                action_angle = int(api_result.get("angle", task_allowed_angles[0]))
                                model_response_text = api_result.get("raw_text", "")
                                model_cot_text = api_result.get("cot", "")
                                model_system_prompt = api_result.get("system_prompt", "")
                                model_prompt_text = api_result.get("prompt_text", "")
                                usage = api_result.get("usage", {}) or {}
                                step_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                                step_completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                                step_total_tokens = int(usage.get("total_tokens", 0) or 0)
                                if debug_dir_this_attempt is not None:
                                    step_llm_debug_dir_rel = str(debug_dir_this_attempt.relative_to(root)).replace("\\", "/")
                                sample_prompt_tokens += step_prompt_tokens
                                sample_completion_tokens += step_completion_tokens
                                sample_total_tokens += step_total_tokens
                                break
                            except Exception as e:
                                if attempt_idx < retries:
                                    sleep_s, retry_reason = compute_api_retry_sleep(str(e), attempt_idx, backoff)
                                    print(
                                        f"[step02][api-retry] sample={sample_id} step={step} "
                                        f"attempt={attempt_idx + 1}/{retries + 1} "
                                        f"reason={retry_reason} sleep={sleep_s:.1f}s error={e}",
                                        flush=True,
                                    )
                                    if sample_llm_debug_dir is not None:
                                        retry_record = {
                                            "timestamp": datetime.now(timezone.utc).isoformat(),
                                            "sample_id": str(sample_id),
                                            "environment_step": int(step),
                                            "attempt": int(attempt_idx + 1),
                                            "max_attempts": int(retries + 1),
                                            "reason": str(retry_reason),
                                            "sleep_seconds": float(sleep_s),
                                            "error_type": type(e).__name__,
                                        }
                                        with (sample_llm_debug_dir / "retry_waits.jsonl").open("a", encoding="utf-8") as retry_file:
                                            retry_file.write(json.dumps(retry_record, ensure_ascii=False) + "\n")
                                    time.sleep(sleep_s)
                                else:
                                    api_error_count += 1
                                    api_error_last = str(e)
                                    raise RuntimeError(
                                        f"API retries exhausted for sample={sample_id} step={step}: {api_error_last}"
                                    ) from e
                        dt_policy_api = perf_counter() - t_policy
                    else:
                        t_policy = perf_counter()
                        action, action_angle = choose_action_simulated(
                            rng,
                            current,
                            target_pose,
                            task["stream_type"],
                            recent_actions,
                            allowed_actions,
                            task_allowed_angles,
                        )
                        model_response_text = f"SIMULATED_POLICY: action={action}, angle={action_angle}"
                        model_cot_text = ""
                        model_prompt_text = ""
                        dt_policy_sim = perf_counter() - t_policy
                timing_report["aggregate"]["step_policy_api_s"] += dt_policy_api
                timing_report["aggregate"]["step_policy_sim_s"] += dt_policy_sim

                if args.save_frames and sample_frame_dir is not None and panel_img is not None:
                    panel_frame_path = sample_frame_dir / f"panel_{step:03d}.png"
                    save_triptych_panel(
                        panel_frame_path,
                        panel_img,
                    )
                    panel_frame_rel = str(panel_frame_path.relative_to(root)).replace("\\", "/")
                    panel_text_path = sample_frame_dir / f"panel_{step:03d}.txt"
                    panel_text_path.write_text(
                        f"action={action}\n"
                        f"action_angle={int(action_angle)}\n"
                        f"iou={score:.6f}\n"
                        f"tau_success={tau:.6f}\n"
                        f"used_steps={max(0, step - 1)}\n"
                        f"step_limit={step_limit}\n"
                        f"model_prompt={model_prompt_text}\n"
                        f"model_cot={model_cot_text}\n"
                        f"model_response_raw={model_response_text}\n",
                        encoding="utf-8",
                    )
                    panel_text_rel = str(panel_text_path.relative_to(root)).replace("\\", "/")
                    panel_meta_path = sample_frame_dir / f"panel_{step:03d}.json"
                    write_json(
                        panel_meta_path,
                        {
                            "sample_id": sample_id,
                            "step": step,
                            "action": action,
                            "action_angle": int(action_angle),
                            "model_response_raw": model_response_text,
                            "model_cot": model_cot_text,
                            "model_prompt": model_prompt_text,
                            "iou": round(score, 6),
                            "tau_success": round(tau, 6),
                            "used_steps": max(0, step - 1),
                            "step_limit": step_limit,
                            "pose_before_action": current.as_dict(),
                            "backend": args.backend,
                            "api_image_mode": args.api_image_mode if args.backend == "gemini_api" else "",
                            "llm_debug_dir": step_llm_debug_dir_rel,
                            "panel_response_file": panel_text_rel,
                        },
                    )
                    panel_meta_rel = str(panel_meta_path.relative_to(root)).replace("\\", "/")

                t_event = perf_counter()
                record = {
                    "timestamp_ms": now_ms(),
                    "sample_id": sample_id,
                    "step": step,
                    "action": action,
                    "action_angle": int(action_angle),
                    "pose": current.as_dict(),
                    "iou": round(score, 6),
                    "iou_full_precision": float(score),
                    "iou_display": f"{score:.3f}",
                    "render_backend": get_last_render_backend(),
                    "target_frame": task["target_image"],
                    "oblique_frame": "",
                    "projection_frame": "",
                    "panel_frame": panel_frame_rel,
                    "panel_meta": panel_meta_rel,
                    "panel_response_file": panel_text_rel,
                    "model_response_raw": model_response_text,
                    "model_cot": model_cot_text,
                    "model_system_prompt": model_system_prompt,
                    "model_prompt": model_prompt_text,
                    "backend": args.backend,
                    "agent_tool_mode": args.agent_tool_mode if args.api_orchestration in {"magentic", "react"} else "none",
                    "agent_tool_protocol": "python_workspace_view_image_v4" if agent_workspace is not None else "",
                    "agent_tool_stats": agent_workspace.stats() if agent_workspace is not None else {},
                    "prompt_profile": args.prompt_profile if args.backend == "gemini_api" else "",
                    "prompt_protocol": args.prompt_protocol if args.backend == "gemini_api" else "",
                    "metric_feedback_mode": metric_feedback_mode if args.backend == "gemini_api" else "",
                    "metric_feedback_mode": metric_feedback_mode if args.backend == "gemini_api" else "",
                    "hide_iou_feedback": bool(args.hide_iou_feedback) if args.backend == "gemini_api" else False,
                    "stop_protocol": "legacy_stop_as_visual_success" if args.legacy_stop_as_success else "auto_success_stop_no_further_idea_v1",
                    "prompt_tokens": int(step_prompt_tokens) if args.backend == "gemini_api" else 0,
                    "completion_tokens": int(step_completion_tokens) if args.backend == "gemini_api" else 0,
                    "total_tokens": int(step_total_tokens) if args.backend == "gemini_api" else 0,
                    "llm_debug_dir": step_llm_debug_dir_rel,
                }
                trajectory.append(record)
                events_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                dt_event = perf_counter() - t_event
                timing_report["aggregate"]["step_event_write_s"] += dt_event
                if panel_data_url:
                    recent_panel_data_urls.append(panel_data_url)
                    if args.history_window == 0:
                        recent_panel_data_urls = []
                    elif args.history_window > 0 and len(recent_panel_data_urls) > args.history_window:
                        recent_panel_data_urls = recent_panel_data_urls[-args.history_window:]
                state_summary = f"step={step} action={action} angle={int(action_angle)}"
                if args.include_history_cot and model_cot_text:
                    state_summary += f" cot={model_cot_text}"
                if metric_feedback_mode == "full":
                    state_summary += f" iou={score:.6f}"
                recent_state_summaries.append(state_summary)
                if args.history_window == 0:
                    recent_state_summaries = []
                elif args.history_window > 0 and len(recent_state_summaries) > args.history_window:
                    recent_state_summaries = recent_state_summaries[-args.history_window:]

                dt_apply = 0.0

                if action == "STOP":
                    success, terminated_reason = resolve_stop_outcome(score, tau)
                else:
                    t_apply = perf_counter()
                    current = apply_action_with_angle(current, action, int(action_angle))
                    recent_actions.append(action)
                    if args.history_window == 0:
                        recent_actions = []
                    elif args.history_window > 0 and len(recent_actions) > args.history_window:
                        recent_actions = recent_actions[-args.history_window:]
                    last_action = action
                    dt_apply = perf_counter() - t_apply
                timing_report["aggregate"]["step_apply_action_s"] += dt_apply
                dt_step_total = perf_counter() - t_step_total
                timing_report["aggregate"]["step_total_s"] += dt_step_total
                previous_iou = score

                if debug_timing:
                    step_timing = {
                        "step": step,
                        "render_s": round(dt_step_render, 6),
                        "iou_s": round(dt_step_iou, 6),
                        "frame_io_s": round(dt_step_frame_io, 6),
                        "encode_s": round(dt_step_encode, 6),
                        "policy_api_s": round(dt_policy_api, 6),
                        "policy_sim_s": round(dt_policy_sim, 6),
                        "event_write_s": round(dt_event, 6),
                        "apply_action_s": round(dt_apply, 6),
                        "total_s": round(dt_step_total, 6),
                        "action": action,
                        "angle": int(action_angle),
                        "iou": round(score, 6),
                    }
                    task_timing["steps"].append(step_timing)
                    print(
                        f"[timing][step02][{sample_id}][step={step}] "
                        f"render_ms={dt_step_render * 1000:.2f} "
                        f"iou_ms={dt_step_iou * 1000:.2f} "
                        f"encode_ms={dt_step_encode * 1000:.2f} "
                        f"policy_api_ms={dt_policy_api * 1000:.2f} "
                        f"policy_sim_ms={dt_policy_sim * 1000:.2f} "
                        f"event_ms={dt_event * 1000:.2f} "
                        f"apply_ms={dt_apply * 1000:.2f} "
                        f"total_ms={dt_step_total * 1000:.2f} "
                        f"action={action} iou={score:.4f}"
                    )
                if action == "STOP":
                    break

            if trajectory and trajectory[-1]["action"] != "STOP" and len(trajectory) >= step_limit:
                _, final_mask_eval, _ = render_views_limited(
                    task,
                    current,
                    display_size=display_size,
                    eval_size=eval_size,
                    need_oblique=False,
                )
                final_iou = iou(target_mask_eval, final_mask_eval)
                success = bool(final_iou >= tau)
                if success:
                    terminated_reason = "threshold_reached"
            if terminated_reason == "max_steps" and not budget_exhaustion_mode:
                budget_exhaustion_mode = "fixed_step_limit"

            elapsed = now_ms() - start_ms
            dt_task_total = perf_counter() - t_task_start
            steps = len(trajectory)
            agent_workspace_archive = ""
            agent_tool_stats = {}
            if agent_workspace is not None:
                agent_tool_stats = agent_workspace.stats()
                agent_workspace_archive = _rel_path(agent_workspace.archive(), root)
            row = {
                "sample_id": sample_id,
                "stream_type": task["stream_type"],
                "model_name": model_name,
                "checkpoint_path": str(args.checkpoint_path),
                "backend": args.backend,
                "agent_tool_mode": args.agent_tool_mode if args.api_orchestration in {"magentic", "react"} else "none",
                "agent_tool_protocol": str(agent_tool_stats.get("protocol", "")),
                "agent_tool_calls": int(agent_tool_stats.get("calls", 0)),
                "agent_tool_failed_calls": int(agent_tool_stats.get("failed_calls", 0)),
                "agent_tool_runtime_s": float(agent_tool_stats.get("runtime_s", 0.0)),
                "agent_workspace_archive": agent_workspace_archive,
                "success": int(success),
                "steps": steps,
                "step_limit": step_limit,
                "iou_final": round(final_iou, 6),
                "iou_final_display": f"{final_iou:.3f}",
                "projection_distance_final": round(1.0 - final_iou, 6),
                "tau_success": round(tau, 6),
                "latency_ms": elapsed,
                "terminated_reason": terminated_reason,
                "seed": seed,
                "api_seed": int(args.api_seed) if args.backend == "gemini_api" else 0,
                "api_fallback_count": api_fallback_count,
                "api_error_count": api_error_count,
                "api_error_last": api_error_last,
                "prompt_tokens": int(sample_prompt_tokens) if args.backend == "gemini_api" else 0,
                "completion_tokens": int(sample_completion_tokens) if args.backend == "gemini_api" else 0,
                "total_tokens": int(sample_total_tokens) if args.backend == "gemini_api" else 0,
                "trajectory_path": _rel_path(traj_root / f"{sample_id}.json", root),
                "attempt_tag": args.attempt_tag,
                "api_image_mode": args.api_image_mode if args.backend == "gemini_api" else "",
                "prompt_profile": args.prompt_profile if args.backend == "gemini_api" else "",
                "prompt_protocol": args.prompt_protocol if args.backend == "gemini_api" else "",
                "metric_feedback_mode": metric_feedback_mode if args.backend == "gemini_api" else "",
                "render_environment": json.dumps(render_metadata, sort_keys=True, separators=(",", ":")),
                "hide_iou_feedback": bool(args.hide_iou_feedback) if args.backend == "gemini_api" else False,
                "stop_protocol": "legacy_stop_as_visual_success" if args.legacy_stop_as_success else "auto_success_stop_no_further_idea_v1",
                "stagnation_patience": int(args.stagnation_patience),
                "min_best_iou_gain": float(args.min_best_iou_gain),
                "nonimproving_rotations_at_end": int(nonimproving_rotations),
                "budget_exhaustion_mode": budget_exhaustion_mode,
                "history_window": int(args.history_window) if args.backend == "gemini_api" else 0,
                "step_limit_multiplier": float(args.step_limit_multiplier),
                "api_display_width": int(args.api_display_width) if args.backend == "gemini_api" else 0,
                "api_display_height": int(args.api_display_height) if args.backend == "gemini_api" else 0,
                "model_version": args.api_model if args.backend == "gemini_api" else model_name,
                "allowed_angles": ",".join(str(x) for x in task_allowed_angles),
                "render_backends": ",".join(sorted(render_backends)),
                "target_eval_sha256": target_eval_sha256,
                "initial_eval_sha256": initial_eval_sha256,
                "initial_iou_full_precision": initial_iou_full_precision,
                "initial_iou_display": f"{float(initial_iou_full_precision or 0.0):.3f}",
                "target_pose": json.dumps(target_pose.as_dict(), separators=(",", ":")),
                "init_pose": json.dumps(Pose.from_dict(task["init_pose"]).as_dict(), separators=(",", ":")),
                "eval_size": eval_size,
                "llm_debug_dir": _rel_path(sample_llm_debug_dir, root) if sample_llm_debug_dir is not None else "",
            }
            rows.append(row)
            if debug_timing:
                task_timing["task_total_s"] = round(dt_task_total, 6)
                task_timing["steps_executed"] = steps
                task_timing["terminated_reason"] = terminated_reason
                timing_report["tasks"].append(task_timing)
                print(
                    f"[timing][step02][task={sample_id}] "
                    f"setup_ms={dt_setup * 1000:.2f} "
                    f"target_render_ms={dt_target_render * 1000:.2f} "
                    f"target_encode_ms={dt_target_encode * 1000:.2f} "
                    f"steps={steps} task_total_ms={dt_task_total * 1000:.2f} "
                    f"reason={terminated_reason}"
                )

            write_json(
                traj_root / f"{sample_id}.json",
                {
                    "sample_id": sample_id,
                    "model_name": model_name,
                    "checkpoint_path": str(args.checkpoint_path),
                    "backend": args.backend,
                    "seed": seed,
                    "agent_tool_mode": args.agent_tool_mode if args.api_orchestration in {"magentic", "react"} else "none",
                    "agent_tool_protocol": str(agent_tool_stats.get("protocol", "")),
                    "agent_tool_stats": agent_tool_stats,
                    "agent_workspace_archive": agent_workspace_archive,
                    "api_seed": int(args.api_seed) if args.backend == "gemini_api" else 0,
                    "api_fallback_count": api_fallback_count,
                    "api_error_count": api_error_count,
                    "api_error_last": api_error_last,
                    "prompt_profile": args.prompt_profile if args.backend == "gemini_api" else "",
                    "prompt_protocol": args.prompt_protocol if args.backend == "gemini_api" else "",
                    "hide_iou_feedback": bool(args.hide_iou_feedback) if args.backend == "gemini_api" else False,
                    "stop_protocol": "legacy_stop_as_visual_success" if args.legacy_stop_as_success else "auto_success_stop_no_further_idea_v1",
                    "stagnation_patience": int(args.stagnation_patience),
                    "min_best_iou_gain": float(args.min_best_iou_gain),
                    "nonimproving_rotations_at_end": int(nonimproving_rotations),
                    "budget_exhaustion_mode": budget_exhaustion_mode,
                    "render_backends": sorted(render_backends),
                    "render_environment": render_metadata,
                    "target_eval_sha256": target_eval_sha256,
                    "initial_eval_sha256": initial_eval_sha256,
                    "initial_iou_full_precision": initial_iou_full_precision,
                    "initial_iou_display": f"{float(initial_iou_full_precision or 0.0):.3f}",
                    "target_pose": target_pose.as_dict(),
                    "init_pose": Pose.from_dict(task["init_pose"]).as_dict(),
                    "eval_size": eval_size,
                    "llm_debug_dir": _rel_path(sample_llm_debug_dir, root) if sample_llm_debug_dir is not None else "",
                    "records": trajectory,
                },
            )
            write_csv_rows(out_csv, rows)
            completed_ids_runtime.add(sample_id)
            checkpoint = {
                "version": 1,
                "config_fingerprint": cfg_fp,
                "backend": args.backend,
                "model_name": model_name,
                "tasks_file": _rel_path(tasks_path, root),
                "output_csv": _rel_path(out_csv, root),
                "raw_events_file": _rel_path(raw_events_path, root),
                "trajectory_dir": _rel_path(traj_root, root),
                "attempt_tag": args.attempt_tag,
                "seed": seed,
                "api_seed": int(args.api_seed) if args.backend == "gemini_api" else 0,
                "selected_task_count": len(selected_task_ids),
                "selected_task_ids": list(selected_task_ids),
                "completed_count": len(completed_ids_runtime),
                "completed_sample_ids": sorted(completed_ids_runtime),
                "existing_rows_count": len(rows),
                "last_completed_sample_id": sample_id,
                "allowed_angles": list(global_allowed_angles),
                "prompt_profile": args.prompt_profile if args.backend == "gemini_api" else "",
                "prompt_protocol": args.prompt_protocol if args.backend == "gemini_api" else "",
                "hide_iou_feedback": bool(args.hide_iou_feedback) if args.backend == "gemini_api" else False,
                "stop_protocol": "legacy_stop_as_visual_success" if args.legacy_stop_as_success else "auto_success_stop_no_further_idea_v1",
                "stagnation_patience": int(args.stagnation_patience),
                "min_best_iou_gain": float(args.min_best_iou_gain),
                "history_window": int(args.history_window) if args.backend == "gemini_api" else 0,
                "api_display_width": int(args.api_display_width) if args.backend == "gemini_api" else 0,
                "api_display_height": int(args.api_display_height) if args.backend == "gemini_api" else 0,
                "model_version": args.api_model if args.backend == "gemini_api" else model_name,
                "step_limit_multiplier": float(args.step_limit_multiplier),
                "max_step_cap": int(args.max_step_cap),
                "min_step_limit": int(args.min_step_limit),
                "save_frames": bool(args.save_frames),
                "save_llm_debug": bool(args.save_llm_debug) if args.backend == "gemini_api" else False,
                "llm_debug_dir": _rel_path(llm_debug_root, root) if (args.save_llm_debug and args.backend == "gemini_api") else "",
                "rng_state": encode_rng_state(rng.getstate()),
            }
            write_json(checkpoint_path, checkpoint)

    def _to_int(v) -> int:
        try:
            return int(v)
        except Exception:
            try:
                return int(float(v))
            except Exception:
                return 0

    write_csv_rows(out_csv, rows)

    summary = {
        "model_name": model_name,
        "model_version": args.api_model if args.backend == "gemini_api" else model_name,
        "checkpoint_path": str(args.checkpoint_path),
        "backend": args.backend,
        "seed": seed,
        "samples": len(rows),
        "resume_enabled": bool(args.resume),
        "existing_samples_before_run": len(existing_rows),
        "new_samples_this_run": max(0, len(rows) - len(existing_rows)),
        "pending_samples_run": len(tasks),
        "checkpoint_file": _rel_path(checkpoint_path, root),
        "checkpoint_completed_count": len(completed_ids_runtime),
        "success_rate": round(sum(_to_int(r.get("success", 0)) for r in rows) / len(rows), 6) if rows else 0.0,
        "prompt_tokens": sum(_to_int(r.get("prompt_tokens", 0)) for r in rows) if rows else 0,
        "completion_tokens": sum(_to_int(r.get("completion_tokens", 0)) for r in rows) if rows else 0,
        "total_tokens": sum(_to_int(r.get("total_tokens", 0)) for r in rows) if rows else 0,
        "total_api_fallback_count": sum(_to_int(r.get("api_fallback_count", 0)) for r in rows) if rows else 0,
        "total_api_error_count": sum(_to_int(r.get("api_error_count", 0)) for r in rows) if rows else 0,
        "output": _rel_path(out_csv, root),
        "attempt_tag": args.attempt_tag,
        "api_image_mode": args.api_image_mode if args.backend == "gemini_api" else "",
        "prompt_profile": args.prompt_profile if args.backend == "gemini_api" else "",
        "prompt_protocol": args.prompt_protocol if args.backend == "gemini_api" else "",
        "metric_feedback_mode": metric_feedback_mode if args.backend == "gemini_api" else "",
        "render_environment": render_metadata,
        "hide_iou_feedback": bool(args.hide_iou_feedback) if args.backend == "gemini_api" else False,
        "stop_protocol": "legacy_stop_as_visual_success" if args.legacy_stop_as_success else "auto_success_stop_no_further_idea_v1",
        "stagnation_patience": int(args.stagnation_patience),
        "min_best_iou_gain": float(args.min_best_iou_gain),
        "history_window": int(args.history_window) if args.backend == "gemini_api" else 0,
        "api_temperature": float(args.api_temperature) if args.backend == "gemini_api" else 0.0,
        "api_seed": int(args.api_seed) if args.backend == "gemini_api" else 0,
        "api_max_tokens": int(args.api_max_tokens) if args.backend == "gemini_api" else 0,
        "api_response_format": str(args.api_response_format) if args.backend == "gemini_api" else "",
        "api_orchestration": str(args.api_orchestration) if args.backend == "gemini_api" else "",
        "agent_tool_mode": str(args.agent_tool_mode) if args.api_orchestration in {"magentic", "react"} else "none",
        "agent_tool_protocol": "python_workspace_view_image_v4" if args.api_orchestration in {"magentic", "react"} and args.agent_tool_mode == "python" else "",
        "api_stream": False,
        "magentic_max_rounds": int(args.magentic_max_rounds) if args.backend == "gemini_api" else 0,
        "magentic_max_resets": int(args.magentic_max_resets) if args.backend == "gemini_api" else 0,
        "timeout_seconds": int(args.timeout) if args.backend == "gemini_api" else 0,
        "api_retries": int(args.api_retries) if args.backend == "gemini_api" else 0,
        "api_retry_backoff_seconds": float(args.api_retry_backoff) if args.backend == "gemini_api" else 0.0,
        "step_limit_multiplier": float(args.step_limit_multiplier),
        "api_display_width": int(args.api_display_width) if args.backend == "gemini_api" else 0,
        "api_display_height": int(args.api_display_height) if args.backend == "gemini_api" else 0,
        "allowed_angles": ",".join(str(x) for x in global_allowed_angles),
        "save_llm_debug": bool(args.save_llm_debug) if args.backend == "gemini_api" else False,
        "llm_debug_dir": _rel_path(llm_debug_root, root) if (args.save_llm_debug and args.backend == "gemini_api") else "",
    }
    if debug_timing:
        timing_path = output_reports / "step_02_timing.json"
        timing_report["aggregate"] = {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in timing_report["aggregate"].items()
        }
        write_json(timing_path, timing_report)
        summary["timing_report"] = _rel_path(timing_path, root)
        agg = timing_report["aggregate"]
        print(
            "[timing][step02][summary] "
            f"task_setup_s={agg['task_setup_s']:.4f} "
            f"target_render_s={agg['target_render_s']:.4f} "
            f"target_encode_s={agg['target_encode_s']:.4f} "
            f"step_render_s={agg['step_render_s']:.4f} "
            f"step_iou_s={agg['step_iou_s']:.4f} "
            f"step_frame_io_s={agg['step_frame_io_s']:.4f} "
            f"step_encode_s={agg['step_encode_s']:.4f} "
            f"step_policy_api_s={agg['step_policy_api_s']:.4f} "
            f"step_policy_sim_s={agg['step_policy_sim_s']:.4f} "
            f"step_event_write_s={agg['step_event_write_s']:.4f} "
            f"step_apply_action_s={agg['step_apply_action_s']:.4f} "
            f"step_total_s={agg['step_total_s']:.4f}"
        )
    write_json(output_reports / "step_02_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
