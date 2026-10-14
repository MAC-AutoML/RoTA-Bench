#!/usr/bin/env python3
"""Public single-task projection alignment demo for ``single_00005``.

Run locally with ``python single_task_public_demo.py``.  For a public HTTPS
deployment, bind to localhost and put the existing reverse proxy in front of
it; add ``--secure-cookie --require-gpu`` to the production command.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import secrets
import sys
import threading
import time
import zlib
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from benchmark_core import (
    Pose,
    apply_action,
    get_last_render_backend,
    iou,
    np,
    render_views_fast,
)


ROOT = Path(__file__).resolve().parent
CONTROLLER_ASSET_PATH = ROOT / "assets" / "xbox-controller.png"
TASKS_PATH = ROOT / "tasks" / "tasks.jsonl"
SAMPLE_ID = "single_00627"
ALLOWED_ACTIONS = {
    "PITCH_POS",
    "PITCH_NEG",
    "YAW_POS",
    "YAW_NEG",
    "ROLL_POS",
    "ROLL_NEG",
    "STOP",
}
ALLOWED_ANGLES = {30, 10, 5, 1}
SUCCESS_THRESHOLD = 0.9
MAX_STEPS = 48
SESSION_TTL_SECONDS = 30 * 60
SESSION_CLEANUP_SECONDS = 60
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_REQUESTS = 30
MAX_JSON_BYTES = 1024
DISPLAY_SIZE = (480, 270)


class ClientError(ValueError):
    """An invalid public API request."""


class RateLimitError(ClientError):
    """The visitor sent too many state-changing requests."""


class TerminalStateError(ClientError):
    """The current game has already ended."""


def load_single_task(path: Path, sample_id: str = SAMPLE_ID) -> Dict:
    """Read the configured JSONL file one line at a time and find one sample."""
    matches = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("sample_id") == sample_id:
                matches.append({key: value for key, value in row.items() if key not in {"answer_actions", "answer_poses"}})
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {sample_id!r} task, found {len(matches)}")


def load_tasks(path: Path, sample_ids) -> Dict[str, Dict]:
    """Read the JSONL benchmark once instead of rescanning it per task."""
    wanted = set(sample_ids)
    found: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            # Avoid decoding hundreds of large icon-point arrays that are not
            # part of the frozen split.
            if not any(f'"sample_id": "{sample_id}"' in line for sample_id in wanted):
                continue
            item = json.loads(line)
            sample_id = item.get("sample_id")
            if sample_id in wanted:
                found[sample_id] = item
    missing = wanted.difference(found)
    if missing:
        raise RuntimeError(f"Missing benchmark tasks: {sorted(missing)}")
    return found
    return matches[0]


def png_data_url(image) -> str:
    """Encode a grayscale or RGB image without exposing files to the browser."""
    if np is not None:
        array = np.asarray(image, dtype=np.uint8)
        height, width = array.shape[:2]
        color_type = 2 if array.ndim == 3 else 0
        rows = [b"\x00" + array[y, :width].tobytes() for y in range(height)]
    else:  # pragma: no cover - benchmark_core normally provides NumPy.
        height = len(image)
        width = len(image[0]) if height else 0
        color_type = 0
        rows = [b"\x00" + bytes(row[:width]) for row in image]

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return len(payload).to_bytes(4, "big") + kind + payload + zlib.crc32(kind + payload).to_bytes(4, "big")

    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            width.to_bytes(4, "big")
            + height.to_bytes(4, "big")
            + bytes((8, color_type, 0, 0, 0)),
        )
        + chunk(b"IDAT", zlib.compress(b"".join(rows), level=6))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


@dataclass
class VisitorState:
    pose: Pose
    task_id: str = SAMPLE_ID
    used_steps: int = 0
    terminated: bool = False
    success: bool = False
    last_access: float = field(default_factory=time.monotonic)
    requests: Deque[float] = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)


class DemoService:
    def __init__(self, tasks: Dict[str, Dict], *, require_gpu: bool, logger: logging.Logger) -> None:
        self.tasks = tasks
        self.task = tasks[SAMPLE_ID]
        self.require_gpu = require_gpu
        self.logger = logger
        self.sessions: Dict[str, VisitorState] = {}
        self.sessions_lock = threading.Lock()
        self.render_lock = threading.Lock()
        self.last_cleanup = 0.0
        self.target_display = None
        self.target_eval = None
        self.target_cache: Dict[str, Tuple] = {}
        # Keep the public renderer's evaluation resolution aligned with the
        # selected benchmark task; ``task`` is not a local variable here.
        self.eval_size = int(self.task.get("eval_size", 320))
        self.ready = False

    def startup_check(self) -> None:
        """Validate the renderer without blocking startup on CPU pre-rendering.

        Targets and observations are rendered lazily on the first request for
        each task. This keeps a CPU Space responsive while preserving the same
        render path and evaluation protocol.
        """
        if np is None:
            raise RuntimeError("NumPy is required for the public demo renderer")
        backend = get_last_render_backend()
        if self.require_gpu and backend not in (None, "cuda"):
            raise RuntimeError(f"GPU renderer required, got {backend!r}")
        self.ready = True
        self.logger.info("Renderer ready (lazy CPU rendering; backend=%s)", backend)

    def _cleanup_expired_sessions(self, now: float) -> None:
        if now - self.last_cleanup < SESSION_CLEANUP_SECONDS:
            return
        with self.sessions_lock:
            expired = [sid for sid, state in self.sessions.items() if now - state.last_access >= SESSION_TTL_SECONDS]
            for sid in expired:
                del self.sessions[sid]
            self.last_cleanup = now
        if expired:
            self.logger.info("Expired %d idle anonymous sessions", len(expired))

    def get_session(self, cookie_sid: Optional[str], task_id: Optional[str] = None) -> Tuple[str, VisitorState, bool]:
        now = time.monotonic()
        self._cleanup_expired_sessions(now)
        with self.sessions_lock:
            state = self.sessions.get(cookie_sid or "")
            if state is not None:
                if task_id in self.tasks and state.task_id != task_id:
                    state.task_id = task_id
                    state.pose = Pose.from_dict(self.tasks[task_id]["init_pose"])
                    state.used_steps = 0
                    state.terminated = False
                    state.success = False
                state.last_access = now
                return cookie_sid or "", state, False
            selected = task_id if task_id in self.tasks else SAMPLE_ID
            sid = secrets.token_urlsafe(32)
            state = VisitorState(pose=Pose.from_dict(self.tasks[selected]["init_pose"]), task_id=selected, last_access=now)
            self.sessions[sid] = state
            return sid, state, True

    def _rate_limit(self, state: VisitorState, now: float) -> None:
        while state.requests and now - state.requests[0] >= RATE_LIMIT_WINDOW_SECONDS:
            state.requests.popleft()
        if len(state.requests) >= RATE_LIMIT_REQUESTS:
            raise RateLimitError("操作过于频繁，请稍后再试。")
        state.requests.append(now)

    def _target(self, task: Dict):
        task_id = str(task.get("sample_id", SAMPLE_ID))
        cached = self.target_cache.get(task_id)
        if cached is None:
            target_pose = Pose.from_dict(task["target_pose"])
            display, evaluation, _ = render_views_fast(task, target_pose, display_size=DISPLAY_SIZE, eval_size=self.eval_size, need_oblique=False)
            cached = (display, evaluation)
            self.target_cache[task_id] = cached
        return cached

    def _render(self, pose: Pose, task: Dict):
        with self.render_lock:
            target_display, target_eval = self._target(task)
            current_display, current_eval, current_oblique = render_views_fast(
                task,
                pose,
                display_size=DISPLAY_SIZE,
                eval_size=self.eval_size,
                need_oblique=True,
            )
            score = iou(target_eval, current_eval)
        return current_display, current_oblique, score

    def state_payload(self, state: VisitorState) -> Dict:
        with state.lock:
            state.last_access = time.monotonic()
            task = self.tasks[state.task_id]
            target_display, _ = self._target(task)
            current_display, current_oblique, _ = self._render(state.pose, task)
            return {
                "target_image": png_data_url(target_display),
                "projection_image": png_data_url(current_display),
                "object_image": png_data_url(current_oblique),
                "used_steps": state.used_steps,
                "max_steps": MAX_STEPS,
                "terminated": state.terminated,
                "success": state.success,
                "task_id": state.task_id,
            }

    def apply_action(self, state: VisitorState, action: object, angle: object) -> Dict:
        if not isinstance(action, str) or action not in ALLOWED_ACTIONS:
            raise ClientError("不支持的旋转方向。")
        if isinstance(angle, bool) or not isinstance(angle, int) or angle not in ALLOWED_ANGLES:
            raise ClientError("不支持的旋转角度。")
        with state.lock:
            now = time.monotonic()
            state.last_access = now
            self._rate_limit(state, now)
            if state.terminated:
                raise TerminalStateError("本轮已结束，请重新开始。")
            _, _, current_score = self._render(state.pose, self.tasks[state.task_id])
            if action == "STOP":
                state.used_steps += 1
                state.terminated = True
                state.success = current_score >= SUCCESS_THRESHOLD
                return self.state_payload_locked(state)
            state.pose = apply_action(state.pose, action, angle)
            state.used_steps += 1
            _, _, score = self._render(state.pose, self.tasks[state.task_id])
            if score >= SUCCESS_THRESHOLD:
                state.terminated = True
                state.success = True
            elif state.used_steps >= MAX_STEPS:
                state.terminated = True
            return self.state_payload_locked(state)

    def state_payload_locked(self, state: VisitorState) -> Dict:
        task = self.tasks[state.task_id]
        target_display, _ = self._target(task)
        current_display, current_oblique, _ = self._render(state.pose, task)
        return {
            "target_image": png_data_url(target_display),
            "projection_image": png_data_url(current_display),
            "object_image": png_data_url(current_oblique),
            "used_steps": state.used_steps,
            "max_steps": MAX_STEPS,
            "terminated": state.terminated,
            "success": state.success,
            "task_id": state.task_id,
        }

    def reset(self, state: VisitorState) -> Dict:
        with state.lock:
            now = time.monotonic()
            state.last_access = now
            self._rate_limit(state, now)
            state.pose = Pose.from_dict(self.tasks[state.task_id]["init_pose"])
            state.used_steps = 0
            state.terminated = False
            state.success = False
            return self.state_payload_locked(state)


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="theme-color" content="#f5f5f7">
  <title>投影对齐</title>
  <style>
    :root { color-scheme: dark; --ink:#f5f7ff; --muted:#9ba7bd; --line:rgba(160,184,220,.18); --canvas:#101827; --wash:#070b13; --blue:#54a8ff; --green:#54d38a; }
    * { box-sizing:border-box; }
    html, body { overflow:hidden; }
    body { margin:0; min-width:320px; background:radial-gradient(circle at 50% -20%,#17263c 0%,#070b13 52%,#05070c 100%); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text",sans-serif; -webkit-font-smoothing:antialiased; }
    button { font:inherit; }
    .shell { position:relative; width:min(1440px,100%); min-height:100svh; margin:auto; padding:18px clamp(16px,4vw,56px) 28px; isolation:isolate; }
    .shell::before { content:""; position:fixed; inset:0; z-index:-1; pointer-events:none; opacity:.28; background-image:linear-gradient(rgba(116,164,224,.11) 1px,transparent 1px),linear-gradient(90deg,rgba(116,164,224,.11) 1px,transparent 1px); background-size:44px 44px; mask-image:linear-gradient(to bottom,black,transparent 78%); }
    .topbar { min-height:44px; display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid var(--line); box-shadow:0 1px 0 rgba(84,168,255,.08); }
    .top-actions { display:flex; align-items:center; gap:12px; }
    .brand { font-size:17px; font-weight:600; letter-spacing:-.35px; }
    .steps { color:var(--blue); font-size:14px; font-variant-numeric:tabular-nums; text-shadow:0 0 14px rgba(84,168,255,.35); }
    main { padding-top:clamp(28px,5vh,64px); }
    .intro { max-width:640px; margin-bottom:clamp(30px,5vh,60px); }
    h1 { margin:0; font-size:clamp(34px,5vw,56px); line-height:1.07; letter-spacing:-.04em; font-weight:600; }
    .intro p { margin:12px 0 0; color:var(--muted); font-size:clamp(17px,2vw,21px); line-height:1.35; letter-spacing:-.015em; }
    .views { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:clamp(12px,2vw,28px); }
    figure { margin:0; min-width:0; }
    figcaption { display:flex; align-items:center; justify-content:space-between; min-height:28px; margin-bottom:9px; color:var(--muted); font-size:14px; font-weight:600; letter-spacing:-.15px; }
    .media { overflow:hidden; background:#02050a; border:1px solid rgba(123,169,224,.34); border-radius:10px; aspect-ratio:16 / 9; display:grid; place-items:center; box-shadow:0 12px 34px rgba(0,0,0,.28), inset 0 0 24px rgba(40,106,180,.08); }
    .media.object { background:#02050a; border-color:rgba(123,169,224,.34); }
    .media img { display:block; width:100%; height:100%; object-fit:fill; image-rendering:auto; }
    .controls { margin-top:clamp(32px,6vh,72px); padding-top:20px; border-top:1px solid var(--line); }
    .text-button { appearance:none; min-height:44px; border:1px solid var(--line); background:var(--canvas); color:var(--ink); cursor:pointer; transition:background-color 180ms ease, color 180ms ease, transform 180ms ease; }
    .controller { position:relative; width:min(860px,100%); min-height:320px; margin:0 auto 14px; border:1px solid rgba(142,185,239,.3); border-radius:18px; background:linear-gradient(145deg,#1a2a42 0%,#0d1524 48%,#080d17 100%); box-shadow:0 22px 46px rgba(0,0,0,.42), 0 0 38px rgba(36,117,210,.1), inset 0 1px 0 rgba(255,255,255,.16), inset 0 -2px 0 rgba(0,0,0,.8); }
    .controller::before { content:""; position:absolute; inset:10px; border:1px solid rgba(148,194,246,.16); border-radius:12px; pointer-events:none; }
    .controller-grip, .controller-body { display:contents; }
    .shoulder-button { appearance:none; position:absolute; z-index:2; top:34px; width:92px; height:38px; border:1px solid #b8b8bd; border-radius:10px; background:linear-gradient(145deg,#fff 0%,#f3f3f5 58%,#d8d8dd 100%); color:#1d1d1f; cursor:pointer; box-shadow:0 4px 0 #84848a, inset 0 1px 0 rgba(255,255,255,.96), inset 0 -2px 0 rgba(0,0,0,.12); font-size:20px; font-weight:650; line-height:1; transition:background-color 180ms ease, transform 180ms ease, box-shadow 180ms ease; }
    .shoulder-left { left:calc(50% - 94px); }
    .shoulder-right { left:calc(50% + 2px); }
    .dpad { --key-size:58px; position:absolute; z-index:2; top:94px; left:76px; display:grid; grid-template-columns:repeat(3,var(--key-size)); grid-template-rows:repeat(3,var(--key-size)); filter:drop-shadow(0 4px 2px rgba(0,0,0,.55)); }
    .controller-button { appearance:none; display:grid; place-items:center; width:var(--key-size); height:var(--key-size); border:1px solid #b8b8bd; background:linear-gradient(145deg,#fff 0%,#f3f3f5 58%,#d8d8dd 100%); color:#1d1d1f; cursor:pointer; box-shadow:0 4px 0 #84848a, inset 0 1px 0 rgba(255,255,255,.96), inset 0 -2px 0 rgba(0,0,0,.12); font-size:27px; font-weight:650; line-height:1; transition:background-color 180ms ease, transform 180ms ease, box-shadow 180ms ease; }
    .controller-button:hover, .shoulder-button:hover { background:linear-gradient(145deg,#fff 0%,#fff 55%,#e7e7eb 100%); }
    .dpad-up { grid-column:2; grid-row:1; border-radius:7px 7px 3px 3px; }
    .dpad-left { grid-column:1; grid-row:2; border-radius:7px 3px 3px 7px; }
    .dpad-center { grid-column:2; grid-row:2; width:28px; height:28px; place-self:center; border:1px solid #080808; border-radius:50%; background:#0a0a0b; box-shadow:inset 0 1px 0 rgba(255,255,255,.1); }
    .dpad-right { grid-column:3; grid-row:2; border-radius:3px 7px 7px 3px; }
    .dpad-down { grid-column:2; grid-row:3; border-radius:3px 3px 7px 7px; }
    .arcade-buttons { --key-size:58px; position:absolute; z-index:2; top:72px; right:88px; display:grid; grid-template-columns:repeat(3,var(--key-size)); grid-template-rows:repeat(3,var(--key-size)); filter:drop-shadow(0 4px 2px rgba(0,0,0,.55)); }
    .arcade-button { border-radius:50%; font-size:18px; }
    .angle-x { grid-column:2; grid-row:1; }
    .angle-y { grid-column:1; grid-row:2; }
    .angle-b { grid-column:3; grid-row:2; }
    .angle-a { grid-column:2; grid-row:3; }
    .angle-button[aria-pressed="true"] { border-color:#0071e3; background:linear-gradient(145deg,#fff 0%,#eef6ff 62%,#c9e4ff 100%); box-shadow:0 4px 0 #0066c7, inset 0 1px 0 rgba(255,255,255,.96), 0 0 16px rgba(0,113,227,.48); }
    .text-button:hover { background:#e8e8ed; }
    button:focus-visible { outline:3px solid rgba(0,102,204,.35); outline-offset:2px; }
    button:active { transform:scale(.96); }
    .controller-button:active { transform:translateY(2px); box-shadow:0 1px 0 #84848a, inset 0 1px 0 rgba(0,0,0,.16); }
    .shoulder-left:active, .shoulder-right:active { transform:translateY(2px); box-shadow:0 1px 0 #84848a, inset 0 1px 0 rgba(0,0,0,.16); }
    button:disabled { cursor:not-allowed; opacity:.42; transform:none; }
    .utility { display:flex; justify-content:space-between; gap:16px; align-items:center; margin-top:20px; }
    .utility-actions { display:flex; flex-wrap:wrap; gap:8px; }
    .text-button { border:0; border-radius:999px; padding:0 16px; color:var(--blue); font-size:15px; }
    .status { min-height:20px; color:var(--muted); font-size:14px; text-align:right; }
    .status.error { color:#b42318; }
    .help { max-width:620px; margin-top:18px; color:var(--muted); font-size:14px; line-height:1.45; }
    .help p { margin:0; }
    .overlay { position:fixed; inset:0; display:none; align-items:center; justify-content:center; padding:24px; background:rgba(245,245,247,.82); backdrop-filter:blur(12px); z-index:10; }
    .overlay.show { display:flex; animation:appear 240ms ease-out; }
    .success { width:min(420px,100%); padding:32px; border-radius:8px; background:var(--canvas); border:1px solid rgba(36,138,61,.3); text-align:center; }
    .success .check { color:var(--green); font-size:36px; line-height:1; }
    .success h2 { margin:12px 0 8px; font-size:30px; letter-spacing:-.035em; }
    .success p { margin:0 0 24px; color:var(--muted); }
    .success .text-button { background:var(--blue); color:#fff; border:1px solid var(--blue); }
    @keyframes appear { from { opacity:0; transform:scale(1.025); } to { opacity:1; transform:scale(1); } }
    @media (max-width:760px) { .shell { padding:14px 16px 24px; } .views { grid-template-columns:1fr; gap:24px; } .utility { align-items:flex-start; flex-direction:column; } .status { text-align:left; } }
    @media (max-width:520px) { .controller { min-height:274px; border-radius:14px; } .shoulder-button { top:28px; width:72px; height:34px; font-size:17px; } .shoulder-left { left:calc(50% - 70px); } .shoulder-right { left:calc(50% + 2px); } .dpad { --key-size:42px; top:86px; left:18px; } .arcade-buttons { --key-size:42px; top:80px; right:18px; } .dpad-center { width:22px; height:22px; } .controller-button { font-size:22px; } .arcade-button { font-size:14px; } }
    @media (prefers-color-scheme:dark) { .text-button:hover { background:#1b2b43; } .overlay { background:rgba(0,0,0,.78); } }
    @media (prefers-reduced-motion:reduce) { *, *::before, *::after { animation-duration:.01ms !important; transition-duration:.01ms !important; } }
    /* Match the research portal's pixel arcade presentation. */
    :root { --ink:#E5EDF4; --muted:#91A3B5; --line:#2A3B50; --canvas:#101A28; --wash:#080C14; --blue:#55D6E8; --green:#78D6A3; --pink:#FF4FD8; }
    body { background:radial-gradient(ellipse at 50% -15%,#13243a 0,transparent 56%),#080C14; color:var(--ink); font-family:Consolas,"SFMono-Regular",monospace; }
    .shell::before { opacity:.48; background-image:linear-gradient(rgba(48,94,130,.14) 1px,transparent 1px),linear-gradient(90deg,rgba(48,94,130,.14) 1px,transparent 1px); background-size:24px 24px; }
    .topbar { border-bottom:2px solid var(--blue); box-shadow:0 3px 0 #11263a; }
    .brand,.steps { color:var(--blue); text-shadow:0 0 12px rgba(85,214,232,.16); letter-spacing:.06em; }
    h1 { font-family:Consolas,"SFMono-Regular",monospace; text-transform:uppercase; letter-spacing:.015em; text-shadow:3px 3px 0 #153148; }
    .intro p { color:#a8bfce; font-size:17px; }
    figcaption { color:var(--blue); font-family:Consolas,monospace; text-transform:uppercase; letter-spacing:.06em; }
    .media,.media.object { background:#02050a; border:2px solid #284961; border-radius:0; box-shadow:4px 4px 0 #08111d,0 0 16px rgba(73,231,255,.06); }
    .controller { border:2px solid #31516d; border-radius:0; background:linear-gradient(135deg,#13233a,#091321 65%); box-shadow:6px 6px 0 #06101c,0 0 26px rgba(73,231,255,.08),inset 0 1px 0 #45617d; }
    .controller::before { border:1px dashed #33506b; border-radius:0; }
    .controller-button,.shoulder-button { border-color:#a2acb6; background:linear-gradient(145deg,#fff,#d9e0e5); color:#07101b; }
    .controller-button:hover,.shoulder-button:hover { background:linear-gradient(145deg,#fff,#ebffff); }
    .angle-button[aria-pressed="true"] { border-color:var(--pink); background:linear-gradient(145deg,#fff,#ffe0fa); box-shadow:0 4px 0 #7d3276,0 0 12px rgba(255,79,216,.28); }
    .text-button { border:1px solid var(--blue); border-radius:0; background:#101d2e; color:var(--blue); box-shadow:3px 3px 0 #123f4e; font-family:Consolas,monospace; text-transform:uppercase; }
    .text-button:hover { background:#193047; }
    button:focus-visible { outline:2px solid var(--pink); outline-offset:3px; }
    .help { max-width:860px; color:#a4bac9; border-left:3px solid var(--blue); background:#101A28; padding:13px 16px; }
    .overlay { background:rgba(3,8,15,.84); }
    .success { border:2px solid var(--green); border-radius:0; background:#101A28; box-shadow:5px 5px 0 #183c30; }
    :root { --amber:#FF4FD8; }
    .angle-button[aria-pressed="true"] { border-color:#FF4FD8; background:linear-gradient(145deg,#fff,#ffe0fa); box-shadow:0 4px 0 #7d3276,0 0 12px rgba(255,79,216,.28); }
    button:focus-visible { outline-color:#FF4FD8; }
    .media,.media.object { box-shadow:4px 4px 0 #08111d,0 0 12px rgba(85,214,232,.04); }
    .controller { box-shadow:5px 5px 0 #06101c,0 0 18px rgba(85,214,232,.05),inset 0 1px 0 #45617d; }
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar"><div class="brand" data-i18n="brand">转转看</div><div class="top-actions"><button id="languageToggle" class="text-button" type="button" aria-label="Switch language" hidden>EN</button><div id="steps" class="steps">0 / 48 步</div></div></header>
    <main>
      <section class="intro"><h1 data-i18n="headline">交互式投影对齐</h1><p data-i18n="subhead">观察目标轮廓，逐步旋转三维物体，使当前投影与目标对齐。</p></section>
      <section class="views" aria-label="投影对齐画面" data-i18n-aria="views">
        <figure><figcaption data-i18n="reference">参考图</figcaption><div class="media"><img id="targetImage" alt="参考图" data-i18n-aria="reference"></div></figure>
        <figure><figcaption data-i18n="object">现在的样子</figcaption><div id="objectSurface" class="media object"><img id="objectImage" alt="现在的物体" data-i18n-aria="objectAlt"></div></figure>
        <figure><figcaption data-i18n="projection">转出来的影子</figcaption><div class="media"><img id="projectionImage" alt="转出来的影子" data-i18n-aria="projection"></div></figure>
      </section>
      <section class="controls" aria-label="旋转控制" data-i18n-aria="controls">
        <div class="controller" role="group" aria-label="选择怎么转" data-i18n-aria="directionGroup"><span class="controller-grip left" aria-hidden="true"></span><span class="controller-grip right" aria-hidden="true"></span><div class="controller-body"><button class="shoulder-button shoulder-left action" type="button" data-action="ROLL_NEG" data-i18n-title="rollNeg" title="逆时针" aria-label="逆时针">↺</button><button class="shoulder-button shoulder-right action" type="button" data-action="ROLL_POS" data-i18n-title="rollPos" title="顺时针" aria-label="顺时针">↻</button><div class="dpad"><button class="controller-button dpad-up action" type="button" data-action="PITCH_POS" data-i18n-title="pitchPos" title="往上翻" aria-label="往上翻">↑</button><button class="controller-button dpad-left action" type="button" data-action="YAW_NEG" data-i18n-title="yawNeg" title="往左转" aria-label="往左转">←</button><span class="dpad-center" aria-hidden="true"></span><button class="controller-button dpad-right action" type="button" data-action="YAW_POS" data-i18n-title="yawPos" title="往右转" aria-label="往右转">→</button><button class="controller-button dpad-down action" type="button" data-action="PITCH_NEG" data-i18n-title="pitchNeg" title="往下翻" aria-label="往下翻">↓</button></div><div id="angles" class="arcade-buttons" role="group" aria-label="选择旋转角度" data-i18n-aria="angleGroup"><button class="controller-button arcade-button angle-button angle-x" type="button" data-angle="30" aria-pressed="true">30</button><button class="controller-button arcade-button angle-button angle-y" type="button" data-angle="10" aria-pressed="false">10</button><button class="controller-button arcade-button angle-button angle-b" type="button" data-angle="5" aria-pressed="false">5</button><button class="controller-button arcade-button angle-button angle-a" type="button" data-angle="1" aria-pressed="false">1</button></div></div></div>
      </section>
      <div class="utility"><div class="utility-actions"><button id="reset" class="text-button" type="button" data-i18n="reset">从头再来</button></div><div id="status" class="status" role="status" aria-live="polite"></div></div>
      <section class="help"><p id="helpCopy" data-i18n="helpCopy">左边十字键控制上下和左右；上面两个肩键控制逆时针和顺时针；右边四个圆键选每次转动的角度。对上后会自动结束；还能继续转，就说明还没对上。</p></section>
    </main>
  </div>
  <div id="overlay" class="overlay" aria-modal="true" role="dialog" aria-labelledby="successTitle">
    <div class="success"><div class="check">✓</div><h2 id="successTitle" data-i18n="successTitle">对上了！</h2><p data-i18n="successCopy">两个影子已经一样了。</p><button id="playAgain" class="text-button" type="button" data-i18n="again">再来一次</button></div>
  </div>
  <script>
    const imageIds = { target_image:'targetImage', object_image:'objectImage', projection_image:'projectionImage' };
    const copy = { zh: { brand:'RoTA-Bench · 投影对齐', headline:'交互式投影对齐', subhead:'观察目标轮廓，逐步旋转三维物体，使当前投影与目标对齐。', views:'三视图观察', controls:'旋转操作', angleGroup:'选择旋转角度', directionGroup:'选择旋转方向', reference:'目标轮廓', object:'当前三维物体', objectAlt:'当前三维物体', projection:'当前投影', pitchNeg:'向下翻转', pitchPos:'向上翻转', yawNeg:'向左转动', yawPos:'向右转动', rollNeg:'逆时针旋转', rollPos:'顺时针旋转', reset:'重新开始', helpCopy:'使用方向键选择旋转方向，肩键控制顺时针或逆时针，角度键选择每次旋转幅度。达到成功阈值后任务会自动结束；未结束时可以继续调整。', successTitle:'投影已对齐', successCopy:'当前投影达到任务成功阈值。', again:'再试一次', step:'步', updating:'正在旋转…', restarting:'正在重新开始…', finished:'本轮已结束。点击“重新开始”进行下一轮。' }, en: { brand:'RoTA-Bench · Projection Alignment', headline:'Interactive Projection Alignment', subhead:'Observe the target silhouette and rotate the 3D object until its current projection aligns with the target.', views:'Three-view observation', controls:'Rotation controls', angleGroup:'Select rotation angle', directionGroup:'Select rotation direction', reference:'Target silhouette', object:'Current 3D object', objectAlt:'Current 3D object', projection:'Current projection', pitchNeg:'Rotate downward', pitchPos:'Rotate upward', yawNeg:'Rotate left', yawPos:'Rotate right', rollNeg:'Rotate counter-clockwise', rollPos:'Rotate clockwise', reset:'Restart task', helpCopy:'Use the direction pad to choose an axis direction, the shoulder keys for clockwise or counter-clockwise rotation, and the angle keys to set the step size. The task ends automatically at the success threshold; before then, you can keep adjusting.', successTitle:'Projection aligned', successCopy:'The current projection has reached the task success threshold.', again:'Try again', step:'steps', updating:'Rotating…', restarting:'Restarting…', finished:'This round has ended. Select “Restart task” to begin again.' } };
    copy.zh.brand = '姿态校准';
    copy.zh.headline = '校准空间姿态。';
    copy.zh.subhead = '以参考投影为基准，精确调整当前模型的三维朝向。';
    copy.zh.views = '投影对照';
    copy.zh.controls = '姿态控制';
    copy.zh.helpCopy = '十字键用于俯仰与偏航；顶部肩键用于滚转；右侧数字键设定每次调整的角度。投影达到匹配阈值后会自动完成；仍可继续操作，表示姿态尚未收敛。';
    copy.en.brand = 'Pose Calibration';
    copy.en.headline = 'Calibrate spatial orientation.';
    copy.en.subhead = 'Use the reference projection to precisely align the model in three dimensions.';
    copy.en.views = 'Projection comparison';
    copy.en.controls = 'Pose controls';
    copy.en.helpCopy = 'Use the directional pad for pitch and yaw, the top shoulder buttons for roll, and the numbered buttons to set each adjustment angle. The task completes automatically when the projection reaches its match threshold.';
    copy.zh.object = '当前三维图';
    copy.zh.objectAlt = '当前三维物体';
    copy.zh.projection = '当前投影图';
    copy.zh.subhead = '旋转中间的三维物体，让右侧投影与左侧目标重合。';
    copy.zh.helpCopy = '方向键控制俯仰与偏航，顶部按键控制顺时针和逆时针旋转，数字键设置调整角度。左侧是目标投影，中间是当前三维姿态，右侧是当前投影；达到匹配阈值后自动完成。';
    copy.en.object = 'Current 3D view';
    copy.en.objectAlt = 'Current 3D object';
    copy.en.projection = 'Current projection';
    copy.en.subhead = 'Rotate the 3D object in the middle until its projection matches the target on the left.';
    copy.en.helpCopy = 'Use the directional pad for pitch and yaw, the top buttons for clockwise and counter-clockwise rotation, and the numbered buttons to set the adjustment angle. The left panel is the target projection, the middle panel is the current 3D pose, and the right panel is the current projection.';
    const actionButtons = [...document.querySelectorAll('.action')];
    const angleButtons = [...document.querySelectorAll('[data-angle]')];
    let angle = 30, busy = false, state = null, touchStart = null, language = 'zh';
    const status = document.getElementById('status');
    function t(key) { return copy[language][key]; }
    function updateSteps() { document.getElementById('steps').textContent = state ? `${state.used_steps} / ${state.max_steps} ${t('step')}` : `0 / 48 ${t('step')}`; }
    function localize() { document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en'; document.querySelectorAll('[data-i18n]').forEach(el => { el.textContent = t(el.dataset.i18n); }); document.querySelectorAll('[data-i18n-aria]').forEach(el => { el.setAttribute('aria-label', t(el.dataset.i18nAria)); }); document.querySelectorAll('[data-i18n-title]').forEach(el => { const text = t(el.dataset.i18nTitle); el.title = text; el.setAttribute('aria-label', text); }); document.getElementById('languageToggle').textContent = language === 'zh' ? 'EN' : '中文'; updateSteps(); }
    window.addEventListener('message', event => { if (event.origin !== location.origin || event.data?.type !== 'rota-language' || !['zh','en'].includes(event.data.language)) return; language = event.data.language; localize(); });
    if (window.parent === window) document.getElementById('languageToggle').hidden = false;
    function setStatus(message='', error=false) { status.textContent = message; status.classList.toggle('error', error); }
    function setBusy(value) { busy = value; actionButtons.forEach(button => button.disabled = value || Boolean(state?.terminated)); document.getElementById('reset').disabled = value; angleButtons.forEach(button => button.disabled = value); }
    async function render(next) {
      state = next;
      Object.entries(imageIds).forEach(([key, id]) => { document.getElementById(id).src = next[key]; });
      await Promise.all(Object.values(imageIds).map(id => {
        const image = document.getElementById(id);
        return image.decode ? image.decode().catch(() => {}) : new Promise(resolve => { image.onload = image.onerror = resolve; });
      }));
      updateSteps();
      document.getElementById('overlay').classList.toggle('show', Boolean(next.success));
      setBusy(false);
      if (next.terminated && !next.success) setStatus(t('finished')); else setStatus('');
    }
    async function request(path, options={}) {
      const response = await fetch(path, { credentials:'same-origin', headers:{ 'Content-Type':'application/json', ...(options.headers || {}) }, ...options });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || '请求未完成。');
      return body;
    }
    async function refresh() { setBusy(true); try { await render(await request('/api/state', { headers:{} })); return true; } catch (error) { setBusy(false); setStatus(error.message, true); return false; } }
    async function rotate(action) { if (busy || state?.terminated) return; setBusy(true); setStatus(t('updating')); try { await render(await request('/api/action', { method:'POST', body:JSON.stringify({action, angle}) })); } catch (error) { setBusy(false); setStatus(error.message, true); } }
    async function reset() { if (busy) return; setBusy(true); setStatus(t('restarting')); try { await render(await request('/api/reset', { method:'POST', body:'{}' })); } catch (error) { setBusy(false); setStatus(error.message, true); } }
    angleButtons.forEach(button => button.addEventListener('click', () => { angle = Number(button.dataset.angle); angleButtons.forEach(item => item.setAttribute('aria-pressed', String(item === button))); }));
    actionButtons.forEach(button => button.addEventListener('click', () => rotate(button.dataset.action)));
    document.getElementById('reset').addEventListener('click', reset); document.getElementById('playAgain').addEventListener('click', reset); document.getElementById('languageToggle').addEventListener('click', () => { language = language === 'zh' ? 'en' : 'zh'; localize(); if (window.parent !== window) window.parent.postMessage({ type: 'rota-language-request', language }, location.origin); });
    document.addEventListener('keydown', event => { if (event.metaKey || event.ctrlKey || event.altKey || busy) return; const map = {w:'PITCH_POS',s:'PITCH_NEG',a:'YAW_NEG',d:'YAW_POS',q:'ROLL_NEG',e:'ROLL_POS'}; const action = map[event.key.toLowerCase()]; if (action) { event.preventDefault(); rotate(action); } });
    const objectSurface = document.getElementById('objectSurface');
    objectSurface.addEventListener('pointerdown', event => { touchStart = {x:event.clientX, y:event.clientY}; objectSurface.setPointerCapture?.(event.pointerId); });
    objectSurface.addEventListener('pointerup', event => { if (!touchStart) return; const dx = event.clientX-touchStart.x, dy = event.clientY-touchStart.y; touchStart = null; if (Math.max(Math.abs(dx),Math.abs(dy)) < 18) return; rotate(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 'YAW_POS' : 'YAW_NEG') : (dy > 0 ? 'PITCH_POS' : 'PITCH_NEG')); });
    localize(); refresh().then(ok => { if (window.parent !== window) window.parent.postMessage({type:'rota-demo-ready', ok}, location.origin); });
  </script>
</body>
</html>"""


def configure_logger() -> logging.Logger:
    logger = logging.getLogger("single_task_public_demo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    log_path = ROOT / "logs" / "single_task_public_demo.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rotating = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    rotating.setFormatter(formatter)
    logger.addHandler(rotating)
    return logger


def make_handler(service: DemoService, *, secure_cookie: bool):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ProjectionDemo"

        def log_message(self, _format: str, *args) -> None:
            return

        def _session(self) -> Tuple[str, VisitorState, bool]:
            cookie = SimpleCookie()
            cookie.load(self.headers.get("Cookie", ""))
            value = cookie.get("projection_demo_sid")
            query = parse_qs(urlparse(self.path).query)
            requested = query.get("task", [None])[0]
            return service.get_session(value.value if value else None, requested)

        def _set_session_cookie(self, sid: str) -> None:
            parts = [f"projection_demo_sid={sid}", "Path=/", "HttpOnly", "SameSite=Lax", f"Max-Age={SESSION_TTL_SECONDS}"]
            if secure_cookie:
                parts.append("Secure")
            self.send_header("Set-Cookie", "; ".join(parts))

        def _headers(self, content_type: str, length: int, new_sid: Optional[str] = None) -> None:
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "same-origin")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
            if new_sid:
                self._set_session_cookie(new_sid)

        def _json(self, payload: Dict, status: HTTPStatus = HTTPStatus.OK, new_sid: Optional[str] = None) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self._headers("application/json; charset=utf-8", len(body), new_sid)
            self.end_headers()
            self.wfile.write(body)

        def _html(self, new_sid: Optional[str]) -> None:
            body = HTML_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self._headers("text/html; charset=utf-8", len(body), new_sid)
            self.end_headers()
            self.wfile.write(body)

        def _controller_asset(self) -> None:
            body = CONTROLLER_ASSET_PATH.read_bytes()
            self.send_response(HTTPStatus.OK)
            self._headers("image/png", len(body))
            self.end_headers()
            self.wfile.write(body)

        def _check_origin(self) -> None:
            origin = self.headers.get("Origin")
            if not origin:
                return
            parsed = urlparse(origin)
            if parsed.scheme not in {"http", "https"} or parsed.netloc != self.headers.get("Host", ""):
                raise ClientError("请求来源无效。")

        def _read_json(self) -> Dict:
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                raise ClientError("请求内容类型必须为 application/json。")
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError as exc:
                raise ClientError("请求长度无效。") from exc
            if length < 0 or length > MAX_JSON_BYTES:
                raise ClientError("请求体过大。")
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ClientError("请求 JSON 无效。") from exc
            if not isinstance(payload, dict):
                raise ClientError("请求 JSON 必须为对象。")
            return payload

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/healthz":
                self._json({"ok": True})
                return
            if path == "/readyz":
                if service.ready:
                    self._json({"ready": True})
                else:
                    self._json({"ready": False}, HTTPStatus.SERVICE_UNAVAILABLE)
                return
            if path == "/":
                sid, _, fresh = self._session()
                self._html(sid if fresh else None)
                return
            if path == "/assets/xbox-controller.png":
                self._controller_asset()
                return
            if path == "/api/preview":
                query = parse_qs(urlparse(self.path).query)
                task_id = query.get("task", [SAMPLE_ID])[0]
                if task_id not in service.tasks:
                    self._json({"error": "未找到该任务。"}, HTTPStatus.NOT_FOUND)
                    return
                try:
                    target_display, _ = service._target(service.tasks[task_id])
                    body = base64.b64decode(png_data_url(target_display).split(",", 1)[1])
                    self.send_response(HTTPStatus.OK)
                    self._headers("image/png", len(body))
                    self.end_headers()
                    self.wfile.write(body)
                except Exception:
                    service.logger.exception("Task preview render failed: %s", task_id)
                    self._json({"error": "预览暂不可用。"}, HTTPStatus.SERVICE_UNAVAILABLE)
                return
            if path == "/api/state":
                sid, state, fresh = self._session()
                try:
                    self._json(service.state_payload(state), new_sid=sid if fresh else None)
                except Exception:
                    service.logger.exception("State render failed")
                    self._json({"error": "当前画面暂不可用，请稍后重试。"}, HTTPStatus.SERVICE_UNAVAILABLE, sid if fresh else None)
                return
            self._json({"error": "未找到该地址。"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            sid, state, fresh = self._session()
            try:
                self._check_origin()
                payload = self._read_json()
                if path == "/api/action":
                    output = service.apply_action(state, payload.get("action"), payload.get("angle"))
                elif path == "/api/reset":
                    output = service.reset(state)
                else:
                    self._json({"error": "未找到该地址。"}, HTTPStatus.NOT_FOUND, sid if fresh else None)
                    return
                self._json(output, new_sid=sid if fresh else None)
            except TerminalStateError as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT, sid if fresh else None)
            except RateLimitError as exc:
                self._json({"error": str(exc)}, HTTPStatus.TOO_MANY_REQUESTS, sid if fresh else None)
            except ClientError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST, sid if fresh else None)
            except Exception:
                service.logger.exception("Action request failed: path=%s", path)
                self._json({"error": "操作暂不可用，请稍后重试。"}, HTTPStatus.SERVICE_UNAVAILABLE, sid if fresh else None)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the public single-task projection demo.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; keep localhost behind a reverse proxy in production.")
    parser.add_argument("--port", type=int, default=8787, help="Bind port.")
    parser.add_argument("--secure-cookie", action="store_true", help="Mark the session cookie Secure (required behind HTTPS).")
    parser.add_argument("--require-gpu", action="store_true", help="Fail startup unless the renderer uses CUDA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("Port must be in the range 1-65535.")
    logger = configure_logger()
    split_path = ROOT / "tasks" / "test.json"
    split = json.loads(split_path.read_text(encoding="utf-8")) if split_path.exists() else {"sample_ids": [SAMPLE_ID]}
    tasks = load_tasks(TASKS_PATH, split.get("sample_ids", [SAMPLE_ID]))
    service = DemoService(tasks, require_gpu=args.require_gpu, logger=logger)
    service.startup_check()
    handler = make_handler(service, secure_cookie=bool(args.secure_cookie))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    logger.info("Serving %d test tasks on http://%s:%d (secure_cookie=%s)", len(tasks), args.host, args.port, bool(args.secure_cookie))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
