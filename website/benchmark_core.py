#!/usr/bin/env python3
"""Shared utilities for the 6-step interactive shadow benchmark pipeline."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union
from render_protocol import RENDER_PARAMETER_SCHEMA, STANDARD_IMAGE_SIZE
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

THETA = 15
ACTIONS = [
    "PITCH_POS",
    "PITCH_NEG",
    "YAW_POS",
    "YAW_NEG",
    "ROLL_POS",
    "ROLL_NEG",
    "STOP",
]

_LAST_RENDER_BACKEND = "uninitialized"


def get_last_render_backend() -> str:
    return _LAST_RENDER_BACKEND


def _set_last_render_backend(value: str) -> None:
    global _LAST_RENDER_BACKEND
    _LAST_RENDER_BACKEND = value


@dataclass
class Pose:
    roll: float
    pitch: float
    yaw: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "roll": round(float(self.roll), 6),
            "pitch": round(float(self.pitch), 6),
            "yaw": round(float(self.yaw), 6),
        }

    @staticmethod
    def from_dict(data: Dict[str, float]) -> "Pose":
        return Pose(float(data["roll"]), float(data["pitch"]), float(data["yaw"]))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _resolve_size(size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(size, int):
        return size, size
    return int(size[0]), int(size[1])


def _generate_base_points(sample: Dict) -> List[Tuple[float, float]]:
    if sample.get("source_dataset") == "game_icons" and sample.get("icon_points"):
        icon_scale = float(sample.get("icon_scale", 1.0))
        pts = [(float(p[0]) * icon_scale, float(p[1]) * icon_scale) for p in sample["icon_points"]]
        if not pts:
            return pts
        center_mode = str(sample.get("icon_center_mode", "bbox")).lower()
        if center_mode == "centroid":
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            mec_radius = max((math.hypot(p[0] - cx, p[1] - cy) for p in pts), default=1e-6)
        elif center_mode == "mec" and cv2 is not None and np is not None:
            arr = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
            (cx, cy), mec_radius = cv2.minEnclosingCircle(arr)
            cx = float(cx)
            cy = float(cy)
            mec_radius = float(max(1e-6, mec_radius))
        else:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx = (min(xs) + max(xs)) * 0.5
            cy = (min(ys) + max(ys)) * 0.5
            mec_radius = max((math.hypot(p[0] - cx, p[1] - cy) for p in pts), default=1e-6)
        centered = [(p[0] - cx, p[1] - cy) for p in pts]
        if sample.get("surface_mode") == "spherical_icon" and bool(sample.get("sphere_fit_icon", True)):
            sphere_radius = float(sample.get("sphere_radius", 28.0))
            sphere_xy_ratio = float(sample.get("sphere_xy_ratio", 0.74))
            target_r = max(1e-6, sphere_radius * sphere_xy_ratio)
            if mec_radius > 1e-6:
                fit = min(1.0, target_r / mec_radius)
                centered = [(p[0] * fit, p[1] * fit) for p in centered]
        return centered

    random.seed(sample["shape_seed"])
    base = sample["base_shape"]
    complexity = sample.get("complexity", 1.0)
    points: List[Tuple[float, float]] = []
    if base == "rect":
        for x in range(-18, 19):
            for y in range(-10, 11):
                points.append((x, y))
    elif base == "tri":
        for y in range(-18, 19):
            width = int((18 - abs(y)) * 1.1)
            for x in range(-width, width + 1):
                points.append((x, y))
    elif base == "lshape":
        for x in range(-18, -6):
            for y in range(-18, 19):
                points.append((x, y))
        for x in range(-18, 19):
            for y in range(7, 19):
                points.append((x, y))
    else:  # blob
        for _ in range(1600):
            x = random.gauss(0, 11 * complexity)
            y = random.gauss(0, 8 * complexity)
            points.append((x, y))
    return points


_SAMPLE_ARRAY_CACHE: Dict[str, Tuple] = {}
_SPHERICAL_BASE_CACHE_KEY = None
_SPHERICAL_BASE_CACHE_VALUE = None
_SPHERICAL_CUDA_BASE_CACHE_KEY = None
_SPHERICAL_CUDA_BASE_CACHE_VALUE = None
_SPHERICAL_CUDA_CACHE_KEY = None
_SPHERICAL_CUDA_CACHE_VALUE = None
_TORCH_CUDA = None
_TORCH_CUDA_CHECKED = False
_CUDA_OFFSET_CACHE = {}


def _sample_cache_key(sample: Dict) -> str:
    return "|".join(
        [
            str(sample.get("sample_id", "")),
            str(sample.get("icon_slug", "")),
            str(sample.get("shape_seed", "")),
            str(sample.get("surface_mode", "")),
            str(sample.get("spike_freq", "")),
            str(sample.get("spike_strength", "")),
            str(sample.get("roughness", "")),
            str(sample.get("cavity_strength", "")),
            str(sample.get("sphere_radius", "")),
            str(sample.get("radial_jitter", "")),
            str(sample.get("icon_scale", "")),
            str(sample.get("icon_center_mode", "")),
            str(sample.get("sphere_bi_extent", "")),
            str(sample.get("sphere_double_sided", "")),
            str(sample.get("sphere_solid", "")),
            str(sample.get("sphere_fill_layers", "")),
            str(sample.get("sphere_xy_ratio", "")),
            str(sample.get("sphere_fit_icon", "")),
            json.dumps(sample.get("surface_holes", []), ensure_ascii=False, sort_keys=True),
        ]
    )


def _spherical_base_cache_key(sample: Dict) -> str:
    """Cache geometry fields that do not change while hole candidates are tested."""
    return "|".join(
        [
            str(sample.get("sample_id", "")),
            str(sample.get("icon_slug", "")),
            str(sample.get("shape_seed", "")),
            str(sample.get("complexity", "")),
            str(sample.get("sphere_radius", "")),
            str(sample.get("sphere_depth_scale", "")),
            str(sample.get("sphere_fill_layers", "")),
            str(sample.get("icon_scale", "")),
            str(sample.get("icon_center_mode", "")),
            str(sample.get("sphere_xy_ratio", "")),
            str(sample.get("sphere_fit_icon", "")),
            str(len(sample.get("icon_points", []))),
        ]
    )


def _coord_noise_np(x, y, seed: float):
    v = np.sin((x * 12.9898 + y * 78.233 + seed * 0.137) * 0.91543) * 43758.5453
    frac = v - np.floor(v)
    return frac * 2.0 - 1.0


def _hole_basis(nx: float, ny: float, nz: float):
    n_norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if n_norm < 1e-8:
        return None
    nx, ny, nz = nx / n_norm, ny / n_norm, nz / n_norm
    if abs(nz) < 0.9:
        rx, ry, rz = 0.0, 0.0, 1.0
    else:
        rx, ry, rz = 0.0, 1.0, 0.0
    ux = ry * nz - rz * ny
    uy = rz * nx - rx * nz
    uz = rx * ny - ry * nx
    un = math.sqrt(ux * ux + uy * uy + uz * uz)
    if un < 1e-8:
        return None
    ux, uy, uz = ux / un, uy / un, uz / un
    vx = ny * uz - nz * uy
    vy = nz * ux - nx * uz
    vz = nx * uy - ny * ux
    return (nx, ny, nz, ux, uy, uz, vx, vy, vz)


def _filter_surface_holes_np(px, py, pz, holes, rmax_override=None):
    if not holes:
        return px, py, pz
    rr = np.sqrt(px * px + py * py + pz * pz)
    if rmax_override is None:
        rmax = float(np.max(rr)) if rr.size else 1.0
    else:
        rmax = float(rmax_override)
    keep = np.ones(px.shape[0], dtype=np.bool_)
    eps = 1e-8
    qx = px / (rr + eps)
    qy = py / (rr + eps)
    qz = pz / (rr + eps)
    for h in holes:
        basis = _hole_basis(float(h.get("nx", 0.0)), float(h.get("ny", 0.0)), float(h.get("nz", 1.0)))
        if basis is None:
            continue
        nx, ny, nz, ux, uy, uz, vx, vy, vz = basis
        half_w = float(h.get("half_w", 0.12))
        half_h = float(h.get("half_h", 0.12))
        radius = float(h.get("radius", max(half_w, half_h)))
        shear = float(h.get("shear", 0.0))
        min_n = float(h.get("min_n", 0.96))
        shell_ratio = float(h.get("shell_ratio", 0.78))
        shape = str(h.get("shape", "rect")).lower()
        a = qx * ux + qy * uy + qz * uz
        b = qx * vx + qy * vy + qz * vz
        c = qx * nx + qy * ny + qz * nz
        if shape in ("circle", "disk", "sphere"):
            footprint = (a * a + b * b) <= (radius * radius)
        elif shape in ("parallelogram", "para"):
            b_skew = b - a * shear
            footprint = (np.abs(a) <= half_w) & (np.abs(b_skew) <= half_h)
        else:
            footprint = (np.abs(a) <= half_w) & (np.abs(b) <= half_h)
        remove = (rr >= shell_ratio * rmax) & (c >= min_n) & footprint
        keep &= ~remove
    return px[keep], py[keep], pz[keep]


def _point_kept_after_holes(sample: Dict, px: float, py: float, pz: float, rmax: float) -> bool:
    holes = sample.get("surface_holes", [])
    if not holes:
        return True
    rr = math.sqrt(px * px + py * py + pz * pz)
    if rr < 1e-8:
        return True
    qx, qy, qz = px / rr, py / rr, pz / rr
    for h in holes:
        basis = _hole_basis(float(h.get("nx", 0.0)), float(h.get("ny", 0.0)), float(h.get("nz", 1.0)))
        if basis is None:
            continue
        nx, ny, nz, ux, uy, uz, vx, vy, vz = basis
        half_w = float(h.get("half_w", 0.12))
        half_h = float(h.get("half_h", 0.12))
        radius = float(h.get("radius", max(half_w, half_h)))
        shear = float(h.get("shear", 0.0))
        min_n = float(h.get("min_n", 0.96))
        shell_ratio = float(h.get("shell_ratio", 0.78))
        shape = str(h.get("shape", "rect")).lower()
        a = qx * ux + qy * uy + qz * uz
        b = qx * vx + qy * vy + qz * vz
        c = qx * nx + qy * ny + qz * nz
        if shape in ("circle", "disk", "sphere"):
            in_footprint = (a * a + b * b) <= (radius * radius)
        elif shape in ("parallelogram", "para"):
            in_footprint = abs(a) <= half_w and abs(b - a * shear) <= half_h
        else:
            in_footprint = abs(a) <= half_w and abs(b) <= half_h
        if rr >= shell_ratio * rmax and c >= min_n and in_footprint:
            return False
    return True


def _prepare_sample_arrays(sample: Dict):
    if np is None:
        return None
    key = _sample_cache_key(sample)
    if key in _SAMPLE_ARRAY_CACHE:
        return _SAMPLE_ARRAY_CACHE[key]

    pts = _generate_base_points(sample)
    if not pts:
        return None
    arr = np.asarray(pts, dtype=np.float32)
    px = arr[:, 0]
    py = arr[:, 1]

    mode = sample.get("surface_mode", "solid_smooth")
    c = float(sample.get("complexity", 1.0))
    seed = float(sample.get("shape_seed", 0))
    r = np.sqrt(px * px + py * py) / 24.0
    a = np.arctan2(py, px)
    base_dome = np.clip(1.0 - r * r, 0.0, None)

    if mode == "spiky_irregular":
        spike_freq = float(sample.get("spike_freq", 16.0))
        spike_strength = float(sample.get("spike_strength", 6.5))
        roughness = float(sample.get("roughness", 2.8))
        cavity_strength = float(sample.get("cavity_strength", 2.5))
        sphere_radius = float(sample.get("sphere_radius", 26.0))
        radial_jitter = float(sample.get("radial_jitter", 1.8))

        radial_spikes = np.abs(np.sin(a * spike_freq + _coord_noise_np(px, py, seed) * 1.7))
        coarse = _coord_noise_np(px * 0.32, py * 0.32, seed)
        fine = _coord_noise_np(px * 1.45, py * 1.45, seed + 17.0)
        cavity = np.clip(-np.sin((px + py) * 0.24 + fine * 1.8), 0.0, None)
        sphere_core = np.sqrt(np.clip(sphere_radius * sphere_radius - (px * px + py * py), 0.0, None)) * 0.22
        radial_field = _coord_noise_np(px * 0.46, py * 0.46, seed + 23.0) * radial_jitter
        pz = (
            sphere_core
            + base_dome * 1.6
            + radial_spikes * spike_strength * (1.0 - r * 0.45)
            + coarse * roughness
            + fine * 1.2
            + radial_field
            - cavity * cavity_strength
        ) * c
    elif mode == "spherical_icon":
        sphere_radius = float(sample.get("sphere_radius", 28.0))
        sphere_scale = float(sample.get("sphere_depth_scale", 0.26))
        sphere_core = np.sqrt(np.clip(sphere_radius * sphere_radius - (px * px + py * py), 0.0, None)) * sphere_scale * c
        double_sided = bool(sample.get("sphere_double_sided", True))
        solid = bool(sample.get("sphere_solid", True))
        fill_layers = int(sample.get("sphere_fill_layers", 31))
        if double_sided and solid:
            layers = max(5, fill_layers)
            if layers % 2 == 0:
                layers += 1
            idx = np.arange(px.shape[0], dtype=np.float32)[None, :]
            lid = np.arange(layers, dtype=np.float32)[:, None]
            seed = float(sample.get("shape_seed", 0)) * 0.0137
            rnd = np.sin((idx * 12.9898 + lid * 78.233 + seed) * 1.12345) * 43758.5453
            frac = rnd - np.floor(rnd)
            t = frac * 2.0 - 1.0
            # Pin two layers at boundaries to keep closed outer surface.
            t[0, :] = -1.0
            t[1, :] = 1.0
            pz = (t * sphere_core[None, :]).reshape(-1).astype(np.float32)
            px = np.tile(px, layers).astype(np.float32)
            py = np.tile(py, layers).astype(np.float32)
        elif double_sided:
            px = np.concatenate([px, px], axis=0)
            py = np.concatenate([py, py], axis=0)
            pz = np.concatenate([sphere_core, -sphere_core], axis=0)
        else:
            pz = sphere_core
    else:
        smooth_ripple = np.sin(px * 0.18 + py * 0.15 + seed * 0.01) * 0.9
        smooth_noise = _coord_noise_np(px * 0.18, py * 0.18, seed) * 0.5
        pz = (base_dome * 2.2 + smooth_ripple + smooth_noise) * c

    px, py, pz = _filter_surface_holes_np(px.astype(np.float32), py.astype(np.float32), pz.astype(np.float32), sample.get("surface_holes", []))

    out = (px.astype(np.float32), py.astype(np.float32), pz.astype(np.float32))
    _SAMPLE_ARRAY_CACHE[key] = out
    return out


def _coord_noise(x: float, y: float, seed: int) -> float:
    # Deterministic pseudo-noise in [-1, 1] without external deps.
    v = math.sin((x * 12.9898 + y * 78.233 + seed * 0.137) * 0.91543) * 43758.5453
    frac = v - math.floor(v)
    return frac * 2.0 - 1.0


def _depth_value(sample: Dict, px: float, py: float) -> float:
    mode = sample.get("surface_mode", "solid_smooth")
    c = float(sample.get("complexity", 1.0))
    seed = int(sample.get("shape_seed", 0))

    r = math.sqrt(px * px + py * py) / 24.0
    a = math.atan2(py, px)
    base_dome = max(0.0, 1.0 - r * r)

    if mode == "spiky_irregular":
        sphere_radius = float(sample.get("sphere_radius", 26.0))
        spike_freq = float(sample.get("spike_freq", 16.0))
        spike_strength = float(sample.get("spike_strength", 6.5))
        roughness = float(sample.get("roughness", 2.8))
        cavity_strength = float(sample.get("cavity_strength", 2.5))
        radial_jitter = float(sample.get("radial_jitter", 1.8))

        radial_spikes = abs(math.sin(a * spike_freq + _coord_noise(px, py, seed) * 1.7))
        coarse = _coord_noise(px * 0.32, py * 0.32, seed)
        fine = _coord_noise(px * 1.45, py * 1.45, seed + 17)
        cavity = max(0.0, -math.sin((px + py) * 0.24 + fine * 1.8))
        sphere_core = math.sqrt(max(0.0, sphere_radius * sphere_radius - (px * px + py * py))) * 0.22
        radial_field = _coord_noise(px * 0.46, py * 0.46, seed + 23.0) * radial_jitter

        z = (
            sphere_core
            + base_dome * 1.6
            + radial_spikes * spike_strength * (1.0 - r * 0.45)
            + coarse * roughness
            + fine * 1.2
            + radial_field
            - cavity * cavity_strength
        )
        return z * c

    if mode == "spherical_icon":
        sphere_radius = float(sample.get("sphere_radius", 28.0))
        sphere_scale = float(sample.get("sphere_depth_scale", 0.26))
        z = math.sqrt(max(0.0, sphere_radius * sphere_radius - (px * px + py * py))) * sphere_scale
        return z * c

    # solid_smooth: thicker than paper but not spiky.
    smooth_ripple = math.sin(px * 0.18 + py * 0.15 + seed * 0.01) * 0.9
    smooth_noise = _coord_noise(px * 0.18, py * 0.18, seed) * 0.5
    z = base_dome * 2.2 + smooth_ripple + smooth_noise
    return z * c


def _depth_values(sample: Dict, px: float, py: float) -> List[float]:
    mode = sample.get("surface_mode", "solid_smooth")
    z = _depth_value(sample, px, py)
    if mode != "spherical_icon":
        return [z]
    if not bool(sample.get("sphere_double_sided", True)):
        return [z]
    if bool(sample.get("sphere_solid", True)):
        layers = max(5, int(sample.get("sphere_fill_layers", 31)))
        if layers % 2 == 0:
            layers += 1
        out: List[float] = []
        seed = float(sample.get("shape_seed", 0))
        phase = (_coord_noise(px * 0.137, py * 0.173, int(seed) + 41) + 1.0) * 0.5
        for i in range(layers):
            t = ((phase + i * 0.61803398875) % 1.0) * 2.0 - 1.0
            out.append(z * t)
        out[0] = -abs(z)
        out[1] = abs(z)
        return out
    return [z, -z]


def _rotate_xyz(px: float, py: float, pz: float, pose: Pose) -> Tuple[float, float, float]:
    roll = math.radians(pose.roll)
    pitch = math.radians(pose.pitch)
    yaw = math.radians(pose.yaw)

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cyw, syw = math.cos(yaw), math.sin(yaw)

    x1, y1, z1 = px, py * cr - pz * sr, py * sr + pz * cr
    x2, y2, z2 = x1 * cp + z1 * sp, y1, -x1 * sp + z1 * cp
    x3, y3, z3 = x2 * cyw - y2 * syw, x2 * syw + y2 * cyw, z2
    return x3, y3, z3


def _binary_close(mask: List[List[int]], iterations: int = 1) -> List[List[int]]:
    if not mask:
        return mask
    h = len(mask)
    w = len(mask[0])
    out = [row[:] for row in mask]
    # Dilate
    for _ in range(iterations):
        nxt = [row[:] for row in out]
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if out[y][x]:
                    continue
                if (
                    out[y - 1][x]
                    or out[y + 1][x]
                    or out[y][x - 1]
                    or out[y][x + 1]
                    or out[y - 1][x - 1]
                    or out[y - 1][x + 1]
                    or out[y + 1][x - 1]
                    or out[y + 1][x + 1]
                ):
                    nxt[y][x] = 1
        out = nxt
    # Erode
    for _ in range(iterations):
        nxt = [row[:] for row in out]
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if not out[y][x]:
                    continue
                n = (
                    out[y - 1][x]
                    + out[y + 1][x]
                    + out[y][x - 1]
                    + out[y][x + 1]
                    + out[y - 1][x - 1]
                    + out[y - 1][x + 1]
                    + out[y + 1][x - 1]
                    + out[y + 1][x + 1]
                )
                if n <= 3:
                    nxt[y][x] = 0
        out = nxt
    return out


def _blur_gray(img: List[List[int]]) -> List[List[int]]:
    if not img:
        return img
    h = len(img)
    w = len(img[0])
    out = [row[:] for row in img]
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            s = (
                img[y][x] * 4
                + img[y - 1][x] * 2
                + img[y + 1][x] * 2
                + img[y][x - 1] * 2
                + img[y][x + 1] * 2
                + img[y - 1][x - 1]
                + img[y - 1][x + 1]
                + img[y + 1][x - 1]
                + img[y + 1][x + 1]
            )
            out[y][x] = int(s / 16)
    return out


if njit is not None:
    @njit(cache=True)
    def _rasterize_mask_numba(xi, yi, zz, width, height, radius):
        zbuf = np.full((height, width), -1e9, dtype=np.float32)
        r2 = radius * radius
        n = xi.shape[0]
        for i in range(n):
            x = xi[i]
            y = yi[i]
            z = zz[i]
            if radius <= 1:
                if 0 <= x < width and 0 <= y < height:
                    if z > zbuf[y, x]:
                        zbuf[y, x] = z
            else:
                for oy in range(-radius, radius + 1):
                    yy = y + oy
                    if yy < 0 or yy >= height:
                        continue
                    for ox in range(-radius, radius + 1):
                        if ox * ox + oy * oy > r2:
                            continue
                        xx = x + ox
                        if 0 <= xx < width and z > zbuf[yy, xx]:
                            zbuf[yy, xx] = z
        out = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                if zbuf[y, x] > -1e8:
                    out[y, x] = 255
        return out

    @njit(cache=True)
    def _rasterize_oblique_numba(xi, yi, shade, width, height, radius):
        out = np.full((height, width), 18, dtype=np.uint8)
        r2 = radius * radius
        n = xi.shape[0]
        for i in range(n):
            x = xi[i]
            y = yi[i]
            s = shade[i]
            if radius <= 1:
                if 0 <= x < width and 0 <= y < height and s > out[y, x]:
                    out[y, x] = s
            else:
                for oy in range(-radius, radius + 1):
                    yy = y + oy
                    if yy < 0 or yy >= height:
                        continue
                    for ox in range(-radius, radius + 1):
                        d2 = ox * ox + oy * oy
                        if d2 > r2:
                            continue
                        xx = x + ox
                        if 0 <= xx < width:
                            dec = int(math.sqrt(d2) * 2.2)
                            v = s - dec
                            if v < 0:
                                v = 0
                            if v > out[yy, xx]:
                                out[yy, xx] = v
        return out


def create_mask(
    sample: Dict,
    pose: Pose,
    size: Union[int, Tuple[int, int]] = 64,
    fill_holes: bool = True,
) -> List[List[int]]:
    # Synthetic projection renderer with pseudo-3D shape depth (z-buffer rasterization).
    width, height = _resolve_size(size)
    cx = width / 2
    cy = height / 2
    zbuf = [[-1e9 for _ in range(width)] for _ in range(height)]
    mask = [[0 for _ in range(width)] for _ in range(height)]

    points = _generate_base_points(sample)
    rmax = 1.0
    if points:
        # Upper bound for hole shell filter in slow path.
        zs_tmp = [_depth_value(sample, px, py) for px, py in points[: min(len(points), 5000)]]
        rmax = max((math.sqrt(px * px + py * py + z * z) for (px, py), z in zip(points[: min(len(points), 5000)], zs_tmp)), default=1.0)

    scale_fit = min(width, height) / 64.0
    # Use smoother radius for solid icons; avoid blocky stamp look.
    if sample.get("source_dataset") == "game_icons":
        stamp_radius = max(1, int(round(scale_fit * 0.36)))
    else:
        stamp_radius = max(1, int(round(scale_fit * 0.2)))
    for px, py in points:
        z_candidates = _depth_values(sample, px, py)
        for z_local in z_candidates:
            if not _point_kept_after_holes(sample, px, py, z_local, rmax):
                continue
            x3, y3, z3 = _rotate_xyz(px, py, z_local, pose)

            if sample.get("surface_mode") == "spherical_icon":
                jitter_scale = 0.0
            else:
                jitter_scale = 0.35 if sample.get("stream_type") == "single" else 0.25
            jitter = jitter_scale * math.sin(math.radians(pose.pitch - pose.yaw)) * (1.0 + abs(z3) * 0.05)

            # No extra perspective term here: keep target silhouette recognizable.
            xi = int(round(cx + (x3 + jitter) * scale_fit))
            yi = int(round(cy + (y3 - jitter) * scale_fit))
            if 0 <= xi < width and 0 <= yi < height:
                if stamp_radius <= 1:
                    if z3 > zbuf[yi][xi]:
                        zbuf[yi][xi] = z3
                else:
                    r2 = stamp_radius * stamp_radius
                    for oy in range(-stamp_radius, stamp_radius + 1):
                        yy = yi + oy
                        if yy < 0 or yy >= height:
                            continue
                        for ox in range(-stamp_radius, stamp_radius + 1):
                            if ox * ox + oy * oy > r2:
                                continue
                            xx = xi + ox
                            if 0 <= xx < width:
                                if z3 > zbuf[yy][xx]:
                                    zbuf[yy][xx] = z3

    for y in range(height):
        for x in range(width):
            if zbuf[y][x] > -1e8:
                mask[y][x] = 1

    if fill_holes:
        mask = _binary_close(mask, iterations=2 if width * height <= 1200000 else 1)
    return mask


def create_oblique_view(sample: Dict, pose: Pose, size: Union[int, Tuple[int, int]] = 64) -> List[List[int]]:
    # Synthetic oblique rendering with depth shading for model/human observation.
    width, height = _resolve_size(size)
    img = [[18 for _ in range(width)] for _ in range(height)]
    cx = width / 2
    cy = height / 2
    pts2d = _generate_base_points(sample)
    rmax = 1.0
    if pts2d:
        zs_tmp = [_depth_value(sample, px, py) for px, py in pts2d[: min(len(pts2d), 5000)]]
        rmax = max((math.sqrt(px * px + py * py + z * z) for (px, py), z in zip(pts2d[: min(len(pts2d), 5000)], zs_tmp)), default=1.0)

    scale_fit = min(width, height) / 64.0
    stamp_radius = max(1, int(round(scale_fit * (0.33 if sample.get("source_dataset") == "game_icons" else 0.18))))
    for px, py in pts2d:
        z_candidates = _depth_values(sample, px, py)
        for z_local in z_candidates:
            if not _point_kept_after_holes(sample, px, py, z_local, rmax):
                continue
            x3, y3, z3 = _rotate_xyz(px, py, z_local, pose)

            if sample.get("surface_mode") == "spherical_icon":
                proj_x = int(round(cx + x3 * 0.9 * scale_fit))
                proj_y = int(round(cy + y3 * 0.9 * scale_fit))
            else:
                proj_x = int(round(cx + x3 * 0.9 * scale_fit))
                proj_y = int(round(cy + (y3 * 0.65 - z3 * 0.35) * scale_fit))
            if 0 <= proj_x < width and 0 <= proj_y < height:
                shade = int(clamp(110 + (z3 * 6) + (x3 * 1.8), 40, 235))
                if stamp_radius <= 1:
                    if shade > img[proj_y][proj_x]:
                        img[proj_y][proj_x] = shade
                else:
                    r2 = stamp_radius * stamp_radius
                    for oy in range(-stamp_radius, stamp_radius + 1):
                        yy = proj_y + oy
                        if yy < 0 or yy >= height:
                            continue
                        for ox in range(-stamp_radius, stamp_radius + 1):
                            if ox * ox + oy * oy > r2:
                                continue
                            xx = proj_x + ox
                            if 0 <= xx < width:
                                decay = int((ox * ox + oy * oy) ** 0.5 * 2.2)
                                img[yy][xx] = max(img[yy][xx], max(0, shade - decay))
    img = _blur_gray(_blur_gray(img))
    return img


def _prepare_spherical_base(sample: Dict):
    global _SPHERICAL_BASE_CACHE_KEY, _SPHERICAL_BASE_CACHE_VALUE
    cache_key = _spherical_base_cache_key(sample)
    if _SPHERICAL_BASE_CACHE_KEY == cache_key:
        return _SPHERICAL_BASE_CACHE_VALUE

    pts = _generate_base_points(sample)
    if not pts:
        return None
    arr = np.asarray(pts, dtype=np.float32)
    px_base = arr[:, 0]
    py_base = arr[:, 1]
    c = float(sample.get("complexity", 1.0))
    sphere_radius = float(sample.get("sphere_radius", 28.0))
    sphere_scale = float(sample.get("sphere_depth_scale", 0.26))
    sphere_core = np.sqrt(
        np.clip(sphere_radius * sphere_radius - (px_base * px_base + py_base * py_base), 0.0, None)
    )
    sphere_core = sphere_core.astype(np.float32) * np.float32(sphere_scale * c)
    rmax_global = (
        float(np.max(np.sqrt(px_base * px_base + py_base * py_base + sphere_core * sphere_core)))
        if px_base.size
        else 1.0
    )
    _SPHERICAL_BASE_CACHE_KEY = cache_key
    _SPHERICAL_BASE_CACHE_VALUE = (px_base, py_base, sphere_core, rmax_global)
    return _SPHERICAL_BASE_CACHE_VALUE


def _get_torch_cuda():
    global _TORCH_CUDA, _TORCH_CUDA_CHECKED
    if os.environ.get("PROJECTION_RENDER_DEVICE", "auto").strip().lower() == "cpu":
        return None
    if not _TORCH_CUDA_CHECKED:
        _TORCH_CUDA_CHECKED = True
        try:
            import torch

            _TORCH_CUDA = torch if torch.cuda.is_available() else None
        except Exception:
            _TORCH_CUDA = None
    return _TORCH_CUDA


def _filter_surface_holes_torch(geometry, holes, rmax: float):
    torch = _get_torch_cuda()
    if torch is None or not holes:
        return geometry
    px, py, pz = geometry[:, 0], geometry[:, 1], geometry[:, 2]
    rr = torch.sqrt(px * px + py * py + pz * pz)
    qx = px / (rr + 1e-8)
    qy = py / (rr + 1e-8)
    qz = pz / (rr + 1e-8)
    keep = torch.ones(geometry.shape[0], dtype=torch.bool, device=geometry.device)
    for hole in holes:
        basis = _hole_basis(
            float(hole.get("nx", 0.0)),
            float(hole.get("ny", 0.0)),
            float(hole.get("nz", 1.0)),
        )
        if basis is None:
            continue
        nx, ny, nz, ux, uy, uz, vx, vy, vz = basis
        a = qx * ux + qy * uy + qz * uz
        b = qx * vx + qy * vy + qz * vz
        c = qx * nx + qy * ny + qz * nz
        half_w = float(hole.get("half_w", 0.12))
        half_h = float(hole.get("half_h", 0.12))
        radius = float(hole.get("radius", max(half_w, half_h)))
        shape = str(hole.get("shape", "rect")).lower()
        if shape in ("circle", "disk", "sphere"):
            footprint = (a * a + b * b) <= radius * radius
        elif shape in ("parallelogram", "para"):
            footprint = (torch.abs(a) <= half_w) & (
                torch.abs(b - a * float(hole.get("shear", 0.0))) <= half_h
            )
        else:
            footprint = (torch.abs(a) <= half_w) & (torch.abs(b) <= half_h)
        remove = (
            (rr >= float(hole.get("shell_ratio", 0.78)) * rmax)
            & (c >= float(hole.get("min_n", 0.96)))
            & footprint
        )
        keep &= ~remove
    return geometry[keep]


def _prepare_spherical_cuda_geometry(sample: Dict):
    global _SPHERICAL_CUDA_BASE_CACHE_KEY, _SPHERICAL_CUDA_BASE_CACHE_VALUE
    global _SPHERICAL_CUDA_CACHE_KEY, _SPHERICAL_CUDA_CACHE_VALUE
    torch = _get_torch_cuda()
    if torch is None:
        return None
    cache_key = _sample_cache_key(sample)
    if _SPHERICAL_CUDA_CACHE_KEY == cache_key:
        return _SPHERICAL_CUDA_CACHE_VALUE

    prepared = _prepare_spherical_base(sample)
    if prepared is None:
        return None
    px_base, py_base, sphere_core, rmax_global = prepared
    base_key = _spherical_base_cache_key(sample)
    if _SPHERICAL_CUDA_BASE_CACHE_KEY != base_key:
        layers = max(5, int(sample.get("sphere_fill_layers", 31)))
        if layers % 2 == 0:
            layers += 1
        px_tensor = torch.as_tensor(px_base, dtype=torch.float32, device="cuda")
        py_tensor = torch.as_tensor(py_base, dtype=torch.float32, device="cuda")
        core_tensor = torch.as_tensor(sphere_core, dtype=torch.float32, device="cuda")
        idx = torch.arange(px_tensor.shape[0], dtype=torch.float32, device="cuda")[None, :]
        lid = torch.arange(layers, dtype=torch.float32, device="cuda")[:, None]
        seed = float(sample.get("shape_seed", 0)) * 0.0137
        rnd = torch.sin((idx * 12.9898 + lid * 78.233 + seed) * 1.12345) * 43758.5453
        t = (rnd - torch.floor(rnd)) * 2.0 - 1.0
        t[0, :] = -1.0
        t[1, :] = 1.0
        _SPHERICAL_CUDA_BASE_CACHE_VALUE = torch.stack(
            [
                px_tensor.repeat(layers),
                py_tensor.repeat(layers),
                (t * core_tensor[None, :]).reshape(-1),
            ],
            dim=1,
        )
        _SPHERICAL_CUDA_BASE_CACHE_KEY = base_key

    geometry = _filter_surface_holes_torch(
        _SPHERICAL_CUDA_BASE_CACHE_VALUE,
        sample.get("surface_holes", []),
        rmax_global,
    )
    if geometry.shape[0] == 0:
        return None
    _SPHERICAL_CUDA_CACHE_KEY = cache_key
    _SPHERICAL_CUDA_CACHE_VALUE = geometry
    return _SPHERICAL_CUDA_CACHE_VALUE


def _cuda_disk_offsets(radius: int, *, with_decay: bool = False):
    torch = _get_torch_cuda()
    key = (int(radius), bool(with_decay))
    cached = _CUDA_OFFSET_CACHE.get(key)
    if cached is not None:
        return cached
    offsets = [(0, 0)]
    if radius > 1:
        r2 = radius * radius
        offsets = [
            (ox, oy)
            for oy in range(-radius, radius + 1)
            for ox in range(-radius, radius + 1)
            if ox * ox + oy * oy <= r2
        ]
    ox = torch.tensor([p[0] for p in offsets], dtype=torch.int64, device="cuda")
    oy = torch.tensor([p[1] for p in offsets], dtype=torch.int64, device="cuda")
    decay = None
    if with_decay:
        decay = torch.tensor(
            [int(math.sqrt(x * x + y * y) * 2.2) for x, y in offsets],
            dtype=torch.int16,
            device="cuda",
        )
    cached = (ox, oy, decay)
    _CUDA_OFFSET_CACHE[key] = cached
    return cached


def _render_spherical_icon_solid_cuda(
    sample: Dict,
    pose: Pose,
    *,
    display_size: Union[int, Tuple[int, int]],
    eval_size: int,
    need_oblique: bool = True,
):
    torch = _get_torch_cuda()
    geometry = _prepare_spherical_cuda_geometry(sample)
    if torch is None or geometry is None:
        return None

    dw, dh = _resolve_size(display_size)
    ew = eh = int(eval_size)
    dscale = min(dw, dh) / 64.0
    escale = min(ew, eh) / 64.0
    dcx, dcy = dw / 2.0, dh / 2.0
    ecx, ecy = ew / 2.0, eh / 2.0
    icon_radius = 0.36 if sample.get("source_dataset") == "game_icons" else 0.2
    d_radius = max(1, int(round(dscale * icon_radius)))
    e_radius = max(1, int(round(escale * icon_radius)))
    o_radius = max(
        1,
        int(round(dscale * (0.33 if sample.get("source_dataset") == "game_icons" else 0.18))),
    )
    d_offsets = _cuda_disk_offsets(d_radius)
    e_offsets = _cuda_disk_offsets(e_radius)
    o_offsets = _cuda_disk_offsets(o_radius, with_decay=True)

    proj_disp = torch.zeros(dh * dw, dtype=torch.uint8, device="cuda")
    proj_eval = torch.zeros(eh * ew, dtype=torch.uint8, device="cuda")
    oblique = torch.full((dh * dw,), 18, dtype=torch.int16, device="cuda") if need_oblique else None

    roll, pitch, yaw = map(math.radians, (pose.roll, pose.pitch, pose.yaw))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cyw, syw = math.cos(yaw), math.sin(yaw)
    point_batch_size = max(10000, int(sample.get("cuda_point_batch_size", 800000)))

    for start in range(0, geometry.shape[0], point_batch_size):
        points = geometry[start : start + point_batch_size]
        px, py, pz = points[:, 0], points[:, 1], points[:, 2]
        y1 = py * cr - pz * sr
        z1 = py * sr + pz * cr
        x2 = px * cp + z1 * sp
        z2 = -px * sp + z1 * cp
        x3 = x2 * cyw - y1 * syw
        y3 = x2 * syw + y1 * cyw

        for output, width, height, cx, cy, scale, offsets in (
            (proj_disp, dw, dh, dcx, dcy, dscale, d_offsets),
            (proj_eval, ew, eh, ecx, ecy, escale, e_offsets),
        ):
            xi = torch.round(cx + x3 * scale).to(dtype=torch.int64)
            yi = torch.round(cy + y3 * scale).to(dtype=torch.int64)
            ox, oy, _ = offsets
            xx = xi[:, None] + ox[None, :]
            yy = yi[:, None] + oy[None, :]
            valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            output[(yy[valid] * width + xx[valid]).long()] = 255

        if need_oblique:
            xi = torch.round(dcx + x3 * 0.9 * dscale).to(dtype=torch.int64)
            yi = torch.round(dcy + y3 * 0.9 * dscale).to(dtype=torch.int64)
            shade = torch.clamp(110.0 + z2 * 6.0 + x3 * 1.8, 40.0, 235.0).to(dtype=torch.int16)
            ox, oy, decay = o_offsets
            xx = xi[:, None] + ox[None, :]
            yy = yi[:, None] + oy[None, :]
            values = shade[:, None] - decay[None, :]
            valid = (xx >= 0) & (xx < dw) & (yy >= 0) & (yy < dh)
            oblique.scatter_reduce_(
                0,
                (yy[valid] * dw + xx[valid]).long(),
                values.expand_as(xx)[valid],
                reduce="amax",
                include_self=True,
            )

    proj_disp = proj_disp.reshape(dh, dw).cpu().numpy()
    proj_eval = proj_eval.reshape(eh, ew).cpu().numpy()
    proj_disp = cv2.morphologyEx(
        proj_disp,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    proj_eval = cv2.morphologyEx(
        proj_eval,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    if not need_oblique:
        return proj_disp, proj_eval, None
    oblique = oblique.reshape(dh, dw).to(dtype=torch.uint8).cpu().numpy()
    oblique = cv2.GaussianBlur(oblique, (3, 3), 0)
    oblique = cv2.GaussianBlur(oblique, (3, 3), 0)
    return proj_disp, proj_eval, oblique


def _render_spherical_icon_solid_batched(
    sample: Dict,
    pose: Pose,
    *,
    display_size: Union[int, Tuple[int, int]],
    eval_size: int,
    need_oblique: bool = True,
):
    # Batch rendering path to avoid materializing huge (points * layers) arrays at once.
    if np is None or njit is None or cv2 is None:
        return None

    prepared = _prepare_spherical_base(sample)
    if prepared is None:
        return None
    px_base, py_base, sphere_core, rmax_global = prepared

    layers = max(5, int(sample.get("sphere_fill_layers", 31)))
    if layers % 2 == 0:
        layers += 1
    batch_layers = max(1, int(sample.get("render_layer_batch_size", 5)))

    roll = math.radians(pose.roll)
    pitch = math.radians(pose.pitch)
    yaw = math.radians(pose.yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cyw, syw = math.cos(yaw), math.sin(yaw)

    dw, dh = _resolve_size(display_size)
    dscale = min(dw, dh) / 64.0
    dcx = dw / 2.0
    dcy = dh / 2.0
    d_radius = max(1, int(round(dscale * (0.36 if sample.get("source_dataset") == "game_icons" else 0.2))))

    ew = eh = int(eval_size)
    escale = min(ew, eh) / 64.0
    ecx = ew / 2.0
    ecy = eh / 2.0
    e_radius = max(1, int(round(escale * (0.36 if sample.get("source_dataset") == "game_icons" else 0.2))))

    proj_disp = np.zeros((dh, dw), dtype=np.uint8)
    proj_eval = np.zeros((eh, ew), dtype=np.uint8)
    oblique = np.full((dh, dw), 18, dtype=np.uint8) if need_oblique else None

    idx = np.arange(px_base.shape[0], dtype=np.float32)[None, :]
    seed = float(sample.get("shape_seed", 0)) * 0.0137

    for start in range(0, layers, batch_layers):
        end = min(layers, start + batch_layers)
        lid = np.arange(start, end, dtype=np.float32)[:, None]
        rnd = np.sin((idx * 12.9898 + lid * 78.233 + seed) * 1.12345) * 43758.5453
        frac = rnd - np.floor(rnd)
        t = frac * 2.0 - 1.0
        if start <= 0 < end:
            t[0 - start, :] = -1.0
        if start <= 1 < end:
            t[1 - start, :] = 1.0

        b = end - start
        pz = (t * sphere_core[None, :]).reshape(-1).astype(np.float32)
        px = np.tile(px_base, b).astype(np.float32)
        py = np.tile(py_base, b).astype(np.float32)
        px, py, pz = _filter_surface_holes_np(px, py, pz, sample.get("surface_holes", []), rmax_override=rmax_global)
        if px.size == 0:
            continue

        # Rotate xyz (vectorized)
        x1 = px
        y1 = py * cr - pz * sr
        z1 = py * sr + pz * cr
        x2 = x1 * cp + z1 * sp
        y2 = y1
        z2 = -x1 * sp + z1 * cp
        x3 = x2 * cyw - y2 * syw
        y3 = x2 * syw + y2 * cyw
        z3 = z2

        dxi = np.rint(dcx + x3 * dscale).astype(np.int32)
        dyi = np.rint(dcy + y3 * dscale).astype(np.int32)
        proj_disp_chunk = _rasterize_mask_numba(dxi, dyi, z3.astype(np.float32), dw, dh, d_radius)
        proj_disp = np.maximum(proj_disp, proj_disp_chunk)

        exi = np.rint(ecx + x3 * escale).astype(np.int32)
        eyi = np.rint(ecy + y3 * escale).astype(np.int32)
        proj_eval_chunk = _rasterize_mask_numba(exi, eyi, z3.astype(np.float32), ew, eh, e_radius)
        proj_eval = np.maximum(proj_eval, proj_eval_chunk)

        if need_oblique:
            oxi = np.rint(dcx + x3 * 0.9 * dscale).astype(np.int32)
            oyi = np.rint(dcy + y3 * 0.9 * dscale).astype(np.int32)
            shade = np.clip((110.0 + z3 * 6.0 + x3 * 1.8), 40.0, 235.0).astype(np.uint8)
            o_radius = max(1, int(round(dscale * (0.33 if sample.get("source_dataset") == "game_icons" else 0.18))))
            oblique_chunk = _rasterize_oblique_numba(oxi, oyi, shade, dw, dh, o_radius)
            oblique = np.maximum(oblique, oblique_chunk)

    proj_disp = cv2.morphologyEx(
        proj_disp,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    proj_eval = cv2.morphologyEx(
        proj_eval,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    if need_oblique and oblique is not None:
        oblique = cv2.GaussianBlur(oblique, (3, 3), 0)
        oblique = cv2.GaussianBlur(oblique, (3, 3), 0)
    return proj_disp, proj_eval, oblique


def render_views_fast(
    sample: Dict,
    pose: Pose,
    *,
    display_size: Union[int, Tuple[int, int]],
    eval_size: int,
    need_oblique: bool = True,
):
    # Accelerated rendering path using numpy + numba + opencv.
    if np is None or njit is None or cv2 is None:
        proj_disp = create_mask(sample, pose, size=display_size, fill_holes=False)
        proj_eval = create_mask(sample, pose, size=eval_size, fill_holes=True)
        oblique = create_oblique_view(sample, pose, size=display_size) if need_oblique else None
        _set_last_render_backend("python_fallback")
        return (
            np.asarray(proj_disp, dtype=np.uint8) if np is not None else proj_disp,
            np.asarray(proj_eval, dtype=np.uint8) if np is not None else proj_eval,
            np.asarray(oblique, dtype=np.uint8) if (np is not None and oblique is not None) else oblique,
        )

    if (
        sample.get("surface_mode") == "spherical_icon"
        and bool(sample.get("sphere_double_sided", True))
        and bool(sample.get("sphere_solid", True))
    ):
        cuda_render = _render_spherical_icon_solid_cuda(
            sample,
            pose,
            display_size=display_size,
            eval_size=eval_size,
            need_oblique=need_oblique,
        )
        if cuda_render is not None:
            _set_last_render_backend("cuda")
            return cuda_render
        batched = _render_spherical_icon_solid_batched(
            sample,
            pose,
            display_size=display_size,
            eval_size=eval_size,
            need_oblique=need_oblique,
        )
        if batched is not None:
            _set_last_render_backend("numpy_batched")
            return batched

    prepared = _prepare_sample_arrays(sample)
    if prepared is None:
        proj_disp = create_mask(sample, pose, size=display_size, fill_holes=False)
        proj_eval = create_mask(sample, pose, size=eval_size, fill_holes=True)
        oblique = create_oblique_view(sample, pose, size=display_size) if need_oblique else None
        _set_last_render_backend("numpy_opencv_fallback")
        return (
            np.asarray(proj_disp, dtype=np.uint8),
            np.asarray(proj_eval, dtype=np.uint8),
            np.asarray(oblique, dtype=np.uint8) if oblique is not None else None,
        )

    px, py, pz = prepared
    roll = math.radians(pose.roll)
    pitch = math.radians(pose.pitch)
    yaw = math.radians(pose.yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cyw, syw = math.cos(yaw), math.sin(yaw)

    # Rotate xyz (vectorized)
    x1 = px
    y1 = py * cr - pz * sr
    z1 = py * sr + pz * cr
    x2 = x1 * cp + z1 * sp
    y2 = y1
    z2 = -x1 * sp + z1 * cp
    x3 = x2 * cyw - y2 * syw
    y3 = x2 * syw + y2 * cyw
    z3 = z2

    if sample.get("surface_mode") == "spherical_icon":
        jitter_scale = 0.0
    else:
        jitter_scale = 0.35 if sample.get("stream_type") == "single" else 0.25
    jitter = jitter_scale * math.sin(math.radians(pose.pitch - pose.yaw)) * (1.0 + np.abs(z3) * 0.05)

    dw, dh = _resolve_size(display_size)
    dscale = min(dw, dh) / 64.0
    dcx = dw / 2.0
    dcy = dh / 2.0
    d_radius = max(1, int(round(dscale * (0.36 if sample.get("source_dataset") == "game_icons" else 0.2))))

    dxi = np.rint(dcx + (x3 + jitter) * dscale).astype(np.int32)
    dyi = np.rint(dcy + (y3 - jitter) * dscale).astype(np.int32)
    proj_disp = _rasterize_mask_numba(dxi, dyi, z3.astype(np.float32), dw, dh, d_radius)
    proj_disp = cv2.morphologyEx(
        proj_disp,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    ew = eh = int(eval_size)
    escale = min(ew, eh) / 64.0
    ecx = ew / 2.0
    ecy = eh / 2.0
    e_radius = max(1, int(round(escale * (0.36 if sample.get("source_dataset") == "game_icons" else 0.2))))
    exi = np.rint(ecx + (x3 + jitter) * escale).astype(np.int32)
    eyi = np.rint(ecy + (y3 - jitter) * escale).astype(np.int32)
    proj_eval = _rasterize_mask_numba(exi, eyi, z3.astype(np.float32), ew, eh, e_radius)
    proj_eval = cv2.morphologyEx(
        proj_eval,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    oblique = None
    if need_oblique:
        oxi = np.rint(dcx + x3 * 0.9 * dscale).astype(np.int32)
        if sample.get("surface_mode") == "spherical_icon":
            oyi = np.rint(dcy + y3 * 0.9 * dscale).astype(np.int32)
        else:
            oyi = np.rint(dcy + (y3 * 0.65 - z3 * 0.35) * dscale).astype(np.int32)
        shade = np.clip((110.0 + z3 * 6.0 + x3 * 1.8), 40.0, 235.0).astype(np.uint8)
        o_radius = max(1, int(round(dscale * (0.33 if sample.get("source_dataset") == "game_icons" else 0.18))))
        oblique = _rasterize_oblique_numba(oxi, oyi, shade, dw, dh, o_radius)
        oblique = cv2.GaussianBlur(oblique, (3, 3), 0)
        oblique = cv2.GaussianBlur(oblique, (3, 3), 0)

    _set_last_render_backend("numpy_numba_opencv")
    return proj_disp, proj_eval, oblique


def iou(mask_a: List[List[int]], mask_b: List[List[int]]) -> float:
    if np is not None and isinstance(mask_a, np.ndarray) and isinstance(mask_b, np.ndarray):
        a = mask_a > 0
        b = mask_b > 0
        inter = int(np.count_nonzero(a & b))
        union = int(np.count_nonzero(a | b))
        if union == 0:
            return 1.0
        return inter / union

    inter = 0
    union = 0
    h = len(mask_a)
    w = len(mask_a[0]) if h else 0
    for y in range(h):
        for x in range(w):
            a = mask_a[y][x]
            b = mask_b[y][x]
            if a or b:
                union += 1
                if a and b:
                    inter += 1
    if union == 0:
        return 1.0
    return inter / union


def apply_action(pose: Pose, action: str, theta: int = THETA) -> Pose:
    # Observer-frame control:
    # action rotations are defined around camera-fixed x/y/z axes (pitch/yaw/roll).
    # Compose as left-multiplication in matrix space, then convert back to Euler used by renderer.
    step = float(theta)
    if action == "PITCH_POS":
        delta = _rot_x(step)
    elif action == "PITCH_NEG":
        delta = _rot_x(-step)
    elif action == "YAW_POS":
        delta = _rot_y(step)
    elif action == "YAW_NEG":
        delta = _rot_y(-step)
    elif action == "ROLL_POS":
        delta = _rot_z(step)
    elif action == "ROLL_NEG":
        delta = _rot_z(-step)
    else:
        return Pose(pose.roll, pose.pitch, pose.yaw)

    cur = _euler_to_matrix(pose.roll, pose.pitch, pose.yaw)
    nxt = _mat_mul(delta, cur)
    roll, pitch, yaw = _matrix_to_euler(nxt)
    return Pose(
        roll=normalize_angle(roll),
        pitch=normalize_angle(pitch),
        yaw=normalize_angle(yaw),
    )


def normalize_angle(v: float) -> float:
    out = ((v + 180) % 360) - 180
    return float(out)


def _rot_x(deg: float):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return (
        (1.0, 0.0, 0.0),
        (0.0, c, -s),
        (0.0, s, c),
    )


def _rot_y(deg: float):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return (
        (c, 0.0, s),
        (0.0, 1.0, 0.0),
        (-s, 0.0, c),
    )


def _rot_z(deg: float):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return (
        (c, -s, 0.0),
        (s, c, 0.0),
        (0.0, 0.0, 1.0),
    )


def _mat_mul(a, b):
    return (
        (
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ),
        (
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ),
        (
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ),
    )


def _euler_to_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float):
    rr = math.radians(float(roll_deg))
    rp = math.radians(float(pitch_deg))
    ry = math.radians(float(yaw_deg))
    cr, sr = math.cos(rr), math.sin(rr)
    cp, sp = math.cos(rp), math.sin(rp)
    cy, sy = math.cos(ry), math.sin(ry)
    # Renderer order: Rx(roll) -> Ry(pitch) -> Rz(yaw), i.e. R = Rz * Ry * Rx
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _matrix_to_euler(m):
    # Inverse of R = Rz * Ry * Rx
    v = max(-1.0, min(1.0, -m[2][0]))
    pitch = math.degrees(math.asin(v))
    cp = math.cos(math.radians(pitch))
    if abs(cp) < 1e-8:
        roll = 0.0
        yaw = math.degrees(math.atan2(-m[0][1], m[1][1]))
    else:
        roll = math.degrees(math.atan2(m[2][1], m[2][2]))
        yaw = math.degrees(math.atan2(m[1][0], m[0][0]))
    return roll, pitch, yaw


def pose_distance_steps(a: Pose, b: Pose, theta: int = THETA) -> int:
    def d(x: int, y: int) -> int:
        dd = abs(normalize_angle(x - y))
        return int(round(dd / theta))

    return d(a.roll, b.roll) + d(a.pitch, b.pitch) + d(a.yaw, b.yaw)


def now_ms() -> int:
    import time

    return int(time.time() * 1000)


def write_png_gray(path: Path, mask: List[List[int]]) -> None:
    ensure_dir(path.parent)
    if np is not None and cv2 is not None and isinstance(mask, np.ndarray):
        arr = mask
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        cv2.imwrite(str(path), arr)
        return
    h = len(mask)
    w = len(mask[0]) if h else 0

    raw = bytearray()
    for y in range(h):
        raw.append(0)  # no filter
        row = mask[y]
        for x in range(w):
            v = row[x]
            if isinstance(v, bool):
                px = 255 if v else 0
            else:
                iv = int(v)
                if iv in (0, 1):
                    px = iv * 255
                else:
                    px = int(clamp(iv, 0, 255))
            raw.append(px)

    compressor = zlib.compressobj()
    compressed = compressor.compress(bytes(raw)) + compressor.flush()

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
        + b"\x08"  # bit depth
        + b"\x00"  # grayscale
        + b"\x00"  # compression
        + b"\x00"  # filter
        + b"\x00"  # interlace
    )
    png.extend(chunk(b"IHDR", ihdr))
    png.extend(chunk(b"IDAT", compressed))
    png.extend(chunk(b"IEND", b""))

    path.write_bytes(bytes(png))


def load_default_config() -> Dict:
    return {
        "theta": 15,
        "T_max": 24,
        "tau_success": 0.9,
        "random_seed": 20260302,
        "sample_count": 240,
        "eval_size": STANDARD_IMAGE_SIZE,
        "display_width": STANDARD_IMAGE_SIZE,
        "display_height": STANDARD_IMAGE_SIZE,
        "icon_categories": ["all"],
        "icon_pool_limit": 1200,
        "icon_min_connected_ratio": 0.985,
        "icon_min_circle_iou": 0.28,
        "icon_min_bbox_fill": 0.08,
        "icon_max_aspect_ratio": 2.2,
        "icon_min_pixel_points": 1800,
        "surface_hole_attempts": 30,
        "render_layer_batch_size": 5,
        "model_name": "simulated_mllm_v1",
    }


def get_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", type=str, default=".", help="Project root path")
    parser.add_argument(
        "--config",
        type=str,
        default="results/artifacts/config/run_config.json",
        help="Config path relative to root",
    )
    return parser.parse_args()


def resolve_paths(root: Path) -> Dict[str, Path]:
    return {
        "root": root,
        "artifacts": root / "artifacts",
        "config": root / "artifacts" / "config" / "run_config.json",
        "data": root / "data",
        "tasks": root / "tasks" / "tasks.jsonl",
        "logs": root / "logs",
        "reports": root / "reports",
    }
