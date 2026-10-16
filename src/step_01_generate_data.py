#!/usr/bin/env python3
"""Step 01: generate single-stream data and tasks from open-source solid icon database."""

from __future__ import annotations

import io
import argparse
import base64
import json
import hashlib
import math
import pickle
import random
import subprocess
import benchmark_core as core
from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Set, Tuple
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable

from benchmark_core import (
    ACTIONS,
    Pose,
    apply_action,
    create_mask,
    create_oblique_view,
    ensure_dir,
    iou,
    load_default_config,
    pose_distance_steps,
    render_views_fast,
    read_json,
    read_jsonl,
    resolve_paths,
    write_json,
    write_jsonl,
    write_png_gray,
)
from render_protocol import STANDARD_DISPLAY_SIZE, STANDARD_IMAGE_SIZE

INIT_THETA_DEG = 30
ALLOWED_STEP_ANGLES = [30, 10, 5]
MAX_ICON_POINTS = 90000
ICON_CACHE_FORMAT_VERSION = 2

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 01 generate data")
    parser.add_argument("--root", type=str, default=".", help="Project root path")
    parser.add_argument(
        "--config",
        type=str,
        default="results/artifacts/config/run_config.json",
        help="Config path relative to root",
    )
    parser.add_argument("--save-init-images", action="store_true", help="Save init projection PNGs")
    parser.add_argument("--save-init-oblique-images", action="store_true", help="Save init oblique PNGs")
    parser.add_argument("--sample-count", type=int, default=0, help="Override sample_count in config when > 0")
    parser.add_argument("--resume", dest="resume", action="store_true", help="Resume Step1 from checkpoint")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start Step1 from scratch")
    parser.add_argument(
        "--no-save-target-images",
        dest="save_target_images",
        action="store_false",
        help="Do not save target PNGs (may break API-based Step02)",
    )
    parser.set_defaults(save_target_images=True)
    parser.set_defaults(resume=False)
    return parser.parse_args()


def random_pose(rng: random.Random) -> Pose:
    choices = list(range(-120, 121, 15))
    return Pose(roll=rng.choice(choices), pitch=rng.choice(choices), yaw=rng.choice(choices))


def _slug(svg_path: Path) -> str:
    return f"{svg_path.parent.name}-{svg_path.stem}".lower().replace(" ", "-")


def uniform_downsample_points(points: List[List[float]], max_points: int) -> List[List[float]]:
    """Keep the full scan range while deterministically limiting point count."""
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if len(points) <= max_points:
        return points
    if max_points == 1:
        return [points[0]]
    last = len(points) - 1
    denominator = max_points - 1
    return [points[(index * last) // denominator] for index in range(max_points)]


def svg_to_icon_points(svg_path: Path, max_points: int = MAX_ICON_POINTS) -> List[List[float]]:
    import cairosvg
    from PIL import Image

    png = cairosvg.svg2png(url=str(svg_path), output_width=512, output_height=512)
    img = Image.open(io.BytesIO(png)).convert("RGBA")
    w, h = img.size
    px = img.load()
    corners = [
        px[0, 0][:3],
        px[w - 1, 0][:3],
        px[0, h - 1][:3],
        px[w - 1, h - 1][:3],
    ]
    bg_r = int(sum(c[0] for c in corners) / 4)
    bg_g = int(sum(c[1] for c in corners) / 4)
    bg_b = int(sum(c[2] for c in corners) / 4)

    points: List[List[float]] = []
    stride = 1
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            r, g, b, a = px[x, y]
            color_delta = abs(int(r) - bg_r) + abs(int(g) - bg_g) + abs(int(b) - bg_b)
            if a > 8 and color_delta > 30:
                nx = (x - w / 2) / (min(w, h) / 2) * 24.0
                ny = (y - h / 2) / (min(w, h) / 2) * 24.0
                points.append([round(nx, 4), round(ny, 4)])

    # Deterministic full-range downsample to control JSON size and runtime.
    points = uniform_downsample_points(points, max_points)

    if not points:
        raise RuntimeError(f"Empty icon points for {svg_path}")
    return points


def _circle_from_two(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float, float]:
    cx = (a[0] + b[0]) / 2.0
    cy = (a[1] + b[1]) / 2.0
    r = math.hypot(a[0] - b[0], a[1] - b[1]) / 2.0
    return cx, cy, r


def _circle_from_three(
    a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]
) -> Optional[Tuple[float, float, float]]:
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-8:
        return None
    ux = (
        (ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by)
    ) / d
    uy = (
        (ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax)
    ) / d
    r = math.hypot(ux - ax, uy - ay)
    return ux, uy, r


def _in_circle(p: Tuple[float, float], c: Tuple[float, float, float]) -> bool:
    return math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] + 1e-7


def minimum_enclosing_circle(points: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    if not points:
        return 0.0, 0.0, 0.0
    pts = points[:]
    random.Random(20260302).shuffle(pts)
    circle: Optional[Tuple[float, float, float]] = None
    for i, p in enumerate(pts):
        if circle is not None and _in_circle(p, circle):
            continue
        circle = (p[0], p[1], 0.0)
        for j in range(i):
            q = pts[j]
            if _in_circle(q, circle):
                continue
            circle = _circle_from_two(p, q)
            for k in range(j):
                r = pts[k]
                if _in_circle(r, circle):
                    continue
                c3 = _circle_from_three(p, q, r)
                if c3 is not None:
                    circle = c3
    if circle is None:
        return 0.0, 0.0, 0.0
    return circle


def _rasterize_points(points: List[List[float]], size: int = 112, pad: int = 6) -> Tuple[Set[Tuple[int, int]], float]:
    if not points:
        return set(), 1.0
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)
    scale = min((size - 1 - 2 * pad) / span_x, (size - 1 - 2 * pad) / span_y)
    occ: Set[Tuple[int, int]] = set()
    for p in points:
        x = int(round((float(p[0]) - min_x) * scale + pad))
        y = int(round((float(p[1]) - min_y) * scale + pad))
        if 0 <= x < size and 0 <= y < size:
            occ.add((x, y))
    return occ, scale


def _largest_component_ratio(occ: Set[Tuple[int, int]]) -> float:
    if not occ:
        return 0.0
    unvisited = set(occ)
    largest = 0
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while unvisited:
        start = unvisited.pop()
        q = deque([start])
        size = 1
        while q:
            x, y = q.popleft()
            for dx, dy in neighbors:
                nxt = (x + dx, y + dy)
                if nxt in unvisited:
                    unvisited.remove(nxt)
                    q.append(nxt)
                    size += 1
        if size > largest:
            largest = size
    return largest / max(1, len(occ))


def icon_quality_stats(points: List[List[float]]) -> Dict[str, float]:
    if not points:
        return {"connected_ratio": 0.0, "circle_iou": 0.0, "bbox_fill": 0.0, "aspect_ratio": 99.0}
    occ, scale = _rasterize_points(points, size=112, pad=6)
    if not occ:
        return {"connected_ratio": 0.0, "circle_iou": 0.0, "bbox_fill": 0.0, "aspect_ratio": 99.0}
    xs = [p[0] for p in occ]
    ys = [p[1] for p in occ]
    bw = max(xs) - min(xs) + 1
    bh = max(ys) - min(ys) + 1
    aspect = max(bw, bh) / max(1, min(bw, bh))
    bbox_fill = len(occ) / max(1, bw * bh)
    connected_ratio = _largest_component_ratio(occ)

    step = max(1, len(points) // 3500)
    sampled = [(float(p[0]), float(p[1])) for p in points[::step]]
    _, _, r = minimum_enclosing_circle(sampled)
    circle_area = math.pi * (max(1e-6, r * scale) ** 2)
    circle_iou = min(1.0, len(occ) / max(1e-6, circle_area))
    return {
        "connected_ratio": float(connected_ratio),
        "circle_iou": float(circle_iou),
        "bbox_fill": float(bbox_fill),
        "aspect_ratio": float(aspect),
    }


def build_icon_db(
    root: Path,
    categories: List[str],
    pool_limit: int = 1200,
    quality_cfg: Optional[Dict] = None,
) -> Tuple[Dict[str, List[Dict]], Dict]:
    base = root / "data_sources" / "game-icons"
    if not base.exists():
        ensure_dir(base.parent)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/game-icons/icons", str(base)],
            check=True,
            cwd=str(root),
        )

    cache_dir = root / "data_sources" / "game-icons-cache-v3"
    ensure_dir(cache_dir)

    all_svgs = list(base.rglob("*.svg"))

    def matches(cat: str, name: str) -> bool:
        n = name.lower()
        if cat == "all":
            return True
        if cat == "spider":
            return "spider" in n and "web" not in n
        if cat == "bird":
            return "bird" in n and "bird-house" not in n and "bird-house" not in n.replace("_", "-")
        if cat == "house":
            return ("house" in n or "home" in n) and "bird-house" not in n
        return cat in n

    db: Dict[str, List[Dict]] = {c: [] for c in categories}

    quality_cfg = quality_cfg or {}
    min_connected_ratio = float(quality_cfg.get("icon_min_connected_ratio", 0.985))
    min_circle_iou = float(quality_cfg.get("icon_min_circle_iou", 0.28))
    min_bbox_fill = float(quality_cfg.get("icon_min_bbox_fill", 0.08))
    max_aspect_ratio = float(quality_cfg.get("icon_max_aspect_ratio", 2.2))
    min_pixel_points = int(quality_cfg.get("icon_min_pixel_points", 1800))

    timing = {
        "build_icon_db_s": 0.0,
        "categories": {},
        "totals": {
            "candidate_svgs": 0,
            "processed_svgs": 0,
            "cache_hit": 0,
            "cache_miss": 0,
            "cache_refresh": 0,
            "passed_quality": 0,
            "rejected_quality": 0,
            "time_svg_to_points_s": 0.0,
            "time_quality_eval_s": 0.0,
        },
    }
    t_build_start = perf_counter()
    for cat in tqdm(categories, desc="Build icon DB", unit="cat"):
        t_cat_start = perf_counter()
        candidates = [p for p in all_svgs if matches(cat, p.stem)]
        # Keep only stable shape-like icons (drop a few metadata-like outliers).
        candidates = [p for p in candidates if "keys" not in p.stem.lower() and "scepter" not in p.stem.lower()]
        if not candidates:
            raise RuntimeError(f"No icons found for category: {cat}")
        timing["totals"]["candidate_svgs"] += len(candidates)

        # Truncate BEFORE caching to avoid processing all files.
        candidates = sorted(candidates, key=lambda p: _slug(p))
        if cat == "all":
            limit = int(pool_limit)
            if limit >= 0:
                candidates = candidates[: max(1, limit)]
        else:
            candidates = candidates[:24]

        for svg in tqdm(candidates, desc=f"Cache {cat}", unit="svg", leave=False):
            slug = _slug(svg)
            cache_file = cache_dir / f"{slug}.json"
            timing["totals"]["processed_svgs"] += 1
            cached_payload = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else None
            cache_is_current = (
                isinstance(cached_payload, dict)
                and int(cached_payload.get("format_version", 0)) >= ICON_CACHE_FORMAT_VERSION
                and isinstance(cached_payload.get("points"), list)
            )
            legacy_points = cached_payload if isinstance(cached_payload, list) else None
            legacy_is_safe = legacy_points is not None and len(legacy_points) < MAX_ICON_POINTS
            if cache_is_current:
                timing["totals"]["cache_hit"] += 1
                icon_points = cached_payload["points"]
            elif legacy_is_safe:
                timing["totals"]["cache_hit"] += 1
                icon_points = legacy_points
            else:
                timing["totals"]["cache_miss"] += 1
                if cache_file.exists():
                    timing["totals"]["cache_refresh"] += 1
                t_svg = perf_counter()
                icon_points = svg_to_icon_points(svg)
                timing["totals"]["time_svg_to_points_s"] += perf_counter() - t_svg
                cache_file.write_text(
                    json.dumps(
                        {
                            "format_version": ICON_CACHE_FORMAT_VERSION,
                            "sampling": "uniform_full_scan_range",
                            "max_points": MAX_ICON_POINTS,
                            "points": icon_points,
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

            t_q = perf_counter()
            q = icon_quality_stats(icon_points)
            timing["totals"]["time_quality_eval_s"] += perf_counter() - t_q
            is_quality_ok = (
                len(icon_points) >= min_pixel_points
                and q["connected_ratio"] >= min_connected_ratio
                and q["circle_iou"] >= min_circle_iou
                and q["bbox_fill"] >= min_bbox_fill
                and q["aspect_ratio"] <= max_aspect_ratio
            )
            if not is_quality_ok:
                timing["totals"]["rejected_quality"] += 1
                continue
            timing["totals"]["passed_quality"] += 1

            db[cat].append(
                {
                    "icon_slug": slug,
                    "icon_svg": str(svg.relative_to(root)).replace("\\", "/"),
                    "icon_cache_file": str(cache_file.relative_to(root)).replace("\\", "/"),
                    "icon_points": icon_points,
                    "quality": {
                        "connected_ratio": round(q["connected_ratio"], 5),
                        "circle_iou": round(q["circle_iou"], 5),
                        "bbox_fill": round(q["bbox_fill"], 5),
                        "aspect_ratio": round(q["aspect_ratio"], 5),
                    },
                }
            )

        if not db[cat]:
            raise RuntimeError(
                f"No icons passed quality filters for category={cat}. "
                f"Try lowering thresholds: connected>={min_connected_ratio}, circle_iou>={min_circle_iou}, "
                f"bbox_fill>={min_bbox_fill}, aspect<={max_aspect_ratio}."
            )
        db[cat] = sorted(db[cat], key=lambda x: x["icon_slug"])
        timing["categories"][cat] = {
            "candidates": len(candidates),
            "passed": len(db[cat]),
            "elapsed_s": round(perf_counter() - t_cat_start, 6),
        }

    timing["build_icon_db_s"] = round(perf_counter() - t_build_start, 6)
    return db, timing


def inverse_action(action: str) -> str:
    mapping = {
        "PITCH_POS": "PITCH_NEG",
        "PITCH_NEG": "PITCH_POS",
        "YAW_POS": "YAW_NEG",
        "YAW_NEG": "YAW_POS",
        "ROLL_POS": "ROLL_NEG",
        "ROLL_NEG": "ROLL_POS",
    }
    return mapping[action]


def _random_unit_vec(rng: random.Random) -> Tuple[float, float, float]:
    while True:
        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-1.0, 1.0)
        z = rng.uniform(-1.0, 1.0)
        n = math.sqrt(x * x + y * y + z * z)
        if n > 1e-6:
            return x / n, y / n, z / n


def _same_mask(a, b) -> bool:
    if np is not None:
        return bool(np.array_equal(a, b))
    return a == b


def _sample_surface_hole(rng: random.Random) -> Dict:
    nx, ny, nz = _random_unit_vec(rng)
    shape = rng.choices(
        ["rect", "parallelogram", "circle"],
        weights=[0.4, 0.3, 0.3],
        k=1,
    )[0]

    hole = {
        "shape": shape,
        "nx": round(nx, 6),
        "ny": round(ny, 6),
        "nz": round(nz, 6),
        "min_n": round(rng.uniform(0.90, 0.97), 6),
        "shell_ratio": round(rng.uniform(0.45, 0.72), 6),
    }

    if shape == "circle":
        hole["radius"] = round(rng.uniform(0.09, 0.24), 6)
        return hole

    long_half = rng.uniform(0.18, 0.34)
    short_half = rng.uniform(0.07, 0.15)
    if rng.random() < 0.5:
        half_w, half_h = long_half, short_half
    else:
        half_w, half_h = short_half, long_half
    hole["half_w"] = round(half_w, 6)
    hole["half_h"] = round(half_h, 6)
    if shape == "parallelogram":
        hole["shear"] = round(rng.uniform(-0.85, 0.85), 6)
    return hole


def _print_icon_filter_report(cfg: Dict, icon_db_timing: Dict, icon_db: Dict[str, List[Dict]]) -> None:
    print("[step01][icon-filter] quality thresholds:")
    print(f"  - icon_categories: {cfg.get('icon_categories', ['all'])}")
    print(f"  - icon_min_connected_ratio: {float(cfg.get('icon_min_connected_ratio', 0.985))}")
    print(f"  - icon_min_circle_iou: {float(cfg.get('icon_min_circle_iou', 0.28))}")
    print(f"  - icon_min_bbox_fill: {float(cfg.get('icon_min_bbox_fill', 0.08))}")
    print(f"  - icon_max_aspect_ratio: {float(cfg.get('icon_max_aspect_ratio', 2.2))}")
    print(f"  - icon_min_pixel_points: {int(cfg.get('icon_min_pixel_points', 1800))}")

    print("[step01][icon-filter] remaining icons after quality filter:")
    categories_timing = icon_db_timing.get("categories", {})
    total_passed = 0
    total_candidates = 0
    for cat in cfg.get("icon_categories", ["all"]):
        cat_info = categories_timing.get(cat, {})
        candidates = int(cat_info.get("candidates", 0))
        passed = int(cat_info.get("passed", len(icon_db.get(cat, []))))
        total_candidates += candidates
        total_passed += passed
        print(f"  - {cat}: {passed}/{candidates}")
    print(f"  - total: {total_passed}/{total_candidates}")


def append_jsonl(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def scan_sample_metadata(path: Path) -> Tuple[List[str], List[str]]:
    sample_ids: List[str] = []
    icon_slugs: List[str] = []
    if not path.exists():
        return sample_ids, icon_slugs
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_ids.append(str(row.get("sample_id", "")))
            icon_slugs.append(str(row.get("icon_slug", "")))
    return sample_ids, icon_slugs


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def truncate_jsonl_rows(path: Path, row_count: int) -> None:
    """Keep only the first non-empty rows without loading a JSONL file into memory."""
    if row_count < 0:
        raise ValueError("row_count must be non-negative")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".truncate.tmp")
    kept = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        if path.exists():
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    if kept >= row_count:
                        break
                    target.write(line.rstrip("\r\n") + "\n")
                    kept += 1
    if kept != row_count:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Cannot keep {row_count} rows from {path}; only found {kept}")
    temporary.replace(path)


def config_fingerprint(cfg: Dict) -> str:
    keys = [
        "random_seed",
        "T_max",
        "tau_success",
        "eval_size",
        "display_width",
        "display_height",
        "icon_categories",
        "icon_pool_limit",
        "icon_min_connected_ratio",
        "icon_min_circle_iou",
        "icon_min_bbox_fill",
        "icon_max_aspect_ratio",
        "icon_min_pixel_points",
        "surface_hole_attempts",
    ]
    payload = {k: cfg.get(k) for k in keys}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def encode_rng_state(state) -> str:
    return base64.b64encode(pickle.dumps(state)).decode("ascii")


def decode_rng_state(text: str):
    return pickle.loads(base64.b64decode(text.encode("ascii")))


def load_checkpoint(path: Path) -> Dict:
    if not path.exists():
        return {}
    return read_json(path)


def infer_next_sample_index(existing_samples: List[Dict]) -> int:
    max_idx = -1
    for row in existing_samples:
        sid = str(row.get("sample_id", "")).strip()
        if sid.startswith("single_"):
            try:
                max_idx = max(max_idx, int(sid.split("_")[-1]))
            except Exception:
                continue
    return max_idx + 1 if max_idx >= 0 else len(existing_samples)


def build_unique_icon_pool(
    cfg: Dict,
    icon_db: Dict[str, List[Dict]],
) -> List[Dict]:
    categories = list(cfg.get("icon_categories", ["all"]))
    if not categories:
        categories = ["all"]
    icon_pool: List[Dict] = []
    seen = set()
    for cat in categories:
        for icon in icon_db.get(cat, []):
            slug = icon.get("icon_slug", "")
            if slug and slug not in seen:
                seen.add(slug)
                pool_item = {"category": cat, **icon}
                pool_item.pop("icon_points", None)
                icon_pool.append(pool_item)
    return icon_pool


def materialize_icon_points(root: Path, icon: Dict) -> Dict:
    """Load one icon's cached points on demand instead of retaining the whole pool."""
    if isinstance(icon.get("icon_points"), list):
        return icon
    cache_ref = str(icon.get("icon_cache_file", "")).strip()
    if not cache_ref:
        raise RuntimeError(f"Icon {icon.get('icon_slug', '')} has no cached points reference")
    payload = json.loads((root / cache_ref).read_text(encoding="utf-8"))
    points = payload.get("points") if isinstance(payload, dict) else payload
    if not isinstance(points, list) or not points:
        raise RuntimeError(f"Invalid cached points for icon {icon.get('icon_slug', '')}")
    materialized = dict(icon)
    materialized["icon_points"] = points
    return materialized


def carve_hole_variants_without_target_change(
    sample: Dict,
    cfg: Dict,
    target_pose: Pose,
    attempt_counts: List[int],
) -> Dict[int, Dict]:
    counts = sorted({max(0, int(value)) for value in attempt_counts})
    if not counts:
        return {}
    attempts = counts[-1]
    eval_size = int(cfg.get("surface_hole_eval_size", 220))
    rng = random.Random(int(sample["shape_seed"]) + 7919)
    accepted: List[Dict] = []
    snapshots: Dict[int, Dict] = {}

    base_sample = dict(sample)
    base_sample["surface_holes"] = []
    _, baseline_eval, _ = render_views_fast(
        base_sample,
        target_pose,
        display_size=eval_size,
        eval_size=eval_size,
        need_oblique=False,
    )

    if 0 in counts:
        snapshots[0] = []
    for attempt_index in range(1, attempts + 1):
        hole = _sample_surface_hole(rng)
        cand = dict(sample)
        cand["surface_holes"] = accepted + [hole]
        _, cand_eval, _ = render_views_fast(
            cand,
            target_pose,
            display_size=eval_size,
            eval_size=eval_size,
            need_oblique=False,
        )
        if _same_mask(cand_eval, baseline_eval):
            accepted.append(hole)
        # If target projection changes, this attempt is reverted by design.

        if attempt_index in counts:
            snapshots[attempt_index] = [dict(item) for item in accepted]

    variants: Dict[int, Dict] = {}
    for count in counts:
        out = dict(sample)
        out["surface_holes"] = snapshots[count]
        out["surface_hole_attempts"] = count
        out["surface_hole_kept"] = len(snapshots[count])
        out["surface_hole_consecutive_failures"] = 0
        variants[count] = out
    return variants


def carve_holes_without_target_change(sample: Dict, cfg: Dict, target_pose: Pose) -> Dict:
    attempts = int(cfg.get("surface_hole_attempts", 30))
    variants = carve_hole_variants_without_target_change(
        sample,
        cfg,
        target_pose,
        [attempts],
    )
    return variants[attempts]


def generate_single_sample(
    index: int,
    cfg: Dict,
    icon: Dict,
    rng: random.Random,
) -> Dict:
    allowed_actions = [a for a in ACTIONS if a != "STOP"]
    category = icon["category"]

    # Keep target icon recognizable: canonical frontal pose.
    target = Pose(roll=0, pitch=0, yaw=0)
    scramble_len = rng.randint(6, 16)
    scramble_actions: List[str] = []
    current = Pose(roll=target.roll, pitch=target.pitch, yaw=target.yaw)
    for _ in range(scramble_len):
        action = rng.choice(allowed_actions)
        if scramble_actions and inverse_action(scramble_actions[-1]) == action:
            action = rng.choice([a for a in allowed_actions if a != inverse_action(scramble_actions[-1])])
        scramble_actions.append(action)
        current = apply_action(current, action, int(INIT_THETA_DEG))
    init = current
    answer_actions = [inverse_action(a) for a in reversed(scramble_actions)]

    sid = f"single_{index:05d}"
    sample = {
        "sample_id": sid,
        "stream_type": "single",
        "shape_seed": cfg["random_seed"] + index,
        "source_dataset": "game_icons",
        "base_shape": "icon",
        "category": category,
        "icon_slug": icon["icon_slug"],
        "icon_svg": icon["icon_svg"],
        "icon_points": icon["icon_points"],
        "icon_quality": icon.get("quality", {}),
        "complexity": 1.0,
        "icon_scale": 1.24,
        "icon_center_mode": "mec",
        "surface_mode": "spherical_icon",
        "sphere_radius": 28.0,
        "sphere_depth_scale": 1.0,
        "sphere_fit_icon": True,
        "sphere_xy_ratio": 0.98,
        "sphere_double_sided": True,
        "sphere_solid": True,
        "sphere_fill_layers": 31,
        "radial_jitter": 0.0,
        "rim_softness": 0.65,
        "fine_noise": 0.0,
        "target_pose": target.as_dict(),
        "init_pose": init.as_dict(),
        "scramble_actions": scramble_actions,
        "answer_actions": answer_actions,
        "theta": int(INIT_THETA_DEG),
        "allowed_angles": list(ALLOWED_STEP_ANGLES),
        "T_max": cfg["T_max"],
        "tau_success": cfg["tau_success"],
        "meta": {
            "difficulty_bucket": len(scramble_actions),
            "generator_mode": "single_stream_spherical_icon_inverse_path",
        },
    }
    return carve_holes_without_target_change(sample, cfg, target_pose=target)


def build_task_row(
    *,
    sample: Dict,
    root: Path,
    paths: Dict[str, Path],
    args: argparse.Namespace,
    display_size: Tuple[int, int],
    eval_size: int,
) -> Tuple[Dict, Dict[str, object]]:
    sample_id = sample["sample_id"]
    stream = "single"
    target_pose = Pose.from_dict(sample["target_pose"])
    init_pose = Pose.from_dict(sample["init_pose"])

    t_rt = perf_counter()
    target_mask, target_mask_eval, _ = render_views_fast(
        sample,
        target_pose,
        display_size=display_size,
        eval_size=eval_size,
        need_oblique=False,
    )
    dt_rt = perf_counter() - t_rt

    t_ri = perf_counter()
    init_mask, init_mask_eval, init_oblique = render_views_fast(
        sample,
        init_pose,
        display_size=display_size,
        eval_size=eval_size,
        need_oblique=True,
    )
    dt_ri = perf_counter() - t_ri

    target_path = paths["data"] / stream / "target" / f"{sample_id}.png"
    init_path = paths["data"] / stream / "init" / f"{sample_id}.png"
    init_oblique_path = paths["data"] / stream / "init_oblique" / f"{sample_id}.png"
    t_wp = perf_counter()
    if args.save_target_images:
        write_png_gray(target_path, target_mask)
    if args.save_init_images:
        write_png_gray(init_path, init_mask)
    if args.save_init_oblique_images:
        write_png_gray(init_oblique_path, init_oblique)
    dt_wp = perf_counter() - t_wp

    t_pack = perf_counter()
    optimal_steps = pose_distance_steps(target_pose, init_pose, sample["theta"])
    init_iou = iou(target_mask_eval, init_mask_eval)

    task = {
        "sample_id": sample_id,
        "stream_type": stream,
        "theta": sample["theta"],
        "T_max": sample["T_max"],
        "tau_success": sample["tau_success"],
        "target_pose": sample["target_pose"],
        "init_pose": sample["init_pose"],
        "shape_seed": sample["shape_seed"],
        "base_shape": sample["base_shape"],
        "source_dataset": sample["source_dataset"],
        "category": sample["category"],
        "icon_slug": sample["icon_slug"],
        "icon_svg": sample["icon_svg"],
        "icon_points": sample["icon_points"],
        "icon_quality": sample.get("icon_quality", {}),
        "complexity": sample["complexity"],
        "surface_mode": sample["surface_mode"],
        "icon_scale": sample["icon_scale"],
        "icon_center_mode": sample.get("icon_center_mode", "bbox"),
        "sphere_radius": sample["sphere_radius"],
        "sphere_depth_scale": sample.get("sphere_depth_scale", 0.26),
        "sphere_fit_icon": bool(sample.get("sphere_fit_icon", True)),
        "sphere_xy_ratio": float(sample.get("sphere_xy_ratio", 0.72)),
        "sphere_double_sided": bool(sample.get("sphere_double_sided", True)),
        "sphere_solid": bool(sample.get("sphere_solid", True)),
        "sphere_fill_layers": int(sample.get("sphere_fill_layers", 11)),
        "radial_jitter": sample["radial_jitter"],
        "rim_softness": sample["rim_softness"],
        "fine_noise": sample["fine_noise"],
        "surface_holes": sample.get("surface_holes", []),
        "surface_hole_attempts": int(sample.get("surface_hole_attempts", 0)),
        "surface_hole_kept": int(sample.get("surface_hole_kept", 0)),
        "surface_hole_consecutive_failures": int(sample.get("surface_hole_consecutive_failures", 0)),
        "target_image": str(target_path.relative_to(root)).replace("\\", "/") if args.save_target_images else "",
        "init_image": str(init_path.relative_to(root)).replace("\\", "/") if args.save_init_images else "",
        "init_oblique_image": str(init_oblique_path.relative_to(root)).replace("\\", "/")
        if args.save_init_oblique_images
        else "",
        "display_size": {"width": display_size[0], "height": display_size[1]},
        "eval_size": eval_size,
        "optimal_steps": optimal_steps,
        "answer_actions": sample["answer_actions"],
        "scramble_actions": sample["scramble_actions"],
        "init_iou": round(init_iou, 6),
        "meta": sample["meta"],
    }
    dt_pack = perf_counter() - t_pack

    return task, {
        "render_target_s": dt_rt,
        "render_init_s": dt_ri,
        "write_png_s": dt_wp,
        "task_pack_s": dt_pack,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    paths = resolve_paths(root)

    cfg_path = root / args.config
    if cfg_path.exists():
        cfg = read_json(cfg_path)
    else:
        cfg = load_default_config()
    if int(args.sample_count) > 0:
        cfg["sample_count"] = int(args.sample_count)
        write_json(cfg_path, cfg)

    if "icon_categories" not in cfg:
        cfg["icon_categories"] = ["all"]
        write_json(cfg_path, cfg)
    debug_timing = bool(cfg.get("debug_timing", False))
    timing_report: Dict[str, object] = {}

    runtime_caps = {
        "numpy": bool(getattr(core, "np", None) is not None),
        "numba": bool(getattr(core, "njit", None) is not None),
        "opencv": bool(getattr(core, "cv2", None) is not None),
    }
    fast_render_enabled = runtime_caps["numpy"] and runtime_caps["numba"] and runtime_caps["opencv"]
    print(
        "[runtime][step01] "
        f"numpy={runtime_caps['numpy']} "
        f"numba={runtime_caps['numba']} "
        f"opencv={runtime_caps['opencv']} "
        f"fast_render={fast_render_enabled}"
    )
    if not fast_render_enabled:
        print(
            "[runtime][step01][warn] "
            "Fast render path disabled. This can make Generate single-stream very slow."
        )
    timing_report["runtime_caps"] = runtime_caps
    timing_report["fast_render_enabled"] = bool(fast_render_enabled)

    pool_limit = int(cfg.get("icon_pool_limit", 1200))
    t_icon_db = perf_counter()
    icon_db, icon_db_timing = build_icon_db(
        root,
        cfg.get("icon_categories", ["all"]),
        pool_limit=pool_limit,
        quality_cfg=cfg,
    )
    _print_icon_filter_report(cfg, icon_db_timing, icon_db)
    timing_report["build_icon_db_total_s"] = round(perf_counter() - t_icon_db, 6)
    timing_report["build_icon_db"] = icon_db_timing

    single_path = paths["data"] / "single" / "samples.jsonl"
    if args.save_target_images:
        ensure_dir(paths["data"] / "single" / "target")
    if args.save_init_images:
        ensure_dir(paths["data"] / "single" / "init")
    if args.save_init_oblique_images:
        ensure_dir(paths["data"] / "single" / "init_oblique")
    eval_size = STANDARD_IMAGE_SIZE
    display_size = STANDARD_DISPLAY_SIZE
    checkpoint_path = paths["reports"] / "step_01_checkpoint.json"
    sample_count = int(cfg.get("sample_count", 0))
    if sample_count <= 0:
        sample_count = 1000

    icon_pool = build_unique_icon_pool(cfg, icon_db=icon_db)
    icon_passed_by_category = {
        cat: int(icon_db_timing.get("categories", {}).get(cat, {}).get("passed", 0))
        for cat in cfg.get("icon_categories", [])
    }
    # The pool keeps only cache references; release all decoded points before rendering.
    del icon_db
    if sample_count > len(icon_pool):
        raise RuntimeError(
            f"Requested sample_count={sample_count}, but unique icons={len(icon_pool)} "
            f"for categories={cfg.get('icon_categories', ['all'])}. Increase categories or reduce sample_count."
        )

    checkpoint = load_checkpoint(checkpoint_path) if args.resume else {}
    cfg_fp = config_fingerprint(cfg)
    completed_sample_ids: List[str] = []
    used_icon_slugs: List[str] = []
    completed_count = 0
    rng = random.Random(cfg["random_seed"] + 1)
    icon_order_slugs: List[str] = []
    icon_lookup = {str(item["icon_slug"]): item for item in icon_pool}
    icon_cursor = 0
    next_sample_index = 0
    skipped_trivial_init_count = 0

    if args.resume and checkpoint:
        if checkpoint.get("config_fingerprint", "") != cfg_fp:
            raise RuntimeError("Checkpoint config fingerprint mismatch. Refuse to resume with different Step1 config.")
        prev_target = int(checkpoint.get("sample_count_target", 0))
        if prev_target > sample_count:
            raise RuntimeError(
                f"Checkpoint target sample_count={prev_target} is larger than current requested sample_count={sample_count}."
            )
        icon_order_slugs = [str(x) for x in checkpoint.get("icon_order", [])]
        if len(icon_order_slugs) != len(icon_pool):
            raise RuntimeError("Checkpoint icon order size mismatch. Refuse to resume.")
        missing = [slug for slug in icon_order_slugs if slug not in icon_lookup]
        if missing:
            raise RuntimeError(f"Checkpoint icon order contains missing slugs, e.g. {missing[:3]}")
        rng.setstate(decode_rng_state(str(checkpoint.get("rng_state", ""))))
        completed_sample_ids, used_icon_slugs = scan_sample_metadata(single_path)
        task_rows = count_jsonl_rows(paths["tasks"])
        completed_count = min(
            int(checkpoint.get("completed_count", 0)),
            len(completed_sample_ids),
            task_rows,
        )
        # A process may stop between the two JSONL appends and the checkpoint write.
        # Reconcile both files to the last fully checkpointed pair before continuing.
        if len(completed_sample_ids) != completed_count:
            truncate_jsonl_rows(single_path, completed_count)
        if task_rows != completed_count:
            truncate_jsonl_rows(paths["tasks"], completed_count)
        checkpoint_count = int(checkpoint.get("completed_count", 0))
        if completed_count != checkpoint_count:
            raise RuntimeError(
                "Checkpoint is ahead of one or both JSONL files; RNG state cannot be recovered safely. "
                "Restart with --no-resume."
            )
        icon_cursor = int(checkpoint.get("icon_cursor", completed_count))
        next_sample_index = int(checkpoint.get("next_sample_index", completed_count))
        skipped_trivial_init_count = int(checkpoint.get("skipped_trivial_init_count", 0))
        checkpoint["sample_count_target"] = sample_count
        completed_sample_ids = completed_sample_ids[:completed_count]
        used_icon_slugs = used_icon_slugs[:completed_count]
    else:
        rng.shuffle(icon_pool)
        icon_order_slugs = [str(item["icon_slug"]) for item in icon_pool]
        write_jsonl(single_path, [])
        write_jsonl(paths["tasks"], [])
        checkpoint = {
            "version": 1,
            "mode": "step01_single_stream_resume",
            "config_fingerprint": cfg_fp,
            "sample_count_target": sample_count,
            "completed_count": 0,
            "icon_cursor": 0,
            "next_sample_index": 0,
            "used_icon_slugs": [],
            "completed_sample_ids": [],
            "icon_order": icon_order_slugs,
            "skipped_trivial_init_count": 0,
            "rng_state": encode_rng_state(rng.getstate()),
            "samples_file": str(single_path.relative_to(root)).replace("\\", "/"),
            "tasks_file": str(paths["tasks"].relative_to(root)).replace("\\", "/"),
        }
        write_json(checkpoint_path, checkpoint)

    icon_pool = [icon_lookup[slug] for slug in icon_order_slugs]
    render_timing = {
        "render_target_s": 0.0,
        "render_init_s": 0.0,
        "write_png_s": 0.0,
        "task_pack_s": 0.0,
        "total_loop_s": 0.0,
        "tasks": sample_count,
        "completed_before_resume": completed_count,
        "skipped_trivial_init_before_resume": skipped_trivial_init_count,
    }
    per_sample_timing: List[Dict[str, float]] = []
    t_render_loop = perf_counter()

    t_gen_samples = 0.0
    t_write_samples = 0.0
    progress = tqdm(total=sample_count, initial=completed_count, desc="Generate+render tasks", unit="task")
    while completed_count < sample_count:
        if icon_cursor >= len(icon_pool):
            raise RuntimeError(
                f"Step1 exhausted unique icon pool at accepted={completed_count}, skipped_trivial_init={skipped_trivial_init_count}. "
                "Reduce sample_count or loosen icon filters."
            )
        t_sample = perf_counter()
        icon = materialize_icon_points(root, icon_pool[icon_cursor])
        sample = generate_single_sample(next_sample_index, cfg, icon, rng)
        t_gen_samples += perf_counter() - t_sample

        task, sample_t = build_task_row(
            sample=sample,
            root=root,
            paths=paths,
            args=args,
            display_size=display_size,
            eval_size=eval_size,
        )
        render_timing["render_target_s"] += float(sample_t["render_target_s"])
        render_timing["render_init_s"] += float(sample_t["render_init_s"])
        render_timing["write_png_s"] += float(sample_t["write_png_s"])
        render_timing["task_pack_s"] += float(sample_t["task_pack_s"])

        icon_cursor += 1
        tau = float(task.get("tau_success", cfg.get("tau_success", 0.9)))
        if float(task.get("init_iou", 0.0)) >= tau:
            skipped_trivial_init_count += 1
            if args.save_target_images:
                target_path = root / task.get("target_image", "")
                if str(task.get("target_image", "")).strip() and target_path.exists():
                    target_path.unlink()
            if args.save_init_images:
                init_path = root / task.get("init_image", "")
                if str(task.get("init_image", "")).strip() and init_path.exists():
                    init_path.unlink()
            if args.save_init_oblique_images:
                init_oblique_path = root / task.get("init_oblique_image", "")
                if str(task.get("init_oblique_image", "")).strip() and init_oblique_path.exists():
                    init_oblique_path.unlink()
            checkpoint["completed_count"] = completed_count
            checkpoint["icon_cursor"] = icon_cursor
            checkpoint["next_sample_index"] = next_sample_index
            checkpoint["skipped_trivial_init_count"] = skipped_trivial_init_count
            checkpoint["rng_state"] = encode_rng_state(rng.getstate())
            write_json(checkpoint_path, checkpoint)
            if debug_timing:
                dt_sample = perf_counter() - t_sample
                print(
                    f"[timing][step01][skip][{sample['sample_id']}] "
                    f"init_iou={float(task.get('init_iou', 0.0)):.4f} "
                    f"tau={tau:.4f} total_ms={dt_sample * 1000:.2f}"
                )
            continue

        t_ws = perf_counter()
        append_jsonl(single_path, sample)
        t_write_samples += perf_counter() - t_ws
        append_jsonl(paths["tasks"], task)
        completed_sample_ids.append(str(sample["sample_id"]))
        used_icon_slugs.append(str(sample["icon_slug"]))

        completed_count += 1
        next_sample_index += 1
        progress.update(1)

        checkpoint["completed_count"] = completed_count
        checkpoint["icon_cursor"] = icon_cursor
        checkpoint["next_sample_index"] = next_sample_index
        checkpoint["used_icon_slugs"] = used_icon_slugs
        checkpoint["completed_sample_ids"] = completed_sample_ids
        checkpoint["skipped_trivial_init_count"] = skipped_trivial_init_count
        checkpoint["rng_state"] = encode_rng_state(rng.getstate())
        write_json(checkpoint_path, checkpoint)

        if debug_timing:
            dt_sample = perf_counter() - t_sample
            sample_timing = {
                "sample_id": sample["sample_id"],
                "render_target_s": round(float(sample_t["render_target_s"]), 6),
                "render_init_s": round(float(sample_t["render_init_s"]), 6),
                "write_png_s": round(float(sample_t["write_png_s"]), 6),
                "task_pack_s": round(float(sample_t["task_pack_s"]), 6),
                "total_s": round(dt_sample, 6),
            }
            per_sample_timing.append(sample_timing)
            print(
                f"[timing][step01][{sample['sample_id']}] "
                f"target_render_ms={float(sample_t['render_target_s']) * 1000:.2f} "
                f"init_render_ms={float(sample_t['render_init_s']) * 1000:.2f} "
                f"write_png_ms={float(sample_t['write_png_s']) * 1000:.2f} "
                f"task_pack_ms={float(sample_t['task_pack_s']) * 1000:.2f} "
                f"total_ms={dt_sample * 1000:.2f}"
            )
    progress.close()

    render_timing["total_loop_s"] = round(perf_counter() - t_render_loop, 6)
    timing_report["generate_single_stream_s"] = round(t_gen_samples, 6)
    timing_report["write_samples_jsonl_s"] = round(t_write_samples, 6)
    timing_report["render_tasks"] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in render_timing.items()}
    if debug_timing:
        timing_report["render_tasks"]["per_sample"] = per_sample_timing
        if per_sample_timing:
            avg_total_s = sum(s["total_s"] for s in per_sample_timing) / len(per_sample_timing)
            slowest = sorted(per_sample_timing, key=lambda x: x["total_s"], reverse=True)[:3]
            timing_report["render_tasks"]["avg_sample_total_s"] = round(avg_total_s, 6)
            timing_report["render_tasks"]["slowest_samples_top3"] = slowest
            print(f"[timing][step01][summary] avg_sample_total_ms={avg_total_s * 1000:.2f}")
            for item in slowest:
                print(
                    f"[timing][step01][slow] sample={item['sample_id']} "
                    f"total_ms={item['total_s'] * 1000:.2f}"
                )

    timing_report["write_tasks_jsonl_s"] = 0.0

    summary = {
        "total_tasks": completed_count,
        "single_tasks": completed_count,
        "engine_mode": "single_stream_spherical_icon_inverse_path",
        "source_dataset": "game_icons",
        "display_size": {"width": display_size[0], "height": display_size[1]},
        "eval_size": eval_size,
        "icon_categories": cfg.get("icon_categories", []),
        "icon_filter_thresholds": {
            "icon_min_connected_ratio": float(cfg.get("icon_min_connected_ratio", 0.985)),
            "icon_min_circle_iou": float(cfg.get("icon_min_circle_iou", 0.28)),
            "icon_min_bbox_fill": float(cfg.get("icon_min_bbox_fill", 0.08)),
            "icon_max_aspect_ratio": float(cfg.get("icon_max_aspect_ratio", 2.2)),
            "icon_min_pixel_points": int(cfg.get("icon_min_pixel_points", 1800)),
        },
        "icon_filter_counts": {
            "per_category": {
                cat: {
                    "candidates": int(icon_db_timing.get("categories", {}).get(cat, {}).get("candidates", 0)),
                    "passed": int(icon_passed_by_category.get(cat, 0)),
                }
                for cat in cfg.get("icon_categories", [])
            },
            "total_candidates": int(
                sum(int(icon_db_timing.get("categories", {}).get(cat, {}).get("candidates", 0)) for cat in cfg.get("icon_categories", []))
            ),
            "total_passed": int(
                sum(
                    int(icon_passed_by_category.get(cat, 0))
                    for cat in cfg.get("icon_categories", [])
                )
            ),
        },
        "save_target_images": bool(args.save_target_images),
        "save_init_images": bool(args.save_init_images),
        "save_init_oblique_images": bool(args.save_init_oblique_images),
        "tasks_file": str(paths["tasks"].relative_to(root)).replace("\\", "/"),
        "resume_enabled": bool(args.resume),
        "checkpoint_file": str(checkpoint_path.relative_to(root)).replace("\\", "/"),
        "used_icon_slugs_count": len(set(used_icon_slugs)),
        "skipped_trivial_init_count": skipped_trivial_init_count,
    }
    write_json(paths["reports"] / "step_01_summary.json", summary)
    if debug_timing:
        write_json(paths["reports"] / "step_01_timing.json", timing_report)
        summary["timing_report"] = str((paths["reports"] / "step_01_timing.json").relative_to(root)).replace("\\", "/")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
