"""Frozen render protocol values shared by active benchmark entrypoints."""

STANDARD_IMAGE_SIZE = 320
STANDARD_DISPLAY_SIZE = (STANDARD_IMAGE_SIZE, STANDARD_IMAGE_SIZE)
STANDARD_POINT_BATCH_SIZE = 100_000
STANDARD_CANDIDATE_BATCH_SIZE = 256
STANDARD_PANEL_SEPARATOR = 6
STANDARD_PANEL_SEPARATOR_VALUE = 12

RENDER_PARAMETER_SCHEMA = {
    "image_size": [STANDARD_IMAGE_SIZE, STANDARD_IMAGE_SIZE],
    "projection_scale_reference": 64.0,
    "geometry_point_sampling": {
        "svg_max_points_per_icon": 90000,
        "cuda_point_batch_size": STANDARD_POINT_BATCH_SIZE,
        "beam_candidate_batch_size": STANDARD_CANDIDATE_BATCH_SIZE,
    },
    "accelerated_mask_fill": "solid_geometry_disk_splat",
    "python_fallback_display_fill_holes": False,
    "python_fallback_evaluation_fill_holes": True,
    "mask_morphology": {
        "operation": "MORPH_CLOSE",
        "kernel": "ELLIPSE",
        "kernel_size": [3, 3],
        "iterations": 1,
    },
    "oblique_blur": {"operation": "GaussianBlur", "kernel_size": [3, 3], "passes": 2},
    "renderer_backend_policy": "cuda_then_existing_cpu_fallback",
    "formal_entrypoint_backend_requirement": "cuda",
    "panel_separator_pixels": STANDARD_PANEL_SEPARATOR,
    "panel_separator_value": STANDARD_PANEL_SEPARATOR_VALUE,
}


def require_standard_square_size(value, *, name="image_size"):
    """Reject protocol drift instead of silently rendering a different size."""
    if isinstance(value, (tuple, list)):
        actual = tuple(int(item) for item in value)
        expected = STANDARD_DISPLAY_SIZE
    else:
        actual = int(value)
        expected = STANDARD_IMAGE_SIZE
    if actual != expected:
        raise ValueError(f"{name} is frozen at 320x320; got {actual!r}")
    return value


def require_cuda_backend(actual_backend, *, context="formal renderer"):
    if actual_backend != "cuda":
        raise RuntimeError(
            f"{context} requires CUDA rendering; actual backend={actual_backend!r}. "
            "Run with the proj environment and a CUDA-capable GPU."
        )
