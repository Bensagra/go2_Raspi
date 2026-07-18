#!/usr/bin/env python3
"""Run a deterministic AdaPoinTr inference smoke test on the configured server."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.lidar_models import AdaPoinTrBackend  # noqa: E402


def synthetic_partial_cloud(point_count: int) -> np.ndarray:
    """Create a stable partial surface with metric scale and nonzero extent."""
    rng = np.random.default_rng(20260717)
    angles = rng.uniform(-0.85 * np.pi, 0.45 * np.pi, point_count)
    heights = rng.uniform(-0.7, 0.9, point_count)
    radii = 1.25 + rng.normal(0.0, 0.035, point_count)
    return np.column_stack(
        (radii * np.cos(angles), radii * np.sin(angles), heights)
    ).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository", default=str(PROJECT_ROOT / "third_party" / "PoinTr")
    )
    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT
            / "third_party"
            / "PoinTr"
            / "cfgs"
            / "Projected_ShapeNet55_models"
            / "AdaPoinTr.yaml"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "models" / "AdaPoinTr_Projected_ShapeNet55.pth"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-points", type=int, default=6000)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.source_points < 512:
        parser.error("--source-points must be >= 512")
    return args


def main() -> None:
    args = parse_args()
    source = synthetic_partial_cloud(args.source_points)
    started = time.perf_counter()
    backend = AdaPoinTrBackend(
        repository=args.repository,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        input_points=2048,
    )
    load_seconds = time.perf_counter() - started
    if backend.device.type == "cuda":
        backend.torch.cuda.reset_peak_memory_stats(backend.device)
    started = time.perf_counter()
    completed = backend.complete(source)
    inference_seconds = time.perf_counter() - started

    if completed.shape != (8192, 3):
        raise RuntimeError(f"unexpected AdaPoinTr output shape: {completed.shape}")
    if not np.isfinite(completed).all():
        raise RuntimeError("AdaPoinTr output contains non-finite coordinates")
    if float(np.ptp(completed, axis=0).min()) <= 0.01:
        raise RuntimeError("AdaPoinTr output is spatially degenerate")

    if args.output:
        target = Path(args.output).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, observed=source, completed=completed)

    peak_mib = 0.0
    gpu = None
    if backend.device.type == "cuda":
        peak_mib = backend.torch.cuda.max_memory_allocated(backend.device) / 1024**2
        gpu = backend.torch.cuda.get_device_name(backend.device)
    print(
        json.dumps(
            {
                "status": "PASS",
                "device": str(backend.device),
                "gpu": gpu,
                "input_points": int(source.shape[0]),
                "output_points": int(completed.shape[0]),
                "finite": True,
                "load_seconds": round(load_seconds, 3),
                "inference_seconds": round(inference_seconds, 3),
                "cuda_peak_mib": round(peak_mib, 1),
                "pointnet2_fallback": backend.using_pointnet_fallback,
                "output": str(Path(args.output).resolve()) if args.output else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
