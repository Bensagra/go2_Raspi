"""Lazy adapters around the official TULIP and AdaPoinTr repositories.

The projects are intentionally not vendored here. The server receives their
repository and checkpoint paths, imports the official model definitions once,
and keeps the models resident for repeated inference.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _load_torch_checkpoint(torch: Any, path: Path) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before the weights_only argument
        checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"unsupported checkpoint structure: {path}")
    return checkpoint


def _install_pointnet2_fallback(torch: Any) -> bool:
    """Install the small PointNet2 subset AdaPoinTr needs for inference.

    The official repository normally compiles pointnet2_ops against CUDA. That
    extension is fastest, but difficult to reproduce across current PyTorch/CUDA
    combinations. This torch-only fallback keeps inference functional (and also
    permits CPU smoke tests); the compiled extension is used whenever available.
    """
    try:
        importlib.import_module("pointnet2_ops.pointnet2_utils")
        return False
    except Exception:
        sys.modules.pop("pointnet2_ops", None)
        sys.modules.pop("pointnet2_ops.pointnet2_utils", None)

    package = types.ModuleType("pointnet2_ops")
    utils = types.ModuleType("pointnet2_ops.pointnet2_utils")

    def furthest_point_sample(xyz: Any, npoint: int) -> Any:
        batch, count, _ = xyz.shape
        if npoint > count:
            raise ValueError(f"FPS requested {npoint} points from a cloud of {count}")
        centroids = torch.zeros((batch, npoint), dtype=torch.long, device=xyz.device)
        distance = torch.full((batch, count), float("inf"), device=xyz.device)
        farthest = torch.zeros((batch,), dtype=torch.long, device=xyz.device)
        batch_indices = torch.arange(batch, dtype=torch.long, device=xyz.device)
        for index in range(npoint):
            centroids[:, index] = farthest
            centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
            candidate_distance = torch.sum((xyz - centroid) ** 2, dim=-1)
            distance = torch.minimum(distance, candidate_distance)
            farthest = torch.max(distance, dim=-1).indices
        return centroids

    def gather_operation(features: Any, indices: Any) -> Any:
        expanded = indices.unsqueeze(1).expand(-1, features.shape[1], -1)
        return torch.gather(features, 2, expanded)

    def three_nn(unknown: Any, known: Any) -> Any:
        distances = torch.cdist(unknown, known, p=2)
        values, indices = torch.topk(
            distances, k=3, dim=-1, largest=False, sorted=False
        )
        return values, indices

    def three_interpolate(features: Any, indices: Any, weight: Any) -> Any:
        batch, channels, _ = features.shape
        target_count = indices.shape[1]
        expanded_features = features.unsqueeze(2).expand(-1, -1, target_count, -1)
        expanded_indices = indices.unsqueeze(1).expand(-1, channels, -1, -1)
        neighbors = torch.gather(expanded_features, 3, expanded_indices)
        return torch.sum(neighbors * weight.unsqueeze(1), dim=-1)

    utils.furthest_point_sample = furthest_point_sample
    utils.gather_operation = gather_operation
    utils.three_nn = three_nn
    utils.three_interpolate = three_interpolate
    package.pointnet2_utils = utils
    sys.modules["pointnet2_ops"] = package
    sys.modules["pointnet2_ops.pointnet2_utils"] = utils
    return True


def _install_chamfer_inference_stub(torch: Any) -> bool:
    """Avoid requiring AdaPoinTr's training-only compiled Chamfer extension."""
    try:
        importlib.import_module("extensions.chamfer_dist")
        return False
    except Exception:
        sys.modules.pop("extensions.chamfer_dist", None)

    module = types.ModuleType("extensions.chamfer_dist")

    class ChamferDistanceL1(torch.nn.Module):
        def forward(self, xyz1: Any, xyz2: Any) -> Any:
            distance = torch.cdist(xyz1, xyz2, p=2)
            return (
                distance.min(dim=2).values.mean() + distance.min(dim=1).values.mean()
            ) / 2

    module.ChamferDistanceL1 = ChamferDistanceL1
    module.ChamferDistanceL1_PM = ChamferDistanceL1
    module.ChamferDistanceL2 = ChamferDistanceL1
    module.ChamferDistanceL2_split = ChamferDistanceL1
    sys.modules["extensions.chamfer_dist"] = module
    return True


class AdaPoinTrBackend:
    """Long-lived inference wrapper for the official yuxumin/PoinTr model."""

    def __init__(
        self,
        repository: str,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda:0",
        input_points: int = 2048,
    ) -> None:
        self.repository = Path(repository).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        for path, label in (
            (self.repository / "models" / "AdaPoinTr.py", "AdaPoinTr repository"),
            (self.config_path, "AdaPoinTr config"),
            (self.checkpoint_path, "AdaPoinTr checkpoint"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")
        if input_points < 512:
            raise ValueError("AdaPoinTr input_points must be >= 512")

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("AdaPoinTr requires PyTorch on the server") from exc
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"AdaPoinTr requested {device}, but CUDA is unavailable")
        self.input_points = int(input_points)

        if str(self.repository) not in sys.path:
            sys.path.insert(0, str(self.repository))

        # Bind the repository's namespace packages explicitly. This prevents an
        # unrelated site-package named ``utils`` or ``extensions`` from winning
        # Python's import resolution on a long-lived ML server.
        for package_name in ("utils", "extensions"):
            package_path = self.repository / package_name
            current_package = sys.modules.get(package_name)
            if current_package is None:
                package = types.ModuleType(package_name)
                package.__path__ = [str(package_path)]
                package.__package__ = package_name
                sys.modules[package_name] = package
            elif str(package_path) not in list(
                getattr(current_package, "__path__", [])
            ):
                raise RuntimeError(
                    f"another Python package named '{package_name}' is already loaded"
                )
        pointnet_fallback = _install_pointnet2_fallback(torch)
        chamfer_stub = _install_chamfer_inference_stub(torch)

        # Import only AdaPoinTr. The official models/__init__.py eagerly imports
        # unrelated baselines and their extra compiled extensions.
        current_models = sys.modules.get("models")
        if current_models is None:
            models_package = types.ModuleType("models")
            models_package.__path__ = [str(self.repository / "models")]
            models_package.__package__ = "models"
            sys.modules["models"] = models_package
        elif str(self.repository / "models") not in list(
            getattr(current_models, "__path__", [])
        ):
            raise RuntimeError(
                "another Python package named 'models' is already loaded"
            )

        config_module = importlib.import_module("utils.config")
        build_module = importlib.import_module("models.build")
        importlib.import_module("models.AdaPoinTr")
        previous_directory = Path.cwd()
        try:
            os.chdir(self.repository)
            config = config_module.cfg_from_yaml_file(str(self.config_path))
        finally:
            os.chdir(previous_directory)
        self.model = build_module.build_model_from_cfg(config.model)

        checkpoint = _load_torch_checkpoint(torch, self.checkpoint_path)
        raw_state = checkpoint.get("model", checkpoint.get("base_model"))
        if not isinstance(raw_state, dict):
            raise RuntimeError("AdaPoinTr checkpoint has no model/base_model state")
        state = {key.replace("module.", ""): value for key, value in raw_state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.using_pointnet_fallback = pointnet_fallback
        self.using_chamfer_stub = chamfer_stub

    def _sample_input(self, points: np.ndarray) -> np.ndarray:
        cloud = np.asarray(points, dtype=np.float32)
        cloud = cloud[np.isfinite(cloud).all(axis=1), :3]
        if cloud.shape[0] == 0:
            raise ValueError("AdaPoinTr input cloud is empty")
        rng = np.random.default_rng(0)
        if cloud.shape[0] >= self.input_points:
            indices = rng.choice(cloud.shape[0], self.input_points, replace=False)
            return np.ascontiguousarray(cloud[indices])
        repeats = self.input_points // cloud.shape[0]
        remainder = self.input_points % cloud.shape[0]
        chunks = [np.tile(cloud, (repeats, 1))] if repeats else []
        if remainder:
            chunks.append(cloud[rng.choice(cloud.shape[0], remainder, replace=False)])
        return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)

    def complete(self, points: np.ndarray) -> np.ndarray:
        """Complete one normalized region and return points in its original frame."""
        sampled = self._sample_input(points)
        center = np.mean(sampled, axis=0, dtype=np.float64).astype(np.float32)
        centered = sampled - center
        radius = float(np.max(np.linalg.norm(centered, axis=1)))
        if not math.isfinite(radius) or radius < 1e-6:
            raise ValueError("AdaPoinTr input region has no spatial extent")
        normalized = centered / radius
        tensor = self.torch.from_numpy(normalized).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            output = self.model(tensor)
        dense = output[-1].squeeze(0).detach().float().cpu().numpy()
        dense = dense * radius + center
        dense = dense[np.isfinite(dense).all(axis=1)]
        return np.ascontiguousarray(dense, dtype=np.float32)


class TulipBackend:
    """Experimental KITTI-range-image adapter for the official ETHZ TULIP model.

    Unitree exposes an odometry-frame voxel map rather than the organized scan
    TULIP was trained on. This adapter projects the newest local map to KITTI's
    16x1024 geometry, applies the official 4x model, then transforms predictions
    back to Unitree odometry coordinates. It should be fine-tuned on L1 data for
    measurements where metric fidelity matters.
    """

    LOW_ROWS = 16
    HIGH_ROWS = 64
    COLUMNS = 1024
    MAX_RANGE = 80.0
    VERTICAL_MIN_DEG = -24.8
    VERTICAL_SPAN_DEG = 26.8

    def __init__(
        self,
        repository: str,
        checkpoint_path: str,
        device: str = "cuda:0",
        minimum_range: float = 0.3,
    ) -> None:
        self.repository = Path(repository).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not (self.repository / "tulip" / "model" / "tulip.py").exists():
            raise FileNotFoundError(f"TULIP repository not found: {self.repository}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"TULIP checkpoint not found: {self.checkpoint_path}"
            )
        if minimum_range <= 0 or minimum_range >= self.MAX_RANGE:
            raise ValueError("invalid TULIP minimum range")
        self.minimum_range = float(minimum_range)

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("TULIP requires PyTorch on the server") from exc
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"TULIP requested {device}, but CUDA is unavailable")

        import_root = self.repository / "tulip"
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
        model_module = importlib.import_module("model.tulip")
        self.model = model_module.tulip_base(
            img_size=(self.LOW_ROWS, self.COLUMNS),
            target_img_size=(self.HIGH_ROWS, self.COLUMNS),
            patch_size=(1, 4),
            in_chans=1,
            window_size=[2, 8],
            swin_v2=False,
            pixel_shuffle=True,
            circular_padding=True,
            log_transform=True,
            patch_unmerging=True,
        )
        checkpoint = _load_torch_checkpoint(torch, self.checkpoint_path)
        raw_state = checkpoint.get("model", checkpoint)
        if not isinstance(raw_state, dict):
            raise RuntimeError("TULIP checkpoint has no model state")
        state = {key.replace("module.", ""): value for key, value in raw_state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _world_to_body(points: np.ndarray, pose: Dict[str, float]) -> np.ndarray:
        position = np.array(
            [float(pose.get("x", 0)), float(pose.get("y", 0)), float(pose.get("z", 0))],
            dtype=np.float32,
        )
        yaw = float(pose.get("yaw", 0))
        delta = points[:, :3] - position
        cosine, sine = math.cos(yaw), math.sin(yaw)
        return np.column_stack(
            (
                cosine * delta[:, 0] + sine * delta[:, 1],
                -sine * delta[:, 0] + cosine * delta[:, 1],
                delta[:, 2],
            )
        ).astype(np.float32)

    @staticmethod
    def _body_to_world(points: np.ndarray, pose: Dict[str, float]) -> np.ndarray:
        yaw = float(pose.get("yaw", 0))
        cosine, sine = math.cos(yaw), math.sin(yaw)
        return np.column_stack(
            (
                cosine * points[:, 0] - sine * points[:, 1] + float(pose.get("x", 0)),
                sine * points[:, 0] + cosine * points[:, 1] + float(pose.get("y", 0)),
                points[:, 2] + float(pose.get("z", 0)),
            )
        ).astype(np.float32)

    def _to_low_range_image(self, body_points: np.ndarray) -> np.ndarray:
        radius = np.linalg.norm(body_points, axis=1)
        valid = (
            np.isfinite(radius)
            & (radius >= self.minimum_range)
            & (radius <= self.MAX_RANGE)
        )
        points = body_points[valid]
        radius = radius[valid]
        if points.shape[0] == 0:
            raise ValueError("no Unitree points fall inside the TULIP/KITTI range")
        elevation = np.degrees(np.arcsin(np.clip(points[:, 2] / radius, -1, 1)))
        high_row = np.rint(
            (elevation - self.VERTICAL_MIN_DEG)
            * (self.HIGH_ROWS - 1)
            / self.VERTICAL_SPAN_DEG
        ).astype(np.int32)
        low_row = np.rint(high_row / 4.0).astype(np.int32)
        horizontal = np.degrees(np.arctan2(points[:, 0], points[:, 1]))
        column = (
            np.rint(
                (90.0 - horizontal) * self.COLUMNS / 360.0 + self.COLUMNS / 2.0 - 1.0
            ).astype(np.int32)
            % self.COLUMNS
        )
        valid_pixel = (
            (high_row >= 0)
            & (high_row < self.HIGH_ROWS)
            & (low_row >= 0)
            & (low_row < self.LOW_ROWS)
        )
        flat = np.full(self.LOW_ROWS * self.COLUMNS, np.inf, dtype=np.float32)
        pixel = low_row[valid_pixel] * self.COLUMNS + column[valid_pixel]
        np.minimum.at(flat, pixel, radius[valid_pixel])
        image = flat.reshape(self.LOW_ROWS, self.COLUMNS)
        image[~np.isfinite(image)] = 0
        return image

    def _high_range_to_points(self, image: np.ndarray) -> np.ndarray:
        row, column = np.nonzero(
            (image >= self.minimum_range) & (image <= self.MAX_RANGE)
        )
        radius = image[row, column]
        vertical = np.deg2rad(
            row * self.VERTICAL_SPAN_DEG / (self.HIGH_ROWS - 1) + self.VERTICAL_MIN_DEG
        )
        horizontal = np.deg2rad(
            -(column + 1 - self.COLUMNS / 2.0) * 360.0 / self.COLUMNS + 90.0
        )
        points = np.column_stack(
            (
                np.sin(horizontal) * np.cos(vertical) * radius,
                np.cos(horizontal) * np.cos(vertical) * radius,
                np.sin(vertical) * radius,
            )
        )
        return np.ascontiguousarray(points, dtype=np.float32)

    def upsample_world(self, points: np.ndarray, pose: Dict[str, float]) -> np.ndarray:
        body = self._world_to_body(np.asarray(points, dtype=np.float32), pose)
        low_meters = self._to_low_range_image(body)
        normalized = low_meters / self.MAX_RANGE
        model_input = np.log1p(normalized).astype(np.float32)
        tensor = self.torch.from_numpy(model_input)[None, None].to(self.device)
        target = self.torch.zeros(
            (1, 1, self.HIGH_ROWS, self.COLUMNS),
            dtype=tensor.dtype,
            device=self.device,
        )
        with self.torch.inference_mode():
            prediction, _, _ = self.model(tensor, target, eval=True)
        high_normalized = self.torch.expm1(prediction).squeeze().float().cpu().numpy()
        high_normalized = np.clip(high_normalized, 0, 1)
        # Preserve every measured low-resolution beam exactly, as in the paper's
        # official evaluation routine.
        high_normalized[0::4, :] = normalized
        predicted_body = self._high_range_to_points(high_normalized * self.MAX_RANGE)
        return self._body_to_world(predicted_body, pose)
