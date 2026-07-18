#!/usr/bin/env python3
"""Standalone LiDAR receiver, completion pipeline, and 3D model exporter.

The server keeps measured Unitree points separate from model predictions:

* ``observed/latest.*`` is only sensor data accumulated in a voxel map.
* ``tulip/latest.ply`` is the optional experimental range-image upsampling.
* ``completed/latest.*`` is observed data plus AdaPoinTr predictions.
* ``mesh/latest.ply`` is an optional Open3D surface from the completed cloud.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hmac
import json
import os
import re
import signal
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lidar3d.protocol import ProtocolError, decode_cloud_frame  # noqa: E402

try:
    import websockets
except ImportError as exc:  # pragma: no cover - clearer deployment failure
    raise SystemExit(
        "Missing websockets. Install with: ./go2 install lidar-server"
    ) from exc

try:
    from lidar_models import AdaPoinTrBackend, TulipBackend
except ImportError:
    from server.lidar_models import AdaPoinTrBackend, TulipBackend


ROBOT_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def _atomic_path(target: Path) -> Path:
    return target.with_name(f".{target.stem}.{uuid.uuid4().hex}{target.suffix}")


def write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_path(path)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_npz_atomic(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_path(path)
    with temporary.open("wb") as file_handle:
        np.savez_compressed(file_handle, points=np.asarray(points, dtype="<f4"))
    os.replace(temporary, path)


def write_binary_ply_atomic(path: Path, points: np.ndarray) -> None:
    cloud = np.ascontiguousarray(points[:, :3], dtype="<f4")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment go2_Raspi standalone LiDAR 3D pipeline\n"
        f"element vertex {cloud.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_path(path)
    with temporary.open("wb") as file_handle:
        file_handle.write(header)
        file_handle.write(cloud.tobytes(order="C"))
    os.replace(temporary, path)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float32)
    cloud = cloud[np.isfinite(cloud).all(axis=1), :3]
    if cloud.shape[0] == 0 or voxel_size <= 0:
        return np.ascontiguousarray(cloud)
    keys = np.floor(cloud / voxel_size).astype(np.int32)
    _, index = np.unique(keys, axis=0, return_index=True)
    return np.ascontiguousarray(cloud[np.sort(index)], dtype=np.float32)


def bounds(points: np.ndarray) -> Dict[str, List[float]]:
    if points.shape[0] == 0:
        return {"min": [], "max": []}
    return {
        "min": [round(float(value), 5) for value in points.min(axis=0)],
        "max": [round(float(value), 5) for value in points.max(axis=0)],
    }


class RobotVoxelMap:
    def __init__(
        self,
        voxel_size: float,
        max_voxels: int,
        minimum_z: Optional[float],
        maximum_z: Optional[float],
    ) -> None:
        self.voxel_size = voxel_size
        self.max_voxels = max_voxels
        self.minimum_z = minimum_z
        self.maximum_z = maximum_z
        self.lock = threading.Lock()
        self.voxels: "OrderedDict[Tuple[int, int, int], Tuple[float, float, float]]" = (
            OrderedDict()
        )
        self.latest_points = np.empty((0, 3), dtype=np.float32)
        self.latest_pose: Optional[Dict[str, float]] = None
        self.latest_header: Dict[str, Any] = {}
        self.revision = 0
        self.processed_revision = -1
        self.last_update = 0.0

    def update(self, header: Dict[str, Any], points: np.ndarray) -> Tuple[int, int]:
        cloud = np.asarray(points, dtype=np.float32)
        cloud = cloud[np.isfinite(cloud).all(axis=1), :3]
        if self.minimum_z is not None:
            cloud = cloud[cloud[:, 2] >= self.minimum_z]
        if self.maximum_z is not None:
            cloud = cloud[cloud[:, 2] <= self.maximum_z]
        if cloud.shape[0] == 0:
            return 0, 0
        keys = np.floor(cloud / self.voxel_size).astype(np.int32)
        unique_keys, unique_index = np.unique(keys, axis=0, return_index=True)
        unique_points = cloud[unique_index]
        added = 0
        with self.lock:
            for key_array, point in zip(unique_keys, unique_points):
                key = (int(key_array[0]), int(key_array[1]), int(key_array[2]))
                if key not in self.voxels:
                    added += 1
                self.voxels[key] = (float(point[0]), float(point[1]), float(point[2]))
                self.voxels.move_to_end(key)
            while len(self.voxels) > self.max_voxels:
                self.voxels.popitem(last=False)
            self.latest_points = np.ascontiguousarray(cloud.copy())
            pose = header.get("pose")
            self.latest_pose = dict(pose) if isinstance(pose, dict) else None
            self.latest_header = dict(header)
            self.revision += 1
            self.last_update = time.time()
            total = len(self.voxels)
        return added, total

    def snapshot(
        self,
    ) -> Tuple[int, np.ndarray, np.ndarray, Optional[Dict[str, float]], Dict[str, Any]]:
        with self.lock:
            revision = self.revision
            observed = np.asarray(list(self.voxels.values()), dtype=np.float32).reshape(
                -1, 3
            )
            latest = self.latest_points.copy()
            pose = dict(self.latest_pose) if self.latest_pose is not None else None
            header = dict(self.latest_header)
        return revision, observed, latest, pose, header

    def is_dirty(self) -> bool:
        with self.lock:
            return self.revision != self.processed_revision and bool(self.voxels)

    def mark_processed(self, revision: int) -> None:
        with self.lock:
            self.processed_revision = max(self.processed_revision, revision)


class ReconstructionPipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_root = Path(args.output_dir).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.adapointr: Optional[AdaPoinTrBackend] = None
        self.tulip: Optional[TulipBackend] = None
        self.open3d: Any = None

        if args.adapointr_repo:
            print("[server] loading AdaPoinTr checkpoint...", flush=True)
            self.adapointr = AdaPoinTrBackend(
                args.adapointr_repo,
                args.adapointr_config,
                args.adapointr_checkpoint,
                device=args.device,
                input_points=args.adapointr_input_points,
            )
            fallback = (
                "torch fallback"
                if self.adapointr.using_pointnet_fallback
                else "compiled pointnet2_ops"
            )
            print(f"[server] AdaPoinTr ready ({fallback})", flush=True)
        if args.tulip_repo:
            print("[server] loading TULIP checkpoint...", flush=True)
            self.tulip = TulipBackend(
                args.tulip_repo,
                args.tulip_checkpoint,
                device=args.device,
                minimum_range=args.tulip_minimum_range,
            )
            print(
                "[server] TULIP ready (experimental Unitree->KITTI adapter)", flush=True
            )
        if args.mesh:
            try:
                import open3d
            except ImportError as exc:
                raise RuntimeError(
                    "--mesh requires Open3D; run ./go2 install lidar-server or use --no-mesh"
                ) from exc
            self.open3d = open3d

    def _regions(self, points: np.ndarray) -> List[np.ndarray]:
        if self.args.adapointr_mode == "whole":
            return [points]
        tile_size = self.args.adapointr_tile_size
        tile_keys = np.floor(points[:, :2] / tile_size).astype(np.int32)
        unique, counts = np.unique(tile_keys, axis=0, return_counts=True)
        order = np.argsort(counts)[::-1]
        regions: List[np.ndarray] = []
        margin = self.args.adapointr_tile_overlap
        for key in unique[order]:
            low = key.astype(np.float32) * tile_size - margin
            high = (key.astype(np.float32) + 1) * tile_size + margin
            mask = (
                (points[:, 0] >= low[0])
                & (points[:, 0] <= high[0])
                & (points[:, 1] >= low[1])
                & (points[:, 1] <= high[1])
            )
            region = points[mask]
            if region.shape[0] >= self.args.adapointr_min_region_points:
                regions.append(region)
            if len(regions) >= self.args.adapointr_max_regions:
                break
        return regions

    def _complete(self, source: np.ndarray) -> Tuple[np.ndarray, int]:
        if self.adapointr is None:
            return source, 0
        if source.shape[0] < self.args.adapointr_min_region_points:
            raise RuntimeError(
                "not enough observed points for AdaPoinTr: "
                f"need {self.args.adapointr_min_region_points}, got {source.shape[0]}"
            )
        predictions: List[np.ndarray] = []
        region_count = 0
        for region in self._regions(source):
            prediction = self.adapointr.complete(region)
            low = region.min(axis=0) - self.args.adapointr_prediction_padding
            high = region.max(axis=0) + self.args.adapointr_prediction_padding
            keep = np.all((prediction >= low) & (prediction <= high), axis=1)
            predictions.append(prediction[keep])
            region_count += 1
        if not predictions:
            raise RuntimeError("no map region had enough points for AdaPoinTr")
        completed = voxel_downsample(
            np.vstack([source, *predictions]), self.args.completed_voxel_size
        )
        return completed, region_count

    def _mesh(self, points: np.ndarray, target: Path) -> Dict[str, Any]:
        if self.open3d is None:
            return {}
        if points.shape[0] < self.args.mesh_min_points:
            raise RuntimeError(
                f"mesh needs at least {self.args.mesh_min_points} points; got {points.shape[0]}"
            )
        o3d = self.open3d
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.args.mesh_normal_radius, max_nn=40
            )
        )
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud, depth=self.args.mesh_poisson_depth
        )
        density = np.asarray(densities)
        if density.size:
            threshold = np.quantile(density, self.args.mesh_density_quantile)
            mesh.remove_vertices_by_mask(density < threshold)
        mesh = mesh.crop(cloud.get_axis_aligned_bounding_box())
        mesh.compute_vertex_normals()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = _atomic_path(target)
        if not o3d.io.write_triangle_mesh(str(temporary), mesh, write_ascii=False):
            raise RuntimeError("Open3D failed to write the mesh")
        os.replace(temporary, target)
        return {
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles),
            "poisson_depth": self.args.mesh_poisson_depth,
        }

    def reconstruct(self, robot_id: str, state: RobotVoxelMap) -> Dict[str, Any]:
        started = time.monotonic()
        revision, observed, latest, pose, source_header = state.snapshot()
        robot_dir = self.output_root / robot_id
        observed_dir = robot_dir / "observed"
        write_npz_atomic(observed_dir / "latest.npz", observed)
        write_binary_ply_atomic(observed_dir / "latest.ply", observed)

        metadata: Dict[str, Any] = {
            "robot_id": robot_id,
            "revision": revision,
            "generated_at": time.time(),
            "source_frame_id": source_header.get("frame_id"),
            "coordinate_frame": source_header.get("coordinate_frame", "utlidar_odom"),
            "observed_points": int(observed.shape[0]),
            "observed_bounds": bounds(observed),
            "tulip_enabled": self.tulip is not None,
            "adapointr_enabled": self.adapointr is not None,
            "mesh_enabled": bool(self.open3d),
            "warnings": [
                "Model-generated points are predictions, not metric sensor measurements."
            ],
            "errors": [],
        }

        completion_source = observed
        if self.tulip is not None:
            if pose is None:
                metadata["errors"].append(
                    "TULIP skipped: Unitree robot_pose is unavailable"
                )
            else:
                try:
                    tulip_points = self.tulip.upsample_world(latest, pose)
                    completion_source = voxel_downsample(
                        np.vstack((observed, tulip_points)),
                        self.args.completed_voxel_size,
                    )
                    tulip_dir = robot_dir / "tulip"
                    write_npz_atomic(tulip_dir / "latest.npz", tulip_points)
                    write_binary_ply_atomic(tulip_dir / "latest.ply", tulip_points)
                    metadata["tulip_predicted_points"] = int(tulip_points.shape[0])
                except Exception as exc:
                    metadata["errors"].append(f"TULIP failed: {exc}")

        completed: Optional[np.ndarray] = None
        if self.adapointr is not None:
            try:
                completed, region_count = self._complete(completion_source)
                completed_dir = robot_dir / "completed"
                write_npz_atomic(completed_dir / "latest.npz", completed)
                write_binary_ply_atomic(completed_dir / "latest.ply", completed)
                metadata.update(
                    {
                        "adapointr_mode": self.args.adapointr_mode,
                        "adapointr_regions": region_count,
                        "completed_points": int(completed.shape[0]),
                        "completed_bounds": bounds(completed),
                    }
                )
            except Exception as exc:
                metadata["errors"].append(f"AdaPoinTr failed: {exc}")

        mesh_source = completed if completed is not None else completion_source
        if self.open3d is not None:
            try:
                metadata["mesh"] = self._mesh(
                    mesh_source, robot_dir / "mesh" / "latest.ply"
                )
            except Exception as exc:
                metadata["errors"].append(f"mesh failed: {exc}")

        metadata["processing_seconds"] = round(time.monotonic() - started, 3)
        write_json_atomic(robot_dir / "latest.json", metadata)
        state.mark_processed(revision)
        return metadata


class Lidar3DServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stop_event = asyncio.Event()
        self.maps: Dict[str, RobotVoxelMap] = {}
        self.maps_lock = threading.Lock()
        self.pipeline = ReconstructionPipeline(args)
        self.frames_received = 0

    def request_stop(self) -> None:
        self.stop_event.set()

    def _state(self, robot_id: str) -> RobotVoxelMap:
        with self.maps_lock:
            state = self.maps.get(robot_id)
            if state is None:
                state = RobotVoxelMap(
                    self.args.map_voxel_size,
                    self.args.map_max_voxels,
                    self.args.minimum_z,
                    self.args.maximum_z,
                )
                self.maps[robot_id] = state
            return state

    @staticmethod
    def _request_path(websocket: Any, path: Optional[str]) -> str:
        if path:
            return path
        request = getattr(websocket, "request", None)
        if request is not None and getattr(request, "path", None):
            return str(request.path)
        return str(getattr(websocket, "path", ""))

    def _authorize_path(self, request_path: str) -> str:
        parsed = urlsplit(request_path)
        pieces = [unquote(piece) for piece in parsed.path.split("/") if piece]
        if (
            len(pieces) != 2
            or pieces[0] != "lidar"
            or not ROBOT_ID_PATTERN.fullmatch(pieces[1])
        ):
            raise PermissionError("expected /lidar/<robot_id>")
        supplied = parse_qs(parsed.query).get("token", [""])[0]
        if self.args.token and not hmac.compare_digest(supplied, self.args.token):
            raise PermissionError("invalid token")
        return pieces[1]

    async def websocket_handler(
        self, websocket: Any, path: Optional[str] = None
    ) -> None:
        try:
            robot_id = self._authorize_path(self._request_path(websocket, path))
        except PermissionError as exc:
            await websocket.close(code=1008, reason=str(exc))
            return
        print(f"[server] Raspberry connected for {robot_id}", flush=True)
        state = self._state(robot_id)
        try:
            async for message in websocket:
                if not isinstance(message, bytes):
                    await websocket.close(
                        code=1003, reason="binary LiDAR frames required"
                    )
                    return
                try:
                    header, points = await asyncio.to_thread(
                        decode_cloud_frame,
                        message,
                        max_payload_bytes=self.args.max_compressed_bytes,
                        max_points=self.args.max_frame_points,
                    )
                    header_robot = str(header.get("robot_id", ""))
                    if header_robot != robot_id:
                        raise ProtocolError("URL robot_id and frame robot_id differ")
                    if header.get("coordinate_frame") != "utlidar_odom":
                        raise ProtocolError(
                            "server requires Unitree odometry-frame points"
                        )
                    added, total = await asyncio.to_thread(state.update, header, points)
                except (ProtocolError, ValueError) as exc:
                    print(f"[server] rejected {robot_id} frame: {exc}", flush=True)
                    await websocket.close(code=1007, reason="invalid LiDAR frame")
                    return
                self.frames_received += 1
                print(
                    f"[server] {robot_id} frame={header.get('frame_id')} "
                    f"received={points.shape[0]} new_voxels={added} map={total}",
                    flush=True,
                )
        except Exception as exc:
            # Normal close types differ across websockets versions; all are safe
            # because the Raspberry reconnects and frames are latest-wins.
            print(f"[server] Raspberry {robot_id} disconnected: {exc}", flush=True)

    async def reconstruction_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self.stop_event.wait(), timeout=self.args.reconstruct_interval
                )
            except asyncio.TimeoutError:
                pass
            for robot_id, state in list(self.maps.items()):
                if not state.is_dirty():
                    continue
                print(f"[server] reconstructing {robot_id}...", flush=True)
                try:
                    result = await asyncio.to_thread(
                        self.pipeline.reconstruct, robot_id, state
                    )
                    print(
                        f"[server] {robot_id} model ready: "
                        f"observed={result['observed_points']} "
                        f"completed={result.get('completed_points', 0)} "
                        f"errors={len(result['errors'])} "
                        f"time={result['processing_seconds']}s",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"[server] reconstruction failed for {robot_id}: {exc}",
                        flush=True,
                    )

    async def _final_reconstruction(self) -> None:
        for robot_id, state in list(self.maps.items()):
            if state.is_dirty():
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(self.pipeline.reconstruct, robot_id, state)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self.request_stop)
        reconstruction_task = asyncio.create_task(
            self.reconstruction_loop(), name="3d-reconstruction"
        )
        async with websockets.serve(
            self.websocket_handler,
            self.args.host,
            self.args.port,
            compression=None,
            max_size=self.args.ws_max_size,
            max_queue=4,
            ping_interval=15,
            ping_timeout=15,
            close_timeout=2,
        ):
            print(
                f"[server] listening on ws://{self.args.host}:{self.args.port}/lidar/<robot_id>",
                flush=True,
            )
            await self.stop_event.wait()
        reconstruction_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reconstruction_task
        await self._final_reconstruction()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive Go2 LiDAR, optionally run TULIP + AdaPoinTr, and export a 3D PLY mesh."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default=os.environ.get("GO2_LIDAR_TOKEN", ""))
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).resolve().parent / "lidar_models")
    )
    parser.add_argument("--reconstruct-interval", type=float, default=30.0)
    parser.add_argument("--ws-max-size", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--max-compressed-bytes", type=int, default=24 * 1024 * 1024)
    parser.add_argument("--max-frame-points", type=int, default=500_000)

    parser.add_argument("--map-voxel-size", type=float, default=0.05)
    parser.add_argument("--map-max-voxels", type=int, default=500_000)
    parser.add_argument("--minimum-z", type=float, default=None)
    parser.add_argument("--maximum-z", type=float, default=None)
    parser.add_argument("--completed-voxel-size", type=float, default=0.025)
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--adapointr-repo", default="")
    parser.add_argument("--adapointr-config", default="")
    parser.add_argument("--adapointr-checkpoint", default="")
    parser.add_argument("--adapointr-input-points", type=int, default=2048)
    parser.add_argument("--adapointr-mode", choices=["whole", "tiles"], default="whole")
    parser.add_argument("--adapointr-tile-size", type=float, default=3.0)
    parser.add_argument("--adapointr-tile-overlap", type=float, default=0.35)
    parser.add_argument("--adapointr-min-region-points", type=int, default=512)
    parser.add_argument("--adapointr-max-regions", type=int, default=8)
    parser.add_argument("--adapointr-prediction-padding", type=float, default=0.25)

    parser.add_argument("--tulip-repo", default="")
    parser.add_argument("--tulip-checkpoint", default="")
    parser.add_argument("--tulip-minimum-range", type=float, default=0.3)

    parser.add_argument("--mesh", dest="mesh", action="store_true", default=True)
    parser.add_argument("--no-mesh", dest="mesh", action="store_false")
    parser.add_argument("--mesh-min-points", type=int, default=1000)
    parser.add_argument("--mesh-normal-radius", type=float, default=0.15)
    parser.add_argument("--mesh-poisson-depth", type=int, default=8)
    parser.add_argument("--mesh-density-quantile", type=float, default=0.02)
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.reconstruct_interval <= 0:
        parser.error("--reconstruct-interval must be positive")
    if min(args.ws_max_size, args.max_compressed_bytes, args.max_frame_points) <= 0:
        parser.error("frame limits must be positive")
    if args.map_voxel_size <= 0 or args.completed_voxel_size <= 0:
        parser.error("voxel sizes must be positive")
    if args.map_max_voxels <= 0:
        parser.error("--map-max-voxels must be positive")
    if (
        args.minimum_z is not None
        and args.maximum_z is not None
        and args.minimum_z >= args.maximum_z
    ):
        parser.error("--minimum-z must be lower than --maximum-z")
    ada_values = [args.adapointr_repo, args.adapointr_config, args.adapointr_checkpoint]
    if any(ada_values) and not all(ada_values):
        parser.error(
            "AdaPoinTr requires --adapointr-repo, --adapointr-config, and --adapointr-checkpoint"
        )
    tulip_values = [args.tulip_repo, args.tulip_checkpoint]
    if any(tulip_values) and not all(tulip_values):
        parser.error("TULIP requires --tulip-repo and --tulip-checkpoint")
    if args.adapointr_tile_size <= 0 or args.adapointr_tile_overlap < 0:
        parser.error("invalid AdaPoinTr tile geometry")
    if args.adapointr_min_region_points < 128 or args.adapointr_max_regions <= 0:
        parser.error("invalid AdaPoinTr region limits")
    if args.adapointr_prediction_padding < 0:
        parser.error("--adapointr-prediction-padding must be >= 0")
    if args.mesh_min_points < 100 or args.mesh_normal_radius <= 0:
        parser.error("invalid mesh point/radius settings")
    if not 5 <= args.mesh_poisson_depth <= 12:
        parser.error("--mesh-poisson-depth must be between 5 and 12")
    if not 0 <= args.mesh_density_quantile < 1:
        parser.error("--mesh-density-quantile must be in [0, 1)")
    return args


def main() -> None:
    args = parse_args()
    server = Lidar3DServer(args)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
