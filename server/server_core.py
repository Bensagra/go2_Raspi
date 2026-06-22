#!/usr/bin/env python3
import argparse
import asyncio
import base64
import contextlib
import json
import math
import re
import threading
import time
import uuid
import zlib
from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import paho.mqtt.client as mqtt
import cv2
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROLE_ALLOWED_COMMANDS = {
    "viewer": set(),
    "operator": {
        "move",
        "turn",
        "stop",
        "enter_mode",
        "set_video",
        "set_camera_stream",
        "set_audio",
        "set_lidar",
        "set_lidar_decoder",
        "set_speed_profile",
    },
    "admin": {
        "move",
        "turn",
        "stop",
        "enter_mode",
        "go_to",
        "follow_target",
        "set_video",
        "set_camera_stream",
        "set_audio",
        "set_lidar",
        "set_lidar_decoder",
        "set_speed_profile",
    },
}


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def extract_token_from_auth_header(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None

    value = authorization.strip()
    parts = value.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()

    return value


class CommandIn(BaseModel):
    command_id: Optional[str] = None
    type: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    ttl_ms: int = 1500


SAFE_STORAGE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


class CoreRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.stop_event = asyncio.Event()
        self.heartbeat_task: Optional[asyncio.Task[None]] = None
        self.map_save_task: Optional[asyncio.Task[None]] = None

        self.app = FastAPI(title="Go2 Server Core", version="0.1.0")
        self._setup_cors()
        self._setup_routes()

        self.mqtt_client: Optional[mqtt.Client] = None

        self.frontend_sockets: Set[WebSocket] = set()
        self.frontend_queues: Dict[WebSocket, asyncio.Queue[str]] = {}
        self.frontend_latest_media: Dict[WebSocket, Dict[str, str]] = {}
        self.frontend_ready: Dict[WebSocket, asyncio.Event] = {}
        self.frontend_sender_tasks: Dict[WebSocket, asyncio.Task[None]] = {}
        self.edge_media_sockets: Dict[str, WebSocket] = {}
        self.drive_owners: Dict[str, WebSocket] = {}
        self.speed_profile_owners: Dict[str, WebSocket] = {}

        self.latest_telemetry: Dict[str, Dict[str, Any]] = {}
        self.latest_media: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self.lidar_voxels: Dict[
            str,
            OrderedDict[Tuple[int, int, int], Tuple[float, float, float]],
        ] = defaultdict(OrderedDict)
        self.lidar_latest_packets: Dict[str, Dict[str, Any]] = {}
        self.lidar_events: Dict[str, asyncio.Event] = {}
        self.lidar_tasks: Dict[str, asyncio.Task[None]] = {}
        self.robot_paths: Dict[str, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.args.lidar_path_max_points)
        )
        self.map_data_locks: Dict[str, threading.RLock] = defaultdict(threading.RLock)
        self.map_revisions: Dict[str, int] = defaultdict(int)
        self.map_saved_revisions: Dict[str, int] = defaultdict(int)
        self.map_last_save_monotonic: Dict[str, float] = defaultdict(lambda: 0.0)
        self.persisted_robot_ids: Set[str] = set()
        self.map_storage_dir = Path(self.args.map_storage_dir).expanduser()
        self.map_storage_dir.mkdir(parents=True, exist_ok=True)

        self.telemetry_history: Dict[str, Deque[Dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.args.replay_max_items)
        )
        self.event_history: Dict[str, Deque[Dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.args.replay_max_items)
        )
        self.ack_history: Dict[str, Deque[Dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.args.replay_max_items)
        )
        self.prediction_history: Dict[str, Deque[Dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.args.replay_max_items)
        )

        self.pending_commands: Dict[str, Dict[str, Any]] = {}
        self.last_control_activity: Dict[str, float] = defaultdict(lambda: 0.0)

        self.rate_limit_buckets: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)

        self.api_tokens = self._parse_api_tokens(self.args.api_token)

        self.audit_log_path = Path(self.args.audit_log).expanduser()
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _parse_api_tokens(entries: List[str]) -> Dict[str, Dict[str, str]]:
        mapping: Dict[str, Dict[str, str]] = {}
        for entry in entries:
            parts = entry.split(":")
            if len(parts) != 3:
                continue
            token, role, user_id = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if token and role and user_id:
                mapping[token] = {"role": role, "user_id": user_id}
        return mapping

    def _mqtt_topic(self, robot_id: str, suffix: str) -> str:
        return f"{self.args.mqtt_topic_prefix}/{robot_id}/{suffix}"

    def _known_robots(self) -> List[str]:
        known = set(self.args.robot_id)
        known.update(self.latest_telemetry.keys())
        known.update(self.edge_media_sockets.keys())
        known.update(self.lidar_voxels.keys())
        known.update(self.persisted_robot_ids)
        return sorted(known)

    def _audit(self, event_type: str, payload: Dict[str, Any]) -> None:
        line = {
            "event_type": event_type,
            "ts": time.time(),
            "payload": payload,
        }
        with self.audit_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=True, separators=(",", ":")) + "\n")

    async def _broadcast(self, payload: Dict[str, Any]) -> None:
        if not self.frontend_queues:
            return

        text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        is_latest_media = (
            payload.get("type") == "media"
            and payload.get("stream") in {"video", "lidar"}
        )

        for ws, queue in list(self.frontend_queues.items()):
            if is_latest_media:
                robot_id = str(payload.get("robot_id", ""))
                stream = str(payload.get("stream", ""))
                self.frontend_latest_media.setdefault(ws, {})[
                    f"{robot_id}:{stream}"
                ] = text
            else:
                if queue.full():
                    with contextlib.suppress(asyncio.QueueEmpty):
                        queue.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(text)

            ready = self.frontend_ready.get(ws)
            if ready is not None:
                ready.set()

    def _enqueue_frontend(self, ws: WebSocket, payload: Dict[str, Any]) -> None:
        queue = self.frontend_queues.get(ws)
        if queue is None:
            return

        text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        if payload.get("type") == "media" and payload.get("stream") in {"video", "lidar"}:
            robot_id = str(payload.get("robot_id", ""))
            stream = str(payload.get("stream", ""))
            self.frontend_latest_media.setdefault(ws, {})[
                f"{robot_id}:{stream}"
            ] = text
        else:
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(text)

        ready = self.frontend_ready.get(ws)
        if ready is not None:
            ready.set()

    async def _frontend_sender_loop(self, ws: WebSocket, queue: asyncio.Queue[str]) -> None:
        try:
            while True:
                ready = self.frontend_ready[ws]
                latest = self.frontend_latest_media[ws]
                batch: List[str] = []

                video_keys = [key for key in latest if key.endswith(":video")]
                if video_keys:
                    batch.append(latest.pop(video_keys[-1]))

                with contextlib.suppress(asyncio.QueueEmpty):
                    batch.append(queue.get_nowait())

                lidar_keys = [key for key in latest if key.endswith(":lidar")]
                if lidar_keys:
                    batch.append(latest.pop(lidar_keys[-1]))

                if not batch:
                    ready.clear()
                    if queue.empty() and not latest:
                        await ready.wait()
                    continue

                for text in batch:
                    await asyncio.wait_for(
                        ws.send_text(text),
                        timeout=self.args.frontend_send_timeout_s,
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            self.frontend_sockets.discard(ws)
            self.frontend_queues.pop(ws, None)
            self.frontend_latest_media.pop(ws, None)
            self.frontend_ready.pop(ws, None)
            current_task = asyncio.current_task()
            if self.frontend_sender_tasks.get(ws) is current_task:
                self.frontend_sender_tasks.pop(ws, None)
            with contextlib.suppress(Exception):
                await ws.close()

    def _decode_lidar_points(self, data: Dict[str, Any]) -> np.ndarray:
        point_format = str(data.get("point_format", ""))
        if point_format not in {"f32_xyz_zlib", "i16_xyz_zlib"}:
            raise ValueError("unsupported lidar point format")

        encoded = data.get("points_base64")
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("missing lidar points")

        compressed = base64.b64decode(encoded, validate=True)
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(
            compressed,
            self.args.lidar_max_packet_bytes + 1,
        )
        if decompressor.unconsumed_tail:
            raise ValueError("lidar packet exceeds server limit")
        raw += decompressor.flush()
        if len(raw) > self.args.lidar_max_packet_bytes:
            raise ValueError("lidar packet exceeds server limit")
        if point_format == "f32_xyz_zlib":
            if len(raw) % 12 != 0:
                raise ValueError("invalid float32 xyz payload length")
            points = np.frombuffer(raw, dtype="<f4").reshape(-1, 3)
        else:
            if len(raw) % 6 != 0:
                raise ValueError("invalid int16 xyz payload length")
            scale = float(data.get("quantization_scale", 0.0))
            offset = np.asarray(data.get("quantization_offset", []), dtype=np.float32)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("invalid lidar quantization scale")
            if offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError("invalid lidar quantization offset")
            quantized = np.frombuffer(raw, dtype="<i2").reshape(-1, 3)
            points = quantized.astype(np.float32) * scale + offset

        declared_count = int(data.get("point_count", points.shape[0]))
        if declared_count != points.shape[0]:
            raise ValueError("lidar point count mismatch")

        finite = np.isfinite(points).all(axis=1)
        return np.asarray(points[finite], dtype=np.float32)

    def _update_lidar_voxels(self, robot_id: str, points: np.ndarray) -> None:
        if points.size == 0:
            return

        resolution = self.args.lidar_voxel_size
        keys = np.floor(points / resolution).astype(np.int32)
        _, unique_indices = np.unique(keys, axis=0, return_index=True)
        unique_points = points[unique_indices]
        unique_keys = keys[unique_indices]

        with self.map_data_locks[robot_id]:
            voxels = self.lidar_voxels[robot_id]
            for key_array, point in zip(unique_keys, unique_points):
                key = (int(key_array[0]), int(key_array[1]), int(key_array[2]))
                voxels[key] = (float(point[0]), float(point[1]), float(point[2]))
                voxels.move_to_end(key)

            while len(voxels) > self.args.lidar_map_max_voxels:
                voxels.popitem(last=False)

    def _robot_pose_for_map(
        self,
        robot_id: str,
        points: np.ndarray,
    ) -> Tuple[float, float, float]:
        telemetry = self.latest_telemetry.get(robot_id, {})
        pose = telemetry.get("pose", {}) if isinstance(telemetry, dict) else {}
        try:
            x = float(pose.get("x"))
            y = float(pose.get("y"))
            yaw = float(pose.get("yaw", 0.0) or 0.0)
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw):
                return x, y, yaw
        except (TypeError, ValueError):
            pass

        if points.size:
            center = np.median(points[:, :2], axis=0)
            return float(center[0]), float(center[1]), 0.0
        return 0.0, 0.0, 0.0

    def _render_lidar_map(
        self,
        robot_id: str,
        source_point_count: int,
    ) -> Optional[Dict[str, Any]]:
        with self.map_data_locks[robot_id]:
            voxels = self.lidar_voxels.get(robot_id)
            if not voxels:
                return None
            points = np.asarray(list(voxels.values()), dtype=np.float32)
            voxel_count = len(voxels)
            path = list(self.robot_paths.get(robot_id, ()))

        if points.shape[0] > self.args.lidar_render_max_points:
            step = max(
                int(np.ceil(points.shape[0] / self.args.lidar_render_max_points)),
                1,
            )
            points = points[::step]

        center_x, center_y, yaw = self._robot_pose_for_map(robot_id, points)
        half_range = self.args.lidar_map_range_m / 2.0
        relative = points.copy()
        relative[:, 0] -= center_x
        relative[:, 1] -= center_y
        visible = (
            (np.abs(relative[:, 0]) <= half_range)
            & (np.abs(relative[:, 1]) <= half_range)
            & (relative[:, 2] >= self.args.lidar_min_z)
            & (relative[:, 2] <= self.args.lidar_max_z)
        )
        relative = relative[visible]
        if relative.size == 0:
            return None

        height = self.args.lidar_render_height
        width = self.args.lidar_render_width
        panel_width = width // 2
        canvas = np.full((height, width, 3), (12, 18, 25), dtype=np.uint8)

        z_low, z_high = np.percentile(relative[:, 2], [2, 98])
        if z_high - z_low < 0.1:
            z_high = z_low + 0.1
        z_norm = np.clip((relative[:, 2] - z_low) / (z_high - z_low), 0, 1)
        colors = cv2.applyColorMap(
            (z_norm * 255).astype(np.uint8).reshape(-1, 1),
            cv2.COLORMAP_TURBO,
        ).reshape(-1, 3)

        top_size = min(panel_width, height) - 34
        origin_x = 17 + top_size // 2
        origin_y = 17 + top_size // 2
        top_scale = top_size / self.args.lidar_map_range_m

        for meter in range(
            -int(half_range),
            int(half_range) + 1,
            max(1, self.args.lidar_grid_step_m),
        ):
            offset = int(round(meter * top_scale))
            cv2.line(
                canvas,
                (origin_x + offset, 17),
                (origin_x + offset, 17 + top_size),
                (38, 49, 60),
                1,
            )
            cv2.line(
                canvas,
                (17, origin_y - offset),
                (17 + top_size, origin_y - offset),
                (38, 49, 60),
                1,
            )

        px = (relative[:, 0] * top_scale + origin_x).astype(np.int32)
        py = (origin_y - relative[:, 1] * top_scale).astype(np.int32)
        valid = (
            (px >= 17)
            & (px < 17 + top_size)
            & (py >= 17)
            & (py < 17 + top_size)
        )
        canvas[py[valid], px[valid]] = colors[valid]

        if path:
            path_array = np.asarray(path, dtype=np.float32)
            path_px = ((path_array[:, 0] - center_x) * top_scale + origin_x).astype(np.int32)
            path_py = (origin_y - (path_array[:, 1] - center_y) * top_scale).astype(np.int32)
            path_points = np.column_stack((path_px, path_py))
            inside = (
                (path_px >= 17)
                & (path_px < 17 + top_size)
                & (path_py >= 17)
                & (path_py < 17 + top_size)
            )
            path_points = path_points[inside]
            if path_points.shape[0] >= 2:
                cv2.polylines(
                    canvas,
                    [path_points.reshape(-1, 1, 2)],
                    False,
                    (60, 210, 255),
                    2,
                )

        robot_center = (origin_x, origin_y)
        direction = (
            int(round(math.cos(yaw) * 15)),
            int(round(-math.sin(yaw) * 15)),
        )
        cv2.circle(canvas, robot_center, 6, (30, 30, 255), -1)
        cv2.arrowedLine(
            canvas,
            robot_center,
            (robot_center[0] + direction[0], robot_center[1] + direction[1]),
            (255, 255, 255),
            2,
            tipLength=0.4,
        )

        iso_origin_x = panel_width + panel_width // 2
        iso_origin_y = int(height * 0.72)
        iso_scale = min(panel_width, height) / (self.args.lidar_map_range_m * 1.45)
        iso_x = (
            (relative[:, 0] - relative[:, 1]) * 0.707 * iso_scale
            + iso_origin_x
        ).astype(np.int32)
        iso_y = (
            (relative[:, 0] + relative[:, 1]) * 0.34 * iso_scale
            - relative[:, 2] * iso_scale * 1.8
            + iso_origin_y
        ).astype(np.int32)
        iso_valid = (
            (iso_x >= panel_width + 4)
            & (iso_x < width - 4)
            & (iso_y >= 20)
            & (iso_y < height - 8)
        )
        order = np.argsort(relative[:, 0] + relative[:, 1])
        order = order[iso_valid[order]]
        canvas[iso_y[order], iso_x[order]] = colors[order]

        cv2.line(canvas, (panel_width, 0), (panel_width, height), (58, 72, 86), 1)
        cv2.putText(canvas, "MAPA SUPERIOR", (18, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 225, 238), 1, cv2.LINE_AA)
        cv2.putText(canvas, "VISTA 3D", (panel_width + 18, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 225, 238), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{voxel_count} voxels | frame {source_point_count} pts | rango {self.args.lidar_map_range_m:.0f}m",
            (18, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (145, 170, 190),
            1,
            cv2.LINE_AA,
        )

        ok, encoded = cv2.imencode(
            ".jpg",
            canvas,
            [cv2.IMWRITE_JPEG_QUALITY, self.args.lidar_render_jpeg_quality],
        )
        if not ok or encoded is None:
            return None

        return {
            "image_format": "jpg",
            "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "point_count": int(relative.shape[0]),
            "map_point_count": voxel_count,
            "source_point_count": source_point_count,
            "rendered_ts": time.time(),
            "coordinate_frame": "map",
        }

    def _process_lidar_packet_sync(
        self,
        robot_id: str,
        data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        started = time.monotonic()
        points = self._decode_lidar_points(data)
        self._update_lidar_voxels(robot_id, points)
        rendered = self._render_lidar_map(robot_id, int(points.shape[0]))
        if rendered is not None:
            rendered["processing_ms"] = round(
                (time.monotonic() - started) * 1000.0,
                2,
            )
        return rendered

    def _schedule_lidar_packet(self, robot_id: str, data: Dict[str, Any]) -> None:
        self.lidar_latest_packets[robot_id] = data
        event = self.lidar_events.setdefault(robot_id, asyncio.Event())
        event.set()
        task = self.lidar_tasks.get(robot_id)
        if task is None or task.done():
            self.lidar_tasks[robot_id] = asyncio.create_task(
                self._lidar_worker(robot_id)
            )

    async def _lidar_worker(self, robot_id: str) -> None:
        event = self.lidar_events[robot_id]
        while not self.stop_event.is_set():
            await event.wait()
            event.clear()
            data = self.lidar_latest_packets.pop(robot_id, None)
            if data is None:
                continue

            try:
                rendered = await asyncio.to_thread(
                    self._process_lidar_packet_sync,
                    robot_id,
                    data,
                )
            except Exception as exc:
                self._audit(
                    "lidar_processing_error",
                    {"robot_id": robot_id, "error": str(exc)},
                )
                continue

            self.map_revisions[robot_id] += 1
            if rendered is None:
                continue
            self.latest_media[robot_id]["lidar"] = rendered
            await self._broadcast(
                {
                    "type": "media",
                    "robot_id": robot_id,
                    "stream": "lidar",
                    "data": rendered,
                    "ts": rendered["rendered_ts"],
                }
            )

    @staticmethod
    def _validate_storage_component(value: str, label: str) -> str:
        clean = str(value).strip()
        if not clean or not SAFE_STORAGE_COMPONENT.fullmatch(clean):
            raise ValueError(f"invalid {label}")
        return clean

    def _map_robot_dir(self, robot_id: str, create: bool = False) -> Path:
        safe_robot_id = self._validate_storage_component(robot_id, "robot id")
        path = self.map_storage_dir / safe_robot_id
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _copy_map_arrays(self, robot_id: str) -> Tuple[np.ndarray, np.ndarray]:
        with self.map_data_locks[robot_id]:
            voxels = self.lidar_voxels.get(robot_id)
            if voxels:
                points = np.asarray(list(voxels.values()), dtype="<f4").reshape(-1, 3)
            else:
                points = np.empty((0, 3), dtype="<f4")

            path_values = list(self.robot_paths.get(robot_id, ()))
            if path_values:
                path = np.asarray(path_values, dtype="<f4").reshape(-1, 2)
            else:
                path = np.empty((0, 2), dtype="<f4")

        return points, path

    @staticmethod
    def _map_bounds(points: np.ndarray) -> Tuple[List[float], List[float]]:
        if points.size == 0:
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        return (
            [round(float(value), 4) for value in np.min(points, axis=0)],
            [round(float(value), 4) for value in np.max(points, axis=0)],
        )

    @staticmethod
    def _read_json_file(path: Path) -> Dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _write_map_files(
        self,
        robot_id: str,
        map_id: str,
        points: np.ndarray,
        path: np.ndarray,
        *,
        is_latest: bool,
        created_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        safe_map_id = self._validate_storage_component(map_id, "map id")
        robot_dir = self._map_robot_dir(robot_id, create=True)
        data_path = robot_dir / f"{safe_map_id}.npz"
        metadata_path = robot_dir / f"{safe_map_id}.json"
        now = time.time()

        if created_at is None and metadata_path.exists():
            previous = self._read_json_file(metadata_path)
            with contextlib.suppress(TypeError, ValueError):
                created_at = float(previous.get("created_at"))
        if created_at is None or not math.isfinite(created_at):
            created_at = now

        bounds_min, bounds_max = self._map_bounds(points)
        metadata = {
            "map_id": safe_map_id,
            "robot_id": robot_id,
            "title": (
                "Mapa actual (guardado automático)"
                if is_latest
                else time.strftime("Mapa %Y-%m-%d %H:%M:%S", time.localtime(created_at))
            ),
            "is_latest": is_latest,
            "created_at": created_at,
            "updated_at": now,
            "point_count": int(points.shape[0]),
            "path_point_count": int(path.shape[0]),
            "voxel_size_m": float(self.args.lidar_voxel_size),
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "coordinate_frame": "map",
            "data_file": data_path.name,
        }

        unique = uuid.uuid4().hex
        data_tmp = data_path.with_name(f".{data_path.name}.{unique}.tmp")
        metadata_tmp = metadata_path.with_name(f".{metadata_path.name}.{unique}.tmp")
        try:
            with data_tmp.open("wb") as fh:
                np.savez_compressed(fh, points=points, path=path)
                fh.flush()
            data_tmp.replace(data_path)

            metadata_tmp.write_text(
                json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            metadata_tmp.replace(metadata_path)
        finally:
            with contextlib.suppress(OSError):
                data_tmp.unlink()
            with contextlib.suppress(OSError):
                metadata_tmp.unlink()

        self.persisted_robot_ids.add(robot_id)
        return metadata

    def _save_latest_map_sync(self, robot_id: str) -> Optional[Dict[str, Any]]:
        points, path = self._copy_map_arrays(robot_id)
        if points.size == 0:
            return None
        return self._write_map_files(
            robot_id,
            "latest",
            points,
            path,
            is_latest=True,
        )

    def _prune_map_snapshots(self, robot_id: str) -> None:
        if self.args.map_max_snapshots <= 0:
            return
        robot_dir = self._map_robot_dir(robot_id)
        snapshots = sorted(
            (
                path
                for path in robot_dir.glob("*.json")
                if path.stem != "latest"
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for metadata_path in snapshots[self.args.map_max_snapshots:]:
            with contextlib.suppress(OSError):
                metadata_path.unlink()
            with contextlib.suppress(OSError):
                metadata_path.with_suffix(".npz").unlink()

    def _create_map_snapshot_sync(self, robot_id: str) -> Dict[str, Any]:
        points, path = self._copy_map_arrays(robot_id)
        if points.size == 0:
            raise ValueError("the robot does not have accumulated LiDAR points")

        self._write_map_files(
            robot_id,
            "latest",
            points,
            path,
            is_latest=True,
        )
        created_at = time.time()
        map_id = (
            time.strftime("%Y%m%d-%H%M%S", time.localtime(created_at))
            + "-"
            + uuid.uuid4().hex[:8]
        )
        metadata = self._write_map_files(
            robot_id,
            map_id,
            points,
            path,
            is_latest=False,
            created_at=created_at,
        )
        self._prune_map_snapshots(robot_id)
        return metadata

    def _list_maps_sync(self, robot_id: Optional[str] = None) -> List[Dict[str, Any]]:
        robot_dirs: List[Path]
        if robot_id:
            robot_dirs = [self._map_robot_dir(robot_id)]
        else:
            robot_dirs = [
                path
                for path in self.map_storage_dir.iterdir()
                if path.is_dir() and SAFE_STORAGE_COMPONENT.fullmatch(path.name)
            ]

        maps: List[Dict[str, Any]] = []
        for robot_dir in robot_dirs:
            if not robot_dir.exists():
                continue
            for metadata_path in robot_dir.glob("*.json"):
                metadata = self._read_json_file(metadata_path)
                map_id = metadata.get("map_id")
                if (
                    not map_id
                    or not SAFE_STORAGE_COMPONENT.fullmatch(str(map_id))
                    or not metadata_path.with_suffix(".npz").exists()
                ):
                    continue
                maps.append(metadata)

        maps.sort(
            key=lambda item: (
                bool(item.get("is_latest")),
                float(item.get("updated_at", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return maps

    def _load_map_files(
        self,
        robot_id: str,
        map_id: str,
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        safe_map_id = self._validate_storage_component(map_id, "map id")
        robot_dir = self._map_robot_dir(robot_id)
        data_path = robot_dir / f"{safe_map_id}.npz"
        metadata_path = robot_dir / f"{safe_map_id}.json"
        if not data_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("map not found")

        metadata = self._read_json_file(metadata_path)
        with np.load(data_path, allow_pickle=False) as stored:
            points = np.asarray(stored["points"], dtype="<f4").reshape(-1, 3)
            path = np.asarray(stored["path"], dtype="<f4").reshape(-1, 2)
        finite_points = np.isfinite(points).all(axis=1)
        finite_path = np.isfinite(path).all(axis=1)
        return metadata, points[finite_points], path[finite_path]

    def _map_payload_sync(
        self,
        robot_id: str,
        map_id: str,
        compressed: bool,
    ) -> Dict[str, Any]:
        metadata, points, path = self._load_map_files(robot_id, map_id)
        point_bytes = np.asarray(points, dtype="<f4").tobytes(order="C")
        path_bytes = np.asarray(path, dtype="<f4").tobytes(order="C")
        if compressed:
            point_bytes = zlib.compress(point_bytes, level=6)
            path_bytes = zlib.compress(path_bytes, level=6)
        suffix = "_zlib" if compressed else ""
        return {
            "metadata": metadata,
            "point_format": f"f32_xyz{suffix}",
            "point_count": int(points.shape[0]),
            "points_base64": base64.b64encode(point_bytes).decode("ascii"),
            "path_format": f"f32_xy{suffix}",
            "path_point_count": int(path.shape[0]),
            "path_base64": base64.b64encode(path_bytes).decode("ascii"),
        }

    def _restore_persisted_maps_sync(self) -> None:
        if not self.map_storage_dir.exists():
            return
        for robot_dir in self.map_storage_dir.iterdir():
            if not robot_dir.is_dir() or not SAFE_STORAGE_COMPONENT.fullmatch(robot_dir.name):
                continue
            try:
                _, points, path = self._load_map_files(robot_dir.name, "latest")
            except (FileNotFoundError, KeyError, OSError, ValueError):
                continue

            if points.shape[0] > self.args.lidar_map_max_voxels:
                points = points[-self.args.lidar_map_max_voxels:]
            keys = np.floor(points / self.args.lidar_voxel_size).astype(np.int32)
            restored: OrderedDict[
                Tuple[int, int, int],
                Tuple[float, float, float],
            ] = OrderedDict()
            for key_array, point in zip(keys, points):
                key = (int(key_array[0]), int(key_array[1]), int(key_array[2]))
                restored[key] = (float(point[0]), float(point[1]), float(point[2]))

            with self.map_data_locks[robot_dir.name]:
                self.lidar_voxels[robot_dir.name] = restored
                restored_path = deque(maxlen=self.args.lidar_path_max_points)
                restored_path.extend(
                    (float(point[0]), float(point[1]))
                    for point in path[-self.args.lidar_path_max_points:]
                )
                self.robot_paths[robot_dir.name] = restored_path
            self.persisted_robot_ids.add(robot_dir.name)

    async def _map_persistence_loop(self) -> None:
        sleep_s = min(max(self.args.map_autosave_interval_s / 2.0, 0.5), 2.0)
        while not self.stop_event.is_set():
            await asyncio.sleep(sleep_s)
            now = time.monotonic()
            candidates = [
                robot_id
                for robot_id, revision in list(self.map_revisions.items())
                if revision > self.map_saved_revisions[robot_id]
                and now - self.map_last_save_monotonic[robot_id]
                >= self.args.map_autosave_interval_s
            ]
            for robot_id in candidates:
                target_revision = self.map_revisions[robot_id]
                try:
                    metadata = await asyncio.to_thread(
                        self._save_latest_map_sync,
                        robot_id,
                    )
                except Exception as exc:
                    self._audit(
                        "map_autosave_error",
                        {"robot_id": robot_id, "error": str(exc)},
                    )
                    continue
                if metadata is not None:
                    self.map_saved_revisions[robot_id] = target_revision
                    self.map_last_save_monotonic[robot_id] = time.monotonic()

    async def _save_all_maps(self) -> None:
        for robot_id in list(self.lidar_voxels.keys()):
            try:
                await asyncio.to_thread(self._save_latest_map_sync, robot_id)
            except Exception as exc:
                self._audit(
                    "map_shutdown_save_error",
                    {"robot_id": robot_id, "error": str(exc)},
                )

    def _setup_cors(self) -> None:
        origins = [x.strip() for x in self.args.cors_origin if x.strip()]
        if not origins:
            origins = ["*"]

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=("*" not in origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_mqtt(self) -> None:
        client_id = self.args.mqtt_client_id or f"server-core-{uuid.uuid4().hex[:8]}"
        self.mqtt_client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)

        if self.args.mqtt_username:
            self.mqtt_client.username_pw_set(self.args.mqtt_username, self.args.mqtt_password)

        if self.args.mqtt_tls:
            self.mqtt_client.tls_set()

        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect

        self.mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)
        self.mqtt_client.connect_async(self.args.mqtt_host, self.args.mqtt_port, keepalive=30)
        self.mqtt_client.loop_start()

    def _on_mqtt_connect(self, client, userdata, flags, rc, properties=None) -> None:
        if rc != 0:
            self._audit("mqtt_connect_failed", {"rc": rc})
            return

        topic_pattern = f"{self.args.mqtt_topic_prefix}/+/#"
        client.subscribe(topic_pattern, qos=1)
        self._audit("mqtt_connected", {"host": self.args.mqtt_host, "port": self.args.mqtt_port})

    def _on_mqtt_disconnect(self, client, userdata, rc, properties=None) -> None:
        self._audit("mqtt_disconnected", {"rc": rc})

    def _on_mqtt_message(self, client, userdata, msg) -> None:
        if self.loop is None:
            return

        topic = msg.topic
        raw = msg.payload.decode("utf-8", errors="ignore")

        try:
            payload = json.loads(raw) if raw else {}
        except Exception:
            self._audit("mqtt_bad_json", {"topic": topic})
            return

        parts = topic.split("/")
        if len(parts) < 3:
            return

        prefix, robot_id = parts[0], parts[1]
        if prefix != self.args.mqtt_topic_prefix:
            return

        suffix = "/".join(parts[2:])

        self.loop.call_soon_threadsafe(asyncio.create_task, self._process_mqtt_payload(robot_id, suffix, payload))

    async def _process_mqtt_payload(self, robot_id: str, suffix: str, payload: Dict[str, Any]) -> None:
        if suffix == "telemetry":
            self.latest_telemetry[robot_id] = payload
            self.telemetry_history[robot_id].append(payload)
            pose = payload.get("pose", {}) if isinstance(payload, dict) else {}
            if isinstance(pose, dict):
                with contextlib.suppress(TypeError, ValueError):
                    x = float(pose.get("x"))
                    y = float(pose.get("y"))
                    if math.isfinite(x) and math.isfinite(y):
                        with self.map_data_locks[robot_id]:
                            path = self.robot_paths[robot_id]
                            if not path or math.hypot(x - path[-1][0], y - path[-1][1]) >= 0.05:
                                path.append((x, y))
                                self.map_revisions[robot_id] += 1

            await self._broadcast({
                "type": "telemetry",
                "robot_id": robot_id,
                "data": payload,
            })

            await self._run_prediction_rules(robot_id, payload)
            return

        if suffix == "events":
            self.event_history[robot_id].append(payload)
            await self._broadcast({
                "type": "event",
                "robot_id": robot_id,
                "data": payload,
            })
            self._audit("edge_event", {"robot_id": robot_id, "data": payload})
            return

        if suffix == "commands/ack":
            ack = dict(payload)
            command_id = str(payload.get("command_id", ""))
            pending = self.pending_commands.pop(command_id, None)
            if pending:
                ack.setdefault("command_type", pending.get("type"))
                ack.setdefault("issued_by", pending.get("issued_by"))

            self.ack_history[robot_id].append(ack)

            await self._broadcast({
                "type": "command_ack",
                "robot_id": robot_id,
                "data": ack,
            })
            self._audit("command_ack", {"robot_id": robot_id, "data": ack})
            return

        if suffix == "topic_events":
            await self._broadcast({
                "type": "topic_event",
                "robot_id": robot_id,
                "data": payload,
            })
            return

    async def _run_prediction_rules(self, robot_id: str, telemetry: Dict[str, Any]) -> None:
        alerts = telemetry.get("alerts", []) if isinstance(telemetry, dict) else []
        battery = telemetry.get("battery") if isinstance(telemetry, dict) else None

        predictions: List[Dict[str, Any]] = []

        if isinstance(battery, (int, float)) and battery <= self.args.prediction_low_battery_threshold:
            predictions.append(
                {
                    "type": "low_battery_risk",
                    "severity": "high" if battery <= 15 else "medium",
                    "message": f"Battery at {battery}%",
                }
            )

        if isinstance(alerts, list) and "obstacle_front" in alerts:
            predictions.append(
                {
                    "type": "obstacle_risk",
                    "severity": "high",
                    "message": "Obstacle detected in front path",
                }
            )

        for prediction in predictions:
            event = {
                "robot_id": robot_id,
                "ts": time.time(),
                "prediction": prediction,
            }
            self.prediction_history[robot_id].append(event)
            await self._broadcast({"type": "prediction", "robot_id": robot_id, "data": event})

    def _validate_rate_limit(self, user_id: str, robot_id: str) -> None:
        key = (user_id, robot_id)
        bucket = self.rate_limit_buckets[key]
        now = time.time()

        while bucket and bucket[0] < now - self.args.command_rate_window_s:
            bucket.popleft()

        if len(bucket) >= self.args.command_rate_max:
            raise HTTPException(status_code=429, detail="Rate limit exceeded for robot commands")

        bucket.append(now)

    def _validate_command_by_role(self, role: str, command_type: str) -> None:
        allowed = ROLE_ALLOWED_COMMANDS.get(role, set())
        if command_type not in allowed:
            raise HTTPException(status_code=403, detail=f"Role '{role}' cannot send command '{command_type}'")

    def _speed_profiles(self) -> Dict[str, Dict[str, float]]:
        return {
            "normal": {
                "forward": min(
                    self.args.normal_forward_speed,
                    self.args.max_forward_speed,
                ),
                "reverse": self.args.max_reverse_speed * self.args.normal_speed_scale,
                "lateral": self.args.max_lateral_speed * self.args.normal_speed_scale,
                "angular": self.args.max_angular_speed * self.args.normal_speed_scale,
            },
            "max_api": {
                "forward": self.args.max_forward_speed,
                "reverse": self.args.max_reverse_speed,
                "lateral": self.args.max_lateral_speed,
                "angular": self.args.max_angular_speed,
            },
        }

    def _active_speed_profile(self, robot_id: str) -> str:
        telemetry = self.latest_telemetry.get(robot_id, {})
        speed_control = telemetry.get("speed_control", {}) if isinstance(telemetry, dict) else {}
        profile = str(speed_control.get("profile", "normal"))
        return profile if profile in self._speed_profiles() else "normal"

    def _publish_speed_profile_command(
        self,
        robot_id: str,
        user_id: str,
        profile: str,
    ) -> Optional[str]:
        if self.mqtt_client is None or profile not in self._speed_profiles():
            return None

        command_id = f"profile-{uuid.uuid4().hex[:12]}"
        wire = {
            "command_id": command_id,
            "robot_id": robot_id,
            "type": "set_speed_profile",
            "payload": {"profile": profile},
            "issued_by": user_id,
            "ts": time.time(),
            "ttl_ms": 1500,
        }
        self.last_control_activity[robot_id] = time.time()
        self.pending_commands[command_id] = {
            "command_id": command_id,
            "robot_id": robot_id,
            "issued_by": user_id,
            "type": "set_speed_profile",
            "ts": time.time(),
        }
        result = self.mqtt_client.publish(
            self._mqtt_topic(robot_id, "commands/in"),
            json.dumps(wire, separators=(",", ":")),
            qos=1,
        )
        if getattr(result, "rc", mqtt.MQTT_ERR_SUCCESS) != mqtt.MQTT_ERR_SUCCESS:
            self.pending_commands.pop(command_id, None)
            return None
        self._audit("speed_profile_requested", wire)
        return command_id

    def _publish_realtime_drive(
        self,
        robot_id: str,
        user_id: str,
        payload: Dict[str, Any],
        sequence: int,
    ) -> bool:
        if self.mqtt_client is None:
            return False

        sanitized = self._sanitize_command(
            "move",
            payload,
            speed_profile=self._active_speed_profile(robot_id),
        )
        sanitized["duration_ms"] = int(
            clamp(float(payload.get("duration_ms", 360)), 200, 700)
        )

        wire = {
            "command_id": f"drive-{user_id}-{sequence}",
            "robot_id": robot_id,
            "type": "move",
            "payload": sanitized,
            "issued_by": user_id,
            "ts": time.time(),
            "ttl_ms": 700,
            "streaming": True,
        }
        self.last_control_activity[robot_id] = time.time()
        result = self.mqtt_client.publish(
            self._mqtt_topic(robot_id, "commands/in"),
            json.dumps(wire, separators=(",", ":")),
            qos=0,
        )
        return getattr(result, "rc", mqtt.MQTT_ERR_SUCCESS) == mqtt.MQTT_ERR_SUCCESS

    def _publish_realtime_stop(self, robot_id: str, user_id: str) -> bool:
        if self.mqtt_client is None:
            return False

        wire = {
            "command_id": f"stop-{uuid.uuid4().hex[:12]}",
            "robot_id": robot_id,
            "type": "stop",
            "payload": {},
            "issued_by": user_id,
            "ts": time.time(),
            "ttl_ms": 700,
            "streaming": True,
        }
        self.last_control_activity[robot_id] = time.time()
        result = self.mqtt_client.publish(
            self._mqtt_topic(robot_id, "commands/in"),
            json.dumps(wire, separators=(",", ":")),
            qos=0,
        )
        return getattr(result, "rc", mqtt.MQTT_ERR_SUCCESS) == mqtt.MQTT_ERR_SUCCESS

    def _sanitize_command(
        self,
        command_type: str,
        payload: Dict[str, Any],
        speed_profile: str = "normal",
    ) -> Dict[str, Any]:
        output = dict(payload)

        if command_type == "move":
            limits = self._speed_profiles().get(
                speed_profile,
                self._speed_profiles()["normal"],
            )
            output["linear_x"] = clamp(
                float(output.get("linear_x", 0.0)),
                -limits["reverse"],
                limits["forward"],
            )
            output["lateral_y"] = clamp(
                float(output.get("lateral_y", 0.0)),
                -limits["lateral"],
                limits["lateral"],
            )
            output["angular_z"] = clamp(
                float(output.get("angular_z", 0.0)),
                -limits["angular"],
                limits["angular"],
            )
            output["duration_ms"] = int(output.get("duration_ms", self.args.default_move_duration_ms))

        if command_type == "turn":
            output["angle_deg"] = float(output.get("angle_deg", 0.0))
            output["duration_ms"] = int(output.get("duration_ms", self.args.default_turn_duration_ms))

        if command_type == "enter_mode":
            output["mode"] = str(output.get("mode", "normal"))

        if command_type == "set_video":
            output["enabled"] = bool(output.get("enabled", True))

        if command_type == "set_camera_stream":
            if "enabled" in output:
                output["enabled"] = bool(output["enabled"])

            if "emit_every" in output:
                output["emit_every"] = max(1, int(output["emit_every"]))

            if "jpeg_quality" in output:
                output["jpeg_quality"] = int(clamp(float(output["jpeg_quality"]), 1, 100))

            if "target_fps" in output:
                output["target_fps"] = clamp(float(output["target_fps"]), 1, 30)

            if "max_width" in output:
                max_width = int(output["max_width"])
                output["max_width"] = 0 if max_width <= 0 else max(320, max_width)

        if command_type == "set_audio":
            if "enabled" in output:
                output["enabled"] = bool(output["enabled"])
            if "emit_every" in output:
                output["emit_every"] = max(1, int(output["emit_every"]))
            if "max_bytes" in output:
                output["max_bytes"] = max(0, int(output["max_bytes"]))
            if not output:
                output["enabled"] = True

        if command_type == "set_lidar":
            output["enabled"] = bool(output.get("enabled", True))
            output["subscribe"] = bool(output.get("subscribe", True))
            if "media_hz" in output:
                output["media_hz"] = clamp(float(output["media_hz"]), 0.2, 15)
            if "max_points" in output:
                output["max_points"] = int(
                    clamp(float(output["max_points"]), 500, 100000)
                )
            if "compression_level" in output:
                output["compression_level"] = int(
                    clamp(float(output["compression_level"]), 0, 9)
                )
            if "quantization_cm" in output:
                output["quantization_cm"] = clamp(
                    float(output["quantization_cm"]),
                    0.1,
                    50,
                )

        if command_type == "set_lidar_decoder":
            decoder = str(output.get("decoder", "libvoxel")).strip().lower()
            if decoder not in {"libvoxel", "native"}:
                raise HTTPException(status_code=400, detail="set_lidar_decoder supports only 'libvoxel' or 'native'")
            output["decoder"] = decoder

        if command_type == "set_speed_profile":
            profile = str(output.get("profile", "normal")).strip().lower()
            if profile not in self._speed_profiles():
                raise HTTPException(status_code=400, detail="profile must be 'normal' or 'max_api'")
            output = {"profile": profile}

        return output

    async def _heartbeat_loop(self) -> None:
        while not self.stop_event.is_set():
            if self.mqtt_client is not None:
                now = time.time()
                for robot_id in self._known_robots():
                    active = now - self.last_control_activity.get(robot_id, 0.0) <= self.args.control_session_timeout_s
                    payload = {
                        "server_ts": now,
                        "session_active": active,
                    }
                    self.mqtt_client.publish(self._mqtt_topic(robot_id, "control/heartbeat"), json.dumps(payload), qos=0)

            await asyncio.sleep(self.args.heartbeat_publish_interval_s)

    def _auth_from_token(self, token: str) -> Dict[str, str]:
        auth = self.api_tokens.get(token)
        if not auth:
            raise HTTPException(status_code=401, detail="Invalid token")
        return auth

    async def _auth_dependency(self, authorization: Optional[str] = Header(default=None)) -> Dict[str, str]:
        token = extract_token_from_auth_header(authorization)
        if not token:
            raise HTTPException(status_code=401, detail="Missing bearer token")
        return self._auth_from_token(token)

    def _setup_routes(self) -> None:
        app = self.app

        @app.on_event("startup")
        async def _startup() -> None:
            self.loop = asyncio.get_running_loop()
            await asyncio.to_thread(self._restore_persisted_maps_sync)
            self._setup_mqtt()
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self.map_save_task = asyncio.create_task(self._map_persistence_loop())
            self._audit("server_start", {"pid": str(uuid.uuid4())})

        @app.on_event("shutdown")
        async def _shutdown() -> None:
            self.stop_event.set()
            for task in self.lidar_tasks.values():
                task.cancel()
            for task in self.lidar_tasks.values():
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.heartbeat_task
            if self.map_save_task:
                self.map_save_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.map_save_task

            await self._save_all_maps()

            if self.mqtt_client is not None:
                with contextlib.suppress(Exception):
                    self.mqtt_client.loop_stop()
                with contextlib.suppress(Exception):
                    self.mqtt_client.disconnect()

            self._audit("server_stop", {})

        @app.get("/health")
        async def health() -> Dict[str, Any]:
            return {
                "ok": True,
                "ts": time.time(),
                "known_robots": self._known_robots(),
                "frontend_clients": len(self.frontend_sockets),
                "edge_media_clients": len(self.edge_media_sockets),
                "stored_maps": len(await asyncio.to_thread(self._list_maps_sync)),
            }

        @app.get("/api/robots")
        async def list_robots(auth: Dict[str, str] = Depends(self._auth_dependency)) -> Dict[str, Any]:
            return {
                "robots": self._known_robots(),
            }

        @app.get("/api/maps")
        async def list_maps(
            robot_id: Optional[str] = Query(default=None),
            auth: Dict[str, str] = Depends(self._auth_dependency),
        ) -> Dict[str, Any]:
            try:
                maps = await asyncio.to_thread(self._list_maps_sync, robot_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {
                "robot_id": robot_id,
                "maps": maps,
                "storage": "server",
            }

        @app.get("/api/maps/{robot_id}/{map_id}")
        async def get_map(
            robot_id: str,
            map_id: str,
            compressed: bool = Query(default=True),
            auth: Dict[str, str] = Depends(self._auth_dependency),
        ) -> Dict[str, Any]:
            try:
                return await asyncio.to_thread(
                    self._map_payload_sync,
                    robot_id,
                    map_id,
                    compressed,
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail="Map not found") from exc
            except (KeyError, OSError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @app.get("/api/robots/{robot_id}/state")
        async def robot_state(robot_id: str, auth: Dict[str, str] = Depends(self._auth_dependency)) -> Dict[str, Any]:
            telemetry = self.latest_telemetry.get(robot_id, {})
            media = self.latest_media.get(robot_id, {})
            return {
                "robot_id": robot_id,
                "telemetry": telemetry,
                "media_streams": sorted(media.keys()),
                "pending_commands": [x for x in self.pending_commands.values() if x.get("robot_id") == robot_id],
            }

        @app.get("/api/robots/{robot_id}/capabilities")
        async def robot_capabilities(robot_id: str, auth: Dict[str, str] = Depends(self._auth_dependency)) -> Dict[str, Any]:
            role = auth["role"]
            return {
                "robot_id": robot_id,
                "role": role,
                "allowed_commands": sorted(ROLE_ALLOWED_COMMANDS.get(role, set())),
                "limits": {
                    "max_forward_speed": self.args.max_forward_speed,
                    "max_reverse_speed": self.args.max_reverse_speed,
                    "max_lateral_speed": self.args.max_lateral_speed,
                    "max_angular_speed": self.args.max_angular_speed,
                    "default_move_duration_ms": self.args.default_move_duration_ms,
                    "command_rate_max": self.args.command_rate_max,
                    "command_rate_window_s": self.args.command_rate_window_s,
                },
                "speed_profiles": self._speed_profiles(),
                "active_speed_profile": self._active_speed_profile(robot_id),
            }

        @app.post("/api/robots/{robot_id}/maps/snapshot")
        async def save_map_snapshot(
            robot_id: str,
            auth: Dict[str, str] = Depends(self._auth_dependency),
        ) -> Dict[str, Any]:
            if auth["role"] not in {"operator", "admin"}:
                raise HTTPException(status_code=403, detail="Viewer role cannot save maps")
            try:
                metadata = await asyncio.to_thread(
                    self._create_map_snapshot_sync,
                    robot_id,
                )
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except OSError as exc:
                raise HTTPException(status_code=500, detail="Could not store map") from exc

            self.map_saved_revisions[robot_id] = self.map_revisions[robot_id]
            self.map_last_save_monotonic[robot_id] = time.monotonic()
            self._audit(
                "map_snapshot_saved",
                {
                    "robot_id": robot_id,
                    "map_id": metadata["map_id"],
                    "point_count": metadata["point_count"],
                    "user_id": auth["user_id"],
                },
            )
            return {"ok": True, "map": metadata}

        @app.get("/api/robots/{robot_id}/replay")
        async def robot_replay(
            robot_id: str,
            limit: int = Query(default=100, ge=1, le=5000),
            auth: Dict[str, str] = Depends(self._auth_dependency),
        ) -> Dict[str, Any]:
            return {
                "robot_id": robot_id,
                "telemetry": list(self.telemetry_history[robot_id])[-limit:],
                "events": list(self.event_history[robot_id])[-limit:],
                "acks": list(self.ack_history[robot_id])[-limit:],
                "predictions": list(self.prediction_history[robot_id])[-limit:],
            }

        @app.post("/api/robots/{robot_id}/commands")
        async def send_command(
            robot_id: str,
            command: CommandIn,
            auth: Dict[str, str] = Depends(self._auth_dependency),
        ) -> Dict[str, Any]:
            role = auth["role"]
            user_id = auth["user_id"]

            command_type = command.type.strip()
            self._validate_command_by_role(role, command_type)
            self._validate_rate_limit(user_id, robot_id)

            if self.mqtt_client is None:
                raise HTTPException(status_code=503, detail="MQTT broker is not connected")

            cmd_id = command.command_id or f"cmd-{uuid.uuid4().hex[:12]}"
            sanitized_payload = self._sanitize_command(
                command_type,
                command.payload,
                speed_profile=self._active_speed_profile(robot_id),
            )

            wire = {
                "command_id": cmd_id,
                "robot_id": robot_id,
                "type": command_type,
                "payload": sanitized_payload,
                "issued_by": user_id,
                "ts": time.time(),
                "ttl_ms": int(command.ttl_ms),
            }

            self.last_control_activity[robot_id] = time.time()
            self.pending_commands[cmd_id] = {
                "command_id": cmd_id,
                "robot_id": robot_id,
                "issued_by": user_id,
                "type": command_type,
                "ts": time.time(),
            }

            self.mqtt_client.publish(self._mqtt_topic(robot_id, "commands/in"), json.dumps(wire), qos=1)
            self._audit("command_out", wire)

            await self._broadcast({"type": "command_out", "robot_id": robot_id, "data": wire})

            return {
                "ok": True,
                "command_id": cmd_id,
                "status": "queued",
            }

        @app.post("/api/robots/{robot_id}/control/activate")
        async def activate_control(
            robot_id: str,
            auth: Dict[str, str] = Depends(self._auth_dependency),
        ) -> Dict[str, Any]:
            self.last_control_activity[robot_id] = time.time()
            self._audit("control_activate", {"robot_id": robot_id, "user_id": auth["user_id"]})
            return {"ok": True, "robot_id": robot_id}

        @app.websocket("/ws/live")
        async def ws_live(ws: WebSocket, token: str = Query(default="")) -> None:
            try:
                auth = self._auth_from_token(token)
            except HTTPException:
                await ws.close(code=4401)
                return

            await ws.accept()
            try:
                await asyncio.wait_for(
                    ws.send_text(
                        json.dumps(
                            {
                                "type": "hello",
                                "ts": time.time(),
                                "user": auth,
                                "known_robots": self._known_robots(),
                            },
                            ensure_ascii=True,
                        )
                    ),
                    timeout=self.args.frontend_send_timeout_s,
                )
            except Exception:
                with contextlib.suppress(Exception):
                    await ws.close()
                return

            queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self.args.frontend_queue_size)
            self.frontend_sockets.add(ws)
            self.frontend_queues[ws] = queue
            self.frontend_latest_media[ws] = {}
            self.frontend_ready[ws] = asyncio.Event()
            sender_task = asyncio.create_task(self._frontend_sender_loop(ws, queue))
            self.frontend_sender_tasks[ws] = sender_task
            driven_robots: Set[str] = set()
            max_profile_robots: Set[str] = set()
            last_drive_publish: Dict[str, float] = {}

            for robot_id, telemetry in self.latest_telemetry.items():
                self._enqueue_frontend(
                    ws,
                    {"type": "telemetry", "robot_id": robot_id, "data": telemetry},
                )

            for media_robot_id, media_streams in self.latest_media.items():
                for stream, data in media_streams.items():
                    self._enqueue_frontend(
                        ws,
                        {
                            "type": "media",
                            "robot_id": media_robot_id,
                            "stream": stream,
                            "data": data,
                            "ts": time.time(),
                        },
                    )

            try:
                while True:
                    text = await ws.receive_text()
                    with contextlib.suppress(Exception):
                        message = json.loads(text)
                        op = str(message.get("op", "")).strip()

                        if op == "heartbeat":
                            robot_id = str(message.get("robot_id", "")).strip()
                            if robot_id:
                                self.last_control_activity[robot_id] = time.time()
                            continue

                        if op not in {"drive", "drive_stop", "set_speed_profile"}:
                            continue
                        if auth["role"] not in {"operator", "admin"}:
                            continue

                        robot_id = str(message.get("robot_id", "")).strip()
                        if not robot_id:
                            continue

                        if op == "set_speed_profile":
                            profile = str(message.get("profile", "normal")).strip().lower()
                            if profile not in self._speed_profiles():
                                self._enqueue_frontend(
                                    ws,
                                    {
                                        "type": "speed_profile_status",
                                        "robot_id": robot_id,
                                        "ok": False,
                                        "pending": False,
                                        "profile": profile,
                                        "error": "invalid speed profile",
                                    },
                                )
                                continue

                            self._publish_realtime_stop(robot_id, auth["user_id"])
                            driven_robots.discard(robot_id)
                            command_id = self._publish_speed_profile_command(
                                robot_id,
                                auth["user_id"],
                                profile,
                            )
                            if command_id:
                                if profile == "max_api":
                                    max_profile_robots.add(robot_id)
                                    self.speed_profile_owners[robot_id] = ws

                            self._enqueue_frontend(
                                ws,
                                {
                                    "type": "speed_profile_status",
                                    "robot_id": robot_id,
                                    "ok": bool(command_id),
                                    "pending": bool(command_id),
                                    "profile": profile,
                                    "command_id": command_id or "",
                                    "error": "" if command_id else "MQTT broker is not connected",
                                },
                            )
                            continue

                        if op == "drive":
                            now = time.monotonic()
                            if now - last_drive_publish.get(robot_id, 0.0) < 0.03:
                                continue

                            payload = message.get("payload", {})
                            if not isinstance(payload, dict):
                                continue

                            sequence = int(message.get("sequence", 0))
                            if self._publish_realtime_drive(
                                robot_id,
                                auth["user_id"],
                                payload,
                                sequence,
                            ):
                                last_drive_publish[robot_id] = now
                                driven_robots.add(robot_id)
                                self.drive_owners[robot_id] = ws
                            continue

                        if self._publish_realtime_stop(robot_id, auth["user_id"]):
                            driven_robots.discard(robot_id)
                            if self.drive_owners.get(robot_id) is ws:
                                self.drive_owners.pop(robot_id, None)
            except WebSocketDisconnect:
                pass
            finally:
                for robot_id in driven_robots:
                    if self.drive_owners.get(robot_id) is not ws:
                        continue
                    self._publish_realtime_stop(robot_id, auth["user_id"])
                    self.drive_owners.pop(robot_id, None)
                for robot_id in max_profile_robots:
                    if self.speed_profile_owners.get(robot_id) is not ws:
                        continue
                    self._publish_realtime_stop(robot_id, auth["user_id"])
                    self._publish_speed_profile_command(robot_id, auth["user_id"], "normal")
                    self.speed_profile_owners.pop(robot_id, None)
                self.frontend_sockets.discard(ws)
                self.frontend_queues.pop(ws, None)
                self.frontend_latest_media.pop(ws, None)
                self.frontend_ready.pop(ws, None)
                sender_task = self.frontend_sender_tasks.pop(ws, None)
                if sender_task is not None and not sender_task.done():
                    sender_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await sender_task

        @app.websocket("/ws/edge-media/{robot_id}")
        async def ws_edge_media(robot_id: str, ws: WebSocket, token: str = Query(default="")) -> None:
            if token != self.args.edge_media_token:
                await ws.close(code=4403)
                return

            await ws.accept()
            previous_ws = self.edge_media_sockets.get(robot_id)
            self.edge_media_sockets[robot_id] = ws
            if previous_ws is not None and previous_ws is not ws:
                with contextlib.suppress(Exception):
                    await previous_ws.close(code=1012)
            self._audit("edge_media_connected", {"robot_id": robot_id})

            try:
                while True:
                    text = await ws.receive_text()
                    with contextlib.suppress(Exception):
                        payload = json.loads(text)
                        stream = str(payload.get("stream", "")).strip() or "unknown"
                        data = payload.get("data", {})

                        if stream == "lidar_points" and isinstance(data, dict):
                            self._schedule_lidar_packet(robot_id, data)
                            continue

                        if stream == "video" and isinstance(data, dict):
                            data = dict(data)
                            data["server_received_ts"] = time.time()

                        if isinstance(data, dict) and stream != "audio":
                            self.latest_media[robot_id][stream] = data

                        await self._broadcast(
                            {
                                "type": "media",
                                "robot_id": robot_id,
                                "stream": stream,
                                "data": data,
                                "ts": payload.get("ts", time.time()),
                            }
                        )
            except WebSocketDisconnect:
                pass
            finally:
                if self.edge_media_sockets.get(robot_id) is ws:
                    self.edge_media_sockets.pop(robot_id, None)
                self._audit("edge_media_disconnected", {"robot_id": robot_id})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Go2 Server Core: ingests telemetry/events, validates and dispatches commands, "
            "offers WebSocket realtime feed to frontend, stores replay and audit."
        )
    )

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=[],
        help="Allowed CORS origin. Repeat for multiple origins. Default allows all origins.",
    )

    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-username", default="")
    parser.add_argument("--mqtt-password", default="")
    parser.add_argument("--mqtt-tls", action="store_true")
    parser.add_argument("--mqtt-topic-prefix", default="go2")
    parser.add_argument("--mqtt-client-id", default="")

    parser.add_argument("--robot-id", action="append", default=["go2_01"], help="Known robot ids")

    parser.add_argument(
        "--api-token",
        action="append",
        default=["dev-operator-token:operator:operator_01", "dev-viewer-token:viewer:viewer_01"],
        help="Token format: <token>:<role>:<user_id>",
    )
    parser.add_argument("--edge-media-token", default="edge-media-dev-token")
    parser.add_argument("--frontend-queue-size", type=int, default=16)
    parser.add_argument("--frontend-send-timeout-s", type=float, default=2.0)

    parser.add_argument("--lidar-voxel-size", type=float, default=0.08)
    parser.add_argument("--lidar-map-max-voxels", type=int, default=120000)
    parser.add_argument("--lidar-render-max-points", type=int, default=50000)
    parser.add_argument("--lidar-max-packet-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--lidar-map-range-m", type=float, default=20.0)
    parser.add_argument("--lidar-min-z", type=float, default=-1.5)
    parser.add_argument("--lidar-max-z", type=float, default=3.5)
    parser.add_argument("--lidar-render-width", type=int, default=960)
    parser.add_argument("--lidar-render-height", type=int, default=480)
    parser.add_argument("--lidar-render-jpeg-quality", type=int, default=68)
    parser.add_argument("--lidar-grid-step-m", type=int, default=1)
    parser.add_argument("--lidar-path-max-points", type=int, default=1000)
    parser.add_argument(
        "--map-storage-dir",
        default=str(Path(__file__).resolve().parent / "maps"),
        help="Server directory used to persist accumulated 3D maps.",
    )
    parser.add_argument(
        "--map-autosave-interval-s",
        type=float,
        default=5.0,
        help="Seconds between automatic saves of a changed map.",
    )
    parser.add_argument(
        "--map-max-snapshots",
        type=int,
        default=0,
        help="Maximum snapshots per robot. Use 0 (default) to keep every snapshot.",
    )

    parser.add_argument("--audit-log", default="./server/audit/server_audit.jsonl")
    parser.add_argument("--replay-max-items", type=int, default=2000)

    parser.add_argument("--command-rate-max", type=int, default=120)
    parser.add_argument("--command-rate-window-s", type=float, default=10.0)

    parser.add_argument("--max-forward-speed", type=float, default=3.8)
    parser.add_argument("--max-reverse-speed", type=float, default=2.5)
    parser.add_argument("--max-lateral-speed", type=float, default=1.0)
    parser.add_argument("--max-angular-speed", type=float, default=4.0)
    parser.add_argument("--normal-forward-speed", type=float, default=3.5)
    parser.add_argument("--normal-speed-scale", type=float, default=0.92)
    parser.add_argument("--default-move-duration-ms", type=int, default=700)
    parser.add_argument("--default-turn-duration-ms", type=int, default=400)

    parser.add_argument("--heartbeat-publish-interval-s", type=float, default=0.5)
    parser.add_argument("--control-session-timeout-s", type=float, default=2.0)

    parser.add_argument("--prediction-low-battery-threshold", type=float, default=20.0)

    args = parser.parse_args()

    if args.port <= 0 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")

    if args.mqtt_port <= 0 or args.mqtt_port > 65535:
        parser.error("--mqtt-port must be between 1 and 65535")

    if args.replay_max_items <= 0:
        parser.error("--replay-max-items must be > 0")

    if args.command_rate_max <= 0:
        parser.error("--command-rate-max must be > 0")

    if args.command_rate_window_s <= 0:
        parser.error("--command-rate-window-s must be > 0")

    if args.max_forward_speed <= 0:
        parser.error("--max-forward-speed must be > 0")

    if args.max_reverse_speed <= 0:
        parser.error("--max-reverse-speed must be > 0")

    if args.max_lateral_speed <= 0:
        parser.error("--max-lateral-speed must be > 0")

    if args.max_angular_speed <= 0:
        parser.error("--max-angular-speed must be > 0")

    if args.normal_forward_speed <= 0 or args.normal_forward_speed > args.max_forward_speed:
        parser.error("--normal-forward-speed must be > 0 and <= --max-forward-speed")

    if args.normal_speed_scale <= 0 or args.normal_speed_scale > 1:
        parser.error("--normal-speed-scale must be > 0 and <= 1")

    if args.heartbeat_publish_interval_s <= 0:
        parser.error("--heartbeat-publish-interval-s must be > 0")

    if args.control_session_timeout_s <= 0:
        parser.error("--control-session-timeout-s must be > 0")

    if args.frontend_queue_size <= 0:
        parser.error("--frontend-queue-size must be > 0")

    if args.frontend_send_timeout_s <= 0:
        parser.error("--frontend-send-timeout-s must be > 0")

    if args.lidar_voxel_size <= 0:
        parser.error("--lidar-voxel-size must be > 0")

    if args.lidar_map_max_voxels <= 0 or args.lidar_render_max_points <= 0:
        parser.error("lidar point limits must be > 0")

    if args.lidar_max_packet_bytes <= 0:
        parser.error("--lidar-max-packet-bytes must be > 0")

    if args.lidar_map_range_m <= 1:
        parser.error("--lidar-map-range-m must be > 1")

    if args.lidar_min_z >= args.lidar_max_z:
        parser.error("--lidar-min-z must be lower than --lidar-max-z")

    if args.lidar_render_width < 640 or args.lidar_render_height < 320:
        parser.error("lidar render size must be at least 640x320")

    if args.lidar_render_jpeg_quality < 1 or args.lidar_render_jpeg_quality > 100:
        parser.error("--lidar-render-jpeg-quality must be between 1 and 100")

    if args.lidar_grid_step_m <= 0 or args.lidar_path_max_points <= 0:
        parser.error("lidar grid/path settings must be > 0")

    if not str(args.map_storage_dir).strip():
        parser.error("--map-storage-dir must not be empty")

    if args.map_autosave_interval_s <= 0:
        parser.error("--map-autosave-interval-s must be > 0")

    if args.map_max_snapshots < 0:
        parser.error("--map-max-snapshots must be >= 0")

    return args


def main() -> None:
    args = parse_args()
    runtime = CoreRuntime(args)

    uvicorn.run(
        runtime.app,
        host=args.host,
        port=args.port,
        log_level="info",
        ws_max_queue=16,
        ws_ping_interval=15.0,
        ws_ping_timeout=15.0,
        ws_per_message_deflate=False,
    )


if __name__ == "__main__":
    main()
