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
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

try:
    from perception import Perception
    from autonomy import AutonomyController, AutonomyConfig
except ImportError:  # package import (server.server_core)
    from server.perception import Perception  # type: ignore
    from server.autonomy import AutonomyController, AutonomyConfig  # type: ignore


ROLE_ALLOWED_COMMANDS = {
    "viewer": set(),
    "operator": {
        "move",
        "turn",
        "stop",
        "enter_mode",
        "e_stop",
        "set_autonomy",
        "set_safety",
        "play_audio",
        "set_video",
        "set_camera_stream",
        "set_audio",
        "set_lidar",
        "set_lidar_decoder",
        "set_color",
        "set_speed_profile",
    },
    "admin": {
        "move",
        "turn",
        "stop",
        "enter_mode",
        "go_to",
        "follow_target",
        "drive_velocity",
        "e_stop",
        "set_autonomy",
        "set_safety",
        "play_audio",
        "set_video",
        "set_camera_stream",
        "set_audio",
        "set_lidar",
        "set_lidar_decoder",
        "set_color",
        "set_speed_profile",
    },
}


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _is_h264_video(payload: Dict[str, Any]) -> bool:
    # H.264 carries inter-frame dependencies, so unlike MJPEG/lidar snapshots it
    # must be relayed reliably and in order (never coalesced "latest-wins").
    if payload.get("stream") != "video":
        return False
    data = payload.get("data")
    return isinstance(data, dict) and data.get("image_format") == "h264"


# ---------------------------------------------------------------------------
# Binary media frame codec
# ---------------------------------------------------------------------------
# Layout (little-endian):  magic(1) version(1) header_len(u32) header payload
# The header is compact JSON (metadata); the payload is the raw binary blob
# (zlib'd int16 point cloud, or H.264/WebP bytes). No base64 => ~33% lighter on
# the wire and no encode/decode CPU on either end.
MEDIA_FRAME_MAGIC = 0xA7
MEDIA_FRAME_VERSION = 1


def encode_media_frame(header: Dict[str, Any], payload: bytes) -> bytes:
    header_bytes = json.dumps(
        header, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    out = bytearray(6 + len(header_bytes) + len(payload))
    out[0] = MEDIA_FRAME_MAGIC
    out[1] = MEDIA_FRAME_VERSION
    out[2:6] = len(header_bytes).to_bytes(4, "little")
    out[6 : 6 + len(header_bytes)] = header_bytes
    out[6 + len(header_bytes) :] = payload
    return bytes(out)


def decode_media_frame(buffer: bytes) -> Tuple[Dict[str, Any], bytes]:
    if len(buffer) < 6 or buffer[0] != MEDIA_FRAME_MAGIC or buffer[1] != MEDIA_FRAME_VERSION:
        raise ValueError("invalid media frame")
    header_len = int.from_bytes(buffer[2:6], "little")
    end = 6 + header_len
    if end > len(buffer):
        raise ValueError("invalid media frame header length")
    header = json.loads(buffer[6:end].decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("invalid media frame header")
    return header, bytes(buffer[end:])


def quantize_points_i16(
    points: np.ndarray,
    quantization_cm: float,
    compression_level: int = 6,
) -> Tuple[bytes, float, List[float], int]:
    """Quantize an Nx3 float cloud to int16 + zlib. Mirrors the edge encoder so the
    browser decodes live frames and saved maps through a single path."""
    points = np.ascontiguousarray(points[:, :3], dtype=np.float32)
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    offset = (bounds_min + bounds_max) / 2.0
    scale = max(quantization_cm / 100.0, 0.001)
    max_delta = float(np.max(np.abs(points - offset))) if points.size else 0.0
    if max_delta > 0:
        scale = max(scale, max_delta / 32760.0)
    quantized = np.rint((points - offset) / scale)
    quantized = np.clip(quantized, -32768, 32767).astype("<i2")
    raw = np.ascontiguousarray(quantized).tobytes()
    compressed = zlib.compress(raw, compression_level)
    return compressed, float(scale), offset.tolist(), int(points.shape[0])


def zlib_inflate_limited(compressed: bytes, limit: int) -> bytes:
    decompressor = zlib.decompressobj()
    raw = decompressor.decompress(compressed, limit + 1)
    if decompressor.unconsumed_tail:
        raise ValueError("compressed payload exceeds limit")
    raw += decompressor.flush()
    if len(raw) > limit:
        raise ValueError("compressed payload exceeds limit")
    return raw


def _i16_xyz_to_points(raw: bytes, scale: float, offset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(raw) % 6 != 0:
        raise ValueError("invalid int16 xyz payload length")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid lidar quantization scale")
    offset = np.asarray(offset, dtype=np.float32)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("invalid lidar quantization offset")
    quantized = np.frombuffer(raw, dtype="<i2").reshape(-1, 3)
    points = quantized.astype(np.float32) * scale + offset
    finite = np.isfinite(points).all(axis=1)
    return points, finite


def encode_cloud_payload(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    quantization_cm: float,
    compression_level: int = 6,
) -> Tuple[bytes, str, float, List[float], int]:
    """Quantize Nx3 points to int16+zlib, optionally with per-point RGBA color.
    Colored layout: `u32 geom_len | zlib(int16 xyz) | zlib(uint8 rgba)`."""
    compressed, scale, offset, count = quantize_points_i16(
        points, quantization_cm, compression_level
    )
    if colors is None:
        return compressed, "i16_xyz_zlib", scale, offset, count
    rgba = np.ascontiguousarray(colors, dtype=np.uint8)
    col = zlib.compress(rgba.tobytes(), compression_level)
    payload = len(compressed).to_bytes(4, "little") + compressed + col
    return payload, "i16_xyz_rgb_zlib", scale, offset, count


def decode_cloud_payload(
    header: Dict[str, Any],
    payload: bytes,
    limit: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Inverse of encode_cloud_payload. Returns (points Nx3, colors Nx4 or None),
    both filtered to finite points."""
    fmt = str(header.get("fmt", "i16_xyz_zlib"))
    scale = float(header.get("scale", 0.0) or 0.0)
    offset = np.asarray(header.get("offset", []), dtype=np.float32)

    if fmt == "f32_xyz_zlib":
        raw = zlib_inflate_limited(payload, limit)
        if len(raw) % 12 != 0:
            raise ValueError("invalid float32 xyz payload length")
        points = np.frombuffer(raw, dtype="<f4").reshape(-1, 3)
        finite = np.isfinite(points).all(axis=1)
        return np.asarray(points[finite], dtype=np.float32), None

    if fmt == "i16_xyz_zlib":
        raw = zlib_inflate_limited(payload, limit)
        points, finite = _i16_xyz_to_points(raw, scale, offset)
        return np.asarray(points[finite], dtype=np.float32), None

    if fmt == "i16_xyz_rgb_zlib":
        if len(payload) < 4:
            raise ValueError("invalid colored cloud payload")
        geom_len = int.from_bytes(payload[:4], "little")
        if 4 + geom_len > len(payload):
            raise ValueError("invalid colored cloud geometry length")
        geom = payload[4 : 4 + geom_len]
        col = payload[4 + geom_len :]
        raw = zlib_inflate_limited(geom, limit)
        points, finite = _i16_xyz_to_points(raw, scale, offset)
        craw = zlib_inflate_limited(col, limit)
        colors = np.frombuffer(craw, dtype=np.uint8)
        if colors.size != points.shape[0] * 4:
            raise ValueError("lidar color/point count mismatch")
        colors = colors.reshape(-1, 4)
        return (
            np.asarray(points[finite], dtype=np.float32),
            np.ascontiguousarray(colors[finite], dtype=np.uint8),
        )

    raise ValueError("unsupported lidar point format")


# ---------------------------------------------------------------------------
# Solid mesh reconstruction (pure numpy, no external deps)
# ---------------------------------------------------------------------------
# Builds a watertight-ish surface mesh from the accumulated colored cloud by
# voxelizing it and emitting only the faces that border empty space (the visible
# shell), with shared corner vertices so Laplacian smoothing can round the
# blocky surface. Per-vertex normals + camera color let the browser shade it as
# a solid. It's not photogrammetry, but it reads as the real space.
MESH_BLOB_MAGIC = b"MSH1"

# Per face direction: axis, sign, and the 4 corner offsets (in unit-cube coords)
# ordered counter-clockwise when viewed from outside.
_FACE_DIRS = [
    ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
    ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
    ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
    ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
    ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
    ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
]


def reconstruct_solid_mesh(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    voxel_size: float,
    smooth_iters: int = 8,
    max_vertices: int = 1_500_000,
) -> Optional[Dict[str, Any]]:
    """Return a solid mesh {vertices, normals, colors, faces, bounds} or None."""
    if points is None or points.shape[0] < 16:
        return None

    grid = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    base = grid.min(axis=0)
    rel = grid - base
    dims = rel.max(axis=0) + 1
    dy, dz = int(dims[1]), int(dims[2])

    def pack(a: np.ndarray) -> np.ndarray:
        return (a[:, 0] * dy + a[:, 1]) * dz + a[:, 2]

    packed = pack(rel)
    uniq, inv = np.unique(packed, return_index=False, return_inverse=True)
    voxel_count = int(uniq.shape[0])
    if voxel_count < 8:
        return None

    # Average camera color per voxel (fallback handled later if uncolored).
    vox_rgb = np.zeros((voxel_count, 3), dtype=np.float64)
    vox_has = np.zeros(voxel_count, dtype=np.int64)
    if colors is not None and colors.shape[0] == points.shape[0]:
        valid = colors[:, 3] > 0
        if valid.any():
            np.add.at(vox_rgb, inv[valid], colors[valid, :3].astype(np.float64))
            np.add.at(vox_has, inv[valid], 1)
    colored = vox_has > 0
    vox_rgb[colored] /= vox_has[colored][:, None]

    occupied = set(uniq.tolist())
    # Unpack unique voxel coords (in the shifted grid).
    vz = uniq % dz
    tmp = uniq // dz
    vy = tmp % dy
    vx = tmp // dy
    vcoord = np.stack([vx, vy, vz], axis=1)  # (M,3)

    # Color lookup by packed voxel id (for vertex coloring).
    color_by_id = {int(pid): vox_rgb[i] for i, pid in enumerate(uniq)}

    vertex_index: Dict[Tuple[int, int, int], int] = {}
    vertices: List[Tuple[int, int, int]] = []
    vcolor_accum: List[List[float]] = []
    vcolor_count: List[int] = []
    faces: List[Tuple[int, int, int]] = []

    def corner_id(c: Tuple[int, int, int], rgb: np.ndarray, has: bool) -> int:
        idx = vertex_index.get(c)
        if idx is None:
            idx = len(vertices)
            vertex_index[c] = idx
            vertices.append(c)
            vcolor_accum.append([0.0, 0.0, 0.0])
            vcolor_count.append(0)
        if has:
            vcolor_accum[idx][0] += float(rgb[0])
            vcolor_accum[idx][1] += float(rgb[1])
            vcolor_accum[idx][2] += float(rgb[2])
            vcolor_count[idx] += 1
        return idx

    for (dvec, corners) in _FACE_DIRS:
        neigh = vcoord + np.asarray(dvec, dtype=np.int64)
        in_bounds = (
            (neigh[:, 0] >= 0) & (neigh[:, 0] < dims[0])
            & (neigh[:, 1] >= 0) & (neigh[:, 1] < dims[1])
            & (neigh[:, 2] >= 0) & (neigh[:, 2] < dims[2])
        )
        neigh_packed = np.full(voxel_count, -1, dtype=np.int64)
        if in_bounds.any():
            neigh_packed[in_bounds] = pack(neigh[in_bounds])
        occupied_neighbor = np.isin(neigh_packed, uniq) & in_bounds
        exposed = ~occupied_neighbor  # face borders empty space
        exposed_idx = np.where(exposed)[0]
        for i in exposed_idx:
            pid = int(uniq[i])
            rgb = color_by_id[pid]
            has = bool(colored[i])
            base_v = vcoord[i]
            quad = []
            for off in corners:
                c = (
                    int(base_v[0] + off[0]),
                    int(base_v[1] + off[1]),
                    int(base_v[2] + off[2]),
                )
                quad.append(corner_id(c, rgb, has))
            faces.append((quad[0], quad[1], quad[2]))
            faces.append((quad[0], quad[2], quad[3]))

    if not faces or len(vertices) > max_vertices:
        return None

    verts = (np.asarray(vertices, dtype=np.float64) + base) * voxel_size
    verts = verts.astype(np.float32)
    face_arr = np.asarray(faces, dtype=np.int64)

    # Per-vertex color (camera where available, else a neutral height tint).
    vcount = np.asarray(vcolor_count, dtype=np.float64)
    vcol = np.asarray(vcolor_accum, dtype=np.float64)
    has_col = vcount > 0
    vcol[has_col] /= vcount[has_col][:, None]
    if (~has_col).any():
        z = verts[:, 2]
        z_lo, z_hi = float(z.min()), float(z.max())
        span = max(z_hi - z_lo, 1e-3)
        t = np.clip((z[~has_col] - z_lo) / span, 0, 1)
        # cool->warm neutral tint so uncolored areas still read as solid
        tint = np.stack([90 + 120 * t, 110 + 80 * t, 150 - 60 * t], axis=1)
        vcol[~has_col] = tint
    vcol = np.clip(vcol, 0, 255).astype(np.uint8)

    verts = _laplacian_smooth(verts, face_arr, smooth_iters)
    normals = _vertex_normals(verts, face_arr)

    return {
        "vertices": verts,
        "normals": normals,
        "colors": vcol,
        "faces": face_arr.astype(np.uint32),
        "bounds_min": verts.min(axis=0).tolist(),
        "bounds_max": verts.max(axis=0).tolist(),
        "voxel_count": voxel_count,
    }


def _laplacian_smooth(verts: np.ndarray, faces: np.ndarray, iters: int) -> np.ndarray:
    if iters <= 0 or verts.shape[0] == 0:
        return verts
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
    src = edges[:, 0]
    dst = edges[:, 1]
    deg = np.zeros(verts.shape[0], dtype=np.float64)
    np.add.at(deg, src, 1.0)
    deg = np.maximum(deg, 1.0)
    out = verts.astype(np.float64)
    for _ in range(iters):
        acc = np.zeros_like(out)
        np.add.at(acc, src, out[dst])
        neighbor_avg = acc / deg[:, None]
        out = out + 0.5 * (neighbor_avg - out)
    return out.astype(np.float32)


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros(verts.shape, dtype=np.float64)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for col in range(3):
        np.add.at(normals[:, col], faces[:, 0], fn[:, col])
        np.add.at(normals[:, col], faces[:, 1], fn[:, col])
        np.add.at(normals[:, col], faces[:, 2], fn[:, col])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-9)
    return (normals / lengths).astype(np.float32)


def encode_mesh_blob(mesh: Dict[str, Any], compression_level: int = 6) -> bytes:
    """Pack a mesh into a compact zlib'd binary the browser parses directly."""
    verts = np.ascontiguousarray(mesh["vertices"], dtype="<f4")
    normals = np.clip(np.rint(mesh["normals"] * 127.0), -127, 127).astype("<i1")
    colors = np.ascontiguousarray(mesh["colors"], dtype=np.uint8)
    faces = np.ascontiguousarray(mesh["faces"], dtype="<u4")
    vcount = verts.shape[0]
    fcount = faces.shape[0]
    body = bytearray()
    body += MESH_BLOB_MAGIC
    body += int(vcount).to_bytes(4, "little")
    body += int(fcount).to_bytes(4, "little")
    body += verts.tobytes()
    body += np.ascontiguousarray(normals).tobytes()
    body += colors.tobytes()
    body += faces.tobytes()
    return zlib.compress(bytes(body), compression_level)


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


class AutonomyActionIn(BaseModel):
    action: str = "start"  # start | stop | estop | status
    overrides: Dict[str, Any] = Field(default_factory=dict)


class FaceLabelIn(BaseModel):
    label: str = ""
    known: bool = False


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
        # Queue items are str (JSON telemetry/events/audio) or bytes (binary media
        # frames). The sender loop dispatches each with send_text / send_bytes.
        self.frontend_queues: Dict[WebSocket, "asyncio.Queue[Any]"] = {}
        self.frontend_latest_media: Dict[WebSocket, Dict[str, Any]] = {}
        self.frontend_ready: Dict[WebSocket, asyncio.Event] = {}
        self.frontend_sender_tasks: Dict[WebSocket, asyncio.Task[None]] = {}
        self.edge_media_sockets: Dict[str, WebSocket] = {}
        self.drive_owners: Dict[str, WebSocket] = {}
        self.speed_profile_owners: Dict[str, WebSocket] = {}

        self.latest_telemetry: Dict[str, Dict[str, Any]] = {}
        # Latest coalesced binary media frame per (robot, stream): replayed to new
        # clients on connect (e.g. lidar keyframe, MJPEG/WebP video snapshot).
        self.latest_media_frames: Dict[str, Dict[str, bytes]] = defaultdict(dict)
        self.lidar_voxels: Dict[
            str,
            OrderedDict[Tuple[int, int, int], Tuple[float, float, float]],
        ] = defaultdict(OrderedDict)
        # Last known camera color per voxel (parallel to lidar_voxels).
        self.lidar_voxel_colors: Dict[
            str, Dict[Tuple[int, int, int], Tuple[int, int, int]]
        ] = defaultdict(dict)
        self.lidar_latest_packets: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
        self.lidar_last_keyframe_at: Dict[str, float] = defaultdict(lambda: 0.0)
        self.lidar_events: Dict[str, asyncio.Event] = {}
        self.lidar_tasks: Dict[str, asyncio.Task[None]] = {}
        # Background solid-mesh reconstruction state.
        self.mesh_task: Optional[asyncio.Task[None]] = None
        self.mesh_built_revisions: Dict[str, int] = defaultdict(lambda: -1)
        self.mesh_building: Set[str] = set()
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

        # Server-side perception (people + face recognition). Heavy ML deps are
        # optional: this stays in mapping-only mode until they are installed.
        self.perception: Optional[Perception] = None
        if self.args.enable_perception:
            self.perception = Perception(
                faces_dir=Path(self.args.faces_dir).expanduser(),
                device=self.args.perception_device,
                person_model=self.args.person_model,
                person_conf=self.args.person_conf,
                match_threshold=self.args.face_match_threshold,
                retention_days=self.args.face_retention_days,
                enable_person=self.args.enable_person_detection,
                enable_face=self.args.enable_face_recognition,
            )
        self.autonomy_controllers: Dict[str, AutonomyController] = {}
        self.autonomy_drive_seq: Dict[str, int] = defaultdict(int)
        # Greeter: play the Go2 greeting when a person is detected (manual OR
        # autonomous). One background task per robot while enabled.
        self.greeter_enabled: Set[str] = set()
        self.greeter_tasks: Dict[str, asyncio.Task] = {}
        self.last_greet_sent: Dict[str, float] = defaultdict(lambda: 0.0)

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

    def _wake_socket(self, ws: WebSocket) -> None:
        ready = self.frontend_ready.get(ws)
        if ready is not None:
            ready.set()

    def _coalesce_to_socket(self, ws: WebSocket, key: str, message: Any) -> None:
        """Latest-wins: only the newest message for `key` is kept (snapshots)."""
        self.frontend_latest_media.setdefault(ws, {})[key] = message
        self._wake_socket(ws)

    def _queue_to_socket(self, ws: WebSocket, message: Any) -> None:
        """Ordered delivery; on overflow drop the OLDEST so streams self-heal."""
        queue = self.frontend_queues.get(ws)
        if queue is None:
            return
        if queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(message)
        self._wake_socket(ws)

    async def _broadcast(self, payload: Dict[str, Any]) -> None:
        """Broadcast a JSON message (telemetry, events, audio) to every frontend."""
        if not self.frontend_queues:
            return
        text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        for ws in list(self.frontend_queues.keys()):
            self._queue_to_socket(ws, text)

    def _broadcast_media_frame(
        self,
        robot_id: str,
        stream: str,
        header: Dict[str, Any],
        payload: bytes,
        coalesce: bool,
    ) -> None:
        """Broadcast a binary media frame. Coalesced frames (snapshots/keyframes)
        are also cached so freshly connected clients receive the latest one."""
        frame = encode_media_frame(header, payload)
        if coalesce:
            self.latest_media_frames[robot_id][stream] = frame
        if not self.frontend_queues:
            return
        key = f"{robot_id}:{stream}"
        for ws in list(self.frontend_queues.keys()):
            if coalesce:
                self._coalesce_to_socket(ws, key, frame)
            else:
                self._queue_to_socket(ws, frame)

    def _enqueue_frontend(self, ws: WebSocket, payload: Dict[str, Any]) -> None:
        queue = self.frontend_queues.get(ws)
        if queue is None:
            return
        text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        self._queue_to_socket(ws, text)

    async def _frontend_sender_loop(self, ws: WebSocket, queue: "asyncio.Queue[Any]") -> None:
        try:
            while True:
                ready = self.frontend_ready[ws]
                latest = self.frontend_latest_media[ws]
                batch: List[Any] = []

                # Snapshots first (newest video frame, lidar keyframe), then the
                # ordered queue (telemetry, events, H.264, lidar deltas, audio).
                for key in list(latest.keys()):
                    batch.append(latest.pop(key))

                for _ in range(self.args.frontend_drain_max):
                    try:
                        batch.append(queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                if not batch:
                    ready.clear()
                    if queue.empty() and not latest:
                        await ready.wait()
                    continue

                for message in batch:
                    if isinstance(message, (bytes, bytearray)):
                        await asyncio.wait_for(
                            ws.send_bytes(message),
                            timeout=self.args.frontend_send_timeout_s,
                        )
                    else:
                        await asyncio.wait_for(
                            ws.send_text(message),
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

    def _points_from_raw(
        self,
        raw: bytes,
        point_format: str,
        scale: float,
        offset: np.ndarray,
    ) -> np.ndarray:
        if point_format == "f32_xyz_zlib":
            if len(raw) % 12 != 0:
                raise ValueError("invalid float32 xyz payload length")
            points = np.frombuffer(raw, dtype="<f4").reshape(-1, 3)
        elif point_format == "i16_xyz_zlib":
            if len(raw) % 6 != 0:
                raise ValueError("invalid int16 xyz payload length")
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("invalid lidar quantization scale")
            if offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError("invalid lidar quantization offset")
            quantized = np.frombuffer(raw, dtype="<i2").reshape(-1, 3)
            points = quantized.astype(np.float32) * scale + offset
        else:
            raise ValueError("unsupported lidar point format")

        finite = np.isfinite(points).all(axis=1)
        return np.asarray(points[finite], dtype=np.float32)

    def _decode_lidar_points(self, data: Dict[str, Any]) -> np.ndarray:
        """Legacy JSON/base64 lidar payload (kept for older edges)."""
        point_format = str(data.get("point_format", ""))
        encoded = data.get("points_base64")
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("missing lidar points")
        compressed = base64.b64decode(encoded, validate=True)
        raw = zlib_inflate_limited(compressed, self.args.lidar_max_packet_bytes)
        scale = float(data.get("quantization_scale", 0.0) or 0.0)
        offset = np.asarray(data.get("quantization_offset", []), dtype=np.float32)
        return self._points_from_raw(raw, point_format, scale, offset)

    def _decode_lidar_frame(
        self, header: Dict[str, Any], payload: bytes
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Binary lidar frame from the edge (no base64). Returns (points, colors)."""
        return decode_cloud_payload(header, payload, self.args.lidar_max_packet_bytes)

    def _update_lidar_voxels(
        self,
        robot_id: str,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Insert points into the accumulated voxel map and update per-voxel color.
        Returns (new_points, new_colors) for voxels first seen this frame — the
        delta. Color updates to existing voxels ride the periodic keyframe."""
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32), None

        resolution = self.args.lidar_voxel_size
        keys = np.floor(points / resolution).astype(np.int32)
        _, unique_indices = np.unique(keys, axis=0, return_index=True)
        unique_points = points[unique_indices]
        unique_keys = keys[unique_indices]
        unique_colors = colors[unique_indices] if colors is not None else None

        new_points: List[Tuple[float, float, float]] = []
        new_colors: List[Tuple[int, int, int, int]] = []
        with self.map_data_locks[robot_id]:
            voxels = self.lidar_voxels[robot_id]
            color_map = self.lidar_voxel_colors[robot_id]
            for i in range(unique_keys.shape[0]):
                key_array = unique_keys[i]
                point = unique_points[i]
                key = (int(key_array[0]), int(key_array[1]), int(key_array[2]))
                value = (float(point[0]), float(point[1]), float(point[2]))
                is_new = key not in voxels
                voxels[key] = value
                voxels.move_to_end(key)
                if unique_colors is not None:
                    c = unique_colors[i]
                    if int(c[3]) != 0:
                        color_map[key] = (int(c[0]), int(c[1]), int(c[2]))
                if is_new:
                    new_points.append(value)
                    known = color_map.get(key)
                    if known is not None:
                        new_colors.append((known[0], known[1], known[2], 255))
                    else:
                        new_colors.append((0, 0, 0, 0))

            while len(voxels) > self.args.lidar_map_max_voxels:
                old_key, _ = voxels.popitem(last=False)
                color_map.pop(old_key, None)

        if not new_points:
            return np.empty((0, 3), dtype=np.float32), None
        np_points = np.asarray(new_points, dtype=np.float32)
        np_colors = (
            np.asarray(new_colors, dtype=np.uint8) if colors is not None else None
        )
        return np_points, np_colors

    def _robot_pose_for_map(
        self,
        robot_id: str,
        points: np.ndarray,
    ) -> Tuple[float, float, float]:
        telemetry = self.latest_telemetry.get(robot_id, {})
        # Prefer the LiDAR-frame pose (rt/utlidar/robot_pose): it is in the same
        # frame as the accumulated voxel cloud, so the pose marker + path line up
        # with the map. Fall back to the sport pose for older edges.
        pose = {}
        if isinstance(telemetry, dict):
            mp = telemetry.get("map_pose")
            pose = mp if isinstance(mp, dict) and mp.get("x") is not None else telemetry.get("pose", {})
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

    def _build_lidar_message(
        self,
        robot_id: str,
        mode: str,
        points: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        source_points: int = 0,
    ) -> Optional[Tuple[Dict[str, Any], bytes, bool]]:
        """Build a binary lidar frame. A *keyframe* carries the whole accumulated
        voxel cloud (downsampled to a budget) so a viewer can rebuild the map from
        scratch; a *delta* carries only the voxels first seen this frame. The
        browser renders both in WebGL — no server-side rasterisation. Points carry
        per-voxel camera color (RGBA, alpha=validity) when available."""
        path: Optional[List[Tuple[float, float]]] = None
        if mode == "keyframe":
            with self.map_data_locks[robot_id]:
                voxels = self.lidar_voxels.get(robot_id)
                if not voxels:
                    return None
                items = list(voxels.items())
                color_map = dict(self.lidar_voxel_colors.get(robot_id, {}))
                voxel_count = len(voxels)
                path = list(self.robot_paths.get(robot_id, ()))
            cloud = np.asarray([value for _, value in items], dtype=np.float32)
            if color_map:
                cloud_colors = np.zeros((len(items), 4), dtype=np.uint8)
                for i, (key, _value) in enumerate(items):
                    c = color_map.get(key)
                    if c is not None:
                        cloud_colors[i, 0] = c[0]
                        cloud_colors[i, 1] = c[1]
                        cloud_colors[i, 2] = c[2]
                        cloud_colors[i, 3] = 255
            else:
                cloud_colors = None
            limit = self.args.lidar_cloud_max_points
            if limit > 0 and cloud.shape[0] > limit:
                step = max(int(np.ceil(cloud.shape[0] / limit)), 1)
                cloud = cloud[::step]
                if cloud_colors is not None:
                    cloud_colors = cloud_colors[::step]
        else:
            if points is None or points.shape[0] == 0:
                return None
            cloud = points
            cloud_colors = colors
            with self.map_data_locks[robot_id]:
                voxel_count = len(self.lidar_voxels.get(robot_id, ()))

        if cloud.shape[0] == 0:
            return None

        payload, fmt, scale, offset, count = encode_cloud_payload(
            cloud,
            cloud_colors,
            self.args.lidar_cloud_quantization_cm,
            self.args.lidar_cloud_compression_level,
        )
        center_x, center_y, yaw = self._robot_pose_for_map(robot_id, cloud)
        header: Dict[str, Any] = {
            "type": "media",
            "robot_id": robot_id,
            "stream": "lidar",
            "ts": time.time(),
            "fmt": fmt,
            "mode": mode,
            "count": count,
            "scale": scale,
            "offset": offset,
            "map_points": int(voxel_count),
            "source_points": int(source_points),
            "pose": {"x": center_x, "y": center_y, "yaw": yaw},
        }
        if mode == "keyframe" and path is not None:
            header["path"] = [
                [round(float(px), 3), round(float(py), 3)] for px, py in path
            ]
        return header, payload, mode == "keyframe"

    def _process_lidar_packet_sync(
        self,
        robot_id: str,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
    ) -> List[Tuple[Dict[str, Any], bytes, bool]]:
        started = time.monotonic()
        new_points, new_colors = self._update_lidar_voxels(robot_id, points, colors)
        self.map_revisions[robot_id] += 1

        now = time.monotonic()
        has_cached_keyframe = "lidar" in self.latest_media_frames.get(robot_id, {})
        want_keyframe = (
            not has_cached_keyframe
            or (now - self.lidar_last_keyframe_at[robot_id])
            >= self.args.lidar_keyframe_interval_s
        )

        messages: List[Tuple[Dict[str, Any], bytes, bool]] = []
        if want_keyframe:
            keyframe = self._build_lidar_message(
                robot_id, "keyframe", source_points=int(points.shape[0])
            )
            if keyframe is not None:
                messages.append(keyframe)
                self.lidar_last_keyframe_at[robot_id] = now

        if new_points.shape[0] > 0:
            delta = self._build_lidar_message(
                robot_id,
                "delta",
                points=new_points,
                colors=new_colors,
                source_points=int(points.shape[0]),
            )
            if delta is not None:
                messages.append(delta)

        processing_ms = round((time.monotonic() - started) * 1000.0, 2)
        for header, _payload, _coalesce in messages:
            header["processing_ms"] = processing_ms
        return messages

    def _schedule_lidar_packet(
        self,
        robot_id: str,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
    ) -> None:
        self.lidar_latest_packets[robot_id] = (points, colors)
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
            packet = self.lidar_latest_packets.pop(robot_id, None)
            if packet is None:
                continue
            points, colors = packet

            try:
                messages = await asyncio.to_thread(
                    self._process_lidar_packet_sync,
                    robot_id,
                    points,
                    colors,
                )
            except Exception as exc:
                self._audit(
                    "lidar_processing_error",
                    {"robot_id": robot_id, "error": str(exc)},
                )
                continue

            for header, payload, coalesce in messages:
                self._broadcast_media_frame(
                    robot_id, "lidar", header, payload, coalesce
                )

    def _handle_edge_binary_frame(self, robot_id: str, buffer: bytes) -> None:
        header, payload = decode_media_frame(buffer)
        stream = str(header.get("stream", "")).strip()

        if stream == "lidar":
            points, colors = self._decode_lidar_frame(header, payload)
            if points.size:
                self._schedule_lidar_packet(robot_id, points, colors)
            return

        if stream == "video":
            out_header = dict(header)
            out_header["type"] = "media"
            out_header["stream"] = "video"
            out_header["robot_id"] = robot_id
            out_header["server_received_ts"] = time.time()
            # H.264 deltas can't be decoded out of context, so never coalesce
            # them as a "latest snapshot"; the periodic keyframe resyncs viewers.
            is_h264 = out_header.get("image_format") == "h264"
            # Feed perception only while something consumes it (autonomy session
            # or the greeter); otherwise zero analysis overhead.
            if self.perception is not None and self.perception_active_for(robot_id):
                with contextlib.suppress(Exception):
                    self.perception.ingest(robot_id, header, payload)
            self._broadcast_media_frame(
                robot_id, "video", out_header, payload, coalesce=not is_h264
            )
            return

    async def _handle_edge_text_frame(self, robot_id: str, text: str) -> None:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            return
        stream = str(payload.get("stream", "")).strip() or "unknown"
        data = payload.get("data", {})

        # Legacy edges send lidar as base64 JSON; still feed the point-cloud pipeline.
        if stream == "lidar_points" and isinstance(data, dict):
            points = await asyncio.to_thread(self._decode_lidar_points, data)
            if points.size:
                self._schedule_lidar_packet(robot_id, points)
            return

        # Audio stays JSON end-to-end (small packets, separate playback path).
        await self._broadcast(
            {
                "type": "media",
                "robot_id": robot_id,
                "stream": stream,
                "data": data,
                "ts": payload.get("ts", time.time()),
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

    def _snapshot_cloud_for_mesh(
        self, robot_id: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        with self.map_data_locks[robot_id]:
            voxels = self.lidar_voxels.get(robot_id)
            if not voxels:
                return np.empty((0, 3), dtype=np.float32), None
            items = list(voxels.items())
            color_map = dict(self.lidar_voxel_colors.get(robot_id, {}))
        points = np.asarray([value for _, value in items], dtype=np.float32)
        colors: Optional[np.ndarray] = None
        if color_map:
            colors = np.zeros((len(items), 4), dtype=np.uint8)
            for i, (key, _value) in enumerate(items):
                c = color_map.get(key)
                if c is not None:
                    colors[i, 0] = c[0]
                    colors[i, 1] = c[1]
                    colors[i, 2] = c[2]
                    colors[i, 3] = 255
        return points, colors

    def _mesh_dir(self, robot_id: str, create: bool = False) -> Path:
        path = self._map_robot_dir(robot_id, create=create) / "mesh"
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _build_and_save_mesh_sync(self, robot_id: str) -> Optional[Dict[str, Any]]:
        points, colors = self._snapshot_cloud_for_mesh(robot_id)
        if points.shape[0] < self.args.mesh_min_voxels:
            return None
        started = time.monotonic()
        mesh = reconstruct_solid_mesh(
            points,
            colors,
            self.args.mesh_voxel_size,
            self.args.mesh_smooth_iters,
            self.args.mesh_max_vertices,
        )
        if mesh is None:
            return None
        blob = encode_mesh_blob(mesh, 6)
        mesh_dir = self._mesh_dir(robot_id, create=True)
        blob_path = mesh_dir / "latest.bin"
        meta_path = mesh_dir / "latest.json"
        tmp_blob = blob_path.with_name("latest.bin.tmp")
        tmp_blob.write_bytes(blob)
        tmp_blob.replace(blob_path)
        metadata = {
            "robot_id": robot_id,
            "vertex_count": int(mesh["vertices"].shape[0]),
            "face_count": int(mesh["faces"].shape[0]),
            "voxel_count": int(mesh["voxel_count"]),
            "bounds_min": mesh["bounds_min"],
            "bounds_max": mesh["bounds_max"],
            "voxel_size": float(self.args.mesh_voxel_size),
            "blob_bytes": len(blob),
            "build_ms": round((time.monotonic() - started) * 1000.0, 1),
            "updated_at": time.time(),
        }
        tmp_meta = meta_path.with_name("latest.json.tmp")
        tmp_meta.write_text(json.dumps(metadata), encoding="utf-8")
        tmp_meta.replace(meta_path)
        return metadata

    async def _build_mesh_for_robot(self, robot_id: str) -> Optional[Dict[str, Any]]:
        if robot_id in self.mesh_building:
            return None
        self.mesh_building.add(robot_id)
        try:
            revision = self.map_revisions.get(robot_id, 0)
            metadata = await asyncio.to_thread(self._build_and_save_mesh_sync, robot_id)
            if metadata is not None:
                self.mesh_built_revisions[robot_id] = revision
                self._audit("mesh_built", metadata)
                await self._broadcast(
                    {"type": "mesh_ready", "robot_id": robot_id, "data": metadata}
                )
            return metadata
        finally:
            self.mesh_building.discard(robot_id)

    async def _mesh_reconstruction_loop(self) -> None:
        """Periodically rebuild + save a solid mesh per robot whose map changed."""
        while not self.stop_event.is_set():
            try:
                await asyncio.sleep(self.args.mesh_interval_s)
            except asyncio.CancelledError:
                raise
            if self.stop_event.is_set():
                break
            for robot_id in list(self.lidar_voxels.keys()):
                revision = self.map_revisions.get(robot_id, 0)
                if revision == self.mesh_built_revisions.get(robot_id, -1):
                    continue
                try:
                    await self._build_mesh_for_robot(robot_id)
                except Exception as exc:
                    self._audit("mesh_error", {"robot_id": robot_id, "error": str(exc)})

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
            # Build the path from the LiDAR-frame pose so it overlays the voxel
            # cloud (same frame); fall back to the sport pose for older edges.
            pose = {}
            if isinstance(payload, dict):
                mp = payload.get("map_pose")
                pose = mp if isinstance(mp, dict) and mp.get("x") is not None else payload.get("pose", {})
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

    # ----------------------------------------------------------- autonomy glue
    def autonomy_drive(self, robot_id: str, vx: float, vy: float, wz: float) -> bool:
        """Publish a continuous velocity goal to the edge (filtered there by the
        safety guard). Called by the AutonomyController at its control rate."""
        if self.mqtt_client is None:
            return False
        self.autonomy_drive_seq[robot_id] += 1
        seq = self.autonomy_drive_seq[robot_id]
        limits = self._speed_profiles().get(
            self._active_speed_profile(robot_id), self._speed_profiles()["normal"]
        )
        payload = {
            "vx": clamp(float(vx), -limits["reverse"], limits["forward"]),
            "vy": clamp(float(vy), -limits["lateral"], limits["lateral"]),
            "wz": clamp(float(wz), -limits["angular"], limits["angular"]),
        }
        wire = {
            "command_id": f"auto-{robot_id}-{seq}",
            "robot_id": robot_id,
            "type": "drive_velocity",
            "payload": payload,
            "issued_by": "autonomy",
            "ts": time.time(),
            "ttl_ms": 800,
            "streaming": True,
        }
        self.last_control_activity[robot_id] = time.time()
        result = self.mqtt_client.publish(
            self._mqtt_topic(robot_id, "commands/in"),
            json.dumps(wire, separators=(",", ":")),
            qos=0,
        )
        return getattr(result, "rc", mqtt.MQTT_ERR_SUCCESS) == mqtt.MQTT_ERR_SUCCESS

    def autonomy_command(self, robot_id: str, cmd_type: str, payload: Dict[str, Any]) -> Optional[str]:
        """Publish a one-shot control command on behalf of the autonomy brain
        (set_autonomy / set_lidar / set_video / stop / e_stop)."""
        if self.mqtt_client is None:
            return None
        command_id = f"auto-{cmd_type}-{uuid.uuid4().hex[:8]}"
        wire = {
            "command_id": command_id,
            "robot_id": robot_id,
            "type": cmd_type,
            "payload": payload,
            "issued_by": "autonomy",
            "ts": time.time(),
            "ttl_ms": 2000,
        }
        self.last_control_activity[robot_id] = time.time()
        self.pending_commands[command_id] = {
            "command_id": command_id,
            "robot_id": robot_id,
            "issued_by": "autonomy",
            "type": cmd_type,
            "ts": time.time(),
        }
        self.mqtt_client.publish(
            self._mqtt_topic(robot_id, "commands/in"), json.dumps(wire), qos=1
        )
        self._audit("autonomy_command", wire)
        return command_id

    async def autonomy_event(self, robot_id: str, event_type: str, data: Dict[str, Any]) -> None:
        message = {"type": "autonomy", "robot_id": robot_id, "event": event_type, "data": data, "ts": time.time()}
        await self._broadcast(message)
        self._audit("autonomy_event", {"robot_id": robot_id, "event": event_type, "data": data})

    def start_autonomy(self, robot_id: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        existing = self.autonomy_controllers.get(robot_id)
        if existing and existing.task and not existing.task.done():
            return existing.snapshot()
        cfg = AutonomyConfig(
            control_hz=self.args.autonomy_control_hz,
            v_max=self.args.autonomy_v_max,
            w_max=self.args.autonomy_w_max,
            grid_res=self.args.autonomy_grid_res,
            robot_radius_m=self.args.autonomy_robot_radius_m,
        )
        for key, value in (overrides or {}).items():
            if hasattr(cfg, key) and isinstance(value, (int, float)):
                setattr(cfg, key, float(value))
        controller = AutonomyController(self, robot_id, cfg)
        self.autonomy_controllers[robot_id] = controller
        controller.start()
        # Greet detected people during autonomous roaming too.
        if self.args.greeter_with_autonomy:
            self.start_greeter(robot_id)
        self._audit("autonomy_start", {"robot_id": robot_id})
        return controller.snapshot()

    async def stop_autonomy(self, robot_id: str, estop: bool = False) -> Dict[str, Any]:
        controller = self.autonomy_controllers.get(robot_id)
        if controller:
            await controller.stop("estopped" if estop else "stopped")
        if estop:
            self.autonomy_command(robot_id, "e_stop", {})
        else:
            self.autonomy_command(robot_id, "set_autonomy", {"enabled": False})
        self._audit("autonomy_stop", {"robot_id": robot_id, "estop": estop})
        return controller.snapshot() if controller else {"robot_id": robot_id, "state": "idle", "running": False}

    def autonomy_status(self, robot_id: str) -> Dict[str, Any]:
        controller = self.autonomy_controllers.get(robot_id)
        if controller:
            return controller.snapshot()
        return {"robot_id": robot_id, "state": "idle", "running": False, "captures": 0}

    # -------------------------------------------------------------- greeter
    def perception_active_for(self, robot_id: str) -> bool:
        """Whether the server should be decoding this robot's camera (for any
        perception consumer: autonomy or the greeter)."""
        if robot_id in self.greeter_enabled:
            return True
        controller = self.autonomy_controllers.get(robot_id)
        return bool(controller and controller.task and not controller.task.done())

    def start_greeter(self, robot_id: str) -> Dict[str, Any]:
        self.greeter_enabled.add(robot_id)
        task = self.greeter_tasks.get(robot_id)
        if task is None or task.done():
            self.greeter_tasks[robot_id] = asyncio.create_task(self._greeter_loop(robot_id))
        self._audit("greeter_start", {"robot_id": robot_id})
        return self.greeter_status(robot_id)

    def stop_greeter(self, robot_id: str) -> Dict[str, Any]:
        self.greeter_enabled.discard(robot_id)
        task = self.greeter_tasks.pop(robot_id, None)
        if task:
            task.cancel()
        self._audit("greeter_stop", {"robot_id": robot_id})
        return self.greeter_status(robot_id)

    def greeter_status(self, robot_id: str) -> Dict[str, Any]:
        return {
            "robot_id": robot_id,
            "enabled": robot_id in self.greeter_enabled,
            "available": self.perception is not None and self.perception.person is not None
            and self.perception.person.available,
        }

    async def _greeter_loop(self, robot_id: str) -> None:
        """While enabled, detect people on the latest camera frame and tell the
        edge to play the greeting (rate-limited). Works in manual and autonomy."""
        interval = 1.0 / max(self.args.greeter_detect_hz, 0.5)
        while robot_id in self.greeter_enabled and not self.stop_event.is_set():
            await asyncio.sleep(interval)
            if self.perception is None:
                continue
            try:
                boxes, dims = self.perception.detect_people(robot_id)
            except Exception:
                continue
            if not boxes or dims is None:
                continue
            w, h = dims
            best = max((b.score for b in boxes), default=0.0)
            biggest = max((b.area for b in boxes), default=0.0) / float(max(1, w * h))
            if best < self.args.greeter_person_conf or biggest < self.args.greeter_min_area_frac:
                continue
            now = time.monotonic()
            if now - self.last_greet_sent[robot_id] < self.args.greet_interval_s:
                continue
            command_id = self.autonomy_command(
                robot_id, "play_audio", {"force": True}
            )
            if not command_id:
                continue
            self.last_greet_sent[robot_id] = now
            await self.autonomy_event(
                robot_id, "greet",
                {
                    "command_id": command_id,
                    "person_score": round(best, 2),
                    "area_frac": round(biggest, 3),
                },
            )

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

            if "format" in output:
                image_format = str(output["format"]).strip().lower()
                if image_format not in {"jpg", "webp"}:
                    raise HTTPException(status_code=400, detail="camera format must be 'jpg' or 'webp'")
                output["format"] = image_format

            if "emit_every" in output:
                output["emit_every"] = max(1, int(output["emit_every"]))

            if "jpeg_quality" in output:
                output["jpeg_quality"] = int(clamp(float(output["jpeg_quality"]), 1, 100))

            if "min_quality" in output:
                output["min_quality"] = int(clamp(float(output["min_quality"]), 1, 100))
                requested_quality = int(output.get("jpeg_quality", 100))
                output["min_quality"] = min(output["min_quality"], requested_quality)

            if "target_fps" in output:
                output["target_fps"] = clamp(float(output["target_fps"]), 1, 40)

            if "max_width" in output:
                max_width = int(output["max_width"])
                output["max_width"] = 0 if max_width <= 0 else max(320, max_width)

            if "uplink_max_kbps" in output:
                uplink_max_kbps = int(output["uplink_max_kbps"])
                output["uplink_max_kbps"] = (
                    0
                    if uplink_max_kbps <= 0
                    else int(clamp(uplink_max_kbps, 256, 50000))
                )

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

        if command_type == "set_color":
            if "enabled" in output:
                output["enabled"] = bool(output["enabled"])
            if "fov_deg" in output:
                output["fov_deg"] = clamp(float(output["fov_deg"]), 30.0, 220.0)
            if "pitch_deg" in output:
                output["pitch_deg"] = clamp(float(output["pitch_deg"]), -60.0, 60.0)
            if "height_m" in output:
                output["height_m"] = clamp(float(output["height_m"]), -1.0, 3.0)
            if "forward_m" in output:
                output["forward_m"] = clamp(float(output["forward_m"]), -1.0, 2.0)
            if "max_distance_m" in output:
                output["max_distance_m"] = clamp(float(output["max_distance_m"]), 0.5, 60.0)

        if command_type == "set_speed_profile":
            profile = str(output.get("profile", "normal")).strip().lower()
            if profile not in self._speed_profiles():
                raise HTTPException(status_code=400, detail="profile must be 'normal' or 'max_api'")
            output = {"profile": profile}

        if command_type == "set_autonomy":
            output = {"enabled": bool(output.get("enabled", False))}

        if command_type == "e_stop":
            output = {}

        if command_type == "play_audio":
            output = {"force": bool(output.get("force", True))}

        if command_type == "drive_velocity":
            output["vx"] = float(output.get("vx", 0.0))
            output["vy"] = float(output.get("vy", 0.0))
            output["wz"] = float(output.get("wz", 0.0))

        if command_type == "set_safety":
            allowed = {
                "enabled", "stop_distance_m", "slow_distance_m", "robot_half_width_m",
                "obstacle_min_height_m", "obstacle_max_height_m", "cliff_enabled",
                "cliff_void_enabled", "cliff_lookahead_m", "cliff_drop_m",
                "ground_z_default_m", "ground_z_tolerance_m",
                "min_consider_range_m", "max_consider_radius_m",
                "min_cluster_points", "obstacle_cluster_radius_m",
                "fail_safe_block",
            }
            output = {
                k: (
                    bool(v)
                    if k in {"enabled", "cliff_enabled", "cliff_void_enabled", "fail_safe_block"}
                    else int(v)
                    if k == "min_cluster_points"
                    else float(v)
                )
                for k, v in output.items()
                if k in allowed
            }

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
            if self.args.mesh_interval_s > 0:
                self.mesh_task = asyncio.create_task(self._mesh_reconstruction_loop())
            if self.perception is not None:
                with contextlib.suppress(Exception):
                    removed = self.perception.gallery.enforce_retention()
                    if removed:
                        self._audit("faces_retention_purge", {"removed": removed})
            self._audit("server_start", {"pid": str(uuid.uuid4())})

        @app.on_event("shutdown")
        async def _shutdown() -> None:
            self.stop_event.set()
            for task in list(self.greeter_tasks.values()):
                task.cancel()
            for controller in list(self.autonomy_controllers.values()):
                with contextlib.suppress(Exception):
                    await controller.stop("stopped")
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
            if self.mesh_task:
                self.mesh_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.mesh_task

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

        @app.get("/api/meshes/{robot_id}")
        async def mesh_info(
            robot_id: str,
            auth: Dict[str, str] = Depends(self._auth_dependency),
        ) -> Dict[str, Any]:
            try:
                meta_path = self._mesh_dir(robot_id) / "latest.json"
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            metadata = self._read_json_file(meta_path)
            return {
                "robot_id": robot_id,
                "has_mesh": bool(metadata),
                "building": robot_id in self.mesh_building,
                "metadata": metadata,
            }

        @app.get("/api/meshes/{robot_id}/latest")
        async def get_mesh(
            robot_id: str,
            auth: Dict[str, str] = Depends(self._auth_dependency),
        ) -> Response:
            try:
                blob_path = self._mesh_dir(robot_id) / "latest.bin"
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if not blob_path.exists():
                raise HTTPException(status_code=404, detail="Mesh not found")
            data = await asyncio.to_thread(blob_path.read_bytes)
            return Response(content=data, media_type="application/octet-stream")

        @app.post("/api/meshes/{robot_id}/rebuild")
        async def rebuild_mesh(
            robot_id: str,
            auth: Dict[str, str] = Depends(self._auth_dependency),
        ) -> Dict[str, Any]:
            if auth["role"] not in {"operator", "admin"}:
                raise HTTPException(status_code=403, detail="operator role required")
            metadata = await self._build_mesh_for_robot(robot_id)
            if metadata is None:
                return {"ok": False, "reason": "not enough map data yet or already building"}
            return {"ok": True, "metadata": metadata}

        @app.get("/api/robots/{robot_id}/state")
        async def robot_state(robot_id: str, auth: Dict[str, str] = Depends(self._auth_dependency)) -> Dict[str, Any]:
            telemetry = self.latest_telemetry.get(robot_id, {})
            media = self.latest_media_frames.get(robot_id, {})
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

        # ---------------------------------------------------------- autonomy
        @app.get("/api/perception/capabilities")
        async def perception_caps(auth: Dict[str, str] = Depends(self._auth_dependency)) -> Dict[str, Any]:
            if self.perception is None:
                return {"enabled": False}
            return {"enabled": True, **self.perception.capabilities()}

        @app.get("/api/robots/{robot_id}/autonomy")
        async def autonomy_get(
            robot_id: str, auth: Dict[str, str] = Depends(self._auth_dependency)
        ) -> Dict[str, Any]:
            return self.autonomy_status(robot_id)

        @app.post("/api/robots/{robot_id}/autonomy")
        async def autonomy_post(
            robot_id: str,
            body: AutonomyActionIn,
            auth: Dict[str, str] = Depends(self._auth_dependency),
        ) -> Dict[str, Any]:
            role = auth["role"]
            # Autonomy implies driving the robot: gate on the same permission.
            self._validate_command_by_role(role, "set_autonomy")
            if self.mqtt_client is None:
                raise HTTPException(status_code=503, detail="MQTT broker is not connected")
            action = body.action.strip().lower()
            self.last_control_activity[robot_id] = time.time()
            if action == "start":
                return {"ok": True, **self.start_autonomy(robot_id, body.overrides)}
            if action in {"stop", "estop"}:
                snap = await self.stop_autonomy(robot_id, estop=(action == "estop"))
                return {"ok": True, **snap}
            if action == "status":
                return {"ok": True, **self.autonomy_status(robot_id)}
            raise HTTPException(status_code=400, detail="action must be start|stop|estop|status")

        @app.get("/api/robots/{robot_id}/greeter")
        async def greeter_get(
            robot_id: str, auth: Dict[str, str] = Depends(self._auth_dependency)
        ) -> Dict[str, Any]:
            return self.greeter_status(robot_id)

        @app.post("/api/robots/{robot_id}/greeter")
        async def greeter_post(
            robot_id: str,
            body: AutonomyActionIn,
            auth: Dict[str, str] = Depends(self._auth_dependency),
        ) -> Dict[str, Any]:
            self._validate_command_by_role(auth["role"], "play_audio")
            action = body.action.strip().lower()
            self.last_control_activity[robot_id] = time.time()
            if action in {"start", "on", "enable"}:
                return {"ok": True, **self.start_greeter(robot_id)}
            if action in {"stop", "off", "disable"}:
                return {"ok": True, **self.stop_greeter(robot_id)}
            if action == "status":
                return {"ok": True, **self.greeter_status(robot_id)}
            raise HTTPException(status_code=400, detail="action must be start|stop|status")

        # ------------------------------------------------------- faces gallery
        @app.get("/api/robots/{robot_id}/faces")
        async def faces_list(
            robot_id: str, auth: Dict[str, str] = Depends(self._auth_dependency)
        ) -> Dict[str, Any]:
            if self.perception is None:
                return {"people": []}
            return {"people": self.perception.gallery.list_people(robot_id)}

        @app.get("/api/robots/{robot_id}/faces/{person_id}/image")
        async def faces_image(
            robot_id: str, person_id: str,
            auth: Dict[str, str] = Depends(self._auth_dependency),
        ) -> Any:
            if self.perception is None:
                raise HTTPException(status_code=404, detail="perception disabled")
            path = self.perception.gallery.crop_path(robot_id, person_id)
            if path is None:
                raise HTTPException(status_code=404, detail="no crop for this person")
            return FileResponse(str(path), media_type="image/jpeg")

        @app.post("/api/robots/{robot_id}/faces/{person_id}")
        async def faces_label(
            robot_id: str, person_id: str, body: FaceLabelIn,
            auth: Dict[str, str] = Depends(self._auth_dependency),
        ) -> Dict[str, Any]:
            if auth["role"] not in {"operator", "admin"}:
                raise HTTPException(status_code=403, detail="not allowed")
            if self.perception is None:
                raise HTTPException(status_code=404, detail="perception disabled")
            ok = self.perception.gallery.set_label(robot_id, person_id, body.label, body.known)
            if not ok:
                raise HTTPException(status_code=404, detail="person not found")
            return {"ok": True}

        @app.delete("/api/robots/{robot_id}/faces")
        async def faces_purge_all(
            robot_id: str, auth: Dict[str, str] = Depends(self._auth_dependency)
        ) -> Dict[str, Any]:
            if auth["role"] not in {"operator", "admin"}:
                raise HTTPException(status_code=403, detail="not allowed")
            if self.perception is None:
                return {"ok": True, "removed": 0}
            removed = self.perception.gallery.purge(robot_id)
            self._audit("faces_purged", {"robot_id": robot_id, "removed": removed, "by": auth["user_id"]})
            return {"ok": True, "removed": removed}

        @app.delete("/api/robots/{robot_id}/faces/{person_id}")
        async def faces_purge_one(
            robot_id: str, person_id: str,
            auth: Dict[str, str] = Depends(self._auth_dependency),
        ) -> Dict[str, Any]:
            if auth["role"] not in {"operator", "admin"}:
                raise HTTPException(status_code=403, detail="not allowed")
            if self.perception is None:
                return {"ok": True, "removed": 0}
            removed = self.perception.gallery.purge(robot_id, person_id)
            return {"ok": True, "removed": removed}

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

            queue: "asyncio.Queue[Any]" = asyncio.Queue(maxsize=self.args.frontend_queue_size)
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

            # Replay the latest binary snapshot per stream (lidar keyframe, video
            # frame) so a freshly connected viewer has something to show at once.
            for media_robot_id, frames in self.latest_media_frames.items():
                for stream, frame in frames.items():
                    self._coalesce_to_socket(ws, f"{media_robot_id}:{stream}", frame)

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
                    message = await ws.receive()
                    if message.get("type") == "websocket.disconnect":
                        break
                    raw_bytes = message.get("bytes")
                    raw_text = message.get("text")
                    with contextlib.suppress(Exception):
                        if raw_bytes is not None:
                            self._handle_edge_binary_frame(robot_id, raw_bytes)
                        elif raw_text is not None:
                            await self._handle_edge_text_frame(robot_id, raw_text)
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
    parser.add_argument("--frontend-queue-size", type=int, default=32)
    parser.add_argument("--frontend-send-timeout-s", type=float, default=2.0)
    parser.add_argument(
        "--frontend-drain-max",
        type=int,
        default=48,
        help="Max queued messages flushed to a frontend per sender wake-up.",
    )

    parser.add_argument("--lidar-voxel-size", type=float, default=0.08)
    parser.add_argument("--lidar-map-max-voxels", type=int, default=120000)
    parser.add_argument("--lidar-max-packet-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--lidar-min-z", type=float, default=-1.5)
    parser.add_argument("--lidar-max-z", type=float, default=3.5)
    parser.add_argument("--lidar-path-max-points", type=int, default=1000)
    # Live point-cloud streaming (browser renders it in WebGL; no server raster).
    parser.add_argument(
        "--lidar-keyframe-interval-s",
        type=float,
        default=3.0,
        help="Seconds between full-cloud keyframes (deltas stream in between).",
    )
    parser.add_argument(
        "--lidar-cloud-max-points",
        type=int,
        default=60000,
        help="Max points sent in a keyframe (LOD downsample budget).",
    )
    parser.add_argument("--lidar-cloud-quantization-cm", type=float, default=2.0)
    parser.add_argument("--lidar-cloud-compression-level", type=int, default=6)

    # Automatic solid-mesh reconstruction (background; saved + served to viewer).
    parser.add_argument(
        "--mesh-interval-s",
        type=float,
        default=60.0,
        help="Seconds between background mesh rebuilds (0 disables).",
    )
    parser.add_argument("--mesh-voxel-size", type=float, default=0.08,
                        help="Mesh reconstruction resolution in meters.")
    parser.add_argument("--mesh-min-voxels", type=int, default=300,
                        help="Min accumulated voxels before a mesh is built.")
    parser.add_argument("--mesh-smooth-iters", type=int, default=8,
                        help="Laplacian smoothing iterations (rounds the blocky shell).")
    parser.add_argument("--mesh-max-vertices", type=int, default=1_500_000,
                        help="Skip mesh if it would exceed this many vertices.")
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

    # --- Autonomous exploration ----------------------------------------------
    parser.add_argument("--autonomy-control-hz", type=float, default=5.0,
                        help="Autonomy decision/drive rate.")
    parser.add_argument("--autonomy-v-max", type=float, default=0.35,
                        help="Max forward speed during autonomy (m/s).")
    parser.add_argument("--autonomy-w-max", type=float, default=0.8,
                        help="Max yaw rate during autonomy (rad/s).")
    parser.add_argument("--autonomy-grid-res", type=float, default=0.15,
                        help="Frontier occupancy grid resolution (m).")
    parser.add_argument("--autonomy-robot-radius-m", type=float, default=0.30,
                        help="Robot radius used to inflate obstacles for planning.")

    # --- Server perception (people + faces) ----------------------------------
    parser.add_argument("--enable-perception", dest="enable_perception",
                        action="store_true", default=True,
                        help="Enable server-side perception (default on; ML deps optional).")
    parser.add_argument("--disable-perception", dest="enable_perception",
                        action="store_false", help="Disable server-side perception.")
    parser.add_argument("--enable-person-detection", dest="enable_person_detection",
                        action="store_true", default=True)
    parser.add_argument("--disable-person-detection", dest="enable_person_detection",
                        action="store_false")
    parser.add_argument("--enable-face-recognition", dest="enable_face_recognition",
                        action="store_true", default=True)
    parser.add_argument("--disable-face-recognition", dest="enable_face_recognition",
                        action="store_false")
    parser.add_argument("--perception-device", default="cuda", choices=["cuda", "cpu"],
                        help="Device for YOLO/insightface (GPU server: cuda).")
    parser.add_argument("--person-model", default="yolov8n.pt",
                        help="Ultralytics model for person detection.")
    parser.add_argument("--person-conf", type=float, default=0.45,
                        help="Person detection confidence threshold.")
    parser.add_argument("--face-match-threshold", type=float, default=0.35,
                        help="Cosine similarity to consider two faces the same person.")
    parser.add_argument("--face-retention-days", type=float, default=0.0,
                        help="Auto-delete unknown faces older than N days (0 = keep; privacy).")
    parser.add_argument("--faces-dir", default="./server/faces",
                        help="Where face crops + embeddings are stored.")

    # --- Greeter (play Go2 greeting on person detection) ---------------------
    parser.add_argument("--greet-interval-s", type=float, default=5.0,
                        help="Minimum seconds between greetings.")
    parser.add_argument("--greeter-detect-hz", type=float, default=2.5,
                        help="How often the greeter runs person detection.")
    parser.add_argument("--greeter-person-conf", type=float, default=0.5,
                        help="Person confidence needed to greet.")
    parser.add_argument("--greeter-min-area-frac", type=float, default=0.03,
                        help="Min person bbox area fraction of the frame to greet.")
    parser.add_argument("--greeter-with-autonomy", dest="greeter_with_autonomy",
                        action="store_true", default=True,
                        help="Also greet while autonomous exploration runs (default on).")
    parser.add_argument("--no-greeter-with-autonomy", dest="greeter_with_autonomy",
                        action="store_false")

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

    if args.lidar_map_max_voxels <= 0 or args.lidar_cloud_max_points <= 0:
        parser.error("lidar point limits must be > 0")

    if args.lidar_max_packet_bytes <= 0:
        parser.error("--lidar-max-packet-bytes must be > 0")

    if args.lidar_min_z >= args.lidar_max_z:
        parser.error("--lidar-min-z must be lower than --lidar-max-z")

    if args.lidar_keyframe_interval_s <= 0:
        parser.error("--lidar-keyframe-interval-s must be > 0")

    if args.lidar_cloud_quantization_cm <= 0:
        parser.error("--lidar-cloud-quantization-cm must be > 0")

    if args.lidar_cloud_compression_level < 0 or args.lidar_cloud_compression_level > 9:
        parser.error("--lidar-cloud-compression-level must be between 0 and 9")

    if args.lidar_path_max_points <= 0:
        parser.error("--lidar-path-max-points must be > 0")

    if args.frontend_drain_max <= 0:
        parser.error("--frontend-drain-max must be > 0")

    if args.mesh_voxel_size <= 0:
        parser.error("--mesh-voxel-size must be > 0")

    if args.mesh_min_voxels <= 0 or args.mesh_max_vertices <= 0:
        parser.error("mesh size limits must be > 0")

    if args.mesh_smooth_iters < 0:
        parser.error("--mesh-smooth-iters must be >= 0")

    if not str(args.map_storage_dir).strip():
        parser.error("--map-storage-dir must not be empty")

    if args.map_autosave_interval_s <= 0:
        parser.error("--map-autosave-interval-s must be > 0")

    if args.map_max_snapshots < 0:
        parser.error("--map-max-snapshots must be >= 0")

    if args.greet_interval_s <= 0:
        parser.error("--greet-interval-s must be > 0")

    return args


def raise_fd_limit() -> None:
    """Raise the open-file limit toward the system hard limit. The media/WebSocket
    fan-out plus perception decoding open many descriptors; avoids [Errno 24]."""
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = hard if hard != resource.RLIM_INFINITY else 1048576
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            new_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            print(f"[server] open-file limit raised {soft} -> {new_soft}", flush=True)
    except Exception as exc:
        print(f"[server] could not raise open-file limit: {exc}", flush=True)


def main() -> None:
    raise_fd_limit()
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
