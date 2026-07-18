#!/usr/bin/env python3
"""Standalone Unitree Go2 LiDAR capture and Raspberry -> server uplink.

This process deliberately does only three things: connect to the Go2 WebRTC
service, decode its LiDAR voxel map, and send the newest cloud to the 3D server.
It doesn't start MQTT, camera, audio, control, autonomy, or the dashboard.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress
import json
import math
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lidar3d.protocol import ProtocolError, encode_cloud_frame  # noqa: E402

try:
    import websockets
except ImportError as exc:  # pragma: no cover - clearer runtime failure on the Pi
    raise SystemExit(
        "Missing websockets. Install with: ./go2 install lidar-raspi"
    ) from exc

try:
    from unitree_webrtc_connect import (
        RTC_TOPIC,
        UnitreeWebRTCConnection,
        WebRTCConnectionMethod,
    )
except ImportError:
    try:
        from unitree_webrtc_connect.constants import RTC_TOPIC, WebRTCConnectionMethod
        from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection
    except ImportError as exc:  # pragma: no cover - hardware dependency
        raise SystemExit(
            "Missing unitree_webrtc_connect. Install with: ./go2 install lidar-raspi"
        ) from exc


TOPIC = dict(RTC_TOPIC)
LIDAR_TOPIC = TOPIC.get("ULIDAR_ARRAY", "rt/utlidar/voxel_map_compressed")
POSE_TOPIC = TOPIC.get("ROBOTODOM", "rt/utlidar/robot_pose")
SWITCH_TOPIC = TOPIC.get("ULIDAR_SWITCH", "rt/utlidar/switch")


def patch_nonstandard_unitree_json() -> None:
    """Accept bare inf/nan values emitted by some Go2 LiDAR-state messages."""
    try:
        from unitree_webrtc_connect import webrtc_datachannel
    except Exception:
        return

    real_json = json
    invalid_number = re.compile(
        r"(?<=[:,\[\s])-?(?:inf|nan)(?=[,\]}\s])", re.IGNORECASE
    )

    def replace_number(match: "re.Match[str]") -> str:
        token = match.group(0)
        if token.lower().endswith("nan"):
            return "NaN"
        return "-Infinity" if token.startswith("-") else "Infinity"

    class TolerantJson:
        def __getattr__(self, name: str) -> Any:
            return getattr(real_json, name)

        def loads(self, value: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(value, str) and (
                "inf" in value.lower() or "nan" in value.lower()
            ):
                value = invalid_number.sub(replace_number, value)
            return real_json.loads(value, *args, **kwargs)

    webrtc_datachannel.json = TolerantJson()


patch_nonstandard_unitree_json()


def extract_points(value: Any, depth: int = 0) -> Optional[np.ndarray]:
    """Extract native or libvoxel XYZ output from the Unitree callback payload."""
    if value is None or depth > 8:
        return None
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype == np.uint8 and array.ndim == 1 and array.size % 12 == 0:
            decoded = np.frombuffer(array.tobytes(), dtype="<f4").reshape(-1, 3)
            return decoded if decoded.size else None
        if array.ndim == 2 and array.shape[1] >= 3:
            return array[:, :3].astype(np.float32, copy=False)
        if array.ndim == 1 and array.size % 3 == 0:
            return array.astype(np.float32, copy=False).reshape(-1, 3)
        return None
    if isinstance(value, dict):
        # Native decoder: data.data.points. Libvoxel: data.data.positions as
        # packed float32 bytes exposed in a uint8 ndarray.
        for key in ("points", "positions", "cloud", "cloud_points", "xyz", "data"):
            if key in value:
                found = extract_points(value[key], depth + 1)
                if found is not None:
                    return found
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        with contextlib.suppress(TypeError, ValueError):
            array = np.asarray(value, dtype=np.float32)
            if array.ndim == 2 and array.shape[1] >= 3:
                return array[:, :3]
            if array.ndim == 1 and array.size % 3 == 0:
                return array.reshape(-1, 3)
        for item in value[:8]:
            found = extract_points(item, depth + 1)
            if found is not None:
                return found
    return None


def _xyz(value: Any) -> Optional[Tuple[float, float, float]]:
    if isinstance(value, dict):
        with contextlib.suppress(TypeError, ValueError):
            return (
                float(value.get("x", 0)),
                float(value.get("y", 0)),
                float(value.get("z", 0)),
            )
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        with contextlib.suppress(TypeError, ValueError):
            return (
                float(value[0]),
                float(value[1]),
                float(value[2] if len(value) > 2 else 0),
            )
    return None


def _yaw(orientation: Any, node: Dict[str, Any]) -> float:
    if isinstance(orientation, dict):
        values = [orientation.get(key) for key in ("x", "y", "z", "w")]
    elif isinstance(orientation, (list, tuple)) and len(orientation) >= 4:
        values = list(orientation[:4])
    else:
        values = []
    if len(values) == 4 and all(value is not None for value in values):
        with contextlib.suppress(TypeError, ValueError):
            qx, qy, qz, qw = (float(value) for value in values)
            return math.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )
    rpy = node.get("rpy")
    if isinstance(rpy, (list, tuple)) and len(rpy) >= 3:
        with contextlib.suppress(TypeError, ValueError):
            return float(rpy[2])
    return 0.0


def extract_pose(message: Any) -> Optional[Dict[str, float]]:
    if not isinstance(message, dict):
        return None
    data = message.get("data", message)
    if not isinstance(data, dict):
        return None
    nodes = [data]
    if isinstance(data.get("pose"), dict):
        nodes.insert(0, data["pose"])
    orientation = next(
        (
            node.get("orientation")
            for node in nodes
            if node.get("orientation") is not None
        ),
        None,
    )
    rpy_node = next(
        (node for node in nodes if isinstance(node.get("rpy"), (list, tuple))), {}
    )
    for node in nodes:
        position = _xyz(node.get("position"))
        if position is not None:
            return {
                "x": position[0],
                "y": position[1],
                "z": position[2],
                "yaw": _yaw(orientation, rpy_node),
            }
    return None


def cloud_source_metadata(message: Any) -> Dict[str, Any]:
    data = message.get("data", {}) if isinstance(message, dict) else {}
    if not isinstance(data, dict):
        return {}
    result: Dict[str, Any] = {}
    for key in ("resolution", "stamp", "frame_id", "src_size"):
        value = data.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            if not isinstance(value, float) or math.isfinite(value):
                result[f"unitree_{key}"] = value
    origin = data.get("origin")
    if isinstance(origin, (list, tuple)) and len(origin) >= 3:
        with contextlib.suppress(TypeError, ValueError):
            values = [float(origin[index]) for index in range(3)]
            if all(math.isfinite(value) for value in values):
                result["unitree_origin"] = values
    return result


class LidarOnlySender:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.stop_event = asyncio.Event()
        self.raw_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
        self.wire_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
        self.connection: Optional[UnitreeWebRTCConnection] = None
        self.latest_pose: Optional[Dict[str, float]] = None
        self.frame_id = 0
        self.frames_received = 0
        self.frames_encoded = 0
        self.frames_sent = 0
        self.frames_dropped = 0
        self.last_point_count = 0
        self.last_wire_bytes = 0
        self.last_emit_at = 0.0

    @staticmethod
    def _replace_latest(queue: asyncio.Queue[Any], item: Any) -> None:
        if queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(item)

    def _offer_lidar(self, message: Any) -> None:
        self.frames_received += 1
        if self.raw_queue.full():
            self.frames_dropped += 1
        self._replace_latest(self.raw_queue, message)

    def _lidar_callback(self, message: Any) -> None:
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._offer_lidar, message)

    def _pose_callback(self, message: Any) -> None:
        pose = extract_pose(message)
        if pose is not None and self.loop is not None:
            self.loop.call_soon_threadsafe(setattr, self, "latest_pose", pose)

    def _prepare_frame(self, message: Any) -> Optional[bytes]:
        points = extract_points(message)
        if points is None:
            return None
        cloud = np.asarray(points, dtype=np.float32)
        if cloud.ndim != 2 or cloud.shape[1] < 3:
            return None
        cloud = cloud[:, :3]
        cloud = cloud[np.isfinite(cloud).all(axis=1)]
        if self.args.min_z is not None:
            cloud = cloud[cloud[:, 2] >= self.args.min_z]
        if self.args.max_z is not None:
            cloud = cloud[cloud[:, 2] <= self.args.max_z]
        if cloud.shape[0] == 0:
            return None

        if self.args.edge_voxel_size > 0:
            keys = np.floor(cloud / self.args.edge_voxel_size).astype(np.int32)
            _, indices = np.unique(keys, axis=0, return_index=True)
            cloud = cloud[np.sort(indices)]
        if cloud.shape[0] > self.args.max_points:
            indices = np.linspace(
                0, cloud.shape[0] - 1, self.args.max_points, dtype=np.int64
            )
            cloud = cloud[indices]

        self.frame_id += 1
        metadata: Dict[str, Any] = {
            "robot_id": self.args.robot_id,
            "frame_id": self.frame_id,
            "ts": time.time(),
            "coordinate_frame": "utlidar_odom",
            "source_points": int(np.asarray(points).shape[0]),
        }
        if self.latest_pose is not None:
            metadata["pose"] = dict(self.latest_pose)
        metadata.update(cloud_source_metadata(message))
        wire = encode_cloud_frame(
            cloud,
            metadata,
            quantization_cm=self.args.quantization_cm,
            compression_level=self.args.compression_level,
        )
        self.last_point_count = int(cloud.shape[0])
        self.last_wire_bytes = len(wire)
        return wire

    async def encode_loop(self) -> None:
        interval = 1.0 / self.args.send_hz
        while not self.stop_event.is_set():
            message = await self.raw_queue.get()
            remaining = interval - (time.monotonic() - self.last_emit_at)
            if remaining > 0:
                await asyncio.sleep(remaining)
                # Keep only the newest map accumulated while rate limiting.
                while not self.raw_queue.empty():
                    with contextlib.suppress(asyncio.QueueEmpty):
                        message = self.raw_queue.get_nowait()
            try:
                wire = await asyncio.to_thread(self._prepare_frame, message)
            except (ProtocolError, ValueError) as exc:
                print(f"[raspi] discarded LiDAR frame: {exc}", flush=True)
                continue
            if wire is None:
                print("[raspi] Unitree frame had no decodable XYZ points", flush=True)
                continue
            self.last_emit_at = time.monotonic()
            self.frames_encoded += 1
            self._replace_latest(self.wire_queue, wire)

    def _server_url(self) -> str:
        if self.args.server_ws_url:
            url = self.args.server_ws_url.format(robot_id=self.args.robot_id)
        else:
            address = ipaddress.ip_address(self.args.server_ip)
            host = f"[{address}]" if address.version == 6 else str(address)
            url = f"ws://{host}:{self.args.server_port}/lidar/{self.args.robot_id}"
        if self.args.token:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}token={quote(self.args.token, safe='')}"
        return url

    async def uplink_loop(self) -> None:
        url = self._server_url()
        while not self.stop_event.is_set():
            try:
                async with websockets.connect(
                    url,
                    compression=None,
                    open_timeout=10,
                    close_timeout=2,
                    ping_interval=15,
                    ping_timeout=15,
                    max_size=self.args.ws_max_size,
                    max_queue=2,
                    write_limit=128 * 1024,
                ) as websocket:
                    print(f"[raspi] server connected: {url.split('?')[0]}", flush=True)
                    while not self.stop_event.is_set():
                        wire = await self.wire_queue.get()
                        await asyncio.wait_for(
                            websocket.send(wire), timeout=self.args.send_timeout
                        )
                        self.frames_sent += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[raspi] server unavailable ({exc}); retrying", flush=True)
                await asyncio.sleep(self.args.reconnect_seconds)

    def _build_robot_connection(self) -> UnitreeWebRTCConnection:
        method = (
            WebRTCConnectionMethod.LocalAP
            if self.args.connection == "local-ap"
            else WebRTCConnectionMethod.LocalSTA
        )
        kwargs: Dict[str, Any] = {}
        if method == WebRTCConnectionMethod.LocalSTA:
            kwargs["ip"] = self.args.go2_ip
        if self.args.aes_key:
            kwargs["aes_128_key"] = self.args.aes_key
        return UnitreeWebRTCConnection(method, **kwargs)

    async def robot_loop(self) -> None:
        while not self.stop_event.is_set():
            connection: Optional[UnitreeWebRTCConnection] = None
            try:
                connection = self._build_robot_connection()
                await asyncio.wait_for(
                    connection.connect(), timeout=self.args.robot_connect_timeout
                )
                self.connection = connection
                if not self.args.keep_traffic_saving:
                    await asyncio.wait_for(
                        connection.datachannel.disableTrafficSaving(True), timeout=4
                    )
                connection.datachannel.set_decoder(self.args.decoder)
                connection.datachannel.pub_sub.subscribe(
                    LIDAR_TOPIC, self._lidar_callback
                )
                connection.datachannel.pub_sub.subscribe(
                    POSE_TOPIC, self._pose_callback
                )
                connection.datachannel.pub_sub.publish_without_callback(
                    SWITCH_TOPIC, "on"
                )
                print(
                    f"[raspi] Go2 connected at {self.args.go2_ip}; "
                    f"LiDAR={LIDAR_TOPIC}, decoder={self.args.decoder}",
                    flush=True,
                )
                while not self.stop_event.is_set():
                    await asyncio.sleep(1)
                    pc = getattr(connection, "pc", None)
                    state = getattr(pc, "connectionState", "connected")
                    if state in {"failed", "closed"}:
                        raise ConnectionError(f"WebRTC state is {state}")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[raspi] Go2 connection failed ({exc}); retrying", flush=True)
            finally:
                self.connection = None
                if connection is not None:
                    if self.args.lidar_off_on_exit:
                        with contextlib.suppress(Exception):
                            connection.datachannel.pub_sub.publish_without_callback(
                                SWITCH_TOPIC, "off"
                            )
                    with contextlib.suppress(Exception):
                        await connection.disconnect()
            if not self.stop_event.is_set():
                await asyncio.sleep(self.args.reconnect_seconds)

    async def status_loop(self) -> None:
        while not self.stop_event.is_set():
            await asyncio.sleep(5)
            print(
                "[raspi] "
                f"received={self.frames_received} encoded={self.frames_encoded} "
                f"sent={self.frames_sent} dropped={self.frames_dropped} "
                f"points={self.last_point_count} wire={self.last_wire_bytes / 1024:.1f}KiB",
                flush=True,
            )

    def request_stop(self) -> None:
        self.stop_event.set()

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                self.loop.add_signal_handler(sig, self.request_stop)
        tasks = [
            asyncio.create_task(self.robot_loop(), name="unitree-lidar"),
            asyncio.create_task(self.encode_loop(), name="cloud-encoder"),
            asyncio.create_task(self.uplink_loop(), name="server-uplink"),
            asyncio.create_task(self.status_loop(), name="status"),
        ]
        await self.stop_event.wait()
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture only the Go2 LiDAR on the Raspberry Pi and stream it to the 3D server."
    )
    parser.add_argument("--robot-id", default="go2_01")
    parser.add_argument(
        "--connection", choices=["local-sta", "local-ap"], default="local-sta"
    )
    parser.add_argument("--go2-ip", default="192.168.123.161")
    parser.add_argument(
        "--aes-key",
        default=os.environ.get("UNITREE_AES_KEY", ""),
        help="32-hex per-device key for Go2 firmware >=1.1.15; or UNITREE_AES_KEY.",
    )
    parser.add_argument("--decoder", choices=["native", "libvoxel"], default="native")
    parser.add_argument("--keep-traffic-saving", action="store_true")
    parser.add_argument(
        "--lidar-off-on-exit",
        action="store_true",
        help="Turn the shared robot LiDAR service off when this process exits.",
    )

    server_target = parser.add_mutually_exclusive_group(required=True)
    server_target.add_argument(
        "--server-ip",
        help="Server IPv4/IPv6 entered when starting the Raspberry process.",
    )
    server_target.add_argument(
        "--server-ws-url",
        help="Advanced alternative: ws://host:8765/lidar/{robot_id}",
    )
    parser.add_argument("--server-port", type=int, default=8765)
    parser.add_argument("--token", default=os.environ.get("GO2_LIDAR_TOKEN", ""))
    parser.add_argument("--send-hz", type=float, default=1.0)
    parser.add_argument("--max-points", type=int, default=50_000)
    parser.add_argument("--edge-voxel-size", type=float, default=0.0)
    parser.add_argument("--quantization-cm", type=float, default=1.0)
    parser.add_argument("--compression-level", type=int, default=6)
    parser.add_argument("--min-z", type=float, default=None)
    parser.add_argument("--max-z", type=float, default=None)
    parser.add_argument("--send-timeout", type=float, default=10.0)
    parser.add_argument("--reconnect-seconds", type=float, default=2.0)
    parser.add_argument("--robot-connect-timeout", type=float, default=25.0)
    parser.add_argument("--ws-max-size", type=int, default=32 * 1024 * 1024)
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", args.robot_id):
        parser.error("--robot-id contains unsupported characters")
    if args.aes_key and not re.fullmatch(r"[0-9a-fA-F]{32}", args.aes_key):
        parser.error("--aes-key must contain exactly 32 hexadecimal characters")
    if args.server_ip:
        try:
            ipaddress.ip_address(args.server_ip)
        except ValueError:
            parser.error("--server-ip must be a valid IPv4 or IPv6 address")
    if not 1 <= args.server_port <= 65535:
        parser.error("--server-port must be between 1 and 65535")
    if args.send_hz <= 0 or args.send_hz > 20:
        parser.error("--send-hz must be in (0, 20]")
    if args.max_points < 2048:
        parser.error("--max-points must be >= 2048")
    if args.edge_voxel_size < 0:
        parser.error("--edge-voxel-size must be >= 0")
    if args.quantization_cm <= 0 or args.quantization_cm > 20:
        parser.error("--quantization-cm must be in (0, 20]")
    if not 0 <= args.compression_level <= 9:
        parser.error("--compression-level must be between 0 and 9")
    if args.min_z is not None and args.max_z is not None and args.min_z >= args.max_z:
        parser.error("--min-z must be lower than --max-z")
    if args.send_timeout <= 0 or args.reconnect_seconds <= 0:
        parser.error("timeouts must be positive")
    return args


def main() -> None:
    args = parse_args()
    sender = LidarOnlySender(args)
    try:
        asyncio.run(sender.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
