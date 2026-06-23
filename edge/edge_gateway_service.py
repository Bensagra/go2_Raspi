#!/usr/bin/env python3
import argparse
import asyncio
import base64
import contextlib
import gc
import json
import math
import os
import time
import uuid
import zlib
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import av
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from av.audio.resampler import AudioResampler

try:
    import websockets
except Exception:
    websockets = None


# Binary media frame codec (must match server/server_core.py): magic(1)
# version(1) header_len(u32 LE) header(JSON) payload(raw bytes). Sending the
# compressed/encoded blob as-is (no base64) is ~33% lighter and skips the
# base64 encode on the Pi for every frame.
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


def encode_cloud_payload(points, colors, quantization_cm: float, compression_level: int):
    """Quantize Nx3 points to int16+zlib, optionally with per-point RGBA color.

    Returns (payload, fmt, scale, offset, count). Colored layout is
    `u32 geom_len | zlib(int16 xyz) | zlib(uint8 rgba)` so a single payload
    carries geometry + color; uncolored is just `zlib(int16 xyz)`."""
    points = np.ascontiguousarray(points[:, :3], dtype=np.float32)
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    offset = (bounds_min + bounds_max) / 2.0
    scale = max(quantization_cm / 100.0, 0.001)
    max_delta = float(np.max(np.abs(points - offset))) if points.size else 0.0
    if max_delta > 0:
        scale = max(scale, max_delta / 32760.0)
    quantized = np.clip(np.rint((points - offset) / scale), -32768, 32767).astype("<i2")
    geom = zlib.compress(np.ascontiguousarray(quantized).tobytes(), compression_level)
    count = int(points.shape[0])
    if colors is None:
        return geom, "i16_xyz_zlib", float(scale), offset.tolist(), count
    rgba = np.ascontiguousarray(colors, dtype=np.uint8)
    col = zlib.compress(rgba.tobytes(), compression_level)
    payload = len(geom).to_bytes(4, "little") + geom + col
    return payload, "i16_xyz_rgb_zlib", float(scale), offset.tolist(), count

try:
    from unitree_webrtc_connect import (
        DATA_CHANNEL_TYPE,
        RTC_TOPIC,
        SPORT_CMD,
        UnitreeWebRTCConnection,
        WebRTCConnectionMethod,
    )
except ImportError:
    from unitree_webrtc_connect.constants import (  # type: ignore
        DATA_CHANNEL_TYPE,
        RTC_TOPIC,
        SPORT_CMD,
        WebRTCConnectionMethod,
    )
    from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection  # type: ignore


try:
    from safety_guard import SafetyGuard
except ImportError:  # when imported as a package (edge.edge_gateway_service)
    from edge.safety_guard import SafetyGuard  # type: ignore

try:
    from audio_greeting import (
        AudioGreetingError,
        find_audio_uuid,
        prepare_go2_wav,
        resolve_audio_file,
        resolve_audio_uuid_on_hub,
    )
except ImportError:  # when imported as a package (edge.edge_gateway_service)
    from edge.audio_greeting import (  # type: ignore
        AudioGreetingError,
        find_audio_uuid,
        prepare_go2_wav,
        resolve_audio_file,
        resolve_audio_uuid_on_hub,
    )

try:
    from unitree_webrtc_connect.webrtc_audiohub import WebRTCAudioHub
except Exception:  # optional: greeting/audio playback degrades gracefully
    WebRTCAudioHub = None  # type: ignore


TOPIC_ALIAS_TO_VALUE = dict(RTC_TOPIC)
TOPIC_VALUE_TO_ALIAS = {value: key for key, value in TOPIC_ALIAS_TO_VALUE.items()}

PROFILE_TOPICS = {
    "core": ["LOW_STATE", "LF_SPORT_MOD_STATE", "SPORT_MOD_STATE", "MULTIPLE_STATE", "GAS_SENSOR"],
    "lidar": ["ULIDAR_ARRAY", "ULIDAR_STATE", "ROBOTODOM"],
    "audio": ["AUDIO_HUB_PLAY_STATE"],
    "all": sorted(TOPIC_ALIAS_TO_VALUE.keys()),
}


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


class EdgeGatewayService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.stop_event = asyncio.Event()

        self.conn: Optional[UnitreeWebRTCConnection] = None
        self.video_task: Optional[asyncio.Task[None]] = None
        self.video_encode_task: Optional[asyncio.Task[None]] = None
        self.latest_camera_frame = None
        self.camera_frame_ready = asyncio.Event()
        self.robot_connected_at = 0.0
        self.traffic_saving_disabled = False

        self.command_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=args.command_queue_size)
        self.latest_motion_command: Optional[Dict[str, Any]] = None
        self.motion_marker_queued = False
        self.media_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=args.media_queue_size)
        self.latest_media_by_stream: Dict[str, Dict[str, Any]] = {}
        self.media_ready = asyncio.Event()
        self.media_max_kbps = args.media_max_kbps
        self.media_budget_drops: Dict[str, int] = {}
        self.robot_request_lock = asyncio.Lock()
        self.robot_request_sequence = int(time.time() * 1000) & 0x3FFFFFFF

        self.mqtt_client: Optional[mqtt.Client] = None
        self.last_heartbeat_monotonic = time.monotonic()
        self.last_heartbeat_payload: Dict[str, Any] = {}

        self.subscribed_topics: set[str] = set()
        self.latest_by_topic: Dict[str, Any] = {}

        self.camera_enabled = args.enable_camera
        self.camera_channel_active = False
        self.camera_emit_every = args.camera_emit_every
        self.camera_format = args.camera_format
        self.camera_jpeg_quality = args.camera_jpeg_quality
        self.camera_min_quality = args.camera_min_quality
        self.camera_target_fps = args.camera_target_fps
        self.camera_max_width = args.camera_max_width
        self.camera_frame_count = 0
        self.camera_encoded_count = 0
        self.last_camera_emit_at = 0.0

        # H.264 (inter-frame) encoder state. When camera_format == "h264" the edge
        # encodes a real video bitstream instead of independent WebP/JPEG frames,
        # cutting bandwidth ~7-15x for the same picture.
        self.camera_bitrate_kbps = args.camera_bitrate_kbps
        self.camera_gop = args.camera_gop
        self.video_encoder = None
        self.video_encoder_name = ""
        self.video_encoder_key: Optional[Tuple[Any, ...]] = None
        self.video_codec_string = ""
        self.video_pts = 0
        self.video_drop_count = 0
        self.last_video_encode_error_at = 0.0
        self.video_media_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(
            maxsize=args.video_queue_size
        )

        self.audio_enabled = args.enable_audio
        self.audio_emit_every = args.audio_emit_every
        self.audio_max_bytes = args.audio_max_bytes
        self.audio_frame_count = 0
        self.last_audio_error_at = 0.0
        self.audio_resampler = AudioResampler(
            format="s16",
            layout="mono",
            rate=args.audio_sample_rate,
        )
        self.audio_pcm_buffer = bytearray()
        self.audio_buffered_frames = 0

        self.lidar_enabled = args.enable_lidar
        self.lidar_media_hz = args.lidar_media_hz
        self.lidar_uplink_max_points = args.lidar_uplink_max_points
        self.lidar_compression_level = args.lidar_compression_level
        self.lidar_quantization_cm = args.lidar_quantization_cm
        self.last_lidar_media_at = 0.0

        # LiDAR pipeline counters (so we can see, from telemetry, exactly where
        # the chain breaks: arriving -> extracted -> uploaded).
        self.lidar_frames_in = 0
        self.lidar_points_last = 0
        self.lidar_media_sent = 0
        self.lidar_extract_fail = 0
        self.lidar_debug_dumped = False

        # Camera colorization: project each LiDAR point onto the live camera frame
        # (approximate fisheye, no calibration) and sample its RGB. Extrinsics are
        # tunable live via the set_color command so they can be eyeballed.
        self.colorize_enabled = args.enable_colorization
        self.color_cam_fov_deg = args.color_cam_fov_deg
        self.color_cam_pitch_deg = args.color_cam_pitch_deg
        self.color_cam_height_m = args.color_cam_height_m
        self.color_cam_forward_m = args.color_cam_forward_m
        self.color_max_distance_m = args.color_max_distance_m
        # "world": LiDAR points are in the odom frame (default, matches the
        # accumulated map). "body": already robot-relative (skip pose transform).
        self.color_points_frame = args.color_points_frame
        self.latest_color_frame = None

        self.move_active = False
        self.pending_stop_deadline = 0.0
        self.last_move_command: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.last_move_sent_at = 0.0
        self.speed_profile = "normal"
        self.speed_level = 0
        self.last_speed_profile_reset_attempt = 0.0

        self.last_heartbeat_fault_event_at = 0.0

        # Reactive LiDAR safety guard (hard "never collide / never fall" layer).
        # Runs locally on the raw scan so it survives network loss. It filters
        # EVERY translation command (manual move and autonomous drive) but only
        # while it actually has a fresh LiDAR scan (see _guard_velocity).
        self.safety_enabled = args.enable_safety_guard
        self.safety_points_frame = args.safety_points_frame
        self.safety_update_hz = args.safety_update_hz
        self.last_safety_update_at = 0.0
        self.safety_guard = SafetyGuard(
            obstacle_min_height_m=args.safety_obstacle_min_height_m,
            obstacle_max_height_m=args.safety_obstacle_max_height_m,
            robot_half_width_m=args.safety_robot_half_width_m,
            stop_distance_m=args.safety_stop_distance_m,
            slow_distance_m=args.safety_slow_distance_m,
            motion_cone_half_deg=args.safety_cone_half_deg,
            cliff_enabled=args.safety_cliff_enabled,
            cliff_lookahead_m=args.safety_cliff_lookahead_m,
            cliff_drop_m=args.safety_cliff_drop_m,
            ground_z_default_m=args.safety_ground_z_m,
            max_consider_radius_m=args.safety_max_radius_m,
            scan_timeout_s=args.safety_scan_timeout_s,
        )
        self.last_safety_info: Dict[str, Any] = {}
        self.last_safety_event_at = 0.0

        # Autonomy: continuous velocity target repeated by the drive loop and
        # always passed through the safety guard. Enabled by set_autonomy; the
        # server brain feeds drive_velocity goals.
        self.autonomy_enabled = False
        self.drive_target: Optional[Dict[str, float]] = None
        self.drive_target_set_at = 0.0
        self.drive_target_ttl_s = args.drive_target_ttl_s
        self.autonomy_driving = False

        # Audio greeting: play a file through the Go2 speaker (audio hub). The
        # server triggers it on person detection; the edge owns the playback.
        self.audio_hub = None
        self.audio_hub_play_mode_ready = False
        self.audio_hub_lock = asyncio.Lock()
        self.greet_prewarm_task: Optional[asyncio.Task[None]] = None
        self.greet_audio_file = resolve_audio_file(args.greet_audio_file)
        self.greet_audio_uuid = args.greet_audio_uuid or None
        self.greet_audio_uuid_configured = bool(args.greet_audio_uuid)
        self.greet_audio_resolved = False
        self.last_greet_at = 0.0
        self.greet_min_interval_s = args.greet_min_interval_s

    def _reset_audio_pipeline(self) -> None:
        self.audio_resampler = AudioResampler(
            format="s16",
            layout="mono",
            rate=self.args.audio_sample_rate,
        )
        self.audio_pcm_buffer.clear()
        self.audio_buffered_frames = 0

    def _resolve_topic(self, topic: str) -> str:
        if topic in TOPIC_ALIAS_TO_VALUE:
            return TOPIC_ALIAS_TO_VALUE[topic]
        if topic in TOPIC_VALUE_TO_ALIAS:
            return topic
        raise ValueError(f"Unknown topic: {topic}")

    def _topic_alias(self, topic_value: str) -> str:
        return TOPIC_VALUE_TO_ALIAS.get(topic_value, topic_value)

    def _mqtt_topic(self, suffix: str) -> str:
        return f"{self.args.mqtt_topic_prefix}/{self.args.robot_id}/{suffix}"

    def _mqtt_publish(self, suffix: str, payload: Dict[str, Any], qos: int = 0, retain: bool = False) -> None:
        if self.mqtt_client is None:
            return
        topic = self._mqtt_topic(suffix)
        message = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        self.mqtt_client.publish(topic, message, qos=qos, retain=retain)

    def _publish_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        self._mqtt_publish(
            "events",
            {
                "robot_id": self.args.robot_id,
                "ts": time.time(),
                "event": event_type,
                "data": payload,
            },
            qos=1,
        )

    def _send_command_ack(
        self,
        command: Dict[str, Any],
        status: str,
        reason: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "command_id": str(command.get("command_id", "")),
            "robot_id": self.args.robot_id,
            "command_type": str(command.get("type", "")),
            "status": status,
            "reason": reason,
            "edge_ts": time.time(),
        }
        if extra:
            payload.update(extra)
        self._mqtt_publish("commands/ack", payload, qos=1)

    def _setup_mqtt(self) -> None:
        client_id = self.args.mqtt_client_id or f"edge-{self.args.robot_id}-{uuid.uuid4().hex[:8]}"
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
            self._publish_event("mqtt_connect_failed", {"rc": rc})
            return

        client.subscribe(self._mqtt_topic("commands/in"), qos=1)
        client.subscribe(self._mqtt_topic("control/heartbeat"), qos=0)
        self._publish_event(
            "mqtt_connected",
            {
                "host": self.args.mqtt_host,
                "port": self.args.mqtt_port,
                "commands_topic": self._mqtt_topic("commands/in"),
                "heartbeat_topic": self._mqtt_topic("control/heartbeat"),
            },
        )

    def _on_mqtt_disconnect(self, client, userdata, rc, properties=None) -> None:
        self._publish_event("mqtt_disconnected", {"rc": rc})

    def _on_mqtt_message(self, client, userdata, msg) -> None:
        if self.loop is None:
            return

        topic = msg.topic
        payload_raw = msg.payload.decode("utf-8", errors="ignore")
        try:
            payload = json.loads(payload_raw) if payload_raw else {}
        except Exception:
            self._publish_event("mqtt_bad_json", {"topic": topic})
            return

        if topic.endswith("/control/heartbeat"):
            if isinstance(payload, dict):
                self.last_heartbeat_payload = payload
                if bool(payload.get("session_active", False)):
                    self.last_heartbeat_monotonic = time.monotonic()
                else:
                    self.last_heartbeat_monotonic = 0.0
            return

        if topic.endswith("/commands/in"):
            if not isinstance(payload, dict):
                return

            if "command_id" not in payload:
                payload["command_id"] = f"cmd-{uuid.uuid4().hex[:12]}"
            payload.setdefault("robot_id", self.args.robot_id)

            def enqueue() -> None:
                command_type = str(payload.get("type", "")).strip()

                if command_type in {"move", "turn"}:
                    previous = self.latest_motion_command
                    self.latest_motion_command = payload

                    if self.motion_marker_queued:
                        if previous is not None and not previous.get("streaming"):
                            self._send_command_ack(previous, "rejected", "superseded by newer motion command")
                        return

                    try:
                        self.command_queue.put_nowait({"__motion_marker__": True})
                        self.motion_marker_queued = True
                    except asyncio.QueueFull:
                        self.latest_motion_command = previous
                        self._send_command_ack(payload, "rejected", "edge command queue full")
                    return

                if command_type == "stop" and self.latest_motion_command is not None:
                    pending_motion = self.latest_motion_command
                    self.latest_motion_command = None
                    if not pending_motion.get("streaming"):
                        self._send_command_ack(pending_motion, "rejected", "superseded by stop command")

                try:
                    self.command_queue.put_nowait(payload)
                except asyncio.QueueFull:
                    self._send_command_ack(payload, "rejected", "edge command queue full")

            self.loop.call_soon_threadsafe(enqueue)

    async def _on_topic_message(self, topic_value: str, message: Dict[str, Any]) -> None:
        self.latest_by_topic[topic_value] = message

        topic_alias = self._topic_alias(topic_value)
        event_payload = {
            "topic": topic_value,
            "topic_alias": topic_alias,
            "ts": time.time(),
        }
        self._mqtt_publish("topic_events", event_payload, qos=0)

        if self.lidar_enabled and topic_alias == "ULIDAR_ARRAY":
            self.lidar_frames_in += 1
            pts = self._extract_lidar_points(message)
            n = 0 if pts is None else int(np.asarray(pts).reshape(-1, 3).shape[0]) if getattr(pts, "size", 0) else 0
            self.lidar_points_last = n
            if n == 0:
                self.lidar_extract_fail += 1
                if not self.lidar_debug_dumped:
                    self.lidar_debug_dumped = True
                    self._publish_event("lidar_debug", self._describe_lidar_message(message))
            self._update_safety_from_lidar(message)
            await self._maybe_publish_lidar_media(message)

    async def _subscribe_topic(self, topic_value: str) -> None:
        if self.conn is None:
            return

        if topic_value in self.subscribed_topics:
            return

        def callback(message, t=topic_value):
            if self.loop is None:
                return
            self.loop.call_soon_threadsafe(asyncio.create_task, self._on_topic_message(t, message))

        self.conn.datachannel.pub_sub.subscribe(topic_value, callback)
        self.subscribed_topics.add(topic_value)

    async def _set_video(self, enabled: bool) -> None:
        if self.conn is None:
            raise RuntimeError("Robot connection not ready")
        self.camera_enabled = enabled
        if enabled:
            # Rebuild the encoder so the stream restarts on a keyframe (IDR),
            # otherwise late/re-subscribing viewers could not sync.
            self._reset_video_encoder()
        if enabled != self.camera_channel_active:
            self.conn.video.switchVideoChannel(enabled)
            self.camera_channel_active = enabled

    async def _set_audio(self, enabled: bool) -> None:
        if self.conn is None:
            raise RuntimeError("Robot connection not ready")

        if enabled != self.audio_enabled:
            self._reset_audio_pipeline()

        self.audio_enabled = enabled
        self.conn.audio.switchAudioChannel(enabled)

    async def _ensure_audio_hub(self):
        if WebRTCAudioHub is None or self.conn is None:
            return None
        if self.audio_hub is None:
            self.audio_hub = WebRTCAudioHub(self.conn)
            self.audio_hub_play_mode_ready = False
        if not self.audio_hub_play_mode_ready:
            try:
                await asyncio.wait_for(
                    self.audio_hub.set_play_mode("no_cycle"),
                    timeout=4.0,
                )
            except Exception:
                pass
            finally:
                self.audio_hub_play_mode_ready = True
        return self.audio_hub

    @staticmethod
    def _audio_list_find_uuid(response: Any, name: str) -> Optional[str]:
        return find_audio_uuid(response, name)

    def _prepare_wav(self, path: str) -> Optional[str]:
        """Return a cached 44.1 kHz mono PCM16 WAV accepted by the Go2."""
        return prepare_go2_wav(path)

    async def _resolve_greet_uuid(self) -> Optional[str]:
        if self.greet_audio_uuid:
            self.greet_audio_resolved = True
            return self.greet_audio_uuid
        hub = await self._ensure_audio_hub()
        if hub is None:
            return None

        async with self.audio_hub_lock:
            if self.greet_audio_uuid:
                return self.greet_audio_uuid
            try:
                uid, uploaded = await resolve_audio_uuid_on_hub(
                    hub,
                    self.greet_audio_file,
                    self._prepare_wav,
                )
            except AudioGreetingError as exc:
                self._publish_event(
                    "greet_audio_error",
                    {"stage": exc.stage, "error": str(exc), "file": self.greet_audio_file},
                )
                return None
            self.greet_audio_uuid = uid
            self.greet_audio_resolved = True
            self._publish_event("greet_audio_ready", {"uuid": uid, "uploaded": uploaded})
            return uid

    async def _prewarm_greet(self) -> None:
        await asyncio.sleep(2.0)  # let the data channel settle
        with contextlib.suppress(Exception):
            await self._resolve_greet_uuid()

    async def _play_greet(self, force: bool = False) -> Dict[str, Any]:
        now = time.monotonic()
        if not force and now - self.last_greet_at < self.greet_min_interval_s:
            return {"played": False, "skipped": "rate_limited"}
        uid = self.greet_audio_uuid or await self._resolve_greet_uuid()
        if not uid:
            raise RuntimeError("greet audio not available")
        hub = await self._ensure_audio_hub()
        if hub is None:
            raise RuntimeError("audio hub unavailable")
        try:
            await asyncio.wait_for(hub.play_by_uuid(uid), timeout=6.0)
        except Exception:
            if self.greet_audio_uuid_configured:
                raise
            # Refresh a stale UUID once (for example after a robot reset).
            self.greet_audio_uuid = None
            self.greet_audio_resolved = False
            uid = await self._resolve_greet_uuid()
            if not uid:
                raise RuntimeError("greet audio unavailable after UUID refresh")
            hub = await self._ensure_audio_hub()
            if hub is None:
                raise RuntimeError("audio hub unavailable after UUID refresh")
            await asyncio.wait_for(hub.play_by_uuid(uid), timeout=6.0)
        self.last_greet_at = time.monotonic()
        return {"played": True, "uuid": uid}

    async def _set_lidar(self, enabled: bool) -> None:
        if self.conn is None:
            raise RuntimeError("Robot connection not ready")

        if enabled and not self.traffic_saving_disabled:
            with contextlib.suppress(Exception):
                disabled = await asyncio.wait_for(
                    self.conn.datachannel.disableTrafficSaving(True),
                    timeout=3.0,
                )
                self.traffic_saving_disabled = bool(disabled)

        self.lidar_enabled = enabled
        self.conn.datachannel.pub_sub.publish_without_callback(
            TOPIC_ALIAS_TO_VALUE["ULIDAR_SWITCH"],
            "on" if enabled else "off",
        )

    def _next_robot_request_id(self) -> int:
        self.robot_request_sequence = (self.robot_request_sequence + 1) & 0x7FFFFFFF
        if self.robot_request_sequence == 0:
            self.robot_request_sequence = 1
        return self.robot_request_sequence

    @staticmethod
    def _cleanup_pending_request(connection: UnitreeWebRTCConnection, request_id: int) -> None:
        pub_sub = getattr(getattr(connection, "datachannel", None), "pub_sub", None)
        resolver = getattr(pub_sub, "future_resolver", None)
        if resolver is None:
            return

        pending_callbacks = getattr(resolver, "pending_callbacks", None)
        if isinstance(pending_callbacks, dict):
            pending_callbacks.pop(request_id, None)

        chunk_storage = getattr(resolver, "chunk_data_storage", None)
        if isinstance(chunk_storage, dict):
            chunk_storage.pop(request_id, None)

    async def _robot_request(
        self,
        topic: str,
        api_id: int,
        parameter: Optional[Any] = None,
    ) -> Any:
        async with self.robot_request_lock:
            connection = self.conn
            if connection is None:
                raise RuntimeError("Connection not ready")

            channel = getattr(getattr(connection, "datachannel", None), "channel", None)
            if getattr(channel, "readyState", "closed") != "open":
                raise RuntimeError("Robot data channel is not open")

            pub_sub = connection.datachannel.pub_sub
            resolver = getattr(pub_sub, "future_resolver", None)
            if resolver is None:
                raise RuntimeError("Robot response resolver is not available")

            request_id = self._next_robot_request_id()
            request_payload: Dict[str, Any] = {
                "header": {
                    "identity": {
                        "id": request_id,
                        "api_id": api_id,
                    }
                },
                "parameter": "",
            }
            if parameter is not None:
                request_payload["parameter"] = (
                    parameter if isinstance(parameter, str) else json.dumps(parameter)
                )

            response_future = asyncio.get_running_loop().create_future()
            resolver.save_resolve(
                DATA_CHANNEL_TYPE["REQUEST"],
                topic,
                response_future,
                request_id,
            )

            wire_message = json.dumps(
                {
                    "type": DATA_CHANNEL_TYPE["REQUEST"],
                    "topic": topic,
                    "data": request_payload,
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )

            try:
                channel.send(wire_message)
                return await asyncio.wait_for(
                    asyncio.shield(response_future),
                    timeout=self.args.robot_request_timeout_s,
                )
            except asyncio.TimeoutError as exc:
                self._cleanup_pending_request(connection, request_id)
                response_future.cancel()
                raise TimeoutError(
                    f"Robot request timeout api_id={api_id} id={request_id}"
                ) from exc
            except asyncio.CancelledError:
                self._cleanup_pending_request(connection, request_id)
                response_future.cancel()
                raise
            except Exception:
                self._cleanup_pending_request(connection, request_id)
                response_future.cancel()
                raise

    async def _sport_request(self, api_id: int, parameter: Optional[Any] = None) -> Any:
        return await self._robot_request(
            TOPIC_ALIAS_TO_VALUE["SPORT_MOD"],
            api_id,
            parameter,
        )

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

    async def _set_speed_profile(self, profile: str, stop_first: bool = True) -> Any:
        normalized = str(profile).strip().lower()
        if normalized not in self._speed_profiles():
            raise ValueError("profile must be 'normal' or 'max_api'")

        if stop_first:
            self._sport_send_nowait(int(SPORT_CMD["StopMove"]))
            self.move_active = False
            self.pending_stop_deadline = 0.0

        level = 1 if normalized == "max_api" else 0
        response = await self._sport_request(
            int(SPORT_CMD["SpeedLevel"]),
            {"data": level},
        )
        if isinstance(response, dict):
            response_data = response.get("data", {})
            header = response_data.get("header", {}) if isinstance(response_data, dict) else {}
            status = header.get("status", {}) if isinstance(header, dict) else {}
            code = status.get("code") if isinstance(status, dict) else None
            if isinstance(code, (int, float)) and int(code) != 0:
                raise RuntimeError(f"SpeedLevel rejected by robot with code {int(code)}")
        self.speed_profile = normalized
        self.speed_level = level
        self._publish_event(
            "speed_profile_changed",
            {
                "profile": normalized,
                "speed_level": level,
            },
        )
        return response

    def _sport_send_nowait(self, api_id: int, parameter: Optional[Any] = None) -> None:
        connection = self.conn
        if connection is None:
            raise RuntimeError("Connection not ready")

        channel = getattr(getattr(connection, "datachannel", None), "channel", None)
        if getattr(channel, "readyState", "closed") != "open":
            raise RuntimeError("Robot data channel is not open")

        request_payload: Dict[str, Any] = {
            "header": {
                "identity": {
                    "id": self._next_robot_request_id(),
                    "api_id": api_id,
                }
            },
            "parameter": "",
        }
        if parameter is not None:
            request_payload["parameter"] = (
                parameter if isinstance(parameter, str) else json.dumps(parameter)
            )

        connection.datachannel.pub_sub.publish_without_callback(
            TOPIC_ALIAS_TO_VALUE["SPORT_MOD"],
            request_payload,
            DATA_CHANNEL_TYPE["REQUEST"],
        )

    async def _set_motion_mode(self, mode_name: str) -> Any:
        return await self._robot_request(
            TOPIC_ALIAS_TO_VALUE["MOTION_SWITCHER"],
            1002,
            {"name": mode_name},
        )

    async def _on_video_track(self, track) -> None:
        if self.video_task and not self.video_task.done():
            self.video_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.video_task
        if self.video_encode_task and not self.video_encode_task.done():
            self.video_encode_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.video_encode_task

        self.video_task = asyncio.create_task(self._consume_video_track(track))
        self.video_encode_task = asyncio.create_task(self._camera_encode_loop())
        self._publish_event("video_track_received", {})

    async def _consume_video_track(self, track) -> None:
        while not self.stop_event.is_set():
            frame = await track.recv()
            self.camera_frame_count += 1

            if not self.camera_enabled:
                continue
            if (
                self.camera_emit_every > 1
                and self.camera_frame_count % self.camera_emit_every != 0
            ):
                continue

            self.latest_camera_frame = frame
            # Keep an independent reference for colorization (the encode loop
            # consumes/nulls latest_camera_frame).
            if self.colorize_enabled:
                self.latest_color_frame = frame
            self.camera_frame_ready.set()

    async def _camera_encode_loop(self) -> None:
        while not self.stop_event.is_set():
            await self.camera_frame_ready.wait()
            self.camera_frame_ready.clear()

            frame = self.latest_camera_frame
            self.latest_camera_frame = None
            if frame is None or not self.camera_enabled:
                continue

            min_interval = 1.0 / max(self.camera_target_fps, 1.0)
            delay = min_interval - (time.monotonic() - self.last_camera_emit_at)
            if delay > 0:
                await asyncio.sleep(delay)
                if self.latest_camera_frame is not None:
                    frame = self.latest_camera_frame
                    self.latest_camera_frame = None
                    self.camera_frame_ready.clear()

            if self.camera_format == "h264":
                await self._encode_and_emit_h264(frame)
                continue

            encode_started = time.monotonic()
            encoded, width, height, image_format, used_quality = await asyncio.to_thread(
                self._encode_camera_frame,
                frame,
            )
            if encoded is None:
                continue
            encoded_at = time.time()
            self.last_camera_emit_at = time.monotonic()
            self.camera_encoded_count += 1

            await self._enqueue_media(
                {
                    "stream": "video",
                    "binary": True,
                    "header": {
                        "stream": "video",
                        "frame_index": self.camera_encoded_count,
                        "source_frame_index": self.camera_frame_count,
                        "width": width,
                        "height": height,
                        "image_format": image_format,
                        "encoded_bytes": len(encoded),
                        "quality": used_quality,
                        "encoded_ts": encoded_at,
                        "encode_ms": round((time.monotonic() - encode_started) * 1000.0, 2),
                        "target_fps": self.camera_target_fps,
                        "ts": encoded_at,
                    },
                    "payload": encoded,
                }
            )

    def _reset_video_encoder(self) -> None:
        # Called from the event loop while _encode_camera_frame_h264 may be running
        # in a worker thread. Only drop the reference here: closing it now could free
        # the context mid-encode() and crash ffmpeg. The encoder is torn down by GC
        # once the in-flight thread releases its own local handle.
        self.video_encoder = None
        self.video_encoder_key = None
        self.video_codec_string = ""
        self.video_pts = 0

    @staticmethod
    def _h264_codec_string(annexb: bytes) -> str:
        # Build the WebCodecs "avc1.PPCCLL" identifier from the SPS NAL so the
        # browser configures its decoder with the exact profile/level we emit.
        data = annexb
        length = len(data)
        index = 0
        while index + 4 < length:
            if data[index] == 0 and data[index + 1] == 0 and data[index + 2] == 1:
                nal_start = index + 3
            elif (
                data[index] == 0
                and data[index + 1] == 0
                and data[index + 2] == 0
                and data[index + 3] == 1
            ):
                nal_start = index + 4
            else:
                index += 1
                continue
            if (data[nal_start] & 0x1F) == 7 and nal_start + 4 <= length:
                return "avc1.%02x%02x%02x" % (
                    data[nal_start + 1],
                    data[nal_start + 2],
                    data[nal_start + 3],
                )
            index = nal_start
        return "avc1.42e01e"

    def _make_h264_ctx(self, name: str, width: int, height: int, fps: int, gop: int, bitrate_kbps: int):
        ctx = av.CodecContext.create(name, "w")
        ctx.width = width
        ctx.height = height
        ctx.pix_fmt = "yuv420p"
        ctx.bit_rate = max(64, int(bitrate_kbps)) * 1000
        ctx.gop_size = gop
        ctx.time_base = Fraction(1, fps)
        with contextlib.suppress(Exception):
            ctx.framerate = Fraction(fps, 1)
        if name == "libx264":
            ctx.options = {
                "preset": "ultrafast",
                "tune": "zerolatency",
                "profile": "baseline",
                "x264-params": (
                    f"keyint={gop}:min-keyint={gop}:scenecut=0:bframes=0:"
                    f"nal-hrd=cbr:vbv-maxrate={bitrate_kbps}:"
                    f"vbv-bufsize={bitrate_kbps}"
                ),
            }
        else:
            ctx.options = {"profile": "baseline"}
        return ctx

    def _build_video_encoder(self, width: int, height: int, fps: int, bitrate_kbps: int):
        width -= width % 2
        height -= height % 2
        gop = self.camera_gop if self.camera_gop > 0 else max(1, fps) * 2

        if self.args.camera_h264_encoder == "auto":
            candidates = ["h264_v4l2m2m", "libx264"]
        else:
            candidates = [self.args.camera_h264_encoder]

        last_error: Optional[Exception] = None
        for name in candidates:
            # av.CodecContext.create() does NOT call avcodec_open2(); a broken
            # hardware encoder (e.g. h264_v4l2m2m on some Pi builds) only fails when
            # it first encodes. Probe it with a throwaway frame so we can actually
            # fall back to libx264 instead of looping on encode errors.
            try:
                probe = self._make_h264_ctx(name, width, height, fps, gop, bitrate_kbps)
                test_frame = av.VideoFrame(width, height, "yuv420p")
                for plane in test_frame.planes:
                    plane.update(bytes(plane.buffer_size))
                list(probe.encode(test_frame))
                with contextlib.suppress(Exception):
                    list(probe.encode(None))  # flush
                with contextlib.suppress(Exception):
                    probe.close()
            except Exception as exc:  # noqa: BLE001 - this encoder is unusable
                last_error = exc
                self._publish_event(
                    "video_encoder_unavailable",
                    {"encoder": name, "error": str(exc)},
                )
                continue

            # Probe succeeded: build a fresh context so the real stream starts on a
            # clean keyframe.
            ctx = self._make_h264_ctx(name, width, height, fps, gop, bitrate_kbps)
            self._publish_event(
                "video_encoder_ready",
                {
                    "encoder": name,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "bitrate_kbps": bitrate_kbps,
                    "gop": gop,
                },
            )
            return ctx, name
        raise RuntimeError(f"no usable H.264 encoder: {last_error}")

    def _encode_camera_frame_h264(self, frame) -> List[Tuple[bytes, bool, int, int]]:
        width = int(frame.width)
        height = int(frame.height)
        if self.camera_max_width > 0 and width > self.camera_max_width:
            scale = self.camera_max_width / float(width)
            out_w = self.camera_max_width
            out_h = max(2, int(round(height * scale)))
        else:
            out_w, out_h = width, height
        out_w -= out_w % 2
        out_h -= out_h % 2

        fps = max(1, int(round(self.camera_target_fps)))
        key = (out_w, out_h, fps, int(self.camera_bitrate_kbps))
        encoder = self.video_encoder
        if encoder is None or self.video_encoder_key != key:
            encoder, self.video_encoder_name = self._build_video_encoder(
                out_w, out_h, fps, self.camera_bitrate_kbps
            )
            self.video_encoder = encoder
            self.video_encoder_key = key
            self.video_codec_string = ""
            self.video_pts = 0

        # Hold a local handle so a concurrent _reset_video_encoder() on the event
        # loop can't pull the context out from under encode() mid-call.
        vframe = frame.reformat(width=out_w, height=out_h, format="yuv420p")
        vframe.pts = self.video_pts
        vframe.time_base = encoder.time_base
        self.video_pts += 1

        results: List[Tuple[bytes, bool, int, int]] = []
        for packet in encoder.encode(vframe):
            raw = bytes(packet)
            if not raw:
                continue
            is_key = bool(packet.is_keyframe)
            if is_key and not self.video_codec_string:
                self.video_codec_string = self._h264_codec_string(raw)
            results.append((raw, is_key, out_w, out_h))
        return results

    async def _encode_and_emit_h264(self, frame) -> None:
        encode_started = time.monotonic()
        try:
            packets = await asyncio.to_thread(self._encode_camera_frame_h264, frame)
        except Exception as exc:  # noqa: BLE001
            now = time.monotonic()
            if now - self.last_video_encode_error_at >= 2.0:
                self.last_video_encode_error_at = now
                self._publish_event("video_encode_error", {"error": str(exc)})
            self._reset_video_encoder()
            await asyncio.sleep(0.25)
            return

        if not packets:
            return

        encoded_at = time.time()
        self.last_camera_emit_at = time.monotonic()
        encode_ms = round((time.monotonic() - encode_started) * 1000.0, 2)
        codec = self.video_codec_string or "avc1.42e01e"
        for raw, is_key, width, height in packets:
            self.camera_encoded_count += 1
            await self._enqueue_media(
                {
                    "stream": "video",
                    "binary": True,
                    "header": {
                        "stream": "video",
                        "frame_index": self.camera_encoded_count,
                        "source_frame_index": self.camera_frame_count,
                        "width": width,
                        "height": height,
                        "image_format": "h264",
                        "codec": codec,
                        "key": bool(is_key),
                        "encoded_bytes": len(raw),
                        "encoded_ts": encoded_at,
                        "encode_ms": encode_ms,
                        "target_fps": self.camera_target_fps,
                        "encoder": self.video_encoder_name,
                        "ts": encoded_at,
                    },
                    "payload": raw,
                }
            )

    def _encode_camera_frame(
        self,
        frame,
    ) -> Tuple[Optional[bytes], int, int, str, int]:
        image = frame.to_ndarray(format="bgr24")
        height, width = image.shape[:2]
        if self.camera_max_width > 0 and width > self.camera_max_width:
            scale = self.camera_max_width / float(width)
            width = self.camera_max_width
            height = max(1, int(round(height * scale)))
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        image_format = self.camera_format
        quality = self.camera_jpeg_quality

        def encode(selected_format: str, selected_quality: int):
            try:
                if selected_format == "webp":
                    return cv2.imencode(
                        ".webp",
                        image,
                        [getattr(cv2, "IMWRITE_WEBP_QUALITY", 64), selected_quality],
                    )
                return cv2.imencode(
                    ".jpg",
                    image,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        selected_quality,
                        cv2.IMWRITE_JPEG_OPTIMIZE,
                        1,
                    ],
                )
            except Exception:
                return False, None

        ok, encoded = encode(image_format, quality)
        if (not ok or encoded is None) and image_format == "webp":
            image_format = "jpg"
            ok, encoded = encode(image_format, quality)
        if not ok or encoded is None:
            return None, width, height, image_format, quality

        target_bytes = 0
        if self.media_max_kbps > 0:
            camera_bytes_per_second = self.media_max_kbps * 125.0 * 0.65
            target_bytes = int(
                camera_bytes_per_second / max(self.camera_target_fps, 1.0)
            )

        if (
            target_bytes > 0
            and len(encoded) > target_bytes
            and quality > self.camera_min_quality
        ):
            size_ratio = target_bytes / float(len(encoded))
            adaptive_quality = max(
                self.camera_min_quality,
                min(quality - 1, int(round(quality * size_ratio ** 0.55))),
            )
            ok_retry, encoded_retry = encode(image_format, adaptive_quality)
            if ok_retry and encoded_retry is not None:
                quality = adaptive_quality
                encoded = encoded_retry

            if len(encoded) > target_bytes and quality > self.camera_min_quality:
                ok_min, encoded_min = encode(image_format, self.camera_min_quality)
                if ok_min and encoded_min is not None:
                    quality = self.camera_min_quality
                    encoded = encoded_min

            if len(encoded) > target_bytes and width > 320:
                resize_ratio = max(
                    320.0 / width,
                    min(0.95, (target_bytes / float(len(encoded))) ** 0.5 * 0.95),
                )
                resized_width = max(320, int(round(width * resize_ratio)))
                resized_height = max(1, int(round(height * resized_width / width)))
                image = cv2.resize(
                    image,
                    (resized_width, resized_height),
                    interpolation=cv2.INTER_AREA,
                )
                width, height = resized_width, resized_height
                ok_resized, encoded_resized = encode(image_format, quality)
                if ok_resized and encoded_resized is not None:
                    encoded = encoded_resized

        return encoded.tobytes(), width, height, image_format, quality

    async def _on_audio_frame(self, frame) -> None:
        self.audio_frame_count += 1

        if not self.audio_enabled:
            return

        try:
            raw, sample_rate, channels = self._audio_frame_to_pcm(frame)
        except Exception as exc:
            now = time.monotonic()
            if now - self.last_audio_error_at >= 2.0:
                self.last_audio_error_at = now
                self._publish_event("audio_encode_error", {"error": str(exc)})
            return

        if not raw:
            return

        self.audio_pcm_buffer.extend(raw)
        self.audio_buffered_frames += 1

        target_frames = max(1, self.audio_emit_every)
        target_bytes = int(sample_rate * channels * 2 * 0.1)
        if self.audio_buffered_frames < target_frames and len(self.audio_pcm_buffer) < target_bytes:
            return

        packet = bytes(self.audio_pcm_buffer)
        self.audio_pcm_buffer.clear()
        self.audio_buffered_frames = 0

        if self.audio_max_bytes > 0 and len(packet) > self.audio_max_bytes:
            packet = packet[: self.audio_max_bytes]

        bytes_per_frame = channels * 2
        packet = packet[: len(packet) - (len(packet) % bytes_per_frame)]
        if not packet:
            return

        await self._enqueue_media(
            {
                "robot_id": self.args.robot_id,
                "ts": time.time(),
                "stream": "audio",
                "data": {
                    "frame_index": self.audio_frame_count,
                    "audio_format": "pcm_s16le",
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "audio_base64": base64.b64encode(packet).decode("ascii"),
                },
            }
        )

    def _audio_frame_to_pcm(self, frame) -> Tuple[bytes, int, int]:
        if int(getattr(frame, "sample_rate", 0) or 0) <= 0:
            frame.sample_rate = 48000

        output_frames = self.audio_resampler.resample(frame)
        if not output_frames:
            return b"", self.args.audio_sample_rate, 1

        chunks: List[bytes] = []
        for output_frame in output_frames:
            array = np.asarray(output_frame.to_ndarray())
            if array.size == 0:
                continue
            pcm = np.ascontiguousarray(array.reshape(-1), dtype="<i2")
            chunks.append(pcm.tobytes())

        return b"".join(chunks), self.args.audio_sample_rate, 1

    async def _enqueue_media(self, payload: Dict[str, Any]) -> None:
        if not self.args.media_ws_url:
            return

        stream = str(payload.get("stream", "")).strip()
        header = payload.get("header") if isinstance(payload.get("header"), dict) else None
        is_h264_video = (
            stream == "video"
            and header is not None
            and header.get("image_format") == "h264"
        )

        if is_h264_video:
            # Inter-frame video must be delivered in order. Never coalesce/replace
            # a pending frame: dropping a delta would corrupt every following frame
            # until the next keyframe. Under extreme overflow drop the OLDEST frame
            # so the decoder resyncs at the next keyframe instead of stalling.
            if self.video_media_queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    self.video_media_queue.get_nowait()
                    self.video_drop_count += 1
            with contextlib.suppress(asyncio.QueueFull):
                self.video_media_queue.put_nowait(payload)
        elif stream == "audio":
            if self.media_queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    self.media_queue.get_nowait()

            with contextlib.suppress(asyncio.QueueFull):
                self.media_queue.put_nowait(payload)
        else:
            self.latest_media_by_stream[stream] = payload

        self.media_ready.set()

    def _looks_like_lidar_topic(self, topic_alias: str) -> bool:
        upper = topic_alias.upper()
        return "LIDAR" in upper or "ULIDAR" in upper or "CLOUD" in upper

    def _describe_lidar_message(self, value: Any, depth: int = 0) -> Any:
        """Compact, JSON-safe description of a LiDAR message so we can see its real
        shape in telemetry/events when extraction fails (debugging aid)."""
        if depth > 4:
            return "…"
        if isinstance(value, dict):
            return {str(k): self._describe_lidar_message(v, depth + 1) for k, v in list(value.items())[:12]}
        if isinstance(value, (list, tuple)):
            head = self._describe_lidar_message(value[0], depth + 1) if value else None
            return {"_list_len": len(value), "_first": head}
        if isinstance(value, np.ndarray):
            return {"_ndarray": True, "dtype": str(value.dtype), "shape": list(value.shape), "size": int(value.size)}
        if isinstance(value, (bytes, bytearray)):
            return {"_bytes": len(value)}
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, str):
            return value[:60]
        return f"<{type(value).__name__}>"

    def _extract_lidar_points(self, value: Any, depth: int = 0) -> Optional[np.ndarray]:
        if depth > 8 or value is None:
            return None

        if isinstance(value, np.ndarray):
            arr = np.asarray(value)

            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, : min(arr.shape[1], 3)].astype(np.float32, copy=False)

            if arr.ndim == 1:
                if arr.size % 3 == 0:
                    return arr.astype(np.float32).reshape(-1, 3)
                if arr.size % 2 == 0:
                    return arr.astype(np.float32).reshape(-1, 2)

            return None

        if isinstance(value, dict):
            if all(key in value for key in ("x", "y")):
                with contextlib.suppress(Exception):
                    return np.asarray(
                        [[float(value["x"]), float(value["y"]), float(value.get("z", 0.0))]],
                        dtype=np.float32,
                    )

            if "positions" in value:
                with contextlib.suppress(Exception):
                    positions = np.asarray(value["positions"])
                    if positions.dtype == np.uint8 and positions.ndim == 1:
                        if positions.size % 12 != 0:
                            return None
                        points = np.frombuffer(positions.tobytes(), dtype=np.float32).reshape(-1, 3)
                        return points if points.size else None

            preferred = [
                "points",
                "cloud",
                "cloud_points",
                "xyz",
                "voxel_map",
                "positions",
                "data",
                "items",
            ]
            for key in preferred:
                if key in value:
                    found = self._extract_lidar_points(value[key], depth + 1)
                    if found is not None:
                        return found

            for item in value.values():
                found = self._extract_lidar_points(item, depth + 1)
                if found is not None:
                    return found

            return None

        if isinstance(value, (list, tuple)):
            if not value:
                return None

            with contextlib.suppress(Exception):
                arr = np.asarray(value, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    return arr[:, : min(arr.shape[1], 3)]
                if arr.ndim == 1 and arr.size % 3 == 0:
                    return arr.reshape(-1, 3)
                if arr.ndim == 1 and arr.size % 2 == 0:
                    return arr.reshape(-1, 2)

            for item in value[:10]:
                found = self._extract_lidar_points(item, depth + 1)
                if found is not None:
                    return found

        return None

    def _current_camera_pose(self) -> Tuple[float, float, float, float]:
        """Robot pose (x, y, z, yaw) in the odom/world frame from the sport state."""
        topic = TOPIC_ALIAS_TO_VALUE.get("LF_SPORT_MOD_STATE", "")
        msg = self.latest_by_topic.get(topic, {})
        data = msg.get("data", {}) if isinstance(msg, dict) else {}
        pos = data.get("position") if isinstance(data, dict) else None
        imu = data.get("imu_state", {}) if isinstance(data, dict) else {}
        rpy = imu.get("rpy") if isinstance(imu, dict) else None
        x = y = z = 0.0
        yaw = 0.0
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            x = float(pos[0])
            y = float(pos[1])
            if len(pos) >= 3:
                z = float(pos[2])
        if isinstance(rpy, (list, tuple)) and len(rpy) >= 3:
            yaw = float(rpy[2])
        return x, y, z, yaw

    def _world_points_to_body(self, points: np.ndarray) -> np.ndarray:
        """Transform odom/world-frame LiDAR points into the robot body frame
        (x forward, y left, z up relative to the robot) used by the safety guard."""
        if self.safety_points_frame == "body":
            return points
        x, y, z, yaw = self._current_camera_pose()
        d = points[:, :3] - np.array([x, y, z], dtype=np.float32)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        bx = cos_y * d[:, 0] + sin_y * d[:, 1]
        by = -sin_y * d[:, 0] + cos_y * d[:, 1]
        bz = d[:, 2]
        return np.column_stack((bx, by, bz)).astype(np.float32)

    def _update_safety_from_lidar(self, message: Any) -> None:
        """Feed the reactive safety guard from a raw LiDAR scan, at full sensor
        rate (throttled to safety_update_hz). Cheap: a single downsampled numpy
        pass. This is the input to the local collision/cliff guarantee."""
        if not self.safety_enabled:
            return
        now = time.monotonic()
        if now - self.last_safety_update_at < (1.0 / max(self.safety_update_hz, 0.1)):
            return
        self.last_safety_update_at = now

        points = self._extract_lidar_points(message)
        if points is None:
            return
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            return
        finite = np.isfinite(points).all(axis=1)
        points = points[finite][:, :3]
        if points.shape[0] == 0:
            return
        # Downsample for the guard: it only needs the near field at modest density.
        if points.shape[0] > 2500:
            step = int(np.ceil(points.shape[0] / 2500))
            points = points[::step]
        body = self._world_points_to_body(points)
        self.safety_guard.update(body, now=now)

    def _guard_velocity(
        self, vx: float, vy: float, wz: float
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        """Pass a requested body velocity through the safety guard. Only enforced
        while the guard is armed (safety enabled AND LiDAR running, so we actually
        have a scan); otherwise the request passes through untouched and the
        diagnostics flag that safety is inactive."""
        if not self.safety_enabled or not self.lidar_enabled or not self.safety_guard.is_armed():
            reason = (
                "safety_disabled" if not self.safety_enabled
                else "lidar_off" if not self.lidar_enabled
                else "waiting_lidar"  # enabled but no scan yet -> don't block driving
            )
            info = {"active": False, "blocked": False, "reasons": [reason]}
            self.last_safety_info = info
            return vx, vy, wz, info
        result = self.safety_guard.filter_velocity(vx, vy, wz)
        info = dict(result)
        info["active"] = True
        self.last_safety_info = info
        now = time.monotonic()
        if result["blocked"] and now - self.last_safety_event_at >= 0.5:
            self.last_safety_event_at = now
            self._publish_event(
                "safety_intervention",
                {
                    "requested": {"vx": vx, "vy": vy, "wz": wz},
                    "applied": {"vx": result["vx"], "vy": result["vy"], "wz": result["wz"]},
                    "reasons": result["reasons"],
                    "front_clearance_m": result["front_clearance_m"],
                    "cliff": result["cliff"],
                },
            )
        return result["vx"], result["vy"], result["wz"], info

    def _colorize_points(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Project world-frame points onto the latest camera frame (approximate
        fisheye, no calibration) and sample RGB. Returns uint8 (N,4) RGBA where
        alpha=255 marks a valid color, 0 means "not seen by the camera"."""
        frame = self.latest_color_frame
        if frame is None or points.size == 0:
            return None
        try:
            img = frame.to_ndarray(format="rgb24")
        except Exception:
            return None
        height, width = img.shape[:2]
        if height < 2 or width < 2:
            return None

        fov = math.radians(clamp(self.color_cam_fov_deg, 30.0, 220.0))
        pitch = math.radians(self.color_cam_pitch_deg)

        if self.color_points_frame == "body":
            # Points already robot-relative (x fwd, y left, z up): only the camera
            # mount offset matters, no pose transform.
            d = points[:, :3] - np.array(
                [self.color_cam_forward_m, 0.0, self.color_cam_height_m],
                dtype=np.float32,
            )
            bx, by, bz = d[:, 0], d[:, 1], d[:, 2]
        else:
            x, y, z, yaw = self._current_camera_pose()
            cam_x = x + self.color_cam_forward_m * math.cos(yaw)
            cam_y = y + self.color_cam_forward_m * math.sin(yaw)
            cam_z = z + self.color_cam_height_m
            d = points[:, :3] - np.array([cam_x, cam_y, cam_z], dtype=np.float32)
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            bx = cos_y * d[:, 0] + sin_y * d[:, 1]   # body forward
            by = -sin_y * d[:, 0] + cos_y * d[:, 1]  # body left
            bz = d[:, 2]                             # body up
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        forward = bx * cos_p + bz * sin_p
        up = -bx * sin_p + bz * cos_p

        z_cam = forward          # optical axis
        x_cam = -by              # right
        y_cam = -up              # down
        radial = np.sqrt(x_cam * x_cam + y_cam * y_cam)
        theta = np.arctan2(radial, z_cam)
        focal = (min(width, height) / 2.0) / (fov / 2.0)
        safe_radial = np.where(radial > 1e-6, radial, 1.0)
        u = width / 2.0 + focal * theta * (x_cam / safe_radial)
        v = height / 2.0 + focal * theta * (y_cam / safe_radial)
        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        dist = np.linalg.norm(d, axis=1)

        valid = (
            (z_cam > 0.05)
            & (theta <= fov / 2.0)
            & (dist <= self.color_max_distance_m)
            & (ui >= 0) & (ui < width)
            & (vi >= 0) & (vi < height)
        )
        rgba = np.zeros((points.shape[0], 4), dtype=np.uint8)
        idx = np.where(valid)[0]
        if idx.size:
            rgba[idx, 0:3] = img[vi[idx], ui[idx], :3]
            rgba[idx, 3] = 255
        return rgba

    async def _maybe_publish_lidar_media(self, payload: Any) -> None:
        now = time.monotonic()
        if now - self.last_lidar_media_at < (1.0 / max(self.lidar_media_hz, 0.01)):
            return
        self.last_lidar_media_at = now

        points = self._extract_lidar_points(payload)
        if points is None:
            return

        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 2:
            return

        if points.shape[1] == 2:
            points = np.column_stack(
                (points, np.zeros(points.shape[0], dtype=np.float32))
            )
        else:
            points = points[:, :3]

        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if points.size == 0:
            return

        max_points = self.lidar_uplink_max_points
        if max_points > 0 and points.shape[0] > max_points:
            step = max(int(np.ceil(points.shape[0] / max_points)), 1)
            points = points[::step][:max_points]

        # Sample camera color for each point (approximate fisheye projection).
        colors = self._colorize_points(points) if self.colorize_enabled else None

        payload, fmt, scale, offset, count = await asyncio.to_thread(
            encode_cloud_payload,
            points,
            colors,
            self.lidar_quantization_cm,
            self.lidar_compression_level,
        )

        await self._enqueue_media(
            {
                "stream": "lidar",
                "binary": True,
                "header": {
                    "stream": "lidar",
                    "fmt": fmt,
                    "count": count,
                    "scale": scale,
                    "offset": offset,
                    "coordinate_frame": "map",
                    "ts": time.time(),
                },
                "payload": payload,
            }
        )
        self.lidar_media_sent += 1

    def _telemetry_from_low_state(self) -> Dict[str, Any]:
        topic = TOPIC_ALIAS_TO_VALUE.get("LOW_STATE", "")
        msg = self.latest_by_topic.get(topic, {})
        data = msg.get("data", {}) if isinstance(msg, dict) else {}

        bms = data.get("bms_state", {}) if isinstance(data, dict) else {}
        motor_state = data.get("motor_state", []) if isinstance(data, dict) else []

        motor_temps: List[Any] = []
        if isinstance(motor_state, list):
            for motor in motor_state:
                if isinstance(motor, dict):
                    motor_temps.append(motor.get("temperature"))

        return {
            "battery": data.get("soc", bms.get("soc")) if isinstance(data, dict) else None,
            "power_v": data.get("power_v") if isinstance(data, dict) else None,
            "temperature_ntc1": data.get("temperature_ntc1") if isinstance(data, dict) else None,
            "motor_temperatures": motor_temps,
            "bms_bq_ntc": bms.get("bq_ntc") if isinstance(bms, dict) else None,
            "bms_mcu_ntc": bms.get("mcu_ntc") if isinstance(bms, dict) else None,
        }

    def _telemetry_from_sport_state(self) -> Dict[str, Any]:
        topic = TOPIC_ALIAS_TO_VALUE.get("LF_SPORT_MOD_STATE", "")
        msg = self.latest_by_topic.get(topic, {})
        data = msg.get("data", {}) if isinstance(msg, dict) else {}

        pose = data.get("position") if isinstance(data, dict) else None
        imu = data.get("imu_state", {}) if isinstance(data, dict) else {}
        velocity = data.get("velocity") if isinstance(data, dict) else None

        pose_out = {"x": None, "y": None, "yaw": None}
        if isinstance(pose, (list, tuple)) and len(pose) >= 2:
            pose_out["x"] = pose[0]
            pose_out["y"] = pose[1]

        rpy = imu.get("rpy") if isinstance(imu, dict) else None
        if isinstance(rpy, (list, tuple)) and len(rpy) >= 3:
            pose_out["yaw"] = rpy[2]

        velocity_out = {
            "x": None,
            "y": None,
            "z": None,
            "linear": None,
            "angular": None,
        }
        if isinstance(velocity, dict):
            velocity_out["x"] = velocity.get("x", velocity.get("linear"))
            velocity_out["y"] = velocity.get("y", velocity.get("lateral"))
            velocity_out["z"] = velocity.get("z", velocity.get("angular"))
        elif isinstance(velocity, (list, tuple)) and len(velocity) >= 3:
            velocity_out["x"] = velocity[0]
            velocity_out["y"] = velocity[1]
            velocity_out["z"] = velocity[2]

        velocity_out["linear"] = velocity_out["x"]
        velocity_out["angular"] = velocity_out["z"]

        return {
            "pose": pose_out,
            "velocity": velocity_out,
            "imu_temperature": imu.get("temperature") if isinstance(imu, dict) else None,
        }

    def _build_telemetry(self) -> Dict[str, Any]:
        low = self._telemetry_from_low_state()
        sport = self._telemetry_from_sport_state()

        alerts: List[str] = []
        battery = low.get("battery")
        if isinstance(battery, (int, float)) and battery <= self.args.low_battery_threshold:
            alerts.append("low_battery")

        if self.move_active:
            alerts.append("motion_active")

        heartbeat_age = time.monotonic() - self.last_heartbeat_monotonic
        if heartbeat_age > self.args.heartbeat_timeout_s:
            alerts.append("control_heartbeat_timeout")

        safety_status = self.safety_guard.status() if self.safety_enabled else {}
        safety_armed = self.safety_enabled and self.lidar_enabled and self.safety_guard.is_armed()
        if safety_armed and safety_status.get("cliff"):
            alerts.append("cliff_front")
        last_reasons = self.last_safety_info.get("reasons", []) if isinstance(self.last_safety_info, dict) else []
        if safety_armed and ("obstacle_front" in last_reasons or "cliff_front" in last_reasons):
            alerts.append("obstacle_front")

        return {
            "robot_id": self.args.robot_id,
            "ts": time.time(),
            "mode": self.last_heartbeat_payload.get("mode", "manual"),
            "pose": sport.get("pose", {}),
            "velocity": sport.get("velocity", {}),
            "speed_control": {
                "profile": self.speed_profile,
                "speed_level": self.speed_level,
                "limits": self._speed_profiles().get(
                    self.speed_profile,
                    self._speed_profiles()["normal"],
                ),
                "requested": dict(self.last_move_command),
            },
            "battery": battery,
            "power_v": low.get("power_v"),
            "robot_link": {
                "connected": self.conn is not None,
                "pc_state": str(getattr(getattr(self.conn, "pc", None), "connectionState", "closed")),
                "data_state": str(
                    getattr(
                        getattr(getattr(self.conn, "datachannel", None), "channel", None),
                        "readyState",
                        "closed",
                    )
                ),
            },
            "media": {
                "camera_enabled": self.camera_enabled,
                "camera_format": self.camera_format,
                "lidar_enabled": self.lidar_enabled,
                "audio_enabled": self.audio_enabled,
                "camera_emit_every": self.camera_emit_every,
                "camera_target_fps": self.camera_target_fps,
                "camera_max_width": self.camera_max_width,
                "camera_encoded_frames": self.camera_encoded_count,
                "camera_jpeg_quality": self.camera_jpeg_quality,
                "camera_min_quality": self.camera_min_quality,
                "audio_emit_every": self.audio_emit_every,
                "audio_max_bytes": self.audio_max_bytes,
                "audio_sample_rate": self.args.audio_sample_rate,
                "audio_queue_depth": self.media_queue.qsize(),
                "lidar_media_hz": self.lidar_media_hz,
                "lidar_uplink_max_points": self.lidar_uplink_max_points,
                "lidar_compression_level": self.lidar_compression_level,
                "lidar_quantization_cm": self.lidar_quantization_cm,
                "uplink_max_kbps": self.media_max_kbps,
                "uplink_budget_drops": dict(self.media_budget_drops),
            },
            "lidar_pipeline": {
                "frames_in": self.lidar_frames_in,
                "points_last": self.lidar_points_last,
                "media_sent": self.lidar_media_sent,
                "extract_fail": self.lidar_extract_fail,
                "decoder": self.args.lidar_decoder,
                "subscribed": TOPIC_ALIAS_TO_VALUE.get("ULIDAR_ARRAY", "") in self.subscribed_topics,
            },
            "temperatures": {
                "ntc1": low.get("temperature_ntc1"),
                "imu": sport.get("imu_temperature"),
                "motors": low.get("motor_temperatures"),
                "bms_bq_ntc": low.get("bms_bq_ntc"),
                "bms_mcu_ntc": low.get("bms_mcu_ntc"),
            },
            "safety": {
                "enabled": self.safety_enabled,
                "armed": safety_armed,
                **safety_status,
                "last_intervention": {
                    "blocked": bool(self.last_safety_info.get("blocked")),
                    "reasons": last_reasons,
                } if self.last_safety_info else {},
            },
            "autonomy": {
                "enabled": self.autonomy_enabled,
                "driving": self.autonomy_driving,
                "has_target": self.drive_target is not None,
            },
            "alerts": alerts,
        }

    async def _telemetry_loop(self) -> None:
        interval = 1.0 / max(self.args.telemetry_hz, 0.1)
        while not self.stop_event.is_set():
            self._mqtt_publish("telemetry", self._build_telemetry(), qos=0)
            await asyncio.sleep(interval)

    def _validate_command(self, command: Dict[str, Any]) -> Tuple[bool, str]:
        if command.get("robot_id") != self.args.robot_id:
            return False, "robot_id mismatch"

        cmd_type = str(command.get("type", "")).strip()
        if cmd_type not in {
            "move",
            "turn",
            "stop",
            "enter_mode",
            "follow_target",
            "go_to",
            "drive_velocity",
            "set_autonomy",
            "e_stop",
            "set_safety",
            "play_audio",
            "set_video",
            "set_camera_stream",
            "set_audio",
            "set_lidar",
            "set_lidar_decoder",
            "set_color",
            "set_speed_profile",
        }:
            return False, f"unsupported command type: {cmd_type}"

        now_s = time.time()
        issued_ts = float(command.get("ts", now_s))
        ttl_ms = int(command.get("ttl_ms", self.args.default_command_ttl_ms))
        if ttl_ms > 0 and now_s > issued_ts + (ttl_ms / 1000.0):
            return False, "command expired"

        heartbeat_age = time.monotonic() - self.last_heartbeat_monotonic
        requested_profile = str(
            command.get("payload", {}).get("profile", "normal")
            if isinstance(command.get("payload"), dict)
            else "normal"
        )
        requires_live_heartbeat = (
            cmd_type in {"move", "turn", "enter_mode", "follow_target", "go_to", "drive_velocity"}
            or (cmd_type == "set_speed_profile" and requested_profile == "max_api")
        )
        if requires_live_heartbeat and heartbeat_age > self.args.heartbeat_timeout_s:
            return False, "no recent server heartbeat"

        return True, ""

    async def _execute_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        cmd_type = str(command.get("type", ""))
        payload = command.get("payload", {}) if isinstance(command.get("payload"), dict) else {}

        if cmd_type == "stop":
            self._sport_send_nowait(int(SPORT_CMD["StopMove"]))
            self.move_active = False
            self.pending_stop_deadline = 0.0
            return {"executed": "stop"}

        if cmd_type == "move":
            limits = self._speed_profiles().get(
                self.speed_profile,
                self._speed_profiles()["normal"],
            )
            x = clamp(
                float(payload.get("linear_x", 0.0)),
                -limits["reverse"],
                limits["forward"],
            )
            y = clamp(
                float(payload.get("lateral_y", 0.0)),
                -limits["lateral"],
                limits["lateral"],
            )
            z = clamp(
                float(payload.get("angular_z", 0.0)),
                -limits["angular"],
                limits["angular"],
            )
            duration_ms = int(payload.get("duration_ms", self.args.default_move_duration_ms))

            # Reactive safety: clamp/veto translation that would hit an obstacle
            # or run off a ledge. Rotation is preserved.
            x, y, z, safety_info = self._guard_velocity(x, y, z)

            self.move_active = True
            self.last_move_command = {"x": x, "y": y, "z": z}
            self.last_move_sent_at = time.monotonic()
            if duration_ms > 0:
                self.pending_stop_deadline = time.monotonic() + (duration_ms / 1000.0)

            self._sport_send_nowait(int(SPORT_CMD["Move"]), {"x": x, "y": y, "z": z})

            return {
                "executed": "move",
                "x": x,
                "y": y,
                "z": z,
                "duration_ms": duration_ms,
                "safety": safety_info,
            }

        if cmd_type == "turn":
            angle_deg = float(payload.get("angle_deg", 0.0))
            direction = 1.0 if angle_deg >= 0 else -1.0
            active_angular_limit = self._speed_profiles().get(
                self.speed_profile,
                self._speed_profiles()["normal"],
            )["angular"]
            z = direction * clamp(
                abs(float(payload.get("angular_z", self.args.turn_angular_speed))),
                0.05,
                active_angular_limit,
            )
            duration_ms = int(payload.get("duration_ms", abs(angle_deg) * self.args.turn_ms_per_degree))

            self.move_active = True
            self.last_move_command = {"x": 0.0, "y": 0.0, "z": z}
            self.last_move_sent_at = time.monotonic()
            self.pending_stop_deadline = time.monotonic() + max(duration_ms / 1000.0, 0.1)

            await self._sport_request(int(SPORT_CMD["Move"]), {"x": 0.0, "y": 0.0, "z": z})

            return {
                "executed": "turn",
                "angle_deg": angle_deg,
                "angular_z": z,
                "duration_ms": duration_ms,
            }

        if cmd_type == "enter_mode":
            mode = str(payload.get("mode", "normal"))
            response = await self._set_motion_mode(mode)
            return {"executed": "enter_mode", "mode": mode, "response": response}

        if cmd_type == "set_speed_profile":
            profile = str(payload.get("profile", "normal")).strip().lower()
            response = await self._set_speed_profile(profile, stop_first=True)
            return {
                "executed": "set_speed_profile",
                "profile": self.speed_profile,
                "speed_level": self.speed_level,
                "limits": self._speed_profiles()[self.speed_profile],
                "response": response,
            }

        if cmd_type == "set_video":
            enabled = bool(payload.get("enabled", True))
            await self._set_video(enabled)
            return {
                "executed": "set_video",
                "camera_enabled": self.camera_enabled,
            }

        if cmd_type == "set_camera_stream":
            if "format" in payload:
                image_format = str(payload.get("format", self.camera_format)).lower()
                if image_format not in {"jpg", "webp", "h264"}:
                    raise ValueError("format must be 'jpg', 'webp' or 'h264'")
                if image_format != self.camera_format:
                    self._reset_video_encoder()
                self.camera_format = image_format

            if "bitrate_kbps" in payload:
                bitrate_kbps = int(payload.get("bitrate_kbps", self.camera_bitrate_kbps))
                if bitrate_kbps < 64 or bitrate_kbps > 12000:
                    raise ValueError("bitrate_kbps must be between 64 and 12000")
                self.camera_bitrate_kbps = bitrate_kbps

            if "gop" in payload:
                gop = int(payload.get("gop", self.camera_gop))
                if gop < 0 or gop > 600:
                    raise ValueError("gop must be between 0 and 600")
                if gop != self.camera_gop:
                    self.camera_gop = gop
                    self._reset_video_encoder()

            if "emit_every" in payload:
                emit_every = int(payload.get("emit_every", 1))
                if emit_every <= 0:
                    raise ValueError("emit_every must be > 0")
                self.camera_emit_every = emit_every

            if "jpeg_quality" in payload:
                quality = int(payload.get("jpeg_quality", self.args.camera_jpeg_quality))
                if quality < 1 or quality > 100:
                    raise ValueError("jpeg_quality must be between 1 and 100")
                self.camera_jpeg_quality = quality

            if "min_quality" in payload:
                min_quality = int(payload.get("min_quality", self.camera_min_quality))
                if min_quality < 1 or min_quality > 100:
                    raise ValueError("min_quality must be between 1 and 100")
                self.camera_min_quality = min_quality

            self.camera_min_quality = min(
                self.camera_min_quality,
                self.camera_jpeg_quality,
            )

            if "target_fps" in payload:
                target_fps = float(payload.get("target_fps", self.camera_target_fps))
                if target_fps < 1 or target_fps > 40:
                    raise ValueError("target_fps must be between 1 and 40")
                self.camera_target_fps = target_fps

            if "max_width" in payload:
                max_width = int(payload.get("max_width", self.camera_max_width))
                if max_width < 0 or (0 < max_width < 320):
                    raise ValueError("max_width must be 0 or >= 320")
                self.camera_max_width = max_width

            if "uplink_max_kbps" in payload:
                uplink_max_kbps = int(
                    payload.get("uplink_max_kbps", self.media_max_kbps)
                )
                if uplink_max_kbps < 0 or (
                    uplink_max_kbps > 0 and uplink_max_kbps < 256
                ):
                    raise ValueError("uplink_max_kbps must be 0 or >= 256")
                self.media_max_kbps = uplink_max_kbps

            if "enabled" in payload:
                await self._set_video(bool(payload.get("enabled")))

            return {
                "executed": "set_camera_stream",
                "camera_enabled": self.camera_enabled,
                "format": self.camera_format,
                "bitrate_kbps": self.camera_bitrate_kbps,
                "gop": self.camera_gop,
                "emit_every": self.camera_emit_every,
                "jpeg_quality": self.camera_jpeg_quality,
                "min_quality": self.camera_min_quality,
                "target_fps": self.camera_target_fps,
                "max_width": self.camera_max_width,
                "uplink_max_kbps": self.media_max_kbps,
            }

        if cmd_type == "set_audio":
            if "emit_every" in payload:
                emit_every = int(payload.get("emit_every", 1))
                if emit_every <= 0:
                    raise ValueError("emit_every must be > 0")
                self.audio_emit_every = emit_every

            if "max_bytes" in payload:
                max_bytes = int(payload.get("max_bytes", self.audio_max_bytes))
                if max_bytes < 0:
                    raise ValueError("max_bytes must be >= 0")
                self.audio_max_bytes = max_bytes

            if "enabled" in payload or not payload:
                enabled = bool(payload.get("enabled", True))
                await self._set_audio(enabled)
                if enabled:
                    with contextlib.suppress(Exception):
                        await self._subscribe_topic(TOPIC_ALIAS_TO_VALUE["AUDIO_HUB_PLAY_STATE"])

            return {
                "executed": "set_audio",
                "audio_enabled": self.audio_enabled,
                "emit_every": self.audio_emit_every,
                "max_bytes": self.audio_max_bytes,
            }

        if cmd_type == "set_lidar":
            enabled = bool(payload.get("enabled", True))
            subscribe = bool(payload.get("subscribe", True))

            if "media_hz" in payload:
                media_hz = float(payload.get("media_hz", self.lidar_media_hz))
                if media_hz < 0.2 or media_hz > 15:
                    raise ValueError("media_hz must be between 0.2 and 15")
                self.lidar_media_hz = media_hz

            if "max_points" in payload:
                max_points = int(payload.get("max_points", self.lidar_uplink_max_points))
                if max_points < 500 or max_points > 100000:
                    raise ValueError("max_points must be between 500 and 100000")
                self.lidar_uplink_max_points = max_points

            if "compression_level" in payload:
                compression_level = int(
                    payload.get("compression_level", self.lidar_compression_level)
                )
                if compression_level < 0 or compression_level > 9:
                    raise ValueError("compression_level must be between 0 and 9")
                self.lidar_compression_level = compression_level

            if "quantization_cm" in payload:
                quantization_cm = float(
                    payload.get("quantization_cm", self.lidar_quantization_cm)
                )
                if quantization_cm < 0.1 or quantization_cm > 50:
                    raise ValueError("quantization_cm must be between 0.1 and 50")
                self.lidar_quantization_cm = quantization_cm

            await self._set_lidar(enabled)
            if enabled and subscribe:
                with contextlib.suppress(Exception):
                    await self._subscribe_topic(TOPIC_ALIAS_TO_VALUE["ULIDAR_ARRAY"])
                with contextlib.suppress(Exception):
                    await self._subscribe_topic(TOPIC_ALIAS_TO_VALUE["ULIDAR_STATE"])
            return {
                "executed": "set_lidar",
                "lidar_enabled": self.lidar_enabled,
                "subscribe": subscribe,
                "media_hz": self.lidar_media_hz,
                "max_points": self.lidar_uplink_max_points,
                "compression_level": self.lidar_compression_level,
                "quantization_cm": self.lidar_quantization_cm,
            }

        if cmd_type == "set_color":
            if "enabled" in payload:
                self.colorize_enabled = bool(payload.get("enabled"))
            if "fov_deg" in payload:
                self.color_cam_fov_deg = clamp(float(payload["fov_deg"]), 30.0, 220.0)
            if "pitch_deg" in payload:
                self.color_cam_pitch_deg = clamp(float(payload["pitch_deg"]), -60.0, 60.0)
            if "height_m" in payload:
                self.color_cam_height_m = clamp(float(payload["height_m"]), -1.0, 3.0)
            if "forward_m" in payload:
                self.color_cam_forward_m = clamp(float(payload["forward_m"]), -1.0, 2.0)
            if "max_distance_m" in payload:
                self.color_max_distance_m = clamp(float(payload["max_distance_m"]), 0.5, 60.0)
            return {
                "executed": "set_color",
                "colorize_enabled": self.colorize_enabled,
                "fov_deg": self.color_cam_fov_deg,
                "pitch_deg": self.color_cam_pitch_deg,
                "height_m": self.color_cam_height_m,
                "forward_m": self.color_cam_forward_m,
                "max_distance_m": self.color_max_distance_m,
            }

        if cmd_type == "set_lidar_decoder":
            if self.conn is None:
                raise RuntimeError("Connection not ready")

            decoder = str(payload.get("decoder", self.args.lidar_decoder)).strip().lower()
            if decoder not in {"libvoxel", "native"}:
                raise ValueError("decoder must be 'libvoxel' or 'native'")
            self.conn.datachannel.set_decoder(decoder)
            return {
                "executed": "set_lidar_decoder",
                "decoder": decoder,
            }

        if cmd_type == "set_autonomy":
            enabled = bool(payload.get("enabled", False))
            self.autonomy_enabled = enabled
            if not enabled:
                self.drive_target = None
                with contextlib.suppress(Exception):
                    self._sport_send_nowait(int(SPORT_CMD["StopMove"]))
                self.move_active = False
                self.pending_stop_deadline = 0.0
                self.autonomy_driving = False
            self._publish_event("autonomy_state", {"enabled": enabled})
            return {
                "executed": "set_autonomy",
                "autonomy_enabled": self.autonomy_enabled,
                "lidar_enabled": self.lidar_enabled,
                "safety_enabled": self.safety_enabled,
            }

        if cmd_type == "drive_velocity":
            # Continuous velocity goal from the server brain. The drive loop
            # repeats it (through the safety guard) until it goes stale.
            if not self.autonomy_enabled:
                raise ValueError("autonomy is disabled; send set_autonomy first")
            limits = self._speed_profiles().get(
                self.speed_profile, self._speed_profiles()["normal"]
            )
            vx = clamp(float(payload.get("vx", 0.0)), -limits["reverse"], limits["forward"])
            vy = clamp(float(payload.get("vy", 0.0)), -limits["lateral"], limits["lateral"])
            wz = clamp(float(payload.get("wz", 0.0)), -limits["angular"], limits["angular"])
            self.drive_target = {"vx": vx, "vy": vy, "wz": wz}
            self.drive_target_set_at = time.monotonic()
            return {"executed": "drive_velocity", "target": self.drive_target}

        if cmd_type == "e_stop":
            self.autonomy_enabled = False
            self.drive_target = None
            self.autonomy_driving = False
            self.move_active = False
            self.pending_stop_deadline = 0.0
            for _ in range(2):
                with contextlib.suppress(Exception):
                    self._sport_send_nowait(int(SPORT_CMD["StopMove"]))
            self._publish_event("e_stop", {"source": str(command.get("issued_by", ""))})
            return {"executed": "e_stop"}

        if cmd_type == "set_safety":
            if "enabled" in payload:
                self.safety_enabled = bool(payload.get("enabled"))
            updated = self.safety_guard.apply_settings(payload)
            return {
                "executed": "set_safety",
                "safety_enabled": self.safety_enabled,
                "config": updated,
            }

        if cmd_type == "play_audio":
            force = bool(payload.get("force", False))
            result = await self._play_greet(force=force)
            return {"executed": "play_audio", **result}

        if cmd_type in {"follow_target", "go_to"}:
            raise ValueError(f"{cmd_type} not implemented yet on edge")

        raise ValueError(f"unsupported command type: {cmd_type}")

    async def _command_loop(self) -> None:
        while not self.stop_event.is_set():
            command = await self.command_queue.get()

            if command.get("__motion_marker__"):
                self.motion_marker_queued = False
                command = self.latest_motion_command
                self.latest_motion_command = None
                if command is None:
                    continue

            command_id = str(command.get("command_id", ""))
            streaming = bool(command.get("streaming"))

            valid, reason = self._validate_command(command)
            if not valid:
                if not streaming:
                    self._send_command_ack(command, "rejected", reason)
                continue

            if not streaming:
                self._send_command_ack(command, "accepted")

            try:
                result = await self._execute_command(command)
            except Exception as exc:
                if not streaming:
                    self._send_command_ack(command, "error", str(exc))
                self._publish_event("command_error", {"command_id": command_id, "error": str(exc)})
                continue

            if not streaming:
                self._send_command_ack(command, "executed", extra={"result": result})

    async def _watchdog_loop(self) -> None:
        while not self.stop_event.is_set():
            now = time.monotonic()
            heartbeat_age = now - self.last_heartbeat_monotonic

            deadline_expired = (
                self.move_active
                and self.pending_stop_deadline > 0
                and now >= self.pending_stop_deadline
            )
            heartbeat_expired = self.move_active and heartbeat_age > self.args.heartbeat_timeout_s
            profile_heartbeat_expired = (
                self.speed_profile == "max_api"
                and heartbeat_age > self.args.heartbeat_timeout_s
            )

            if deadline_expired or heartbeat_expired:
                try:
                    self._sport_send_nowait(int(SPORT_CMD["StopMove"]))
                except Exception as exc:
                    self.pending_stop_deadline = time.monotonic() + self.args.stop_retry_interval_s
                    if now - self.last_heartbeat_fault_event_at >= 1.0:
                        self.last_heartbeat_fault_event_at = now
                        self._publish_event(
                            "safe_stop_retry",
                            {
                                "error": str(exc),
                                "heartbeat_age_s": round(heartbeat_age, 3),
                            },
                        )
                else:
                    self.move_active = False
                    self.pending_stop_deadline = 0.0
                    self._publish_event(
                        "safe_stop_executed",
                        {
                            "reason": "heartbeat_timeout" if heartbeat_expired else "duration_timeout",
                            "heartbeat_age_s": round(heartbeat_age, 3),
                        },
                    )

            if (
                profile_heartbeat_expired
                and now - self.last_speed_profile_reset_attempt >= 1.0
            ):
                self.last_speed_profile_reset_attempt = now
                try:
                    self._sport_send_nowait(int(SPORT_CMD["StopMove"]))
                    self.move_active = False
                    self.pending_stop_deadline = 0.0
                    await self._set_speed_profile("normal", stop_first=False)
                except Exception as exc:
                    self._publish_event(
                        "speed_profile_reset_retry",
                        {
                            "reason": "heartbeat_timeout",
                            "error": str(exc),
                            "heartbeat_age_s": round(heartbeat_age, 3),
                        },
                    )

            await asyncio.sleep(0.05)

    async def _autonomy_drive_loop(self) -> None:
        """Repeat the latest drive_velocity goal at a fixed rate, always filtered
        by the safety guard. Decouples smooth motion from MQTT jitter and keeps a
        short local TTL so the robot stops promptly if goals stop arriving or the
        server heartbeat drops (the watchdog is the ultimate failsafe)."""
        interval = 1.0 / max(self.args.autonomy_drive_hz, 1.0)
        while not self.stop_event.is_set():
            await asyncio.sleep(interval)

            if not self.autonomy_enabled:
                continue

            now = time.monotonic()
            heartbeat_age = now - self.last_heartbeat_monotonic
            target = self.drive_target
            target_stale = (
                target is None
                or (now - self.drive_target_set_at) > self.drive_target_ttl_s
            )

            if heartbeat_age > self.args.heartbeat_timeout_s:
                # Lost the server: the watchdog will StopMove. Don't drive.
                self.autonomy_driving = False
                continue

            if target_stale:
                if self.autonomy_driving:
                    with contextlib.suppress(Exception):
                        self._sport_send_nowait(int(SPORT_CMD["StopMove"]))
                    self.move_active = False
                    self.pending_stop_deadline = 0.0
                    self.autonomy_driving = False
                continue

            vx, vy, wz, _info = self._guard_velocity(
                target["vx"], target["vy"], target["wz"]
            )
            try:
                self._sport_send_nowait(
                    int(SPORT_CMD["Move"]), {"x": vx, "y": vy, "z": wz}
                )
            except Exception:
                continue

            self.autonomy_driving = True
            self.move_active = True
            self.last_move_command = {"x": vx, "y": vy, "z": wz}
            self.last_move_sent_at = now
            # Refreshed each cycle so the duration watchdog never fires mid-stream;
            # if this loop dies the deadline lapses and the robot is stopped.
            self.pending_stop_deadline = now + max(2.0 * interval, 0.4)

    async def _media_uplink_loop(self) -> None:
        if not self.args.media_ws_url:
            return

        if websockets is None:
            self._publish_event("media_uplink_disabled", {"reason": "python websockets package missing"})
            return

        ws_url = self.args.media_ws_url.format(robot_id=self.args.robot_id)
        if self.args.media_ws_token:
            sep = "&" if "?" in ws_url else "?"
            ws_url = f"{ws_url}{sep}token={self.args.media_ws_token}"

        while not self.stop_event.is_set():
            budget_tokens = max(self.media_max_kbps, 0) * 125.0
            budget_updated_at = time.monotonic()
            try:
                async with websockets.connect(
                    ws_url,
                    compression=None,
                    open_timeout=10,
                    close_timeout=2,
                    ping_interval=15,
                    ping_timeout=15,
                    max_size=self.args.media_ws_max_size,
                    max_queue=4,
                    write_limit=64 * 1024,
                ) as ws:
                    self._publish_event("media_uplink_connected", {"url": ws_url})

                    while not self.stop_event.is_set():
                        await self.media_ready.wait()
                        self.media_ready.clear()

                        batch: List[Dict[str, Any]] = []

                        lidar_points = self.latest_media_by_stream.pop("lidar", None)
                        if lidar_points is not None:
                            batch.append(lidar_points)

                        # H.264 frames first and in order: they carry inter-frame
                        # dependencies and must never be reordered or dropped here.
                        while True:
                            try:
                                batch.append(self.video_media_queue.get_nowait())
                            except asyncio.QueueEmpty:
                                break

                        video = self.latest_media_by_stream.pop("video", None)
                        if video is not None:
                            batch.append(video)

                        for _ in range(self.args.media_audio_batch_size):
                            try:
                                batch.append(self.media_queue.get_nowait())
                            except asyncio.QueueEmpty:
                                break

                        for stream in list(self.latest_media_by_stream):
                            payload = self.latest_media_by_stream.pop(stream, None)
                            if payload is not None:
                                batch.append(payload)

                        if (
                            not self.media_queue.empty()
                            or not self.video_media_queue.empty()
                            or self.latest_media_by_stream
                        ):
                            self.media_ready.set()

                        for payload in batch:
                            header = payload.get("header") if isinstance(payload.get("header"), dict) else None
                            if payload.get("binary") and header is not None:
                                # Binary media frame (lidar/video): blob rides raw,
                                # no base64. ~33% lighter on the wire and on the Pi CPU.
                                wire: Any = encode_media_frame(header, payload["payload"])
                                payload_size = len(wire)
                                reliable = (
                                    payload.get("stream") == "video"
                                    and header.get("image_format") == "h264"
                                )
                            else:
                                # JSON streams (audio): keep text on the wire.
                                wire = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
                                payload_size = len(wire.encode("utf-8"))
                                reliable = False

                            max_bytes_per_second = max(self.media_max_kbps, 0) * 125.0
                            if max_bytes_per_second > 0:
                                now = time.monotonic()
                                budget_tokens = min(
                                    max_bytes_per_second,
                                    budget_tokens
                                    + (now - budget_updated_at) * max_bytes_per_second,
                                )
                                budget_updated_at = now
                                # H.264 rides its own encoder rate control; never drop
                                # it on the byte budget. Still debit the bucket so the
                                # droppable streams (lidar/audio) yield bandwidth to it.
                                if not reliable and payload_size > budget_tokens:
                                    stream = str(payload.get("stream", "unknown"))
                                    self.media_budget_drops[stream] = (
                                        self.media_budget_drops.get(stream, 0) + 1
                                    )
                                    continue
                                budget_tokens = max(0.0, budget_tokens - payload_size)
                            await asyncio.wait_for(ws.send(wire), timeout=self.args.media_ws_send_timeout_s)

            except Exception as exc:
                self._publish_event("media_uplink_retry", {"error": str(exc)})
                await asyncio.sleep(self.args.media_ws_reconnect_s)

    async def _connect_robot(self) -> None:
        connection = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.args.go2_ip)
        try:
            await connection.connect()
        except BaseException:
            with contextlib.suppress(Exception):
                await connection.disconnect()
            raise

        self.conn = connection
        self.robot_connected_at = time.monotonic()
        self.subscribed_topics.clear()
        self.latest_by_topic.clear()
        self._reset_audio_pipeline()

        self.conn.video.add_track_callback(self._on_video_track)
        self.conn.audio.add_track_callback(self._on_audio_frame)

        try:
            await self._set_speed_profile("normal", stop_first=True)
        except Exception as exc:
            self.speed_profile = "normal"
            self.speed_level = 0
            self._publish_event(
                "speed_profile_init_warning",
                {"error": str(exc)},
            )

        if self.args.disable_traffic_saving:
            with contextlib.suppress(Exception):
                disabled = await asyncio.wait_for(
                    self.conn.datachannel.disableTrafficSaving(True),
                    timeout=3.0,
                )
                self.traffic_saving_disabled = bool(disabled)

        with contextlib.suppress(Exception):
            self.conn.datachannel.set_decoder(self.args.lidar_decoder)

        profiles: List[str] = []
        for profile_entry in self.args.subscribe_profile:
            profiles.extend([x.strip() for x in profile_entry.split(",") if x.strip()])

        extra_topics: List[str] = []
        for topic_entry in self.args.subscribe_topic:
            extra_topics.extend([x.strip() for x in topic_entry.split(",") if x.strip()])

        for profile in profiles:
            aliases = PROFILE_TOPICS.get(profile, [])
            for alias in aliases:
                if alias in TOPIC_ALIAS_TO_VALUE:
                    with contextlib.suppress(Exception):
                        await self._subscribe_topic(TOPIC_ALIAS_TO_VALUE[alias])

        for topic in extra_topics:
            with contextlib.suppress(Exception):
                await self._subscribe_topic(self._resolve_topic(topic))

        if self.camera_enabled:
            await self._set_video(True)
        if self.audio_enabled:
            await self._set_audio(True)
        if self.lidar_enabled:
            await self._set_lidar(True)

        # Pre-warm the greeting audio (upload + resolve uuid) so the first greet
        # plays instantly. Background task: never blocks the connection.
        if (
            WebRTCAudioHub is not None
            and not self.greet_audio_resolved
            and (
                self.greet_prewarm_task is None
                or self.greet_prewarm_task.done()
            )
        ):
            self.greet_prewarm_task = asyncio.create_task(self._prewarm_greet())

        self._publish_event(
            "robot_connected",
            {
                "go2_ip": self.args.go2_ip,
                "camera": self.camera_enabled,
                "audio": self.audio_enabled,
                "lidar": self.lidar_enabled,
                "topics": sorted(self.subscribed_topics),
            },
        )

    async def _disconnect_robot(self) -> None:
        if self.greet_prewarm_task and not self.greet_prewarm_task.done():
            self.greet_prewarm_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.greet_prewarm_task
        self.greet_prewarm_task = None

        if self.video_task and not self.video_task.done():
            self.video_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.video_task
        if self.video_encode_task and not self.video_encode_task.done():
            self.video_encode_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.video_encode_task

        self.video_task = None
        self.video_encode_task = None
        self.latest_camera_frame = None
        self.camera_frame_ready.clear()

        connection = self.conn
        if connection is not None:
            with contextlib.suppress(Exception):
                self._sport_send_nowait(int(SPORT_CMD["StopMove"]))
            if self.speed_profile != "normal":
                with contextlib.suppress(Exception):
                    await self._set_speed_profile("normal", stop_first=False)

        self.conn = None
        self.audio_hub = None
        self.audio_hub_play_mode_ready = False
        self.robot_connected_at = 0.0
        self.traffic_saving_disabled = False
        self.subscribed_topics.clear()
        self.latest_by_topic.clear()
        self._reset_audio_pipeline()
        self.speed_profile = "normal"
        self.speed_level = 0
        self.move_active = False
        self.pending_stop_deadline = 0.0
        self.last_move_command = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.camera_channel_active = False

        if connection is None:
            return

        with contextlib.suppress(Exception):
            connection.video.switchVideoChannel(False)
        with contextlib.suppress(Exception):
            connection.audio.switchAudioChannel(False)
        with contextlib.suppress(Exception):
            await connection.disconnect()

    async def _robot_supervisor_loop(self) -> None:
        bad_since = 0.0
        consecutive_failures = 0

        while not self.stop_event.is_set():
            if self.conn is None:
                try:
                    await asyncio.wait_for(
                        self._connect_robot(),
                        timeout=self.args.robot_connect_timeout_s,
                    )
                    bad_since = 0.0
                    consecutive_failures = 0
                except asyncio.CancelledError:
                    raise
                except (Exception, SystemExit) as exc:
                    await self._disconnect_robot()
                    # A failed WebRTC connect can leak sockets/file descriptors in
                    # the aiortc/ICE stack. Force a GC so they are reclaimed before
                    # the next attempt, and back off exponentially so repeated
                    # failures cannot exhaust the FD limit (Errno 24).
                    gc.collect()
                    consecutive_failures += 1
                    backoff = min(
                        self.args.robot_reconnect_s * (2 ** (consecutive_failures - 1)),
                        self.args.robot_reconnect_max_s,
                    )
                    self._publish_event(
                        "robot_connect_retry",
                        {
                            "error": str(exc),
                            "attempt": consecutive_failures,
                            "next_retry_s": round(backoff, 1),
                        },
                    )
                    await asyncio.sleep(backoff)
                continue

            connection = self.conn
            pc = getattr(connection, "pc", None)
            datachannel = getattr(connection, "datachannel", None)
            channel = getattr(datachannel, "channel", None)

            pc_state = str(getattr(pc, "connectionState", "closed"))
            data_state = str(getattr(channel, "readyState", "closed"))

            heartbeat = getattr(datachannel, "heartbeat", None)
            heartbeat_at = getattr(heartbeat, "heartbeat_response", None)
            connected_age = time.monotonic() - self.robot_connected_at

            heartbeat_stale = False
            if heartbeat_at is None:
                heartbeat_stale = connected_age > self.args.robot_heartbeat_timeout_s
            else:
                heartbeat_stale = time.time() - float(heartbeat_at) > self.args.robot_heartbeat_timeout_s

            link_bad = (
                pc_state in {"failed", "closed", "disconnected"}
                or data_state != "open"
                or heartbeat_stale
            )

            if not link_bad:
                bad_since = 0.0
                await asyncio.sleep(1.0)
                continue

            if bad_since <= 0:
                bad_since = time.monotonic()

            if time.monotonic() - bad_since >= self.args.robot_link_grace_s:
                self._publish_event(
                    "robot_link_lost",
                    {
                        "pc_state": pc_state,
                        "data_state": data_state,
                        "heartbeat_stale": heartbeat_stale,
                    },
                )
                await self._disconnect_robot()
                bad_since = 0.0

            await asyncio.sleep(1.0)

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()

        self._setup_mqtt()

        tasks: List[asyncio.Task[Any]] = [
            asyncio.create_task(self._robot_supervisor_loop()),
            asyncio.create_task(self._command_loop()),
            asyncio.create_task(self._watchdog_loop()),
            asyncio.create_task(self._autonomy_drive_loop()),
            asyncio.create_task(self._telemetry_loop()),
            asyncio.create_task(self._media_uplink_loop()),
        ]

        try:
            while not self.stop_event.is_set():
                await asyncio.sleep(0.2)
        finally:
            for task in tasks:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            await self._disconnect_robot()

            if self.mqtt_client is not None:
                with contextlib.suppress(Exception):
                    self.mqtt_client.loop_stop()
                with contextlib.suppress(Exception):
                    self.mqtt_client.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Go2 Edge Gateway (Raspberry): receives local robot data, publishes telemetry/events to MQTT, "
            "executes validated high-level commands from server, and uploads heavy media in a separate channel."
        )
    )

    parser.add_argument("--robot-id", default="go2_01")
    parser.add_argument("--go2-ip", default="192.168.123.161")

    parser.add_argument("--enable-camera", action="store_true")
    parser.add_argument("--enable-audio", action="store_true")
    parser.add_argument("--enable-lidar", action="store_true")
    parser.add_argument("--disable-traffic-saving", action="store_true")
    parser.add_argument("--lidar-decoder", choices=["libvoxel", "native"], default="native")

    parser.add_argument("--subscribe-profile", action="append", default=["core,lidar,audio"])
    parser.add_argument("--subscribe-topic", action="append", default=[])

    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-username", default="")
    parser.add_argument("--mqtt-password", default="")
    parser.add_argument("--mqtt-tls", action="store_true")
    parser.add_argument("--mqtt-topic-prefix", default="go2")
    parser.add_argument("--mqtt-client-id", default="")

    parser.add_argument("--telemetry-hz", type=float, default=5.0)
    parser.add_argument("--low-battery-threshold", type=float, default=20.0)

    parser.add_argument("--heartbeat-timeout-s", type=float, default=1.5)
    parser.add_argument("--default-command-ttl-ms", type=int, default=1500)
    parser.add_argument("--command-queue-size", type=int, default=256)

    parser.add_argument("--max-forward-speed", type=float, default=3.8)
    parser.add_argument("--max-reverse-speed", type=float, default=2.5)
    parser.add_argument("--max-lateral-speed", type=float, default=1.0)
    parser.add_argument("--max-angular-speed", type=float, default=4.0)
    parser.add_argument("--normal-forward-speed", type=float, default=3.5)
    parser.add_argument("--normal-speed-scale", type=float, default=0.92)
    parser.add_argument("--default-move-duration-ms", type=int, default=700)
    parser.add_argument("--turn-angular-speed", type=float, default=0.8)
    parser.add_argument("--turn-ms-per-degree", type=float, default=15.0)

    parser.add_argument("--camera-emit-every", type=int, default=1)
    parser.add_argument("--camera-format", choices=["jpg", "webp", "h264"], default="h264")
    parser.add_argument("--camera-jpeg-quality", type=int, default=55)
    parser.add_argument("--camera-min-quality", type=int, default=34)
    parser.add_argument("--camera-target-fps", type=float, default=30.0)
    parser.add_argument("--camera-max-width", type=int, default=640)
    # H.264 path (default). Inter-frame compression so 30-40 fps fits a ~2 Mbps link.
    parser.add_argument("--camera-bitrate-kbps", type=int, default=900)
    parser.add_argument(
        "--camera-gop",
        type=int,
        default=0,
        help="keyframe interval in frames; 0 = auto (2 seconds at target fps)",
    )
    parser.add_argument(
        "--camera-h264-encoder",
        choices=["auto", "h264_v4l2m2m", "libx264"],
        default="auto",
        help="auto tries the Pi hardware encoder (h264_v4l2m2m) then falls back to libx264",
    )
    parser.add_argument("--video-queue-size", type=int, default=240)

    parser.add_argument("--audio-emit-every", type=int, default=2)
    parser.add_argument("--audio-max-bytes", type=int, default=24576)
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=16000,
        choices=[8000, 12000, 16000, 24000, 48000],
    )

    parser.add_argument("--lidar-media-hz", type=float, default=0.7)
    parser.add_argument("--lidar-uplink-max-points", type=int, default=2500)
    parser.add_argument("--lidar-compression-level", type=int, default=6)
    parser.add_argument(
        "--lidar-quantization-cm",
        type=float,
        default=2.0,
        help="LiDAR coordinate precision in centimeters before zlib compression.",
    )

    # Camera colorization (approximate fisheye, tunable live via set_color).
    parser.add_argument(
        "--enable-colorization",
        action="store_true",
        help="Color each LiDAR point with the live camera frame (needs the camera on).",
    )
    parser.add_argument("--color-cam-fov-deg", type=float, default=150.0,
                        help="Approx camera field of view (fisheye) in degrees.")
    parser.add_argument("--color-cam-pitch-deg", type=float, default=0.0,
                        help="Camera mount pitch tilt; positive looks down.")
    parser.add_argument("--color-cam-height-m", type=float, default=0.30,
                        help="Camera height above the pose origin (m).")
    parser.add_argument("--color-cam-forward-m", type=float, default=0.25,
                        help="Camera forward offset from the pose origin (m).")
    parser.add_argument("--color-max-distance-m", type=float, default=12.0,
                        help="Max point distance to colorize (m).")
    parser.add_argument("--color-points-frame", choices=["world", "body"], default="world",
                        help="Frame of the LiDAR points: 'world'/odom (default) or 'body'.")

    # --- Reactive safety guard + autonomy ------------------------------------
    parser.add_argument("--enable-safety-guard", dest="enable_safety_guard",
                        action="store_true", default=True,
                        help="Reactive LiDAR collision/cliff guard (default on).")
    parser.add_argument("--disable-safety-guard", dest="enable_safety_guard",
                        action="store_false",
                        help="Disable the local safety guard (NOT recommended).")
    parser.add_argument("--safety-points-frame", choices=["world", "body"], default="world",
                        help="Frame of LiDAR points fed to the guard.")
    parser.add_argument("--safety-update-hz", type=float, default=10.0,
                        help="Max rate the guard ingests LiDAR scans.")
    parser.add_argument("--safety-stop-distance-m", type=float, default=0.50,
                        help="Below this clearance, motion toward it is vetoed.")
    parser.add_argument("--safety-slow-distance-m", type=float, default=1.20,
                        help="Between stop and this distance, speed is ramped down.")
    parser.add_argument("--safety-robot-half-width-m", type=float, default=0.26,
                        help="Robot footprint half width for corridor clearance.")
    parser.add_argument("--safety-cone-half-deg", type=float, default=45.0,
                        help="Half angle of the cone scanned around the motion direction.")
    parser.add_argument("--safety-obstacle-min-height-m", type=float, default=0.06,
                        help="Min height above ground for a point to count as obstacle.")
    parser.add_argument("--safety-obstacle-max-height-m", type=float, default=1.20,
                        help="Max obstacle height (above it the robot passes under).")
    parser.add_argument("--safety-ground-z-m", type=float, default=-0.30,
                        help="Calibrated floor height in body frame (z up).")
    parser.add_argument("--safety-max-radius-m", type=float, default=4.0,
                        help="Ignore obstacle points beyond this radius.")
    parser.add_argument("--safety-scan-timeout-s", type=float, default=1.0,
                        help="Scan older than this blocks translation (fail safe).")
    parser.add_argument("--safety-cliff-enabled", dest="safety_cliff_enabled",
                        action="store_true", default=True,
                        help="Negative-obstacle (ledge/stair) guard (default on).")
    parser.add_argument("--safety-cliff-disabled", dest="safety_cliff_enabled",
                        action="store_false", help="Disable cliff guard.")
    parser.add_argument("--safety-cliff-lookahead-m", type=float, default=0.55,
                        help="How far ahead the cliff strip extends.")
    parser.add_argument("--safety-cliff-drop-m", type=float, default=0.12,
                        help="Floor drop below ground that counts as a ledge.")
    parser.add_argument("--autonomy-drive-hz", type=float, default=8.0,
                        help="Rate the autonomy loop repeats the drive goal.")
    parser.add_argument("--drive-target-ttl-s", type=float, default=0.8,
                        help="A drive_velocity goal older than this is dropped (robot stops).")

    # --- Audio greeting (Go2 speaker via audio hub) --------------------------
    parser.add_argument("--greet-audio-file", default="Escuela-Técnica-Ort-3.wav",
                        help="WAV played through the Go2 speaker (normalized to PCM16 mono 44.1 kHz).")
    parser.add_argument("--greet-audio-uuid", default="",
                        help="Skip upload: play this already-stored audio uuid directly.")
    parser.add_argument("--greet-min-interval-s", type=float, default=5.0,
                        help="Minimum seconds between greetings (rate limit).")

    parser.add_argument(
        "--media-ws-url",
        default="",
        help="Separate heavy-data uplink URL. Example: ws://server:8000/ws/edge-media/{robot_id}",
    )
    parser.add_argument("--media-ws-token", default="")
    parser.add_argument("--media-queue-size", type=int, default=64)
    parser.add_argument("--media-audio-batch-size", type=int, default=4)
    parser.add_argument(
        "--media-max-kbps",
        type=int,
        default=2200,
        help="Total media uplink budget. Frames are dropped instead of queued; 0 disables the cap.",
    )
    parser.add_argument("--media-ws-reconnect-s", type=float, default=2.0)
    parser.add_argument("--media-ws-send-timeout-s", type=float, default=2.0)
    parser.add_argument("--media-ws-max-size", type=int, default=8 * 1024 * 1024)

    parser.add_argument("--robot-connect-timeout-s", type=float, default=20.0)
    parser.add_argument("--robot-reconnect-s", type=float, default=2.0,
                        help="Base delay between reconnect attempts (grows exponentially on repeated failures).")
    parser.add_argument("--robot-reconnect-max-s", type=float, default=30.0,
                        help="Cap for the exponential reconnect backoff.")
    parser.add_argument("--robot-heartbeat-timeout-s", type=float, default=10.0)
    parser.add_argument("--robot-link-grace-s", type=float, default=3.0)
    parser.add_argument("--robot-request-timeout-s", type=float, default=2.0)
    parser.add_argument("--stop-retry-interval-s", type=float, default=0.5)

    args = parser.parse_args()

    if args.telemetry_hz <= 0:
        parser.error("--telemetry-hz must be > 0")

    if args.camera_emit_every <= 0:
        parser.error("--camera-emit-every must be > 0")

    if args.camera_jpeg_quality < 1 or args.camera_jpeg_quality > 100:
        parser.error("--camera-jpeg-quality must be between 1 and 100")

    if args.camera_min_quality < 1 or args.camera_min_quality > args.camera_jpeg_quality:
        parser.error("--camera-min-quality must be between 1 and --camera-jpeg-quality")

    if args.camera_target_fps < 1 or args.camera_target_fps > 40:
        parser.error("--camera-target-fps must be between 1 and 40")

    if args.camera_max_width < 0 or (0 < args.camera_max_width < 320):
        parser.error("--camera-max-width must be 0 or >= 320")

    if args.camera_bitrate_kbps < 64 or args.camera_bitrate_kbps > 12000:
        parser.error("--camera-bitrate-kbps must be between 64 and 12000")

    if args.camera_gop < 0 or args.camera_gop > 600:
        parser.error("--camera-gop must be between 0 and 600")

    if args.video_queue_size < 1:
        parser.error("--video-queue-size must be >= 1")

    if args.audio_emit_every <= 0:
        parser.error("--audio-emit-every must be > 0")

    if args.audio_max_bytes < 0:
        parser.error("--audio-max-bytes must be >= 0")

    if args.greet_min_interval_s <= 0:
        parser.error("--greet-min-interval-s must be > 0")

    if args.heartbeat_timeout_s <= 0:
        parser.error("--heartbeat-timeout-s must be > 0")

    if args.default_command_ttl_ms < 0:
        parser.error("--default-command-ttl-ms must be >= 0")

    if args.command_queue_size <= 0:
        parser.error("--command-queue-size must be > 0")

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

    if args.media_queue_size <= 0:
        parser.error("--media-queue-size must be > 0")

    if args.media_audio_batch_size <= 0:
        parser.error("--media-audio-batch-size must be > 0")

    if args.media_max_kbps < 0 or (0 < args.media_max_kbps < 256):
        parser.error("--media-max-kbps must be 0 or >= 256")

    if args.media_ws_send_timeout_s <= 0:
        parser.error("--media-ws-send-timeout-s must be > 0")

    if args.robot_connect_timeout_s <= 0:
        parser.error("--robot-connect-timeout-s must be > 0")

    if args.robot_reconnect_s <= 0:
        parser.error("--robot-reconnect-s must be > 0")

    if args.robot_reconnect_max_s < args.robot_reconnect_s:
        parser.error("--robot-reconnect-max-s must be >= --robot-reconnect-s")

    if args.robot_heartbeat_timeout_s <= 0:
        parser.error("--robot-heartbeat-timeout-s must be > 0")

    if args.robot_link_grace_s < 0:
        parser.error("--robot-link-grace-s must be >= 0")

    if args.robot_request_timeout_s <= 0:
        parser.error("--robot-request-timeout-s must be > 0")

    if args.stop_retry_interval_s <= 0:
        parser.error("--stop-retry-interval-s must be > 0")

    if args.lidar_media_hz <= 0:
        parser.error("--lidar-media-hz must be > 0")

    if args.lidar_uplink_max_points <= 0:
        parser.error("--lidar-uplink-max-points must be > 0")

    if args.lidar_compression_level < 0 or args.lidar_compression_level > 9:
        parser.error("--lidar-compression-level must be between 0 and 9")

    if args.lidar_quantization_cm < 0.1 or args.lidar_quantization_cm > 50:
        parser.error("--lidar-quantization-cm must be between 0.1 and 50")

    if args.safety_update_hz <= 0:
        parser.error("--safety-update-hz must be > 0")

    if args.safety_stop_distance_m <= 0:
        parser.error("--safety-stop-distance-m must be > 0")

    if args.safety_slow_distance_m <= args.safety_stop_distance_m:
        parser.error("--safety-slow-distance-m must be > --safety-stop-distance-m")

    if args.autonomy_drive_hz <= 0:
        parser.error("--autonomy-drive-hz must be > 0")

    if args.drive_target_ttl_s <= 0:
        parser.error("--drive-target-ttl-s must be > 0")

    if args.mqtt_port <= 0 or args.mqtt_port > 65535:
        parser.error("--mqtt-port must be between 1 and 65535")

    return args


def raise_fd_limit() -> None:
    """Raise this process's open-file limit toward the system hard limit.

    The Go2 WebRTC stack (aiortc/ICE) plus the LiDAR decoder (libvoxel.wasm via
    wasmtime) open many file descriptors; the typical 1024 soft limit on a
    Raspberry Pi is easily exhausted ([Errno 24] Too many open files), which then
    blocks every reconnect. Bumping the soft limit to the hard limit in-process
    means it works without anyone remembering `ulimit -n`."""
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = hard if hard != resource.RLIM_INFINITY else 1048576
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            new_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            print(f"[edge] open-file limit raised {soft} -> {new_soft}", flush=True)
    except Exception as exc:  # non-Unix or not permitted; harmless
        print(f"[edge] could not raise open-file limit: {exc}", flush=True)


def main() -> None:
    raise_fd_limit()
    args = parse_args()
    app = EdgeGatewayService(args)

    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
