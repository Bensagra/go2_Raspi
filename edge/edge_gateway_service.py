#!/usr/bin/env python3
import argparse
import asyncio
import base64
import contextlib
import json
import time
import uuid
import zlib
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from av.audio.resampler import AudioResampler

try:
    import websockets
except Exception:
    websockets = None

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
        self.robot_request_lock = asyncio.Lock()
        self.robot_request_sequence = int(time.time() * 1000) & 0x3FFFFFFF

        self.mqtt_client: Optional[mqtt.Client] = None
        self.last_heartbeat_monotonic = time.monotonic()
        self.last_heartbeat_payload: Dict[str, Any] = {}

        self.subscribed_topics: set[str] = set()
        self.latest_by_topic: Dict[str, Any] = {}

        self.camera_enabled = args.enable_camera
        self.camera_emit_every = args.camera_emit_every
        self.camera_jpeg_quality = args.camera_jpeg_quality
        self.camera_target_fps = args.camera_target_fps
        self.camera_max_width = args.camera_max_width
        self.camera_frame_count = 0
        self.camera_encoded_count = 0
        self.last_camera_emit_at = 0.0

        self.audio_enabled = args.enable_audio
        self.audio_emit_every = args.audio_emit_every
        self.audio_max_bytes = args.audio_max_bytes
        self.audio_frame_count = 0
        self.last_audio_error_at = 0.0
        self.audio_resampler = AudioResampler(format="s16", layout="mono", rate=48000)
        self.audio_pcm_buffer = bytearray()
        self.audio_buffered_frames = 0

        self.lidar_enabled = args.enable_lidar
        self.last_lidar_media_at = 0.0

        self.move_active = False
        self.pending_stop_deadline = 0.0
        self.last_move_command: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.last_move_sent_at = 0.0
        self.speed_profile = "normal"
        self.speed_level = 0
        self.last_speed_profile_reset_attempt = 0.0

        self.last_heartbeat_fault_event_at = 0.0

    def _reset_audio_pipeline(self) -> None:
        self.audio_resampler = AudioResampler(format="s16", layout="mono", rate=48000)
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
        self.conn.video.switchVideoChannel(enabled)

    async def _set_audio(self, enabled: bool) -> None:
        if self.conn is None:
            raise RuntimeError("Robot connection not ready")

        if enabled != self.audio_enabled:
            self._reset_audio_pipeline()

        self.audio_enabled = enabled
        self.conn.audio.switchAudioChannel(enabled)

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

            self.latest_camera_frame = frame
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

            encode_started = time.monotonic()
            encoded, width, height = await asyncio.to_thread(self._encode_camera_frame, frame)
            if encoded is None:
                continue
            encoded_at = time.time()
            self.last_camera_emit_at = time.monotonic()
            self.camera_encoded_count += 1

            await self._enqueue_media(
                {
                    "robot_id": self.args.robot_id,
                    "ts": encoded_at,
                    "stream": "video",
                    "data": {
                        "frame_index": self.camera_encoded_count,
                        "source_frame_index": self.camera_frame_count,
                        "width": width,
                        "height": height,
                        "image_format": "jpg",
                        "image_base64": base64.b64encode(encoded).decode("ascii"),
                        "encoded_ts": encoded_at,
                        "encode_ms": round((time.monotonic() - encode_started) * 1000.0, 2),
                        "target_fps": self.camera_target_fps,
                    },
                }
            )

    def _encode_camera_frame(self, frame) -> Tuple[Optional[bytes], int, int]:
        image = frame.to_ndarray(format="bgr24")
        height, width = image.shape[:2]
        if self.camera_max_width > 0 and width > self.camera_max_width:
            scale = self.camera_max_width / float(width)
            width = self.camera_max_width
            height = max(1, int(round(height * scale)))
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                self.camera_jpeg_quality,
                cv2.IMWRITE_JPEG_OPTIMIZE,
                0,
            ],
        )
        if not ok or encoded is None:
            return None, width, height
        return encoded.tobytes(), width, height

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
            return b"", 48000, 1

        chunks: List[bytes] = []
        for output_frame in output_frames:
            array = np.asarray(output_frame.to_ndarray())
            if array.size == 0:
                continue
            pcm = np.ascontiguousarray(array.reshape(-1), dtype="<i2")
            chunks.append(pcm.tobytes())

        return b"".join(chunks), 48000, 1

    async def _enqueue_media(self, payload: Dict[str, Any]) -> None:
        if not self.args.media_ws_url:
            return

        stream = str(payload.get("stream", "")).strip()

        if stream == "audio":
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

    async def _maybe_publish_lidar_media(self, payload: Any) -> None:
        now = time.monotonic()
        if now - self.last_lidar_media_at < (1.0 / max(self.args.lidar_media_hz, 0.01)):
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

        max_points = self.args.lidar_uplink_max_points
        if max_points > 0 and points.shape[0] > max_points:
            step = max(int(np.ceil(points.shape[0] / max_points)), 1)
            points = points[::step][:max_points]

        raw = np.ascontiguousarray(points, dtype="<f4").tobytes()
        compressed = await asyncio.to_thread(
            zlib.compress,
            raw,
            self.args.lidar_compression_level,
        )
        mins = points.min(axis=0)
        maxs = points.max(axis=0)

        await self._enqueue_media(
            {
                "robot_id": self.args.robot_id,
                "ts": time.time(),
                "stream": "lidar_points",
                "data": {
                    "point_format": "f32_xyz_zlib",
                    "points_base64": base64.b64encode(compressed).decode("ascii"),
                    "point_count": int(points.shape[0]),
                    "uncompressed_bytes": len(raw),
                    "bounds_min": mins.tolist(),
                    "bounds_max": maxs.tolist(),
                    "coordinate_frame": "map",
                },
            }
        )

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
                "lidar_enabled": self.lidar_enabled,
                "audio_enabled": self.audio_enabled,
                "camera_emit_every": self.camera_emit_every,
                "camera_target_fps": self.camera_target_fps,
                "camera_max_width": self.camera_max_width,
                "camera_encoded_frames": self.camera_encoded_count,
                "audio_emit_every": self.audio_emit_every,
                "audio_max_bytes": self.audio_max_bytes,
                "audio_queue_depth": self.media_queue.qsize(),
            },
            "temperatures": {
                "ntc1": low.get("temperature_ntc1"),
                "imu": sport.get("imu_temperature"),
                "motors": low.get("motor_temperatures"),
                "bms_bq_ntc": low.get("bms_bq_ntc"),
                "bms_mcu_ntc": low.get("bms_mcu_ntc"),
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
            "set_video",
            "set_camera_stream",
            "set_audio",
            "set_lidar",
            "set_lidar_decoder",
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
            cmd_type in {"move", "turn", "enter_mode", "follow_target", "go_to"}
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

            if "target_fps" in payload:
                target_fps = float(payload.get("target_fps", self.camera_target_fps))
                if target_fps < 1 or target_fps > 30:
                    raise ValueError("target_fps must be between 1 and 30")
                self.camera_target_fps = target_fps

            if "max_width" in payload:
                max_width = int(payload.get("max_width", self.camera_max_width))
                if max_width < 0 or (0 < max_width < 320):
                    raise ValueError("max_width must be 0 or >= 320")
                self.camera_max_width = max_width

            if "enabled" in payload:
                await self._set_video(bool(payload.get("enabled")))

            return {
                "executed": "set_camera_stream",
                "camera_enabled": self.camera_enabled,
                "emit_every": self.camera_emit_every,
                "jpeg_quality": self.camera_jpeg_quality,
                "target_fps": self.camera_target_fps,
                "max_width": self.camera_max_width,
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

                        video = self.latest_media_by_stream.pop("video", None)
                        if video is not None:
                            batch.append(video)

                        for _ in range(self.args.media_audio_batch_size):
                            try:
                                batch.append(self.media_queue.get_nowait())
                            except asyncio.QueueEmpty:
                                break

                        lidar_points = self.latest_media_by_stream.pop("lidar_points", None)
                        if lidar_points is not None:
                            batch.append(lidar_points)

                        for stream in list(self.latest_media_by_stream):
                            payload = self.latest_media_by_stream.pop(stream, None)
                            if payload is not None:
                                batch.append(payload)

                        if not self.media_queue.empty() or self.latest_media_by_stream:
                            self.media_ready.set()

                        for payload in batch:
                            text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
                            await asyncio.wait_for(ws.send(text), timeout=self.args.media_ws_send_timeout_s)

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

        while not self.stop_event.is_set():
            if self.conn is None:
                try:
                    await asyncio.wait_for(
                        self._connect_robot(),
                        timeout=self.args.robot_connect_timeout_s,
                    )
                    bad_since = 0.0
                except asyncio.CancelledError:
                    raise
                except (Exception, SystemExit) as exc:
                    await self._disconnect_robot()
                    self._publish_event("robot_connect_retry", {"error": str(exc)})
                    await asyncio.sleep(self.args.robot_reconnect_s)
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
    parser.add_argument("--camera-jpeg-quality", type=int, default=45)
    parser.add_argument("--camera-target-fps", type=float, default=15.0)
    parser.add_argument("--camera-max-width", type=int, default=960)

    parser.add_argument("--audio-emit-every", type=int, default=2)
    parser.add_argument("--audio-max-bytes", type=int, default=24576)

    parser.add_argument("--lidar-media-hz", type=float, default=5.0)
    parser.add_argument("--lidar-uplink-max-points", type=int, default=30000)
    parser.add_argument("--lidar-compression-level", type=int, default=1)

    parser.add_argument(
        "--media-ws-url",
        default="",
        help="Separate heavy-data uplink URL. Example: ws://server:8000/ws/edge-media/{robot_id}",
    )
    parser.add_argument("--media-ws-token", default="")
    parser.add_argument("--media-queue-size", type=int, default=64)
    parser.add_argument("--media-audio-batch-size", type=int, default=4)
    parser.add_argument("--media-ws-reconnect-s", type=float, default=2.0)
    parser.add_argument("--media-ws-send-timeout-s", type=float, default=2.0)
    parser.add_argument("--media-ws-max-size", type=int, default=8 * 1024 * 1024)

    parser.add_argument("--robot-connect-timeout-s", type=float, default=20.0)
    parser.add_argument("--robot-reconnect-s", type=float, default=2.0)
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

    if args.camera_target_fps < 1 or args.camera_target_fps > 30:
        parser.error("--camera-target-fps must be between 1 and 30")

    if args.camera_max_width < 0 or (0 < args.camera_max_width < 320):
        parser.error("--camera-max-width must be 0 or >= 320")

    if args.audio_emit_every <= 0:
        parser.error("--audio-emit-every must be > 0")

    if args.audio_max_bytes < 0:
        parser.error("--audio-max-bytes must be >= 0")

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

    if args.media_ws_send_timeout_s <= 0:
        parser.error("--media-ws-send-timeout-s must be > 0")

    if args.robot_connect_timeout_s <= 0:
        parser.error("--robot-connect-timeout-s must be > 0")

    if args.robot_reconnect_s <= 0:
        parser.error("--robot-reconnect-s must be > 0")

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

    if args.mqtt_port <= 0 or args.mqtt_port > 65535:
        parser.error("--mqtt-port must be between 1 and 65535")

    return args


def main() -> None:
    args = parse_args()
    app = EdgeGatewayService(args)

    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
