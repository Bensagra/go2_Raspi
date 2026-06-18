#!/usr/bin/env python3
import argparse
import asyncio
import base64
import contextlib
import json
import time
import uuid
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from aiortc import AudioStreamTrack
from av import AudioFrame
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


class BrowserMicrophoneTrack(AudioStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate: int = 48000, frame_samples: int = 960) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.frame_bytes = frame_samples * 2
        self.max_buffer_bytes = sample_rate * 2
        self.buffer = bytearray()
        self.enabled = False
        self._timestamp = 0
        self._started_at = 0.0
        self.received_packets = 0
        self.dropped_bytes = 0

    def feed(self, pcm_s16le: bytes) -> None:
        if not pcm_s16le:
            return

        aligned = pcm_s16le[: len(pcm_s16le) - (len(pcm_s16le) % 2)]
        if not aligned:
            return

        self.buffer.extend(aligned)
        self.received_packets += 1

        overflow = len(self.buffer) - self.max_buffer_bytes
        if overflow > 0:
            overflow -= overflow % 2
            del self.buffer[:overflow]
            self.dropped_bytes += overflow

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled:
            self.buffer.clear()

    async def recv(self) -> AudioFrame:
        if self._started_at <= 0:
            self._started_at = time.monotonic()
            self._timestamp = 0
        else:
            self._timestamp += self.frame_samples
            deadline = self._started_at + (self._timestamp / self.sample_rate)
            await asyncio.sleep(max(0.0, deadline - time.monotonic()))

        if self.enabled and len(self.buffer) >= self.frame_bytes:
            data = bytes(self.buffer[: self.frame_bytes])
            del self.buffer[: self.frame_bytes]
        else:
            data = bytes(self.frame_bytes)

        frame = AudioFrame(format="s16", layout="mono", samples=self.frame_samples)
        frame.planes[0].update(data)
        frame.sample_rate = self.sample_rate
        frame.pts = self._timestamp
        frame.time_base = Fraction(1, self.sample_rate)
        return frame


class EdgeGatewayService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.stop_event = asyncio.Event()

        self.conn: Optional[UnitreeWebRTCConnection] = None
        self.video_task: Optional[asyncio.Task[None]] = None
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
        self.camera_frame_count = 0

        self.audio_enabled = args.enable_audio
        self.audio_emit_every = args.audio_emit_every
        self.audio_max_bytes = args.audio_max_bytes
        self.audio_frame_count = 0
        self.last_audio_error_at = 0.0
        self.audio_resampler = AudioResampler(format="s16", layout="mono", rate=48000)
        self.audio_pcm_buffer = bytearray()
        self.audio_buffered_frames = 0
        self.talkback_track: Optional[BrowserMicrophoneTrack] = None
        self.talkback_sender = None
        self.talkback_enabled = False

        self.lidar_enabled = args.enable_lidar
        self.last_lidar_media_at = 0.0

        self.move_active = False
        self.pending_stop_deadline = 0.0
        self.last_move_command: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.last_move_sent_at = 0.0

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
            self.last_heartbeat_monotonic = time.monotonic()
            if isinstance(payload, dict):
                self.last_heartbeat_payload = payload
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
                        if previous is not None:
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

        self.video_task = asyncio.create_task(self._consume_video_track(track))
        self._publish_event("video_track_received", {})

    async def _consume_video_track(self, track) -> None:
        while not self.stop_event.is_set():
            frame = await track.recv()
            self.camera_frame_count += 1

            if not self.camera_enabled:
                continue

            if self.camera_frame_count % self.camera_emit_every != 0 and self.camera_frame_count != 1:
                continue

            encoded = await asyncio.to_thread(self._encode_camera_frame, frame)
            if encoded is None:
                continue

            await self._enqueue_media(
                {
                    "robot_id": self.args.robot_id,
                    "ts": time.time(),
                    "stream": "video",
                    "data": {
                        "frame_index": self.camera_frame_count,
                        "width": frame.width,
                        "height": frame.height,
                        "image_format": "jpg",
                        "image_base64": base64.b64encode(encoded).decode("ascii"),
                    },
                }
            )

    def _encode_camera_frame(self, frame) -> Optional[bytes]:
        image = frame.to_ndarray(format="bgr24")
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, self.camera_jpeg_quality],
        )
        if not ok or encoded is None:
            return None
        return encoded.tobytes()

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

    def _attach_talkback_track(self) -> None:
        if self.conn is None:
            raise RuntimeError("Robot connection not ready")

        if self.talkback_track is None:
            self.talkback_track = BrowserMicrophoneTrack()

        pc = getattr(self.conn, "pc", None)
        if pc is None:
            raise RuntimeError("Robot peer connection not ready")

        for transceiver in pc.getTransceivers():
            if getattr(transceiver, "kind", "") != "audio":
                continue
            sender = transceiver.sender
            sender.replaceTrack(self.talkback_track)
            self.talkback_sender = sender
            return

        self.talkback_sender = pc.addTrack(self.talkback_track)

    async def _set_talkback(self, enabled: bool) -> None:
        if enabled:
            self._attach_talkback_track()
            if self.conn is None:
                raise RuntimeError("Robot connection not ready")
            self.conn.audio.switchAudioChannel(True)

        if self.talkback_track is not None:
            self.talkback_track.set_enabled(enabled)

        self.talkback_enabled = enabled
        self._publish_event("talkback_changed", {"enabled": enabled})
        await self._enqueue_media(
            {
                "stream": "talkback_status",
                "data": {
                    "active": enabled,
                    "ok": True,
                    "error": "",
                },
                "ts": time.time(),
            }
        )

    def _feed_talkback_audio(self, data: Dict[str, Any]) -> None:
        if not self.talkback_enabled or self.talkback_track is None:
            return

        sample_rate = int(data.get("sample_rate", 0) or 0)
        channels = int(data.get("channels", 0) or 0)
        audio_format = str(data.get("audio_format", ""))
        encoded = data.get("audio_base64")

        if sample_rate != 48000 or channels != 1 or audio_format != "pcm_s16le":
            raise ValueError("Talkback requires mono pcm_s16le at 48000 Hz")
        if not isinstance(encoded, str) or not encoded:
            return

        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > self.args.talkback_max_packet_bytes:
            raise ValueError("Talkback packet too large")
        self.talkback_track.feed(raw)

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

    def _render_lidar_preview(self, points: np.ndarray) -> Optional[np.ndarray]:
        if points.size == 0:
            return None

        size = self.args.lidar_preview_size
        image = np.zeros((size, size, 3), dtype=np.uint8)

        xy = points[:, :2].astype(np.float32)
        finite = np.isfinite(xy).all(axis=1)
        xy = xy[finite]
        if xy.size == 0:
            return None

        if self.args.lidar_preview_max_points > 0 and xy.shape[0] > self.args.lidar_preview_max_points:
            step = max(xy.shape[0] // self.args.lidar_preview_max_points, 1)
            xy = xy[::step]

        max_abs = np.percentile(np.abs(xy), 98, axis=0)
        span = float(max(max_abs.max(), 0.5))
        scale = (size * 0.45) / span

        px = (xy[:, 0] * scale + size / 2.0).astype(np.int32)
        py = (size / 2.0 - xy[:, 1] * scale).astype(np.int32)

        valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
        image[py[valid], px[valid]] = (60, 230, 60)
        cv2.circle(image, (size // 2, size // 2), 3, (0, 0, 255), -1)

        return image

    async def _maybe_publish_lidar_media(self, payload: Any) -> None:
        now = time.monotonic()
        if now - self.last_lidar_media_at < (1.0 / max(self.args.lidar_media_hz, 0.01)):
            return
        self.last_lidar_media_at = now

        points = self._extract_lidar_points(payload)
        if points is None:
            return

        encoded = await asyncio.to_thread(self._encode_lidar_preview, points)
        if encoded is None:
            return

        await self._enqueue_media(
            {
                "robot_id": self.args.robot_id,
                "ts": time.time(),
                "stream": "lidar",
                "data": {
                    "image_format": "jpg",
                    "image_base64": base64.b64encode(encoded).decode("ascii"),
                    "point_count": int(points.shape[0]),
                },
            }
        )

    def _encode_lidar_preview(self, points: np.ndarray) -> Optional[bytes]:
        preview = self._render_lidar_preview(points)
        if preview is None:
            return None

        ok, encoded = cv2.imencode(
            ".jpg",
            preview,
            [cv2.IMWRITE_JPEG_QUALITY, self.args.lidar_preview_jpeg_quality],
        )
        if not ok or encoded is None:
            return None
        return encoded.tobytes()

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

        velocity_out = {"linear": None, "angular": None}
        if isinstance(velocity, dict):
            velocity_out["linear"] = velocity.get("x", velocity.get("linear"))
            velocity_out["angular"] = velocity.get("z", velocity.get("angular"))
        elif isinstance(velocity, (list, tuple)) and len(velocity) >= 3:
            velocity_out["linear"] = velocity[0]
            velocity_out["angular"] = velocity[2]

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
                "audio_emit_every": self.audio_emit_every,
                "audio_max_bytes": self.audio_max_bytes,
                "audio_queue_depth": self.media_queue.qsize(),
                "talkback_enabled": self.talkback_enabled,
                "talkback_buffer_bytes": (
                    len(self.talkback_track.buffer) if self.talkback_track is not None else 0
                ),
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
        }:
            return False, f"unsupported command type: {cmd_type}"

        now_s = time.time()
        issued_ts = float(command.get("ts", now_s))
        ttl_ms = int(command.get("ttl_ms", self.args.default_command_ttl_ms))
        if ttl_ms > 0 and now_s > issued_ts + (ttl_ms / 1000.0):
            return False, "command expired"

        heartbeat_age = time.monotonic() - self.last_heartbeat_monotonic
        requires_live_heartbeat = cmd_type in {"move", "turn", "enter_mode", "follow_target", "go_to"}
        if requires_live_heartbeat and heartbeat_age > self.args.heartbeat_timeout_s:
            return False, "no recent server heartbeat"

        return True, ""

    async def _execute_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        cmd_type = str(command.get("type", ""))
        payload = command.get("payload", {}) if isinstance(command.get("payload"), dict) else {}

        if cmd_type == "stop":
            await self._sport_request(int(SPORT_CMD["StopMove"]))
            self.move_active = False
            self.pending_stop_deadline = 0.0
            return {"executed": "stop"}

        if cmd_type == "move":
            x = clamp(float(payload.get("linear_x", 0.0)), -self.args.max_linear_speed, self.args.max_linear_speed)
            y = clamp(float(payload.get("lateral_y", 0.0)), -self.args.max_lateral_speed, self.args.max_lateral_speed)
            z = clamp(float(payload.get("angular_z", 0.0)), -self.args.max_angular_speed, self.args.max_angular_speed)
            duration_ms = int(payload.get("duration_ms", self.args.default_move_duration_ms))

            self.move_active = True
            self.last_move_command = {"x": x, "y": y, "z": z}
            self.last_move_sent_at = time.monotonic()
            if duration_ms > 0:
                self.pending_stop_deadline = time.monotonic() + (duration_ms / 1000.0)

            await self._sport_request(int(SPORT_CMD["Move"]), {"x": x, "y": y, "z": z})

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
            z = direction * clamp(abs(float(payload.get("angular_z", self.args.turn_angular_speed))), 0.05, self.args.max_angular_speed)
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

            if "enabled" in payload:
                await self._set_video(bool(payload.get("enabled")))

            return {
                "executed": "set_camera_stream",
                "camera_enabled": self.camera_enabled,
                "emit_every": self.camera_emit_every,
                "jpeg_quality": self.camera_jpeg_quality,
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

            valid, reason = self._validate_command(command)
            if not valid:
                self._send_command_ack(command, "rejected", reason)
                continue

            self._send_command_ack(command, "accepted")

            try:
                result = await self._execute_command(command)
            except Exception as exc:
                self._send_command_ack(command, "error", str(exc))
                self._publish_event("command_error", {"command_id": command_id, "error": str(exc)})
                continue

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

            if deadline_expired or heartbeat_expired:
                try:
                    await self._sport_request(int(SPORT_CMD["StopMove"]))
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

            await asyncio.sleep(0.05)

    async def _media_sender_loop(self, ws) -> None:
        while not self.stop_event.is_set():
            await self.media_ready.wait()
            self.media_ready.clear()

            batch: List[Dict[str, Any]] = []

            for _ in range(self.args.media_audio_batch_size):
                try:
                    batch.append(self.media_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            for stream in ("lidar", "video"):
                payload = self.latest_media_by_stream.pop(stream, None)
                if payload is not None:
                    batch.append(payload)

            for stream in list(self.latest_media_by_stream):
                payload = self.latest_media_by_stream.pop(stream, None)
                if payload is not None:
                    batch.append(payload)

            if not self.media_queue.empty() or self.latest_media_by_stream:
                self.media_ready.set()

            for payload in batch:
                text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
                await asyncio.wait_for(ws.send(text), timeout=self.args.media_ws_send_timeout_s)

    async def _media_receiver_loop(self, ws) -> None:
        async for text in ws:
            try:
                payload = json.loads(text)
            except (TypeError, ValueError) as exc:
                self._publish_event("media_downlink_invalid", {"error": str(exc)})
                continue

            op = str(payload.get("op", "")).strip()

            try:
                if op == "mic_start":
                    await self._set_talkback(True)
                    continue

                if op == "mic_stop":
                    await self._set_talkback(False)
                    continue

                if op == "mic_audio":
                    data = payload.get("data", {})
                    if isinstance(data, dict):
                        self._feed_talkback_audio(data)
            except Exception as exc:
                if op == "mic_start":
                    self.talkback_enabled = False
                    if self.talkback_track is not None:
                        self.talkback_track.set_enabled(False)

                self._publish_event(
                    "talkback_error",
                    {"operation": op, "error": str(exc)},
                )
                await self._enqueue_media(
                    {
                        "stream": "talkback_status",
                        "data": {
                            "active": False,
                            "ok": False,
                            "error": str(exc),
                        },
                        "ts": time.time(),
                    }
                )

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

                    sender_task = asyncio.create_task(self._media_sender_loop(ws))
                    receiver_task = asyncio.create_task(self._media_receiver_loop(ws))
                    done, pending = await asyncio.wait(
                        {sender_task, receiver_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in pending:
                        task.cancel()
                    for task in pending:
                        with contextlib.suppress(asyncio.CancelledError):
                            await task

                    for task in done:
                        task.result()

            except Exception as exc:
                with contextlib.suppress(Exception):
                    await self._set_talkback(False)
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

        self.video_task = None

        connection = self.conn
        self.conn = None
        self.robot_connected_at = 0.0
        self.traffic_saving_disabled = False
        self.subscribed_topics.clear()
        self.latest_by_topic.clear()
        self._reset_audio_pipeline()
        self.talkback_enabled = False
        if self.talkback_track is not None:
            self.talkback_track.set_enabled(False)
            self.talkback_track.stop()
        self.talkback_track = None
        self.talkback_sender = None

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

    parser.add_argument("--max-linear-speed", type=float, default=0.45)
    parser.add_argument("--max-lateral-speed", type=float, default=0.25)
    parser.add_argument("--max-angular-speed", type=float, default=1.0)
    parser.add_argument("--default-move-duration-ms", type=int, default=700)
    parser.add_argument("--turn-angular-speed", type=float, default=0.8)
    parser.add_argument("--turn-ms-per-degree", type=float, default=15.0)

    parser.add_argument("--camera-emit-every", type=int, default=3)
    parser.add_argument("--camera-jpeg-quality", type=int, default=55)

    parser.add_argument("--audio-emit-every", type=int, default=2)
    parser.add_argument("--audio-max-bytes", type=int, default=24576)

    parser.add_argument("--lidar-media-hz", type=float, default=2.0)
    parser.add_argument("--lidar-preview-size", type=int, default=420)
    parser.add_argument("--lidar-preview-max-points", type=int, default=15000)
    parser.add_argument("--lidar-preview-jpeg-quality", type=int, default=60)

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
    parser.add_argument("--talkback-max-packet-bytes", type=int, default=32768)

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

    if args.media_queue_size <= 0:
        parser.error("--media-queue-size must be > 0")

    if args.media_audio_batch_size <= 0:
        parser.error("--media-audio-batch-size must be > 0")

    if args.media_ws_send_timeout_s <= 0:
        parser.error("--media-ws-send-timeout-s must be > 0")

    if args.talkback_max_packet_bytes <= 0:
        parser.error("--talkback-max-packet-bytes must be > 0")

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

    if args.lidar_preview_size < 200:
        parser.error("--lidar-preview-size must be >= 200")

    if args.lidar_preview_jpeg_quality < 1 or args.lidar_preview_jpeg_quality > 100:
        parser.error("--lidar-preview-jpeg-quality must be between 1 and 100")

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
