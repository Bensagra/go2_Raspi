#!/usr/bin/env python3
import argparse
import asyncio
import base64
import contextlib
import json
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

try:
    from unitree_webrtc_connect import (
        AUDIO_API,
        DATA_CHANNEL_TYPE,
        RTC_TOPIC,
        SPORT_CMD,
        UnitreeWebRTCConnection,
        WebRTCConnectionMethod,
    )
except ImportError:
    from unitree_webrtc_connect.constants import (
        AUDIO_API,
        DATA_CHANNEL_TYPE,
        RTC_TOPIC,
        SPORT_CMD,
        WebRTCConnectionMethod,
    )
    from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection


DEFAULT_GO2_IP = "192.168.123.161"

TOPIC_ALIAS_TO_VALUE = dict(RTC_TOPIC)
TOPIC_VALUE_TO_ALIAS = {value: key for key, value in TOPIC_ALIAS_TO_VALUE.items()}

MSG_ALIAS_TO_VALUE = dict(DATA_CHANNEL_TYPE)
MSG_VALUES = set(MSG_ALIAS_TO_VALUE.values())

PROFILE_TOPICS = {
    "core": [
        "LOW_STATE",
        "MULTIPLE_STATE",
        "LF_SPORT_MOD_STATE",
        "SPORT_MOD_STATE",
    ],
    "all_telemetry": [
        "LOW_STATE",
        "MULTIPLE_STATE",
        "SPORT_MOD_STATE",
        "LF_SPORT_MOD_STATE",
        "ULIDAR",
        "ULIDAR_ARRAY",
        "ULIDAR_STATE",
        "ROBOTODOM",
        "UWB_STATE",
        "SELF_TEST",
        "GRID_MAP",
        "SERVICE_STATE",
        "GPT_FEEDBACK",
        "SLAM_QT_NOTICE",
        "SLAM_PC_TO_IMAGE_LOCAL",
        "SLAM_ODOMETRY",
        "ARM_FEEDBACK",
        "AUDIO_HUB_PLAY_STATE",
        "GAS_SENSOR",
        "LIDAR_MAPPING_CLOUD_POINT",
        "LIDAR_MAPPING_ODOM",
        "LIDAR_MAPPING_PCD_FILE",
        "LIDAR_MAPPING_SERVER_LOG",
        "LIDAR_LOCALIZATION_ODOM",
        "LIDAR_NAVIGATION_GLOBAL_PATH",
        "LIDAR_LOCALIZATION_CLOUD_POINT",
    ],
    "lidar": ["ULIDAR_ARRAY", "ULIDAR_STATE", "ROBOTODOM"],
    "audio": ["AUDIO_HUB_PLAY_STATE"],
    "navigation": [
        "SLAM_ODOMETRY",
        "LIDAR_MAPPING_ODOM",
        "LIDAR_LOCALIZATION_ODOM",
        "LIDAR_NAVIGATION_GLOBAL_PATH",
    ],
}


class Go2SshGateway:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.conn: Optional[UnitreeWebRTCConnection] = None

        self.stop_event = asyncio.Event()
        self.command_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self.stdout_lock = asyncio.Lock()

        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.stdin_thread: Optional[threading.Thread] = None

        self.subscribed_topics: set[str] = set()
        self.latest_by_topic: Dict[str, Any] = {}

        self.camera_enabled = args.enable_camera
        self.camera_emit_every = args.camera_emit_every
        self.camera_format = args.camera_format
        self.camera_jpeg_quality = args.camera_jpeg_quality
        self.camera_png_compression = args.camera_png_compression
        self.camera_frame_count = 0

        self.lidar_enabled = args.enable_lidar
        self.audio_enabled = args.enable_audio
        self.audio_emit_every = args.audio_emit_every
        self.audio_max_bytes = args.audio_max_bytes
        self.audio_frame_count = 0
        self.last_audio_error_at = 0.0

        self.video_task: Optional[asyncio.Task[None]] = None

        self.data_out = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout

    async def emit(self, message: Dict[str, Any]) -> None:
        line = json.dumps(message, ensure_ascii=True, separators=(",", ":"))
        async with self.stdout_lock:
            self.data_out.write(line + "\n")
            self.data_out.flush()

    def _sanitize(self, value: Any, depth: int = 0) -> Any:
        if depth > self.args.max_depth:
            return "<max_depth>"

        if value is None or isinstance(value, (bool, int, float, str)):
            return value

        if isinstance(value, bytes):
            return self._sanitize_bytes(value)

        if isinstance(value, dict):
            output: Dict[str, Any] = {}
            for key, item in value.items():
                output[str(key)] = self._sanitize(item, depth + 1)
            return output

        if isinstance(value, (list, tuple)):
            max_items = self.args.max_list_items
            if max_items > 0 and len(value) > max_items:
                return {
                    "__truncated__": True,
                    "total_items": len(value),
                    "items": [self._sanitize(item, depth + 1) for item in value[:max_items]],
                }
            return [self._sanitize(item, depth + 1) for item in value]

        to_list = getattr(value, "tolist", None)
        if callable(to_list):
            with contextlib.suppress(Exception):
                return self._sanitize(value.tolist(), depth + 1)

        if callable(value):
            return "<callable>"

        return repr(value)

    def _sanitize_bytes(self, data: bytes) -> Dict[str, Any]:
        max_bytes = self.args.max_bytes
        if max_bytes > 0 and len(data) > max_bytes:
            encoded = base64.b64encode(data[:max_bytes]).decode("ascii")
            return {
                "__type__": "bytes_base64",
                "truncated": True,
                "total_bytes": len(data),
                "encoded": encoded,
            }

        encoded = base64.b64encode(data).decode("ascii")
        return {
            "__type__": "bytes_base64",
            "truncated": False,
            "total_bytes": len(data),
            "encoded": encoded,
        }

    def _resolve_topic(self, topic: str) -> str:
        if topic in TOPIC_ALIAS_TO_VALUE:
            return TOPIC_ALIAS_TO_VALUE[topic]
        if topic in TOPIC_VALUE_TO_ALIAS:
            return topic
        raise ValueError(f"Unknown topic: {topic}")

    def _resolve_msg_type(self, msg_type: str) -> str:
        if msg_type in MSG_ALIAS_TO_VALUE:
            return MSG_ALIAS_TO_VALUE[msg_type]
        if msg_type in MSG_VALUES:
            return msg_type
        raise ValueError(f"Unknown msg_type: {msg_type}")

    def _topic_alias(self, topic_value: str) -> str:
        return TOPIC_VALUE_TO_ALIAS.get(topic_value, topic_value)

    def _schedule_coro(self, coro) -> None:
        if self.loop is None:
            return
        self.loop.call_soon_threadsafe(asyncio.create_task, coro)

    def _stdin_reader(self) -> None:
        try:
            for line in sys.stdin:
                if self.loop is not None:
                    self.loop.call_soon_threadsafe(self.command_queue.put_nowait, line)
        finally:
            if self.loop is not None:
                self.loop.call_soon_threadsafe(self.command_queue.put_nowait, None)

    async def _on_topic_message(self, topic_value: str, message: Dict[str, Any]) -> None:
        self.latest_by_topic[topic_value] = message

        event = {
            "type": "event",
            "stream": "topic",
            "topic": topic_value,
            "topic_alias": self._topic_alias(topic_value),
            "ts": time.time(),
            "data": self._sanitize(message),
        }
        await self.emit(event)

    async def _subscribe_topic(self, topic_value: str) -> None:
        if self.conn is None:
            raise RuntimeError("Connection not ready")

        if topic_value in self.subscribed_topics:
            return

        def callback(message, t=topic_value):
            self._schedule_coro(self._on_topic_message(t, message))

        self.conn.datachannel.pub_sub.subscribe(topic_value, callback)
        self.subscribed_topics.add(topic_value)

    async def _unsubscribe_topic(self, topic_value: str) -> None:
        if self.conn is None:
            raise RuntimeError("Connection not ready")

        if topic_value not in self.subscribed_topics:
            return

        self.conn.datachannel.pub_sub.unsubscribe(topic_value)
        self.subscribed_topics.discard(topic_value)

    async def _on_video_track(self, track) -> None:
        if self.video_task and not self.video_task.done():
            self.video_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.video_task

        self.video_task = asyncio.create_task(self._consume_video_track(track))
        await self.emit({
            "type": "status",
            "event": "video_track_received",
            "ts": time.time(),
        })

    def _encode_camera_image(self, image):
        if self.camera_format == "jpg":
            return cv2.imencode(
                ".jpg",
                image,
                [cv2.IMWRITE_JPEG_QUALITY, self.camera_jpeg_quality],
            )

        return cv2.imencode(
            ".png",
            image,
            [cv2.IMWRITE_PNG_COMPRESSION, self.camera_png_compression],
        )

    async def _consume_video_track(self, track) -> None:
        while not self.stop_event.is_set():
            frame = await track.recv()
            self.camera_frame_count += 1

            if not self.camera_enabled:
                continue

            if self.camera_frame_count % self.camera_emit_every != 0 and self.camera_frame_count != 1:
                continue

            image = frame.to_ndarray(format="bgr24")
            ok, encoded = self._encode_camera_image(image)
            if not ok or encoded is None:
                await self.emit(
                    {
                        "type": "error",
                        "source": "camera",
                        "message": "Failed to encode camera frame",
                        "ts": time.time(),
                    }
                )
                continue

            frame_format = frame.format.name if frame.format else "unknown"
            event = {
                "type": "event",
                "stream": "camera",
                "ts": time.time(),
                "data": {
                    "frame_index": self.camera_frame_count,
                    "pts": frame.pts,
                    "time_base": str(frame.time_base),
                    "width": frame.width,
                    "height": frame.height,
                    "format": frame_format,
                    "dtype": str(image.dtype),
                    "shape": list(image.shape),
                    "image_format": self.camera_format,
                    "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
                },
            }
            await self.emit(event)

    def _audio_frame_to_pcm_payload(self, frame) -> Dict[str, Any]:
        array = frame.to_ndarray()
        if array is None:
            raise ValueError("Audio frame payload is empty")

        if not isinstance(array, np.ndarray):
            array = np.asarray(array)

        if array.ndim == 0:
            array = np.expand_dims(array, axis=0)

        if array.ndim == 1:
            array = np.expand_dims(array, axis=1)

        if array.ndim > 2:
            array = array.reshape(array.shape[0], -1)

        # aiortc often yields planar audio in (channels, samples) shape.
        if array.ndim == 2 and array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
            array = array.T

        if np.issubdtype(array.dtype, np.floating):
            array = (np.clip(array, -1.0, 1.0) * 32767.0).astype(np.int16)
        else:
            array = array.astype(np.int16, copy=False)

        array = np.ascontiguousarray(array)

        channels = int(array.shape[1]) if array.ndim == 2 else 1
        samples_per_channel = int(array.shape[0])

        pcm_bytes = array.tobytes()
        total_bytes = len(pcm_bytes)

        truncated = False
        if self.audio_max_bytes > 0 and total_bytes > self.audio_max_bytes:
            truncated = True
            pcm_bytes = pcm_bytes[: self.audio_max_bytes]

        bytes_per_sample = 2
        bytes_per_frame = max(bytes_per_sample * channels, bytes_per_sample)
        aligned_len = len(pcm_bytes) - (len(pcm_bytes) % bytes_per_frame)
        pcm_bytes = pcm_bytes[:aligned_len]

        if not pcm_bytes:
            raise ValueError("Audio payload too small after alignment")

        return {
            "pcm_bytes": pcm_bytes,
            "channels": channels,
            "samples_per_channel": samples_per_channel,
            "sample_rate": int(getattr(frame, "sample_rate", 0) or 0),
            "layout": getattr(getattr(frame, "layout", None), "name", "unknown"),
            "frame_format": getattr(getattr(frame, "format", None), "name", "unknown"),
            "total_bytes": total_bytes,
            "truncated": truncated,
        }

    async def _on_audio_frame(self, frame) -> None:
        self.audio_frame_count += 1

        if not self.audio_enabled:
            return

        if self.audio_frame_count % self.audio_emit_every != 0 and self.audio_frame_count != 1:
            return

        try:
            payload = self._audio_frame_to_pcm_payload(frame)
        except Exception as exc:
            now = time.time()
            if now - self.last_audio_error_at > 2.0:
                self.last_audio_error_at = now
                await self.emit(
                    {
                        "type": "error",
                        "source": "audio",
                        "message": f"Failed to encode audio frame: {exc}",
                        "ts": now,
                    }
                )
            return

        event = {
            "type": "event",
            "stream": "audio",
            "ts": time.time(),
            "data": {
                "frame_index": self.audio_frame_count,
                "pts": frame.pts,
                "time_base": str(frame.time_base),
                "audio_format": "pcm_s16le",
                "sample_rate": payload["sample_rate"],
                "channels": payload["channels"],
                "samples_per_channel": payload["samples_per_channel"],
                "layout": payload["layout"],
                "frame_format": payload["frame_format"],
                "total_bytes": payload["total_bytes"],
                "truncated": payload["truncated"],
                "audio_base64": base64.b64encode(payload["pcm_bytes"]).decode("ascii"),
            },
        }
        await self.emit(event)

    async def _set_video(self, enabled: bool) -> None:
        if self.conn is None:
            raise RuntimeError("Connection not ready")
        self.camera_enabled = enabled
        self.conn.video.switchVideoChannel(enabled)

    async def _set_audio(self, enabled: bool) -> None:
        if self.conn is None:
            raise RuntimeError("Connection not ready")
        self.audio_enabled = enabled
        self.conn.audio.switchAudioChannel(enabled)

    async def _set_lidar(self, enabled: bool) -> None:
        if self.conn is None:
            raise RuntimeError("Connection not ready")

        self.lidar_enabled = enabled
        self.conn.datachannel.pub_sub.publish_without_callback(
            TOPIC_ALIAS_TO_VALUE["ULIDAR_SWITCH"],
            "on" if enabled else "off",
        )

    async def _sport_request(self, api_id: int, parameter: Optional[Any] = None) -> Any:
        if self.conn is None:
            raise RuntimeError("Connection not ready")

        options: Dict[str, Any] = {"api_id": api_id}
        if parameter is not None:
            options["parameter"] = parameter

        response = await self.conn.datachannel.pub_sub.publish_request_new(
            TOPIC_ALIAS_TO_VALUE["SPORT_MOD"],
            options,
        )
        return response

    async def _audiohub_request(self, api_id: int, parameter: Optional[Any] = None) -> Any:
        if self.conn is None:
            raise RuntimeError("Connection not ready")

        if parameter is None:
            parameter = {}

        options: Dict[str, Any] = {"api_id": api_id}
        if isinstance(parameter, str):
            options["parameter"] = parameter
        else:
            options["parameter"] = json.dumps(parameter, ensure_ascii=True)

        response = await self.conn.datachannel.pub_sub.publish_request_new(
            TOPIC_ALIAS_TO_VALUE["AUDIO_HUB_REQ"],
            options,
        )
        return response

    async def _handle_audiohub_action(self, command: Dict[str, Any]) -> Any:
        action = str(command.get("action", "")).strip().lower()
        if not action:
            raise ValueError("audiohub requires 'action'")

        if action in {"list", "get_audio_list"}:
            return await self._audiohub_request(int(AUDIO_API["GET_AUDIO_LIST"]), {})

        if action in {"play", "select_start_play"}:
            unique_id = command.get("unique_id", command.get("uuid"))
            if not unique_id:
                raise ValueError("audiohub play requires 'unique_id' or 'uuid'")
            return await self._audiohub_request(
                int(AUDIO_API["SELECT_START_PLAY"]),
                {"unique_id": str(unique_id)},
            )

        if action in {"pause"}:
            return await self._audiohub_request(int(AUDIO_API["PAUSE"]), {})

        if action in {"resume", "unsuspend"}:
            return await self._audiohub_request(int(AUDIO_API["UNSUSPEND"]), {})

        if action in {"prev", "previous"}:
            return await self._audiohub_request(int(AUDIO_API["SELECT_PREV_START_PLAY"]), {})

        if action in {"next"}:
            return await self._audiohub_request(int(AUDIO_API["SELECT_NEXT_START_PLAY"]), {})

        if action in {"set_mode", "set_play_mode"}:
            play_mode = command.get("play_mode", command.get("mode"))
            if not isinstance(play_mode, str) or not play_mode:
                raise ValueError("audiohub set_mode requires 'play_mode' (single_cycle|no_cycle|list_loop)")
            return await self._audiohub_request(int(AUDIO_API["SET_PLAY_MODE"]), {"play_mode": play_mode})

        if action in {"get_mode", "get_play_mode"}:
            return await self._audiohub_request(int(AUDIO_API["GET_PLAY_MODE"]), {})

        if action in {"rename", "select_rename"}:
            unique_id = command.get("unique_id", command.get("uuid"))
            new_name = command.get("new_name", command.get("name"))
            if not unique_id or not isinstance(new_name, str) or not new_name:
                raise ValueError("audiohub rename requires 'unique_id' and 'new_name'")
            return await self._audiohub_request(
                int(AUDIO_API["SELECT_RENAME"]),
                {"unique_id": str(unique_id), "new_name": new_name},
            )

        if action in {"delete", "select_delete"}:
            unique_id = command.get("unique_id", command.get("uuid"))
            if not unique_id:
                raise ValueError("audiohub delete requires 'unique_id' or 'uuid'")
            return await self._audiohub_request(int(AUDIO_API["SELECT_DELETE"]), {"unique_id": str(unique_id)})

        if action in {"enter_megaphone"}:
            return await self._audiohub_request(int(AUDIO_API["ENTER_MEGAPHONE"]), {})

        if action in {"exit_megaphone"}:
            return await self._audiohub_request(int(AUDIO_API["EXIT_MEGAPHONE"]), {})

        if action in {"raw", "request"}:
            api_id = int(command["api_id"])
            parameter = command.get("parameter", {})
            return await self._audiohub_request(api_id, parameter)

        raise ValueError(f"Unknown audiohub action: {action}")

    def _status_snapshot(self) -> Dict[str, Any]:
        return {
            "connected": bool(self.conn and self.conn.isConnected),
            "ip": self.args.ip,
            "subscribed_topics": sorted(self.subscribed_topics),
            "latest_topics": sorted(self.latest_by_topic.keys()),
            "camera": {
                "enabled": self.camera_enabled,
                "emit_every": self.camera_emit_every,
                "format": self.camera_format,
                "jpeg_quality": self.camera_jpeg_quality,
                "png_compression": self.camera_png_compression,
                "frame_count": self.camera_frame_count,
            },
            "lidar": {
                "enabled": self.lidar_enabled,
                "decoder": self.args.lidar_decoder,
            },
            "audio": {
                "enabled": self.audio_enabled,
                "emit_every": self.audio_emit_every,
                "max_bytes": self.audio_max_bytes,
                "frame_count": self.audio_frame_count,
            },
        }

    def _extract_temperatures(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        low_state_topic = TOPIC_ALIAS_TO_VALUE["LOW_STATE"]
        low_state_msg = self.latest_by_topic.get(low_state_topic, {})
        low_state_data = low_state_msg.get("data", {}) if isinstance(low_state_msg, dict) else {}

        motor_state = low_state_data.get("motor_state", []) if isinstance(low_state_data, dict) else []
        motor_temps: List[Any] = []
        if isinstance(motor_state, list):
            for motor in motor_state:
                if isinstance(motor, dict):
                    motor_temps.append(motor.get("temperature"))

        bms_state = low_state_data.get("bms_state", {}) if isinstance(low_state_data, dict) else {}
        out["low_state"] = {
            "temperature_ntc1": low_state_data.get("temperature_ntc1") if isinstance(low_state_data, dict) else None,
            "power_v": low_state_data.get("power_v") if isinstance(low_state_data, dict) else None,
            "motor_temperatures": motor_temps,
            "bms_bq_ntc": bms_state.get("bq_ntc") if isinstance(bms_state, dict) else None,
            "bms_mcu_ntc": bms_state.get("mcu_ntc") if isinstance(bms_state, dict) else None,
        }

        sport_topic = TOPIC_ALIAS_TO_VALUE["LF_SPORT_MOD_STATE"]
        sport_msg = self.latest_by_topic.get(sport_topic, {})
        sport_data = sport_msg.get("data", {}) if isinstance(sport_msg, dict) else {}
        imu_state = sport_data.get("imu_state", {}) if isinstance(sport_data, dict) else {}
        out["lf_sport_mod_state"] = {
            "imu_temperature": imu_state.get("temperature") if isinstance(imu_state, dict) else None,
        }

        gas_topic = TOPIC_ALIAS_TO_VALUE.get("GAS_SENSOR")
        gas_msg = self.latest_by_topic.get(gas_topic, {}) if gas_topic else {}
        gas_data = gas_msg.get("data") if isinstance(gas_msg, dict) else None
        out["gas_sensor"] = gas_data

        return out

    async def _send_response(self, cmd_id: Any, ok: bool, result: Any = None, error: str = "") -> None:
        message = {
            "type": "response",
            "id": cmd_id,
            "ok": ok,
            "ts": time.time(),
        }
        if ok:
            message["result"] = self._sanitize(result)
        else:
            message["error"] = error
        await self.emit(message)

    async def _handle_command(self, command: Dict[str, Any]) -> None:
        cmd_id = command.get("id")
        op = command.get("op")

        if not op:
            await self._send_response(cmd_id, False, error="Missing 'op' in command")
            return

        try:
            if op == "ping":
                await self._send_response(cmd_id, True, {"pong": time.time()})
                return

            if op == "help":
                await self._send_response(
                    cmd_id,
                    True,
                    {
                        "ops": [
                            "ping",
                            "status",
                            "list_topics",
                            "list_sport_cmd",
                            "get_latest",
                            "get_temperatures",
                            "subscribe",
                            "unsubscribe",
                            "subscribe_profile",
                            "request",
                            "publish",
                            "publish_no_ack",
                            "sport",
                            "set_motion_mode",
                            "get_motion_mode",
                            "set_video",
                            "set_camera_stream",
                            "set_audio",
                            "audiohub",
                            "set_lidar",
                            "set_lidar_decoder",
                            "exit",
                        ]
                    },
                )
                return

            if op == "status":
                await self._send_response(cmd_id, True, self._status_snapshot())
                return

            if op == "list_topics":
                await self._send_response(cmd_id, True, TOPIC_ALIAS_TO_VALUE)
                return

            if op == "list_sport_cmd":
                await self._send_response(cmd_id, True, SPORT_CMD)
                return

            if op == "get_latest":
                if "topic" in command:
                    topic_value = self._resolve_topic(str(command["topic"]))
                    payload = self.latest_by_topic.get(topic_value)
                    await self._send_response(
                        cmd_id,
                        True,
                        {
                            "topic": topic_value,
                            "topic_alias": self._topic_alias(topic_value),
                            "data": payload,
                        },
                    )
                else:
                    await self._send_response(cmd_id, True, self.latest_by_topic)
                return

            if op == "get_temperatures":
                await self._send_response(cmd_id, True, self._extract_temperatures())
                return

            if op == "subscribe":
                topic_value = self._resolve_topic(str(command["topic"]))
                await self._subscribe_topic(topic_value)
                await self._send_response(cmd_id, True, {"subscribed": topic_value})
                return

            if op == "unsubscribe":
                topic_value = self._resolve_topic(str(command["topic"]))
                await self._unsubscribe_topic(topic_value)
                await self._send_response(cmd_id, True, {"unsubscribed": topic_value})
                return

            if op == "subscribe_profile":
                profile = str(command.get("profile", "core"))
                aliases = PROFILE_TOPICS.get(profile)
                if aliases is None:
                    raise ValueError(f"Unknown profile: {profile}")
                subscribed: List[str] = []
                for alias in aliases:
                    if alias in TOPIC_ALIAS_TO_VALUE:
                        topic_value = TOPIC_ALIAS_TO_VALUE[alias]
                        await self._subscribe_topic(topic_value)
                        subscribed.append(topic_value)
                await self._send_response(cmd_id, True, {"profile": profile, "subscribed": subscribed})
                return

            if op == "request":
                if self.conn is None:
                    raise RuntimeError("Connection not ready")
                topic_value = self._resolve_topic(str(command["topic"]))
                api_id = int(command["api_id"])
                parameter = command.get("parameter")
                options: Dict[str, Any] = {"api_id": api_id}
                if parameter is not None:
                    options["parameter"] = parameter
                response = await self.conn.datachannel.pub_sub.publish_request_new(topic_value, options)
                await self._send_response(cmd_id, True, response)
                return

            if op == "publish":
                if self.conn is None:
                    raise RuntimeError("Connection not ready")
                topic_value = self._resolve_topic(str(command["topic"]))
                msg_type = self._resolve_msg_type(str(command.get("msg_type", "MSG")))
                data = command.get("data")
                response = await self.conn.datachannel.pub_sub.publish(topic_value, data, msg_type)
                await self._send_response(cmd_id, True, response)
                return

            if op == "publish_no_ack":
                if self.conn is None:
                    raise RuntimeError("Connection not ready")
                topic_value = self._resolve_topic(str(command["topic"]))
                msg_type = self._resolve_msg_type(str(command.get("msg_type", "MSG")))
                data = command.get("data")
                self.conn.datachannel.pub_sub.publish_without_callback(topic_value, data, msg_type)
                await self._send_response(cmd_id, True, {"published": topic_value})
                return

            if op == "sport":
                action = command.get("action")
                if action is None:
                    raise ValueError("sport requires 'action'")

                if isinstance(action, str):
                    if action not in SPORT_CMD:
                        raise ValueError(f"Unknown SPORT_CMD action: {action}")
                    api_id = int(SPORT_CMD[action])
                else:
                    api_id = int(action)

                parameter = command.get("parameter")
                response = await self._sport_request(api_id, parameter)
                await self._send_response(cmd_id, True, response)
                return

            if op == "set_motion_mode":
                if self.conn is None:
                    raise RuntimeError("Connection not ready")
                name = str(command["name"])
                response = await self.conn.datachannel.pub_sub.publish_request_new(
                    TOPIC_ALIAS_TO_VALUE["MOTION_SWITCHER"],
                    {
                        "api_id": 1002,
                        "parameter": {"name": name},
                    },
                )
                await self._send_response(cmd_id, True, response)
                return

            if op == "get_motion_mode":
                if self.conn is None:
                    raise RuntimeError("Connection not ready")
                response = await self.conn.datachannel.pub_sub.publish_request_new(
                    TOPIC_ALIAS_TO_VALUE["MOTION_SWITCHER"],
                    {"api_id": 1001},
                )
                await self._send_response(cmd_id, True, response)
                return

            if op == "set_video":
                enabled = bool(command.get("enabled", True))
                await self._set_video(enabled)
                await self._send_response(cmd_id, True, {"camera_enabled": self.camera_enabled})
                return

            if op == "set_camera_stream":
                if "emit_every" in command:
                    new_emit_every = int(command["emit_every"])
                    if new_emit_every <= 0:
                        raise ValueError("emit_every must be > 0")
                    self.camera_emit_every = new_emit_every

                if "format" in command:
                    fmt = str(command["format"]).lower()
                    if fmt not in {"jpg", "png"}:
                        raise ValueError("format must be 'jpg' or 'png'")
                    self.camera_format = fmt

                if "jpeg_quality" in command:
                    quality = int(command["jpeg_quality"])
                    if quality < 1 or quality > 100:
                        raise ValueError("jpeg_quality must be between 1 and 100")
                    self.camera_jpeg_quality = quality

                if "png_compression" in command:
                    compression = int(command["png_compression"])
                    if compression < 0 or compression > 9:
                        raise ValueError("png_compression must be between 0 and 9")
                    self.camera_png_compression = compression

                if "enabled" in command:
                    await self._set_video(bool(command["enabled"]))

                await self._send_response(cmd_id, True, self._status_snapshot()["camera"])
                return

            if op == "set_lidar":
                enabled = bool(command.get("enabled", True))
                await self._set_lidar(enabled)

                if enabled and bool(command.get("subscribe", True)):
                    await self._subscribe_topic(TOPIC_ALIAS_TO_VALUE["ULIDAR_ARRAY"])
                    await self._subscribe_topic(TOPIC_ALIAS_TO_VALUE["ULIDAR_STATE"])
                await self._send_response(cmd_id, True, {"lidar_enabled": self.lidar_enabled})
                return

            if op == "set_audio":
                if "emit_every" in command:
                    new_emit_every = int(command["emit_every"])
                    if new_emit_every <= 0:
                        raise ValueError("emit_every must be > 0")
                    self.audio_emit_every = new_emit_every

                if "max_bytes" in command:
                    max_bytes = int(command["max_bytes"])
                    if max_bytes < 0:
                        raise ValueError("max_bytes must be >= 0")
                    self.audio_max_bytes = max_bytes

                if "enabled" in command:
                    await self._set_audio(bool(command["enabled"]))

                await self._send_response(cmd_id, True, self._status_snapshot()["audio"])
                return

            if op == "audiohub":
                result = await self._handle_audiohub_action(command)
                await self._send_response(cmd_id, True, result)
                return

            if op == "set_lidar_decoder":
                if self.conn is None:
                    raise RuntimeError("Connection not ready")
                decoder = str(command.get("decoder", "libvoxel"))
                self.conn.datachannel.set_decoder(decoder)
                await self._send_response(cmd_id, True, {"decoder": decoder})
                return

            if op in {"exit", "quit", "stop"}:
                await self._send_response(cmd_id, True, {"stopping": True})
                self.stop_event.set()
                return

            await self._send_response(cmd_id, False, error=f"Unknown op: {op}")

        except Exception as exc:
            await self._send_response(cmd_id, False, error=str(exc))

    async def _command_loop(self) -> None:
        while not self.stop_event.is_set():
            line = await self.command_queue.get()
            if line is None:
                await self.emit(
                    {
                        "type": "status",
                        "event": "stdin_eof",
                        "ts": time.time(),
                    }
                )
                if self.args.exit_on_stdin_eof:
                    self.stop_event.set()
                return

            raw = line.strip()
            if not raw:
                continue

            try:
                command = json.loads(raw)
            except json.JSONDecodeError as exc:
                await self.emit(
                    {
                        "type": "error",
                        "source": "stdin",
                        "message": f"Invalid JSON command: {exc}",
                        "raw": raw,
                        "ts": time.time(),
                    }
                )
                continue

            if not isinstance(command, dict):
                await self.emit(
                    {
                        "type": "error",
                        "source": "stdin",
                        "message": "Command must be a JSON object",
                        "ts": time.time(),
                    }
                )
                continue

            await self._handle_command(command)

    async def _connect(self) -> None:
        self.conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.args.ip)
        await self.conn.connect()

        self.conn.video.add_track_callback(self._on_video_track)
        self.conn.audio.add_track_callback(self._on_audio_frame)

        if self.args.disable_traffic_saving:
            with contextlib.suppress(Exception):
                await self.conn.datachannel.disableTrafficSaving(True)

        with contextlib.suppress(Exception):
            self.conn.datachannel.set_decoder(self.args.lidar_decoder)

        for topic in self.args.subscribe_topics:
            with contextlib.suppress(Exception):
                await self._subscribe_topic(self._resolve_topic(topic))

        for profile in self.args.subscribe_profiles:
            aliases = PROFILE_TOPICS.get(profile, [])
            for alias in aliases:
                with contextlib.suppress(Exception):
                    await self._subscribe_topic(TOPIC_ALIAS_TO_VALUE[alias])

        if self.camera_enabled:
            await self._set_video(True)

        if self.audio_enabled:
            await self._set_audio(True)
            with contextlib.suppress(Exception):
                await self._subscribe_topic(TOPIC_ALIAS_TO_VALUE["AUDIO_HUB_PLAY_STATE"])

        if self.lidar_enabled:
            await self._set_lidar(True)
            with contextlib.suppress(Exception):
                await self._subscribe_topic(TOPIC_ALIAS_TO_VALUE["ULIDAR_ARRAY"])
            with contextlib.suppress(Exception):
                await self._subscribe_topic(TOPIC_ALIAS_TO_VALUE["ULIDAR_STATE"])

    async def run(self) -> None:
        if sys.__stdout__ is not None:
            with contextlib.redirect_stdout(sys.stderr):
                await self._run_impl()
        else:
            await self._run_impl()

    async def _run_impl(self) -> None:
        self.loop = asyncio.get_running_loop()

        await self._connect()

        await self.emit(
            {
                "type": "status",
                "event": "ready",
                "ts": time.time(),
                "status": self._status_snapshot(),
            }
        )

        self.stdin_thread = threading.Thread(target=self._stdin_reader, daemon=True)
        self.stdin_thread.start()

        command_task = asyncio.create_task(self._command_loop())

        try:
            while not self.stop_event.is_set():
                await asyncio.sleep(0.2)
        finally:
            command_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await command_task

            if self.video_task and not self.video_task.done():
                self.video_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.video_task

            if self.conn:
                with contextlib.suppress(Exception):
                    self.conn.video.switchVideoChannel(False)
                with contextlib.suppress(Exception):
                    self.conn.audio.switchAudioChannel(False)
                with contextlib.suppress(Exception):
                    await self.conn.disconnect()


def _split_csv_values(values: List[str]) -> List[str]:
    output: List[str] = []
    for item in values:
        for part in item.split(","):
            value = part.strip()
            if value:
                output.append(value)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Go2 SSH gateway: stream telemetry/camera/lidar/audio as NDJSON on stdout "
            "and receive control commands as NDJSON on stdin."
        )
    )

    parser.add_argument("--ip", default=DEFAULT_GO2_IP, help="Go2 IP in STA mode")
    parser.add_argument(
        "--subscribe-profile",
        action="append",
        default=["core"],
        help="Auto-subscribe profile(s): core, all_telemetry, lidar, audio, navigation (repeat or comma-separated)",
    )
    parser.add_argument(
        "--subscribe-topic",
        action="append",
        default=[],
        help="Auto-subscribe extra topic alias/value (repeat or comma-separated)",
    )

    parser.add_argument("--enable-camera", action="store_true", help="Enable camera streaming at startup")
    parser.add_argument("--camera-emit-every", type=int, default=1, help="Emit one camera frame every N frames")
    parser.add_argument(
        "--camera-format",
        choices=["jpg", "png"],
        default="jpg",
        help="Camera frame encoding in output events",
    )
    parser.add_argument("--camera-jpeg-quality", type=int, default=75, help="JPEG quality 1-100")
    parser.add_argument("--camera-png-compression", type=int, default=3, help="PNG compression 0-9")

    parser.add_argument("--enable-lidar", action="store_true", help="Turn lidar on at startup")
    parser.add_argument(
        "--lidar-decoder",
        choices=["libvoxel", "native"],
        default="libvoxel",
        help="Lidar decoder type",
    )

    parser.add_argument(
        "--disable-traffic-saving",
        action="store_true",
        help="Send disableTrafficSaving(on) after connect (recommended for lidar)",
    )
    parser.add_argument("--enable-audio", action="store_true", help="Enable audio streaming at startup")
    parser.add_argument("--audio-emit-every", type=int, default=1, help="Emit one audio packet every N audio frames")
    parser.add_argument(
        "--audio-max-bytes",
        type=int,
        default=24576,
        help="Max PCM bytes per audio event (0 disables truncation)",
    )
    parser.add_argument(
        "--exit-on-stdin-eof",
        action="store_true",
        help="Stop gateway when stdin closes",
    )

    parser.add_argument(
        "--max-list-items",
        type=int,
        default=0,
        help="Truncate JSON arrays to first N items in output (0 disables truncation)",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=0,
        help="Truncate raw bytes payload before base64 in output (0 disables truncation)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=20,
        help="Max nested depth during JSON sanitization",
    )

    args = parser.parse_args()

    args.subscribe_profiles = _split_csv_values(args.subscribe_profile)
    args.subscribe_topics = _split_csv_values(args.subscribe_topic)

    if args.camera_emit_every <= 0:
        parser.error("--camera-emit-every must be > 0")

    if args.camera_jpeg_quality < 1 or args.camera_jpeg_quality > 100:
        parser.error("--camera-jpeg-quality must be between 1 and 100")

    if args.camera_png_compression < 0 or args.camera_png_compression > 9:
        parser.error("--camera-png-compression must be between 0 and 9")

    if args.audio_emit_every <= 0:
        parser.error("--audio-emit-every must be > 0")

    if args.audio_max_bytes < 0:
        parser.error("--audio-max-bytes must be >= 0")

    if args.max_list_items < 0:
        parser.error("--max-list-items must be >= 0")

    if args.max_bytes < 0:
        parser.error("--max-bytes must be >= 0")

    if args.max_depth <= 0:
        parser.error("--max-depth must be > 0")

    return args


def main() -> None:
    args = parse_args()
    gateway = Go2SshGateway(args)

    try:
        asyncio.run(gateway.run())
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
