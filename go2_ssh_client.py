#!/usr/bin/env python3
import argparse
import base64
import contextlib
import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import sounddevice as sd
except Exception:
    sd = None


LEFT_KEYS = {81, 2424832, 65361}
UP_KEYS = {82, 2490368, 65362}
RIGHT_KEYS = {83, 2555904, 65363}
DOWN_KEYS = {84, 2621440, 65364}

W_KEYS = {ord("w"), ord("W")}
A_KEYS = {ord("a"), ord("A")}
S_KEYS = {ord("s"), ord("S")}
D_KEYS = {ord("d"), ord("D")}

STRAFE_LEFT_KEYS = {ord("z"), ord("Z")}
STRAFE_RIGHT_KEYS = {ord("c"), ord("C")}

STOP_KEYS = {ord(" "), ord("x"), ord("X")}
QUIT_KEYS = {27, ord("q"), ord("Q")}
HELP_KEYS = {ord("h"), ord("H")}

LIDAR_TOPIC_HINTS = {
    "ULIDAR",
    "ULIDAR_ARRAY",
    "ULIDAR_STATE",
    "LIDAR_MAPPING_CLOUD_POINT",
    "LIDAR_LOCALIZATION_CLOUD_POINT",
}


def _print_json(prefix: str, payload: Any) -> None:
    print(f"{prefix}{json.dumps(payload, ensure_ascii=True)}", flush=True)


def _parse_bool_text(value: str) -> bool:
    return value.strip().lower() in {"1", "on", "true", "yes", "y"}


class GatewayClient:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        self.proc: Optional[subprocess.Popen[str]] = None
        self.running = False
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()

        self.reader_thread: Optional[threading.Thread] = None
        self.stderr_thread: Optional[threading.Thread] = None
        self.command_thread: Optional[threading.Thread] = None
        self.audio_thread: Optional[threading.Thread] = None

        self.pending_responses: Dict[str, queue.Queue[dict]] = {}
        self.id_counter = 1
        self.id_lock = threading.Lock()
        self.send_lock = threading.Lock()

        self.last_stderr_lines: deque[str] = deque(maxlen=40)
        self.last_error: str = ""

        self.frame_lock = threading.Lock()
        self.topic_lock = threading.Lock()

        self.latest_camera_frame: Optional[np.ndarray] = None
        self.latest_lidar_image: Optional[np.ndarray] = None
        self.latest_lidar_point_count = 0
        self.latest_topics: Dict[str, Any] = {}

        self.camera_events = 0
        self.topic_events = 0
        self.audio_events = 0

        self.gui_enabled = not args.no_gui
        self.show_video_window = self.gui_enabled and not args.no_video_window
        self.show_lidar_window = self.gui_enabled and not args.no_lidar_window
        self.arrow_teleop_enabled = self.gui_enabled and not args.disable_arrow_teleop

        self.pending_stop_at = 0.0
        self.current_move = (0.0, 0.0, 0.0)
        self.move_active = False
        self.last_move_sent_at = 0.0

        self.play_audio = bool(args.play_audio)
        self.audio_supported = sd is not None
        self.audio_queue: queue.Queue[Optional[Tuple[np.ndarray, int, int]]] = queue.Queue(
            maxsize=max(4, args.audio_queue_size)
        )
        self.audio_dropped_chunks = 0
        self.audio_error_count = 0

        self.last_stats_at = time.monotonic()

    def _next_id(self) -> str:
        with self.id_lock:
            value = f"cmd-{self.id_counter}"
            self.id_counter += 1
            return value

    def _build_remote_gateway_command(self) -> str:
        cmd: List[str] = [
            self.args.remote_python,
            "-u",
            self.args.remote_gateway,
            "--ip",
            self.args.go2_ip,
            "--camera-format",
            self.args.camera_format,
            "--camera-jpeg-quality",
            str(self.args.camera_jpeg_quality),
            "--camera-png-compression",
            str(self.args.camera_png_compression),
            "--camera-emit-every",
            str(self.args.camera_emit_every),
            "--audio-emit-every",
            str(self.args.audio_emit_every),
            "--audio-max-bytes",
            str(self.args.audio_max_bytes),
            "--max-list-items",
            str(self.args.max_list_items),
            "--max-depth",
            str(self.args.max_depth),
        ]

        if self.args.enable_camera:
            cmd.append("--enable-camera")
        if self.args.enable_lidar:
            cmd.append("--enable-lidar")
        if self.args.enable_audio:
            cmd.append("--enable-audio")
        if self.args.disable_traffic_saving:
            cmd.append("--disable-traffic-saving")
        if self.args.max_bytes > 0:
            cmd.extend(["--max-bytes", str(self.args.max_bytes)])
        if self.args.exit_on_stdin_eof:
            cmd.append("--exit-on-stdin-eof")

        for profile in self.args.subscribe_profile:
            cmd.extend(["--subscribe-profile", profile])
        for topic in self.args.subscribe_topic:
            cmd.extend(["--subscribe-topic", topic])

        return " ".join(shlex.quote(x) for x in cmd)

    def start(self) -> None:
        remote_cmd = self._build_remote_gateway_command()

        ssh_cmd: List[str] = ["ssh", "-T"]

        if self.args.identity_file:
            ssh_cmd.extend(["-i", self.args.identity_file])

        batch_mode = "no" if self.args.ssh_password else "yes"

        ssh_cmd.extend(
            [
                "-p",
                str(self.args.ssh_port),
                "-o",
                f"StrictHostKeyChecking={self.args.strict_host_key_checking}",
                "-o",
                f"ConnectTimeout={self.args.ssh_connect_timeout}",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=3",
                "-o",
                f"BatchMode={batch_mode}",
                self.args.remote,
                remote_cmd,
            ]
        )

        popen_env = None
        if self.args.ssh_password:
            if shutil.which("sshpass") is None:
                raise RuntimeError(
                    "--ssh-password requiere sshpass instalado. Instala con: sudo apt install -y sshpass"
                )
            popen_env = os.environ.copy()
            popen_env["SSHPASS"] = self.args.ssh_password
            ssh_cmd = ["sshpass", "-e"] + ssh_cmd

        _print_json("[local] Starting SSH: ", {"cmd": ssh_cmd})

        self.proc = subprocess.Popen(
            ssh_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=popen_env,
        )

        assert self.proc.stdout is not None
        assert self.proc.stderr is not None

        self.running = True
        self.stop_event.clear()
        self.ready_event.clear()
        self.last_error = ""

        self.reader_thread = threading.Thread(target=self._stdout_reader, daemon=True)
        self.reader_thread.start()

        self.stderr_thread = threading.Thread(target=self._stderr_reader, daemon=True)
        self.stderr_thread.start()

    def _stdout_reader(self) -> None:
        assert self.proc is not None
        assert self.proc.stdout is not None

        for line in self.proc.stdout:
            raw = line.strip()
            if not raw:
                continue

            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                print(f"[gateway/raw] {raw}", flush=True)
                continue

            self._handle_gateway_message(message)

        self.running = False
        self.stop_event.set()

    def _stderr_reader(self) -> None:
        assert self.proc is not None
        assert self.proc.stderr is not None

        for line in self.proc.stderr:
            self.last_stderr_lines.append(line.rstrip("\n"))
            if self.args.print_ssh_stderr:
                sys.stderr.write(f"[ssh/stderr] {line}")
                sys.stderr.flush()

    def _handle_gateway_message(self, message: Dict[str, Any]) -> None:
        msg_type = message.get("type")

        if msg_type == "response":
            cmd_id = str(message.get("id"))
            q = self.pending_responses.get(cmd_id)
            if q is not None:
                q.put(message)
            _print_json("[gateway/response] ", message)
            return

        if msg_type == "status":
            if message.get("event") == "ready":
                self.ready_event.set()
            _print_json("[gateway/status] ", message)
            return

        if msg_type == "error":
            _print_json("[gateway/error] ", message)
            return

        if msg_type == "event":
            self._handle_event(message)
            return

        _print_json("[gateway/msg] ", message)

    def _handle_event(self, message: Dict[str, Any]) -> None:
        stream = str(message.get("stream", ""))

        if stream == "camera":
            self._handle_camera_event(message)
            return

        if stream == "topic":
            self._handle_topic_event(message)
            return

        if stream == "audio":
            self._handle_audio_event(message)
            return

        if self.args.print_topic_events:
            _print_json("[gateway/event] ", message)

    def _handle_camera_event(self, message: Dict[str, Any]) -> None:
        data = message.get("data")
        if not isinstance(data, dict):
            return

        image_base64 = data.get("image_base64")
        if not isinstance(image_base64, str) or not image_base64:
            return

        try:
            raw = base64.b64decode(image_base64, validate=False)
        except Exception:
            return

        arr = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            return

        with self.frame_lock:
            self.latest_camera_frame = image

        self.camera_events += 1

        if self.args.print_camera_events:
            payload = {
                "frame_index": data.get("frame_index"),
                "width": data.get("width"),
                "height": data.get("height"),
                "format": data.get("image_format"),
            }
            _print_json("[camera] ", payload)

    def _handle_topic_event(self, message: Dict[str, Any]) -> None:
        topic_alias = str(message.get("topic_alias") or message.get("topic") or "")
        data = message.get("data")

        with self.topic_lock:
            self.latest_topics[topic_alias] = data

        self.topic_events += 1

        if self._looks_like_lidar_topic(topic_alias):
            self._update_lidar_image(data)

        if self.args.print_topic_events:
            _print_json("[topic] ", {"topic_alias": topic_alias, "topic": message.get("topic")})

    def _handle_audio_event(self, message: Dict[str, Any]) -> None:
        if not self.play_audio or not self.audio_supported:
            return

        data = message.get("data")
        if not isinstance(data, dict):
            return

        audio_base64 = data.get("audio_base64")
        if not isinstance(audio_base64, str) or not audio_base64:
            return

        try:
            raw = base64.b64decode(audio_base64, validate=False)
        except Exception:
            return

        sample_rate = int(data.get("sample_rate") or self.args.audio_default_sample_rate)
        if sample_rate <= 0:
            sample_rate = self.args.audio_default_sample_rate

        channels = int(data.get("channels") or 1)
        if channels <= 0:
            channels = 1

        if len(raw) % 2 != 0:
            raw = raw[:-1]

        bytes_per_frame = 2 * channels
        if len(raw) < bytes_per_frame:
            return

        if len(raw) % bytes_per_frame != 0:
            raw = raw[: len(raw) - (len(raw) % bytes_per_frame)]

        if not raw:
            return

        samples = np.frombuffer(raw, dtype=np.int16)
        try:
            samples = samples.reshape(-1, channels)
        except ValueError:
            return

        self._enqueue_audio_chunk(samples, sample_rate, channels)
        self.audio_events += 1

        if self.args.print_audio_events:
            payload = {
                "sample_rate": sample_rate,
                "channels": channels,
                "frames": int(samples.shape[0]),
            }
            _print_json("[audio] ", payload)

    def _enqueue_audio_chunk(self, samples: np.ndarray, sample_rate: int, channels: int) -> None:
        item = (samples.copy(), sample_rate, channels)
        try:
            self.audio_queue.put_nowait(item)
            return
        except queue.Full:
            self.audio_dropped_chunks += 1

        with contextlib.suppress(queue.Empty):
            self.audio_queue.get_nowait()

        with contextlib.suppress(queue.Full):
            self.audio_queue.put_nowait(item)

    def _audio_player_loop(self) -> None:
        if sd is None:
            return

        stream = None
        current_sample_rate = 0
        current_channels = 0

        try:
            while self.running or not self.audio_queue.empty():
                try:
                    item = self.audio_queue.get(timeout=0.2)
                except queue.Empty:
                    if self.stop_event.is_set() and not self.running:
                        break
                    continue

                if item is None:
                    break

                samples, sample_rate, channels = item

                if sample_rate <= 0:
                    sample_rate = self.args.audio_default_sample_rate
                if channels <= 0:
                    channels = 1

                if stream is None or sample_rate != current_sample_rate or channels != current_channels:
                    if stream is not None:
                        with contextlib.suppress(Exception):
                            stream.stop()
                        with contextlib.suppress(Exception):
                            stream.close()

                    stream = sd.OutputStream(samplerate=sample_rate, channels=channels, dtype="int16")
                    stream.start()
                    current_sample_rate = sample_rate
                    current_channels = channels

                stream.write(samples)
        except Exception as exc:
            self.audio_error_count += 1
            print(f"[local/warn] audio playback disabled due to error: {exc}", flush=True)
        finally:
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.stop()
                with contextlib.suppress(Exception):
                    stream.close()

    def start_audio_player(self) -> None:
        if not self.play_audio:
            return

        if not self.audio_supported:
            print("[local/warn] --play-audio requested but sounddevice is not available.", flush=True)
            return

        if self.audio_thread and self.audio_thread.is_alive():
            return

        self.audio_thread = threading.Thread(target=self._audio_player_loop, daemon=True)
        self.audio_thread.start()

    def wait_until_ready(self, timeout: float) -> bool:
        deadline = time.time() + timeout

        while time.time() < deadline:
            if self.ready_event.is_set():
                return True

            if self.proc is not None:
                exit_code = self.proc.poll()
                if exit_code is not None:
                    self.running = False
                    stderr = "\n".join(self.last_stderr_lines)
                    self.last_error = (
                        f"SSH/Gateway finalizo antes de estar listo (exit={exit_code})."
                        + (f"\nUltimo stderr:\n{stderr}" if stderr else "")
                    )
                    return False

            time.sleep(0.1)

        stderr = "\n".join(self.last_stderr_lines)
        self.last_error = (
            f"Timeout esperando evento 'ready' del gateway ({timeout:.1f}s)."
            + (f"\nUltimo stderr:\n{stderr}" if stderr else "")
        )
        return False

    def send(self, payload: Dict[str, Any], wait_response: bool = False, timeout: float = 8.0) -> Optional[dict]:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("SSH process not started")
        if not self.running:
            raise RuntimeError("Gateway is not running")

        if wait_response and not self.ready_event.is_set():
            ready = self.wait_until_ready(timeout=max(timeout, self.args.ready_timeout))
            if not ready:
                raise RuntimeError(self.last_error or "Gateway is not ready")

        payload = dict(payload)
        if "id" not in payload:
            payload["id"] = self._next_id()

        cmd_id = str(payload["id"])

        response_queue: Optional[queue.Queue[dict]] = None
        if wait_response:
            response_queue = queue.Queue(maxsize=1)

        with self.send_lock:
            if wait_response and response_queue is not None:
                self.pending_responses[cmd_id] = response_queue

            try:
                self.proc.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
                self.proc.stdin.flush()
            except Exception:
                self.pending_responses.pop(cmd_id, None)
                raise

        if not wait_response:
            return None

        assert response_queue is not None
        try:
            return response_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        finally:
            self.pending_responses.pop(cmd_id, None)

    def safe_send_no_wait(self, payload: Dict[str, Any]) -> bool:
        try:
            self.send(payload, wait_response=False)
            return True
        except Exception as exc:
            print(f"[local/error] failed to send command: {exc}", flush=True)
            return False

    def _should_wait_response(self, payload: Dict[str, Any]) -> bool:
        op = str(payload.get("op", ""))
        return op in {
            "help",
            "ping",
            "status",
            "list_topics",
            "list_sport_cmd",
            "get_latest",
            "get_temperatures",
            "request",
            "publish",
            "sport",
            "set_motion_mode",
            "get_motion_mode",
            "set_video",
            "set_camera_stream",
            "set_lidar",
            "set_lidar_decoder",
            "set_audio",
            "audiohub",
            "subscribe",
            "unsubscribe",
            "subscribe_profile",
        }

    def _send_move(self, x: float, y: float, z: float) -> None:
        payload = {
            "op": "sport",
            "action": "Move",
            "parameter": {
                "x": float(x),
                "y": float(y),
                "z": float(z),
            },
        }
        if self.safe_send_no_wait(payload):
            self.current_move = (x, y, z)
            self.move_active = True
            self.last_move_sent_at = time.monotonic()

    def _send_stop_move(self) -> None:
        if self.safe_send_no_wait({"op": "sport", "action": "StopMove"}):
            self.move_active = False
            self.current_move = (0.0, 0.0, 0.0)
            self.pending_stop_at = 0.0

    def _movement_from_key(self, key: int) -> Optional[Tuple[float, float, float]]:
        if key in UP_KEYS or key in W_KEYS:
            return (self.args.teleop_linear, 0.0, 0.0)

        if key in DOWN_KEYS or key in S_KEYS:
            return (-self.args.teleop_linear, 0.0, 0.0)

        if key in LEFT_KEYS or key in A_KEYS:
            return (0.0, 0.0, self.args.teleop_yaw)

        if key in RIGHT_KEYS or key in D_KEYS:
            return (0.0, 0.0, -self.args.teleop_yaw)

        if key in STRAFE_LEFT_KEYS:
            return (0.0, self.args.teleop_lateral, 0.0)

        if key in STRAFE_RIGHT_KEYS:
            return (0.0, -self.args.teleop_lateral, 0.0)

        return None

    def _process_teleop(self, key: int) -> None:
        if not self.arrow_teleop_enabled:
            return

        now = time.monotonic()
        refresh_interval = 1.0 / max(self.args.teleop_refresh_hz, 1e-6)

        if key in STOP_KEYS:
            self._send_stop_move()
            return

        movement = self._movement_from_key(key)
        if movement is not None:
            self.pending_stop_at = now + self.args.deadman_timeout
            if (not self.move_active) or (now - self.last_move_sent_at >= refresh_interval):
                self._send_move(*movement)
            return

        if self.move_active and now < self.pending_stop_at and now - self.last_move_sent_at >= refresh_interval:
            self._send_move(*self.current_move)
            return

        if self.move_active and now >= self.pending_stop_at:
            self._send_stop_move()

    def _looks_like_lidar_topic(self, topic_alias: str) -> bool:
        if topic_alias in LIDAR_TOPIC_HINTS:
            return True

        topic_alias_upper = topic_alias.upper()
        return "LIDAR" in topic_alias_upper or "ULIDAR" in topic_alias_upper or "CLOUD" in topic_alias_upper

    def _extract_lidar_points(self, value: Any, depth: int = 0) -> Optional[np.ndarray]:
        if depth > 8:
            return None

        if isinstance(value, dict):
            if value.get("__truncated__") and isinstance(value.get("items"), list):
                found = self._extract_lidar_points(value["items"], depth + 1)
                if found is not None:
                    return found

            preferred_keys = (
                "points",
                "point_cloud",
                "cloud_points",
                "cloud",
                "xyz",
                "voxels",
                "voxel_map",
                "positions",
                "position",
                "data",
            )
            for key in preferred_keys:
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

            first = value[0]

            if isinstance(first, (list, tuple)) and len(first) >= 2:
                with contextlib.suppress(Exception):
                    arr = np.asarray(value, dtype=np.float32)
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        return arr[:, : min(arr.shape[1], 3)]

            if isinstance(first, (int, float)):
                with contextlib.suppress(Exception):
                    arr = np.asarray(value, dtype=np.float32)
                    if arr.ndim == 1 and arr.size >= 6:
                        if arr.size % 3 == 0:
                            return arr.reshape(-1, 3)
                        if arr.size % 2 == 0:
                            return arr.reshape(-1, 2)

            for item in value[:10]:
                found = self._extract_lidar_points(item, depth + 1)
                if found is not None:
                    return found

        return None

    def _render_lidar_image(self, points: np.ndarray) -> Tuple[np.ndarray, int]:
        size = self.args.lidar_window_size
        image = np.zeros((size, size, 3), dtype=np.uint8)

        if points.size == 0:
            cv2.putText(
                image,
                "No lidar points",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 180),
                2,
                cv2.LINE_AA,
            )
            return image, 0

        xy = points[:, :2].astype(np.float32)
        finite = np.isfinite(xy).all(axis=1)
        xy = xy[finite]

        if xy.size == 0:
            cv2.putText(
                image,
                "No finite lidar points",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 180),
                2,
                cv2.LINE_AA,
            )
            return image, 0

        point_count = int(xy.shape[0])

        if self.args.lidar_max_points > 0 and point_count > self.args.lidar_max_points:
            step = max(point_count // self.args.lidar_max_points, 1)
            xy = xy[::step]

        max_abs = np.percentile(np.abs(xy), 98, axis=0)
        span = float(max(max_abs.max(), 0.5))
        scale = (size * 0.45) / span

        px = (xy[:, 0] * scale + size / 2.0).astype(np.int32)
        py = (size / 2.0 - xy[:, 1] * scale).astype(np.int32)

        valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
        image[py[valid], px[valid]] = (60, 230, 60)

        cv2.circle(image, (size // 2, size // 2), 4, (0, 0, 255), -1)
        cv2.putText(
            image,
            f"points={point_count}",
            (15, size - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

        return image, point_count

    def _update_lidar_image(self, payload: Any) -> None:
        points = self._extract_lidar_points(payload)
        if points is None:
            return

        with contextlib.suppress(Exception):
            image, point_count = self._render_lidar_image(points)
            with self.frame_lock:
                self.latest_lidar_image = image
                self.latest_lidar_point_count = point_count

    def _camera_frame_for_display(self) -> np.ndarray:
        with self.frame_lock:
            frame = self.latest_camera_frame.copy() if self.latest_camera_frame is not None else None

        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "Waiting camera frames...",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )

        move_text = f"move x={self.current_move[0]:+.2f} y={self.current_move[1]:+.2f} z={self.current_move[2]:+.2f}"
        cv2.putText(frame, move_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            "Arrows/WASD move | Z/C strafe | Space stop | Q/Esc quit",
            (15, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )

        return frame

    def _lidar_frame_for_display(self) -> np.ndarray:
        with self.frame_lock:
            lidar = self.latest_lidar_image.copy() if self.latest_lidar_image is not None else None
            point_count = self.latest_lidar_point_count

        if lidar is None:
            size = self.args.lidar_window_size
            lidar = np.zeros((size, size, 3), dtype=np.uint8)
            cv2.putText(
                lidar,
                "Waiting lidar points...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 180),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                lidar,
                f"lidar points={point_count}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )

        return lidar

    def _print_stats(self) -> None:
        now = time.monotonic()
        if self.args.stats_every <= 0:
            return
        if now - self.last_stats_at < self.args.stats_every:
            return

        self.last_stats_at = now
        stats = {
            "camera_events": self.camera_events,
            "topic_events": self.topic_events,
            "audio_events": self.audio_events,
            "audio_dropped_chunks": self.audio_dropped_chunks,
            "move_active": self.move_active,
        }
        _print_json("[local/stats] ", stats)

    def _ui_loop(self) -> None:
        print("[local] GUI activo. Usa h para ayuda de controles.", flush=True)

        while self.running and not self.stop_event.is_set():
            key = -1

            try:
                if self.show_video_window:
                    cv2.imshow("Go2 Camera", self._camera_frame_for_display())
                if self.show_lidar_window:
                    cv2.imshow("Go2 LiDAR", self._lidar_frame_for_display())

                if self.show_video_window or self.show_lidar_window:
                    key = cv2.waitKeyEx(1)
            except cv2.error as exc:
                print(f"[local/warn] GUI unavailable, fallback to headless mode: {exc}", flush=True)
                self.show_video_window = False
                self.show_lidar_window = False
                self.gui_enabled = False
                break

            if key in QUIT_KEYS:
                self.stop_event.set()
                break

            if key in HELP_KEYS:
                print_runtime_controls()

            self._process_teleop(key)
            self._print_stats()

            time.sleep(0.005)

        cv2.destroyAllWindows()

    def _headless_loop(self) -> None:
        print("[local] Headless mode. Usa comandos por consola para controlar.", flush=True)
        while self.running and not self.stop_event.is_set():
            self._print_stats()
            time.sleep(0.05)

    def _command_loop(self) -> None:
        while self.running and not self.stop_event.is_set():
            try:
                line = input("go2> ")
            except EOFError:
                self.stop_event.set()
                break

            line = line.strip()
            if not line:
                continue

            try:
                payload = parse_local_command(line)
            except Exception as exc:
                print(f"[local/error] {exc}", flush=True)
                continue

            if payload is None:
                continue

            wait_response = self._should_wait_response(payload)

            try:
                response = self.send(payload, wait_response=wait_response, timeout=self.args.response_timeout)
            except Exception as exc:
                print(f"[local/error] {exc}", flush=True)
                continue

            if wait_response and response is None:
                print("[local/warn] timeout waiting response", flush=True)

            if str(payload.get("op", "")) in {"exit", "quit", "stop"}:
                self.stop_event.set()
                break

    def start_command_console(self) -> None:
        if self.args.no_command_console:
            return

        if self.command_thread and self.command_thread.is_alive():
            return

        self.command_thread = threading.Thread(target=self._command_loop, daemon=True)
        self.command_thread.start()

    def run(self) -> int:
        self.start()

        if not self.wait_until_ready(self.args.ready_timeout):
            print(f"[local/error] {self.last_error}", flush=True)
            return 1

        if self.play_audio:
            self.start_audio_player()

        print_usage()
        print_runtime_controls()
        self.start_command_console()

        if self.gui_enabled and (self.show_video_window or self.show_lidar_window):
            self._ui_loop()
        else:
            self._headless_loop()

        return 0

    def close(self) -> None:
        self.stop_event.set()

        if self.proc is not None:
            try:
                if self.running:
                    with contextlib.suppress(Exception):
                        self.send({"op": "exit"}, wait_response=False)
            except Exception:
                pass

            with contextlib.suppress(Exception):
                if self.proc.stdin:
                    self.proc.stdin.close()

            with contextlib.suppress(Exception):
                self.proc.terminate()
                self.proc.wait(timeout=3)

            self.running = False

        if self.audio_thread and self.audio_thread.is_alive():
            with contextlib.suppress(Exception):
                self.audio_queue.put_nowait(None)
            self.audio_thread.join(timeout=1.5)

        if self.command_thread and self.command_thread.is_alive():
            self.command_thread.join(timeout=0.2)

        cv2.destroyAllWindows()


def parse_local_command(text: str) -> Optional[Dict[str, Any]]:
    raw = text.strip()
    if not raw:
        return None

    if raw.startswith("{"):
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("JSON command must be an object")
        return payload

    parts = raw.split()
    head = parts[0].lower()

    if head in {"quit", "exit"}:
        return {"op": "exit"}

    if head in {"help", "?"}:
        return {"op": "help"}

    if head == "status":
        return {"op": "status"}

    if head == "topics":
        return {"op": "list_topics"}

    if head == "sportcmd":
        return {"op": "list_sport_cmd"}

    if head == "temps":
        return {"op": "get_temperatures"}

    if head == "latest":
        if len(parts) == 1:
            return {"op": "get_latest"}
        return {"op": "get_latest", "topic": parts[1]}

    if head == "sub" and len(parts) >= 2:
        return {"op": "subscribe", "topic": parts[1]}

    if head == "unsub" and len(parts) >= 2:
        return {"op": "unsubscribe", "topic": parts[1]}

    if head == "profile" and len(parts) >= 2:
        return {"op": "subscribe_profile", "profile": parts[1]}

    if head == "motion" and len(parts) >= 2:
        return {"op": "set_motion_mode", "name": parts[1]}

    if head == "video" and len(parts) >= 2:
        return {"op": "set_video", "enabled": _parse_bool_text(parts[1])}

    if head == "lidar" and len(parts) >= 2:
        return {"op": "set_lidar", "enabled": _parse_bool_text(parts[1]), "subscribe": True}

    if head == "audio" and len(parts) >= 2:
        return {"op": "set_audio", "enabled": _parse_bool_text(parts[1])}

    if head == "audiocfg":
        cmd: Dict[str, Any] = {"op": "set_audio"}
        for token in parts[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key == "emit":
                cmd["emit_every"] = int(value)
            elif key == "maxbytes":
                cmd["max_bytes"] = int(value)
            elif key == "enabled":
                cmd["enabled"] = _parse_bool_text(value)
        return cmd

    if head == "audiolist":
        return {"op": "audiohub", "action": "list"}

    if head == "audioplay" and len(parts) >= 2:
        return {"op": "audiohub", "action": "play", "unique_id": parts[1]}

    if head == "audiopause":
        return {"op": "audiohub", "action": "pause"}

    if head == "audioresume":
        return {"op": "audiohub", "action": "resume"}

    if head == "audionext":
        return {"op": "audiohub", "action": "next"}

    if head == "audioprev":
        return {"op": "audiohub", "action": "prev"}

    if head == "audiomode" and len(parts) >= 2:
        return {"op": "audiohub", "action": "set_mode", "play_mode": parts[1]}

    if head == "audiohub" and len(parts) >= 2:
        action = parts[1]
        if action == "raw":
            if len(parts) < 3:
                raise ValueError("audiohub raw requiere api_id")
            cmd = {"op": "audiohub", "action": "raw", "api_id": int(parts[2])}
            if len(parts) >= 4:
                cmd["parameter"] = json.loads(" ".join(parts[3:]))
            return cmd
        return {"op": "audiohub", "action": action}

    if head == "move" and len(parts) >= 4:
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        return {"op": "sport", "action": "Move", "parameter": {"x": x, "y": y, "z": z}}

    if head in {"stop", "halt"}:
        return {"op": "sport", "action": "StopMove"}

    if head == "sport" and len(parts) >= 2:
        cmd = {"op": "sport", "action": parts[1]}
        if len(parts) >= 3:
            param_text = " ".join(parts[2:]).strip()
            if param_text:
                cmd["parameter"] = json.loads(param_text)
        return cmd

    if head == "request" and len(parts) >= 3:
        cmd = {
            "op": "request",
            "topic": parts[1],
            "api_id": int(parts[2]),
        }
        if len(parts) >= 4:
            cmd["parameter"] = json.loads(" ".join(parts[3:]))
        return cmd

    raise ValueError(
        "Unknown command. Use: help, status, temps, sub, sport, move, stop, "
        "motion, video, lidar, audio, audiolist, audioplay, request, exit"
    )


def print_usage() -> None:
    print("\nComandos locales:", flush=True)
    print("  help                               -> lista ops del gateway", flush=True)
    print("  status                             -> estado actual", flush=True)
    print("  temps                              -> resumen de temperaturas", flush=True)
    print("  topics                             -> lista de topics", flush=True)
    print("  sportcmd                           -> lista de SPORT_CMD", flush=True)
    print("  latest                             -> ultimos mensajes cacheados", flush=True)
    print("  latest LOW_STATE                   -> ultimo de un topic", flush=True)
    print("  sub LOW_STATE                      -> suscribirse", flush=True)
    print("  unsub LOW_STATE                    -> desuscribirse", flush=True)
    print("  profile all_telemetry              -> suscribir perfil", flush=True)
    print("  motion normal                      -> set motion mode", flush=True)
    print("  video on|off                       -> activar/desactivar camara", flush=True)
    print("  lidar on|off                       -> activar/desactivar lidar", flush=True)
    print("  audio on|off                       -> activar/desactivar audio stream", flush=True)
    print("  audiocfg emit=1 maxbytes=24576     -> ajustar stream de audio", flush=True)
    print("  audiolist                          -> lista audios del robot", flush=True)
    print("  audioplay <uuid>                   -> reproducir audio por uuid", flush=True)
    print("  audiopause / audioresume           -> pausar/reanudar audio", flush=True)
    print("  audioprev / audionext              -> pista anterior/siguiente", flush=True)
    print("  audiomode list_loop                -> set play_mode", flush=True)
    print("  move 0.3 0 0                       -> Move manual (x y z)", flush=True)
    print("  stop                               -> StopMove", flush=True)
    print("  sport Move {\"x\":0.2,\"y\":0,\"z\":0} -> move", flush=True)
    print("  request MOTION_SWITCHER 1001       -> request generico", flush=True)
    print("  { ...json... }                     -> enviar JSON crudo", flush=True)
    print("  exit                               -> cerrar", flush=True)


def print_runtime_controls() -> None:
    print("\nControles GUI:", flush=True)
    print("  Flechas o WASD                     -> mover", flush=True)
    print("  Z / C                              -> movimiento lateral (y)", flush=True)
    print("  Espacio                            -> StopMove", flush=True)
    print("  H                                  -> imprimir ayuda", flush=True)
    print("  Q o Esc                            -> salir", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive SSH client for go2_ssh_gateway.py. "
            "Run this on your server to visualize video/lidar/audio and send commands concurrently."
        )
    )

    parser.add_argument("--remote", required=True, help="SSH target, for example user@pc-local")
    parser.add_argument("--ssh-port", type=int, default=22, help="SSH port")
    parser.add_argument("--identity-file", help="SSH private key file")
    parser.add_argument("--ssh-password", help="SSH password (requires sshpass installed)")
    parser.add_argument(
        "--strict-host-key-checking",
        choices=["yes", "no", "accept-new"],
        default="accept-new",
        help="SSH StrictHostKeyChecking value",
    )
    parser.add_argument("--ssh-connect-timeout", type=int, default=15, help="SSH connect timeout in seconds")

    parser.add_argument(
        "--remote-python",
        default="/home/bensagra/Documents/go2/.venv/bin/python",
        help="Python executable on remote PC/Raspberry",
    )
    parser.add_argument(
        "--remote-gateway",
        default="/home/bensagra/Documents/go2/go2_ssh_gateway.py",
        help="Path to go2_ssh_gateway.py on remote PC/Raspberry",
    )
    parser.add_argument("--go2-ip", default="192.168.123.161", help="Go2 IP in STA mode")

    parser.add_argument("--subscribe-profile", action="append", default=["core"], help="Gateway subscribe profile")
    parser.add_argument("--subscribe-topic", action="append", default=[], help="Gateway extra subscribe topic")

    parser.add_argument("--enable-camera", action="store_true", help="Enable camera at startup")
    parser.add_argument("--enable-lidar", action="store_true", help="Enable lidar at startup")
    parser.add_argument("--enable-audio", action="store_true", help="Enable audio at startup")

    parser.add_argument("--disable-traffic-saving", action="store_true", help="Enable disableTrafficSaving")
    parser.add_argument("--exit-on-stdin-eof", action="store_true", help="Stop remote gateway when stdin closes")

    parser.add_argument("--camera-emit-every", type=int, default=1, help="Remote camera emit every N frames")
    parser.add_argument("--camera-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--camera-jpeg-quality", type=int, default=75)
    parser.add_argument("--camera-png-compression", type=int, default=3)

    parser.add_argument("--audio-emit-every", type=int, default=1, help="Remote audio emit every N frames")
    parser.add_argument("--audio-max-bytes", type=int, default=24576, help="Max remote PCM bytes per audio event")

    parser.add_argument("--max-list-items", type=int, default=0)
    parser.add_argument("--max-bytes", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=20)

    parser.add_argument("--ready-timeout", type=float, default=45.0, help="Seconds to wait for gateway ready")
    parser.add_argument("--response-timeout", type=float, default=20.0, help="Seconds to wait for command response")

    parser.add_argument("--no-gui", action="store_true", help="Disable OpenCV windows")
    parser.add_argument("--no-video-window", action="store_true", help="Do not show camera window")
    parser.add_argument("--no-lidar-window", action="store_true", help="Do not show lidar window")
    parser.add_argument("--disable-arrow-teleop", action="store_true", help="Disable keyboard teleop from GUI")

    parser.add_argument("--teleop-linear", type=float, default=0.30, help="Linear speed for arrow/W keys")
    parser.add_argument("--teleop-lateral", type=float, default=0.20, help="Lateral speed for Z/C keys")
    parser.add_argument("--teleop-yaw", type=float, default=0.80, help="Yaw speed for arrow/A/D keys")
    parser.add_argument("--teleop-refresh-hz", type=float, default=10.0, help="Refresh rate for Move command")
    parser.add_argument("--deadman-timeout", type=float, default=0.35, help="Seconds before StopMove")

    parser.add_argument("--lidar-window-size", type=int, default=640, help="Lidar viewer image size")
    parser.add_argument("--lidar-max-points", type=int, default=30000, help="Max lidar points rendered")

    parser.add_argument("--play-audio", action="store_true", help="Play remote audio stream locally")
    parser.add_argument(
        "--audio-default-sample-rate",
        type=int,
        default=48000,
        help="Fallback sample rate for audio playback",
    )
    parser.add_argument("--audio-queue-size", type=int, default=64, help="Audio playback queue size")

    parser.add_argument("--no-command-console", action="store_true", help="Disable stdin command console")

    parser.add_argument("--stats-every", type=float, default=5.0, help="Print local stats every N seconds (0 disables)")

    parser.add_argument("--print-camera-events", action="store_true", help="Print decoded camera event summaries")
    parser.add_argument("--print-topic-events", action="store_true", help="Print topic events")
    parser.add_argument("--print-audio-events", action="store_true", help="Print audio event summaries")
    parser.add_argument("--print-ssh-stderr", action="store_true", help="Print SSH stderr lines")

    args = parser.parse_args()

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

    if args.ssh_port <= 0 or args.ssh_port > 65535:
        parser.error("--ssh-port must be between 1 and 65535")
    if args.ssh_connect_timeout <= 0:
        parser.error("--ssh-connect-timeout must be > 0")
    if args.ready_timeout <= 0:
        parser.error("--ready-timeout must be > 0")
    if args.response_timeout <= 0:
        parser.error("--response-timeout must be > 0")

    if args.teleop_refresh_hz <= 0:
        parser.error("--teleop-refresh-hz must be > 0")
    if args.deadman_timeout <= 0:
        parser.error("--deadman-timeout must be > 0")
    if args.lidar_window_size < 200:
        parser.error("--lidar-window-size must be >= 200")
    if args.lidar_max_points < 0:
        parser.error("--lidar-max-points must be >= 0")

    if args.audio_default_sample_rate <= 0:
        parser.error("--audio-default-sample-rate must be > 0")
    if args.audio_queue_size <= 0:
        parser.error("--audio-queue-size must be > 0")

    if args.stats_every < 0:
        parser.error("--stats-every must be >= 0")

    return args


def main() -> None:
    args = parse_args()
    client = GatewayClient(args)

    exit_code = 0
    try:
        exit_code = client.run()
    except KeyboardInterrupt:
        print("[local] Interrupted by user", flush=True)
    finally:
        client.close()

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
