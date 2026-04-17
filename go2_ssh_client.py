#!/usr/bin/env python3
import argparse
import contextlib
from collections import deque
import base64
import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


LEFT_KEYS = {81, 2424832, 65361}
UP_KEYS = {82, 2490368, 65362}
RIGHT_KEYS = {83, 2555904, 65363}
DOWN_KEYS = {84, 2621440, 65364}


def _print_json(prefix: str, payload: Any) -> None:
    print(f"{prefix}{json.dumps(payload, ensure_ascii=True)}", flush=True)


class GatewayClient:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.proc: Optional[subprocess.Popen[str]] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.running = False
        self.ready_event = threading.Event()

        self.pending_responses: Dict[str, queue.Queue[dict]] = {}
        self.id_counter = 1
        self.lock = threading.Lock()
        self.last_stderr_lines: deque[str] = deque(maxlen=30)
        self.last_error: str = ""

        self.frame_lock = threading.Lock()
        self.latest_camera_frame: Optional[np.ndarray] = None
        self.latest_lidar_image: Optional[np.ndarray] = None
        self.latest_lidar_point_count = 0

        self.camera_stream_enabled = bool(args.enable_camera)
        self.lidar_stream_enabled = bool(args.enable_lidar)

        self.gui_enabled = not args.no_gui
        self.show_video_window = self.gui_enabled and not args.no_video_window
        self.show_lidar_window = self.gui_enabled and not args.no_lidar_window
        self.arrow_teleop_enabled = self.gui_enabled and not args.disable_arrow_teleop
        self.pending_stop_at = 0.0

    def _next_id(self) -> str:
        with self.lock:
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
            "--max-list-items",
            str(self.args.max_list_items),
            "--max-depth",
            str(self.args.max_depth),
        ]

        if self.args.enable_camera:
            cmd.append("--enable-camera")
        if self.args.enable_lidar:
            cmd.append("--enable-lidar")
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

        ssh_cmd: List[str] = [
            "ssh",
            "-T",
        ]

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
        self.ready_event.clear()
        self.last_error = ""

        self.reader_thread = threading.Thread(target=self._stdout_reader, daemon=True)
        self.reader_thread.start()

        threading.Thread(target=self._stderr_reader, daemon=True).start()

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

            msg_type = message.get("type")
            if msg_type == "response":
                cmd_id = str(message.get("id"))
                q = self.pending_responses.get(cmd_id)
                if q is not None:
                    q.put(message)
                _print_json("[gateway/response] ", message)
            elif msg_type == "event":
                stream = message.get("stream")
                if stream == "camera" and not self.args.print_camera_events:
                    continue
                _print_json("[gateway/event] ", message)
            elif msg_type == "status":
                if message.get("event") == "ready":
                    self.ready_event.set()
                _print_json("[gateway/status] ", message)
            elif msg_type == "error":
                _print_json("[gateway/error] ", message)
            else:
                _print_json("[gateway/msg] ", message)

        self.running = False

    def _stderr_reader(self) -> None:
        assert self.proc is not None
        assert self.proc.stderr is not None

        for line in self.proc.stderr:
            self.last_stderr_lines.append(line.rstrip("\n"))
            sys.stderr.write(f"[ssh/stderr] {line}")
            sys.stderr.flush()

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
            # Prevent false timeouts while gateway is still connecting to WebRTC.
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
            self.pending_responses[cmd_id] = response_queue

        self.proc.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.proc.stdin.flush()

        if not wait_response:
            return None

        assert response_queue is not None
        try:
            return response_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        finally:
            self.pending_responses.pop(cmd_id, None)

    def close(self) -> None:
        if self.proc is None:
            return

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
        return {"op": "set_video", "enabled": parts[1].lower() in {"1", "on", "true", "yes"}}

    if head == "lidar" and len(parts) >= 2:
        return {"op": "set_lidar", "enabled": parts[1].lower() in {"1", "on", "true", "yes"}, "subscribe": True}

    if head == "sport" and len(parts) >= 2:
        cmd: Dict[str, Any] = {"op": "sport", "action": parts[1]}
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

    raise ValueError("Unknown command. Use: help, status, temps, sub, sport, motion, video, lidar, request, exit")


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
    print("  sport Hello                        -> accion SPORT_CMD", flush=True)
    print("  sport Move {\"x\":0.2,\"y\":0,\"z\":0} -> move", flush=True)
    print("  request MOTION_SWITCHER 1001       -> request generico", flush=True)
    print("  { ...json... }                     -> enviar JSON crudo", flush=True)
    print("  exit                               -> cerrar", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive SSH client for go2_ssh_gateway.py. "
            "Run this on your server to read telemetry/camera/lidar and send actions concurrently."
        )
    )
    parser.add_argument("--remote", required=True, help="SSH target, for example user@pc-local")
    parser.add_argument("--ssh-port", type=int, default=22, help="SSH port")
    parser.add_argument(
        "--identity-file",
        help="SSH private key file (recommended for Raspberry)",
    )
    parser.add_argument(
        "--ssh-password",
        help="SSH password (requires sshpass installed)",
    )
    parser.add_argument(
        "--strict-host-key-checking",
        choices=["yes", "no", "accept-new"],
        default="accept-new",
        help="SSH StrictHostKeyChecking value",
    )
    parser.add_argument(
        "--ssh-connect-timeout",
        type=int,
        default=15,
        help="SSH connect timeout in seconds",
    )

    parser.add_argument(
        "--remote-python",
        default="/home/bensagra/Documents/go2/.venv/bin/python",
        help="Python executable on remote PC",
    )
    parser.add_argument(
        "--remote-gateway",
        default="/home/bensagra/Documents/go2/go2_ssh_gateway.py",
        help="Path to go2_ssh_gateway.py on remote PC",
    )
    parser.add_argument("--go2-ip", default="192.168.123.161", help="Go2 IP in STA mode")

    parser.add_argument("--subscribe-profile", action="append", default=["core"], help="Gateway subscribe profile")
    parser.add_argument("--subscribe-topic", action="append", default=[], help="Gateway extra subscribe topic")
    parser.add_argument("--enable-camera", action="store_true", help="Enable camera at startup")
    parser.add_argument("--enable-lidar", action="store_true", help="Enable lidar at startup")
    parser.add_argument("--disable-traffic-saving", action="store_true", help="Enable disableTrafficSaving")
    parser.add_argument("--exit-on-stdin-eof", action="store_true", help="Stop remote gateway when stdin closes")

    parser.add_argument("--camera-emit-every", type=int, default=1, help="Remote camera emit every N frames")
    parser.add_argument("--camera-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--camera-jpeg-quality", type=int, default=75)
    parser.add_argument("--camera-png-compression", type=int, default=3)

    parser.add_argument("--max-list-items", type=int, default=0)
    parser.add_argument("--max-bytes", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=45.0,
        help="Seconds to wait for gateway ready status",
    )
    parser.add_argument(
        "--response-timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for each command response",
    )

    parser.add_argument(
        "--print-camera-events",
        action="store_true",
        help="Print camera event lines in stdout (can be very verbose)",
    )

    args = parser.parse_args()

    if args.camera_emit_every <= 0:
        parser.error("--camera-emit-every must be > 0")
    if args.camera_jpeg_quality < 1 or args.camera_jpeg_quality > 100:
        parser.error("--camera-jpeg-quality must be between 1 and 100")
    if args.camera_png_compression < 0 or args.camera_png_compression > 9:
        parser.error("--camera-png-compression must be between 0 and 9")
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

    return args


def main() -> None:
    args = parse_args()
    client = GatewayClient(args)

    try:
        client.start()

        if not client.wait_until_ready(args.ready_timeout):
            print(f"[local/error] {client.last_error}", flush=True)
            raise SystemExit(1)

        print_usage()

        while client.running:
            try:
                line = input("go2> ")
            except EOFError:
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

            wait_response = payload.get("op") in {
                "help",
                "status",
                "list_topics",
                "list_sport_cmd",
                "get_latest",
                "get_temperatures",
                "request",
                "sport",
                "set_motion_mode",
                "get_motion_mode",
                "set_video",
                "set_camera_stream",
                "set_lidar",
                "set_lidar_decoder",
                "subscribe",
                "unsubscribe",
                "subscribe_profile",
            }

            response = client.send(payload, wait_response=wait_response, timeout=args.response_timeout)
            if wait_response and response is None:
                print("[local/warn] timeout waiting response", flush=True)

            if payload.get("op") in {"exit", "quit", "stop"}:
                break

            time.sleep(0.02)

    finally:
        client.close()


if __name__ == "__main__":
    main()
