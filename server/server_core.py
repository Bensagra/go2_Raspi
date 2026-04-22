#!/usr/bin/env python3
import argparse
import asyncio
import contextlib
import json
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import paho.mqtt.client as mqtt
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


class CoreRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.stop_event = asyncio.Event()
        self.heartbeat_task: Optional[asyncio.Task[None]] = None

        self.app = FastAPI(title="Go2 Server Core", version="0.1.0")
        self._setup_cors()
        self._setup_routes()

        self.mqtt_client: Optional[mqtt.Client] = None

        self.frontend_sockets: Set[WebSocket] = set()
        self.edge_media_sockets: Dict[str, WebSocket] = {}

        self.latest_telemetry: Dict[str, Dict[str, Any]] = {}
        self.latest_media: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

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
        if not self.frontend_sockets:
            return

        text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        stale: List[WebSocket] = []
        for ws in self.frontend_sockets:
            try:
                await ws.send_text(text)
            except Exception:
                stale.append(ws)

        for ws in stale:
            self.frontend_sockets.discard(ws)

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

        self.mqtt_client.connect(self.args.mqtt_host, self.args.mqtt_port, keepalive=30)
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
            self.ack_history[robot_id].append(payload)
            command_id = str(payload.get("command_id", ""))
            if command_id in self.pending_commands:
                self.pending_commands.pop(command_id, None)

            await self._broadcast({
                "type": "command_ack",
                "robot_id": robot_id,
                "data": payload,
            })
            self._audit("command_ack", {"robot_id": robot_id, "data": payload})
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

    def _sanitize_command(self, command_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        output = dict(payload)

        if command_type == "move":
            output["linear_x"] = clamp(float(output.get("linear_x", 0.0)), -self.args.max_linear_speed, self.args.max_linear_speed)
            output["lateral_y"] = clamp(float(output.get("lateral_y", 0.0)), -self.args.max_lateral_speed, self.args.max_lateral_speed)
            output["angular_z"] = clamp(float(output.get("angular_z", 0.0)), -self.args.max_angular_speed, self.args.max_angular_speed)
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

        if command_type == "set_lidar_decoder":
            decoder = str(output.get("decoder", "libvoxel")).strip().lower()
            if decoder not in {"libvoxel", "native"}:
                raise HTTPException(status_code=400, detail="set_lidar_decoder supports only 'libvoxel' or 'native'")
            output["decoder"] = decoder

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
            self._setup_mqtt()
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._audit("server_start", {"pid": str(uuid.uuid4())})

        @app.on_event("shutdown")
        async def _shutdown() -> None:
            self.stop_event.set()
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.heartbeat_task

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
            }

        @app.get("/api/robots")
        async def list_robots(auth: Dict[str, str] = Depends(self._auth_dependency)) -> Dict[str, Any]:
            return {
                "robots": self._known_robots(),
            }

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
                    "max_linear_speed": self.args.max_linear_speed,
                    "max_lateral_speed": self.args.max_lateral_speed,
                    "max_angular_speed": self.args.max_angular_speed,
                    "default_move_duration_ms": self.args.default_move_duration_ms,
                    "command_rate_max": self.args.command_rate_max,
                    "command_rate_window_s": self.args.command_rate_window_s,
                },
            }

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
            sanitized_payload = self._sanitize_command(command_type, command.payload)

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
            self.frontend_sockets.add(ws)

            await ws.send_text(
                json.dumps(
                    {
                        "type": "hello",
                        "ts": time.time(),
                        "user": auth,
                        "known_robots": self._known_robots(),
                    },
                    ensure_ascii=True,
                )
            )

            for robot_id, telemetry in self.latest_telemetry.items():
                await ws.send_text(
                    json.dumps(
                        {"type": "telemetry", "robot_id": robot_id, "data": telemetry},
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                )

            for media_robot_id, media_streams in self.latest_media.items():
                for stream, data in media_streams.items():
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "media",
                                "robot_id": media_robot_id,
                                "stream": stream,
                                "data": data,
                                "ts": time.time(),
                            },
                            ensure_ascii=True,
                            separators=(",", ":"),
                        )
                    )

            try:
                while True:
                    text = await ws.receive_text()
                    with contextlib.suppress(Exception):
                        message = json.loads(text)
                        if message.get("op") == "heartbeat":
                            robot_id = str(message.get("robot_id", "")).strip()
                            if robot_id:
                                self.last_control_activity[robot_id] = time.time()
            except WebSocketDisconnect:
                pass
            finally:
                self.frontend_sockets.discard(ws)

        @app.websocket("/ws/edge-media/{robot_id}")
        async def ws_edge_media(robot_id: str, ws: WebSocket, token: str = Query(default="")) -> None:
            if token != self.args.edge_media_token:
                await ws.close(code=4403)
                return

            await ws.accept()
            self.edge_media_sockets[robot_id] = ws
            self._audit("edge_media_connected", {"robot_id": robot_id})

            try:
                while True:
                    text = await ws.receive_text()
                    with contextlib.suppress(Exception):
                        payload = json.loads(text)
                        stream = str(payload.get("stream", "")).strip() or "unknown"
                        data = payload.get("data", {})

                        if isinstance(data, dict):
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

    parser.add_argument("--audit-log", default="./server/audit/server_audit.jsonl")
    parser.add_argument("--replay-max-items", type=int, default=2000)

    parser.add_argument("--command-rate-max", type=int, default=120)
    parser.add_argument("--command-rate-window-s", type=float, default=10.0)

    parser.add_argument("--max-linear-speed", type=float, default=0.45)
    parser.add_argument("--max-lateral-speed", type=float, default=0.25)
    parser.add_argument("--max-angular-speed", type=float, default=1.0)
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

    if args.heartbeat_publish_interval_s <= 0:
        parser.error("--heartbeat-publish-interval-s must be > 0")

    if args.control_session_timeout_s <= 0:
        parser.error("--control-session-timeout-s must be > 0")

    return args


def main() -> None:
    args = parse_args()
    runtime = CoreRuntime(args)

    uvicorn.run(runtime.app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
