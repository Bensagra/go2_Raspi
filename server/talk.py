"""Authenticated browser-to-robot PCM relay with one speaker lease per robot."""
import asyncio
import base64
import contextlib
import json
import time
import uuid

from fastapi import HTTPException, Query, WebSocket
import paho.mqtt.client as mqtt

from robot_media_protocol import TALK_MAX_PACKET, TALK_MAX_SECONDS, TALK_RATE, TALK_IDLE_SECONDS


class TalkRelay:
    def __init__(self, runtime):
        self.runtime = runtime
        self.sessions = {}

    def on_status(self, robot_id, message):
        session = self.sessions.get(robot_id)
        if session and message.get("session_id") == session["id"]:
            with contextlib.suppress(asyncio.QueueFull):
                session["status"].put_nowait(message)

    def publish(self, robot_id, session_id, kind, **extra):
        rt = self.runtime
        if not rt._mqtt_connected():
            raise RuntimeError("MQTT disconnected")
        payload = {"type": kind, "session_id": session_id, "ts": time.time(), **extra}
        result = rt.mqtt_client.publish(rt._mqtt_topic(robot_id, "talk/in"), json.dumps(payload),
                                       qos=0 if kind == "pcm" else 1, retain=False)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError("could not send audio to the edge")

    def register(self, app):
        @app.websocket("/ws/talk/{robot_id}")
        async def talk(robot_id: str, ws: WebSocket, token: str = Query(default="")):
            try:
                auth = self.runtime._auth_from_token(token)
            except HTTPException:
                await ws.close(code=4401)
                return
            if auth["role"] not in {"operator", "admin"}:
                await ws.close(code=4403)
                return
            await ws.accept()
            if robot_id not in self.runtime._known_robots() or not self.runtime._control_link_status(robot_id)["ok"]:
                await ws.send_json({"status": "error", "error": "robot control link unavailable"})
                await ws.close(code=4410)
                return
            if robot_id in self.sessions:
                await ws.send_json({"status": "error", "error": "speaker busy"})
                await ws.close(code=4409)
                return
            session = {"id": uuid.uuid4().hex, "status": asyncio.Queue(maxsize=8)}
            self.sessions[robot_id] = session
            try:
                self.publish(robot_id, session["id"], "start")
                ready = await asyncio.wait_for(session["status"].get(), timeout=6)
                await ws.send_json(ready)
                if ready.get("status") != "ready":
                    return
                started = time.monotonic()
                sequence = total = 0
                while True:
                    remaining = TALK_MAX_SECONDS - (time.monotonic() - started)
                    if remaining <= 0:
                        break
                    # Edge errors/timeouts take priority over further audio.
                    if not session["status"].empty():
                        status = session["status"].get_nowait()
                        if status.get("status") in {"stopped", "error"}:
                            await ws.send_json(status)
                            return
                    message = await asyncio.wait_for(ws.receive(), timeout=min(TALK_IDLE_SECONDS, remaining))
                    if message["type"] == "websocket.disconnect":
                        return
                    if message.get("text") is not None:
                        control = json.loads(message["text"])
                        if isinstance(control, dict) and control.get("type") == "stop":
                            break
                        raise ValueError("unsupported audio control message")
                    raw = message.get("bytes")
                    if raw is None or not raw or len(raw) % 2 or len(raw) > TALK_MAX_PACKET:
                        raise ValueError("send binary PCM16 mono 16000 Hz, 2..3200 bytes per packet")
                    total += len(raw)
                    # At most 500 ms burst allowance; also caps total utterance bytes.
                    if total > min((time.monotonic() - started + 0.5) * TALK_RATE * 2,
                                   TALK_MAX_SECONDS * TALK_RATE * 2):
                        raise ValueError("audio rate exceeded")
                    self.publish(robot_id, session["id"], "pcm", sequence=sequence,
                                 pcm=base64.b64encode(raw).decode("ascii"))
                    sequence += 1
                await ws.send_json({"status": "stopped"})
            except asyncio.TimeoutError:
                with contextlib.suppress(Exception):
                    await ws.send_json({"status": "error", "error": "audio timeout"})
            except Exception as exc:
                with contextlib.suppress(Exception):
                    await ws.send_json({"status": "error", "error": str(exc)})
            finally:
                with contextlib.suppress(Exception):
                    self.publish(robot_id, session["id"], "stop")
                if self.sessions.get(robot_id) is session:
                    self.sessions.pop(robot_id, None)
                with contextlib.suppress(Exception):
                    await ws.close()
