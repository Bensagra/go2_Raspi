"""PTT authorization, lifetime, PCM transport, and robot command regressions."""
import asyncio
import base64
import contextlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI, HTTPException
from edge.talk import EdgeTalk, TalkTrack
from edge.edge_gateway_service import EdgeGatewayService, parse_args as edge_args
from robot_media_protocol import flashlight_payload, require_robot_success
from server.talk import TalkRelay
from server.server_core import CoreRuntime, CommandIn, parse_args


class FakeSocket:
    def __init__(self):
        self.messages = asyncio.Queue()
        self.sent = []
        self.closed = None

    async def accept(self):
        pass

    async def receive(self):
        return await self.messages.get()

    async def send_json(self, message):
        self.sent.append(message)

    async def close(self, code=1000):
        self.closed = code


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.005)


class TalkTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runtime = SimpleNamespace(
            _mqtt_connected=lambda: True,
            _mqtt_topic=lambda robot, suffix: f"go2/{robot}/{suffix}",
            _known_robots=lambda: ["dog"],
            _control_link_status=lambda robot: {"ok": True},
            _auth_from_token=lambda token: {"role": token},
        )
        self.relay = TalkRelay(self.runtime)
        self.gateway = SimpleNamespace(
            conn=SimpleNamespace(pc=SimpleNamespace(connectionState="connected")),
            _mqtt_publish=lambda topic, message, **kw: self.relay.on_status("dog", message),
            _publish_event=Mock(),
        )
        self.edge = EdgeTalk(self.gateway)
        self.edge.track = TalkTrack()
        self.published = []

        def publish(topic, payload, **kwargs):
            message = json.loads(payload)
            self.published.append((message, kwargs))
            self.edge.enqueue(message)
            return SimpleNamespace(rc=0)

        self.runtime.mqtt_client = SimpleNamespace(publish=publish)
        app = FastAPI()
        self.relay.register(app)
        self.endpoint = app.routes[-1].endpoint
        self.worker = asyncio.create_task(self.edge.run())

    async def asyncTearDown(self):
        self.worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.worker
        await self.edge.stop()
        self.edge.track.stop()

    async def test_live_pcm_is_sent_and_stop_discards_audio(self):
        ws = FakeSocket()
        task = asyncio.create_task(self.endpoint("dog", ws, "operator"))
        await until(lambda: bool(ws.sent))
        self.assertEqual(ws.sent[0]["status"], "ready")
        pcm = b"\x34\x12" * 640
        await ws.messages.put({"type": "websocket.receive", "bytes": pcm})
        await until(lambda: len(self.edge.track.buffer) == len(pcm))
        frame = await self.edge.track.recv()
        self.assertEqual(bytes(frame.planes[0]), pcm[:640])
        self.assertEqual(frame.sample_rate, 16000)
        await ws.messages.put({"type": "websocket.receive", "text": '{ "type": "stop" }'})
        await task
        await until(lambda: self.edge.session is None)
        self.assertFalse(self.edge.track.active)
        self.assertEqual(self.edge.track.buffer, b"")
        self.assertEqual(bytes((await self.edge.track.recv()).planes[0]), bytes(640))
        self.assertEqual(self.relay.sessions, {})
        self.gateway._publish_event.assert_not_called()
        audio = next((p, options) for p, options in self.published if p["type"] == "pcm")
        self.assertEqual(audio[1], {"qos": 0, "retain": False})

    async def test_viewer_unknown_robot_and_concurrent_speaker_denied(self):
        viewer = FakeSocket()
        await self.endpoint("dog", viewer, "viewer")
        self.assertEqual(viewer.closed, 4403)
        unknown = FakeSocket()
        await self.endpoint("other", unknown, "admin")
        self.assertEqual(unknown.closed, 4410)
        ws = FakeSocket()
        task = asyncio.create_task(self.endpoint("dog", ws, "operator"))
        await until(lambda: bool(ws.sent))
        other = FakeSocket()
        await self.endpoint("dog", other, "admin")
        self.assertEqual(other.closed, 4409)
        await ws.messages.put({"type": "websocket.disconnect"})
        await task
        await until(lambda: self.edge.session is None)

    async def test_bad_pcm_rejected_and_session_released(self):
        for raw in (b"x", bytes(3202), b""):
            ws = FakeSocket()
            task = asyncio.create_task(self.endpoint("dog", ws, "operator"))
            await until(lambda: bool(ws.sent))
            await ws.messages.put({"type": "websocket.receive", "bytes": raw})
            await task
            self.assertTrue(any(m["status"] == "error" for m in ws.sent))
            await until(lambda: self.edge.session is None)

    async def test_edge_timeout_duplicates_stale_and_cross_session_packets(self):
        def packet(kind, **extra):
            return {"type": kind, "session_id": "one", "ts": time.time(), **extra}
        await self.edge.handle(packet("start"))
        deadline = self.edge.deadline
        await self.edge.handle(packet("start"))
        self.assertEqual(self.edge.deadline, deadline)
        pcm = base64.b64encode(b"\x01\x02" * 320).decode()
        await self.edge.handle(packet("pcm", sequence=0, pcm=pcm, session_id="other"))
        await self.edge.handle(packet("pcm", sequence=0, pcm=pcm, ts=time.time() - 5))
        self.assertEqual(self.edge.track.buffer, b"")
        await self.edge.handle(packet("pcm", sequence=0, pcm=pcm))
        await self.edge.handle(packet("pcm", sequence=0, pcm=pcm))
        self.assertEqual(len(self.edge.track.buffer), 640)
        self.edge.deadline = time.monotonic() - 1
        await until(lambda: self.edge.session is None)
        await self.edge.handle(packet("start"))
        self.assertIsNone(self.edge.session)  # late QoS duplicate cannot reopen mic

    async def test_idle_disconnect_watchdog(self):
        await self.edge.handle({"type": "start", "session_id": "idle", "ts": time.time()})
        self.edge.last_packet = time.monotonic() - 3
        await until(lambda: self.edge.session is None)
        self.assertFalse(self.edge.track.active)


class RobotMediaCommandTests(unittest.IsolatedAsyncioTestCase):
    def test_safety_starts_off_and_explicit_enable_still_works(self):
        with patch("sys.argv", ["edge"]):
            self.assertFalse(edge_args().enable_safety_guard)
        with patch("sys.argv", ["edge", "--enable-safety-guard"]):
            self.assertTrue(edge_args().enable_safety_guard)

    def test_flashlight_payload_rejects_invalid_values(self):
        self.assertEqual(flashlight_payload({"enabled": True}), {"brightness": 10})
        self.assertEqual(flashlight_payload({"brightness": 0}), {"brightness": 0})
        for value in (-1, 11, True, "10", 1.5, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                flashlight_payload({"brightness": value})
        with self.assertRaises(RuntimeError):
            require_robot_success({"data": {"header": {"status": {"code": 4}}}})

    async def test_light_confirms_correct_api_and_keeps_state_on_failure(self):
        gateway = object.__new__(EdgeGatewayService)
        gateway.flashlight_brightness = None
        gateway._robot_request = AsyncMock(return_value={"data": {"header": {"status": {"code": 0}}}})
        result = await gateway._execute_command({"type": "set_flashlight", "payload": {"brightness": 7}})
        self.assertEqual(result["brightness"], 7)
        self.assertEqual(gateway._robot_request.call_args.args[1:], (1005, {"brightness": 7}))
        gateway._robot_request.return_value["data"]["header"]["status"]["code"] = 5
        with self.assertRaises(RuntimeError):
            await gateway._execute_command({"type": "set_flashlight", "payload": {"enabled": False}})
        self.assertEqual(gateway.flashlight_brightness, 7)

    async def test_http_light_authorization_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            rt = CoreRuntime(parse_args(["--disable-perception", "--map-storage-dir", directory,
                                         "--audit-log", str(Path(directory) / "audit.jsonl")]))
            rt.mqtt_client = Mock()
            rt.mqtt_client.is_connected.return_value = True
            rt.mqtt_client.publish.return_value = SimpleNamespace(rc=0)
            endpoint = next(r.endpoint for r in rt.app.routes if getattr(r, "path", "") ==
                            "/api/robots/{robot_id}/commands")
            command = CommandIn(type="set_flashlight", payload={"enabled": True})
            with self.assertRaises(HTTPException) as denied:
                await endpoint("dog", command, {"role": "viewer", "user_id": "v"})
            self.assertEqual(denied.exception.status_code, 403)
            response = await endpoint("dog", command, {"role": "operator", "user_id": "o"})
            self.assertEqual(response["status"], "queued")
            self.assertEqual(json.loads(rt.mqtt_client.publish.call_args.args[1])["payload"], {"brightness": 10})
            with self.assertRaises(HTTPException) as bad:
                await endpoint("dog", CommandIn(type="set_flashlight", payload={"brightness": 15}),
                               {"role": "operator", "user_id": "o"})
            self.assertEqual(bad.exception.status_code, 400)
