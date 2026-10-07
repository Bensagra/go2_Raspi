"""Test the exact TIC entrypoint, isolated from MQTT and real robots."""

import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from server.hosting import hosting_args


class HostingConfigTests(unittest.TestCase):
    def test_hosting_paths_and_tokens_replace_local_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            args = hosting_args({
                "DATA_DIR": directory,
                "MQTT_HOST": "broker.example",
                "MQTT_PORT": "8883",
                "MQTT_TLS": "true",
                "MQTT_PASSWORD": "--a-password-with-leading-dashes",
                "API_TOKENS": '["secret:admin:admin_01"]',
                "EDGE_MEDIA_TOKEN": "edge-secret",
                "ROBOT_IDS": "go2_02, go2_03",
                "SERVER_ARGS": '["--map-storage-dir","/read-only","--mesh-interval-s","90"]',
            })
            self.assertEqual(args.api_token, ["secret:admin:admin_01"])
            self.assertEqual(args.edge_media_token, "edge-secret")
            self.assertEqual(args.robot_id, ["go2_02", "go2_03"])
            self.assertEqual(args.map_storage_dir, str(Path(directory).resolve() / "maps"))
            self.assertEqual(args.faces_dir, str(Path(directory).resolve() / "faces"))
            self.assertEqual(args.mission_storage_dir, str(Path(directory).resolve() / "missions"))
            self.assertEqual(args.mqtt_port, 8883)
            self.assertTrue(args.mqtt_tls)
            self.assertEqual(args.mqtt_password, "--a-password-with-leading-dashes")
            self.assertEqual(args.mesh_interval_s, 90)

    def test_unconfigured_hosting_does_not_enable_development_credentials(self):
        args = hosting_args({})
        self.assertEqual(args.api_token, [])
        self.assertEqual(args.edge_media_token, "")

    def test_bad_env_is_rejected_without_echoing_credentials(self):
        for env in (
            {"API_TOKENS": "not-json"},
            {"API_TOKENS": '["secret:root:user"]'},
            {"API_TOKENS": '["secret:admin:user", "secret:viewer:other"]'},
            {"SERVER_ARGS": '"--disable-perception"'},
            {"MQTT_TLS": "maybe"},
        ):
            with self.subTest(env=env), self.assertRaises(ValueError) as error:
                hosting_args(env)
            self.assertNotIn("secret", str(error.exception))


class HostingEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.data = cls.base / "persistent"
        cls.readonly = cls.base / "app"
        cls.readonly.mkdir()
        cls.readonly.chmod(0o555)
        cls.log = (cls.base / "server.log").open("w+")
        cls.addClassCleanup(cls.cleanup_server)
        # Hold an unused TCP port open without listening: MQTT cannot reach a
        # real broker even if the developer has Mosquitto running locally.
        cls.mqtt_guard = socket.socket()
        cls.mqtt_guard.bind(("127.0.0.1", 0))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            cls.port = sock.getsockname()[1]
        cls.url = f"http://127.0.0.1:{cls.port}"
        cls.ws_url = f"ws://127.0.0.1:{cls.port}"
        env = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "DATA_DIR": str(cls.data),
            "MQTT_HOST": "127.0.0.1",
            "MQTT_PORT": str(cls.mqtt_guard.getsockname()[1]),
            "MQTT_TLS": "false",
            "MQTT_USERNAME": "",
            "MQTT_PASSWORD": "",
            "API_TOKENS": '["test-operator:operator:test","test-viewer:viewer:test-viewer"]',
            "EDGE_MEDIA_TOKEN": "test-edge",
            "ROBOT_IDS": "test_robot",
            "SERVER_ARGS": '["--mesh-interval-s","0"]',
        }
        cls.process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(cls.port)],
            cwd=cls.readonly, env=env, stdout=cls.log, stderr=subprocess.STDOUT,
        )
        # First imports of newly installed native wheels can be slow on macOS.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if cls.process.poll() is not None:
                break
            try:
                with urlopen(cls.url + "/health", timeout=0.5) as response:
                    if response.status == 200:
                        return
            except (URLError, TimeoutError):
                time.sleep(0.1)
        cls.log.seek(0)
        raise AssertionError("uvicorn main:app failed to start: " + cls.log.read())

    @classmethod
    def cleanup_server(cls):
        process = getattr(cls, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if hasattr(cls, "mqtt_guard"):
            cls.mqtt_guard.close()
        cls.log.close()
        cls.readonly.chmod(0o755)
        cls.temp.cleanup()

    def request(self, path, token=None, method="GET", payload=None, headers=None):
        headers = dict(headers or {})
        if token:
            headers["Authorization"] = "Bearer " + token
        body = None
        if payload is not None:
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        req = Request(self.url + path, data=body, headers=headers, method=method)
        try:
            response = urlopen(req, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            return response.status, response.read()

    def test_health_auth_and_existing_routes(self):
        status, raw = self.request("/health")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(raw)["ok"])
        self.assertTrue((self.data / "audit/server_audit.jsonl").exists())
        self.assertTrue((self.data / "faces").is_dir())
        self.assertEqual(self.request("/api/robots")[0], 401)
        self.assertEqual(self.request("/api/robots", "dev-operator-token")[0], 401)
        status, raw = self.request("/api/robots", "test-operator")
        self.assertEqual(status, 200)
        self.assertIn("test_robot", json.loads(raw)["robots"])
        self.assertEqual(self.request(
            "/api/robots/test_robot/commands", "test-viewer", "POST", {"type": "move"},
        )[0], 403)

    def test_offline_mqtt_is_reported_and_commands_are_not_accepted(self):
        status, raw = self.request("/health")
        self.assertFalse(json.loads(raw)["mqtt_connected"])
        status, raw = self.request("/api/robots/test_robot/commands", "test-operator", "POST", {"type": "move"})
        self.assertEqual(status, 503)
        status, raw = self.request("/api/robots/test_robot/state", "test-operator")
        state = json.loads(raw)
        self.assertFalse(state["control_link"]["mqtt_connected"])
        self.assertEqual(state["pending_commands"], [])

        async def scenario():
            from websockets.asyncio.client import connect
            async with connect(self.ws_url + "/ws/live?token=test-operator") as viewer:
                await viewer.send(json.dumps({"op": "drive", "robot_id": "test_robot", "sequence": 1, "payload": {"linear_x": 0.1}}))
                while True:
                    packet = await asyncio.wait_for(viewer.recv(), 3)
                    if isinstance(packet, str):
                        message = json.loads(packet)
                        if message.get("type") == "drive_status":
                            self.assertFalse(message["ok"])
                            break
        asyncio.run(scenario())

    def test_mission_recording_survives_viewer_disconnect_and_downloads(self):
        import io
        import zipfile
        import av
        import cv2
        import numpy as np
        from server.server_core import encode_media_frame, decode_media_frame, encode_cloud_payload
        from termica.protocol import encode_csv

        endpoint = "/api/robots/mission_robot/missions"
        self.assertEqual(self.request(endpoint, "test-viewer", "POST", {"name": "Denied"})[0], 403)
        status, raw = self.request(endpoint, "test-operator", "POST", {"name": "Laboratorio"})
        self.assertEqual(status, 201, raw)
        mission_id = json.loads(raw)["mission_id"]
        self.assertEqual(self.request(endpoint, "test-operator", "POST", {"name": "Duplicate"})[0], 409)
        self.assertEqual(self.request(f"/api/missions/{mission_id}/files/mission.zip", "test-operator")[0], 409)
        self.assertEqual(self.request(f"/api/missions/{mission_id}/playback", "test-viewer", "POST")[0], 409)

        async def scenario():
            from websockets.asyncio.client import connect
            async with connect(self.ws_url + "/ws/edge-media/mission_robot?token=test-edge") as edge:
                async with connect(self.ws_url + "/ws/live?token=test-viewer") as viewer:
                    image = np.full((120, 160, 3), 110, np.uint8)
                    jpeg = cv2.imencode(".jpg", image)[1].tobytes()
                    for seq in range(1, 4):
                        await edge.send(encode_media_frame({"stream": "video", "image_format": "jpeg", "ts": time.time()}, jpeg))
                        await edge.send(encode_media_frame({"stream": "arducam", "image_format": "jpg",
                            "width": 160, "height": 120, "ts": time.time(), "seq": seq, "session_id": "csi"}, jpeg))
                        frame = np.full((120, 160), 22, np.float32)
                        frame[30:70, 60:90] = 34
                        await edge.send(encode_media_frame({"stream": "thermal_csv", "format": "csv_zlib", "unit": "celsius",
                            "width": 160, "height": 120, "ts": time.time(), "seq": seq, "session_id": "mission"}, encode_csv(frame)))
                        seen = set()
                        while True:
                            packet = await asyncio.wait_for(viewer.recv(), 3)
                            if isinstance(packet, bytes):
                                header, _ = decode_media_frame(packet)
                                if header.get("robot_id") == "mission_robot":
                                    seen.add(header.get("stream"))
                                if {"thermal", "arducam"} <= seen:
                                    break
                # No browser remains; recording must continue for both cameras.
                await edge.send(encode_media_frame({"stream": "arducam", "image_format": "jpg",
                    "width": 160, "height": 120, "ts": time.time(), "seq": 4, "session_id": "csi"}, jpeg))
                await edge.send(encode_media_frame({"stream": "video", "image_format": "jpeg", "ts": time.time()}, jpeg))
                points = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
                blob, fmt, scale, offset, count = encode_cloud_payload(points, None, 2)
                await edge.send(encode_media_frame({"stream": "lidar", "fmt": fmt, "scale": scale, "offset": offset,
                                                    "count": count, "ts": time.time()}, blob))
                deadline = time.monotonic() + 4
                while time.monotonic() < deadline:
                    _, data = self.request(f"/api/missions/{mission_id}", "test-viewer")
                    current = json.loads(data)
                    if (current["streams"]["camera"]["frames"] == 4 and current["lidar_points"] == 2
                            and current["streams"]["arducam"]["frames"] == 4):
                        break
                    await asyncio.sleep(0.05)
                self.assertEqual(current["status"], "recording")
                self.assertEqual(current["streams"]["camera"]["frames"], 4)
        asyncio.run(scenario())
        self.assertEqual(self.request(f"/api/missions/{mission_id}/stop", "test-viewer", "POST")[0], 403)
        status, raw = self.request(f"/api/missions/{mission_id}/stop", "test-operator", "POST")
        result = json.loads(raw)
        self.assertEqual(status, 200, raw)
        self.assertEqual(result["status"], "completed", raw)
        self.assertEqual(result["missing_streams"], [])
        self.assertTrue((self.data / "missions" / mission_id / "camera.mp4").is_file())
        self.assertEqual(self.request(f"/api/missions/{mission_id}/playback", method="POST")[0], 401)
        status, raw = self.request(f"/api/missions/{mission_id}/playback", "test-viewer", "POST")
        self.assertEqual(status, 200, raw)
        playback = json.loads(raw)
        self.assertEqual(set(playback["videos"]), {"camera", "arducam", "thermal"})
        self.assertEqual(playback["videos"]["camera"]["offset_s"], result["streams"]["camera"]["first_at_s"])
        video_path = playback["videos"]["camera"]["path"]
        video_bytes = (self.data / "missions" / mission_id / "camera.mp4").read_bytes()
        # Native HTML video players seek with byte ranges, without a JS blob download.
        req = Request(self.url + video_path, headers={"Range": "bytes=0-127", "Origin": "https://frontend.example"})
        with urlopen(req) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.headers["Content-Type"], "video/mp4")
            self.assertTrue(response.headers["Content-Disposition"].startswith("inline"))
            self.assertEqual(response.headers["Content-Range"], f"bytes 0-127/{len(video_bytes)}")
            self.assertEqual(response.read(), video_bytes[:128])
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")
        self.assertEqual(self.request(video_path, headers={"Range": "bytes=-64"})[1], video_bytes[-64:])
        self.assertEqual(self.request(video_path, headers={"Range": f"bytes={len(video_bytes)+1}-"})[0], 416)
        self.assertEqual(self.request(video_path, method="HEAD"), (200, b""))
        self.assertEqual(self.request(video_path.replace("camera.mp4", "thermal.mp4"))[0], 401)
        status, raw = self.request(f"/api/missions/{mission_id}/map", "test-viewer")
        self.assertEqual(status, 200, raw)
        self.assertEqual(json.loads(raw)["point_count"], 2)
        status, raw = self.request(f"/api/missions/{mission_id}/download/mission.zip", "test-viewer", "POST")
        self.assertEqual(status, 200, raw)
        download_path = json.loads(raw)["path"]
        status, raw = self.request(download_path)
        self.assertEqual(status, 200)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            with av.open(io.BytesIO(archive.read("camera.mp4"))) as video:
                self.assertEqual(len(list(video.decode(video=0))), 4)
            with av.open(io.BytesIO(archive.read("arducam.mp4"))) as video:
                self.assertEqual(len(list(video.decode(video=0))), 4)
            detections = [json.loads(line) for line in archive.read("thermal_detections.jsonl").splitlines()]
            self.assertTrue(detections[-1]["detection"]["person_present"])
        self.assertEqual(self.request(download_path.replace("mission.zip", "camera.mp4"))[0], 401)
        self.assertEqual(self.request(f"/api/missions/{mission_id}/files/camera.mp4")[0], 401)

    def test_docs_under_tic_prefix(self):
        status, raw = self.request("/docs", headers={"X-Forwarded-Prefix": "/project/server"})
        self.assertEqual(status, 200)
        self.assertIn(b"/project/server/openapi.json", raw)
        status, raw = self.request("/openapi.json", headers={"X-Forwarded-Prefix": "/project/server"})
        self.assertEqual(status, 200)
        schema = json.loads(raw)
        self.assertIn("/api/robots/{robot_id}/commands", schema["paths"])
        self.assertIn({"url": "/project/server"}, schema["servers"])

    def test_thermal_csv_returns_jpeg_on_existing_live_socket(self):
        async def scenario():
            import cv2
            import numpy as np
            from websockets.asyncio.client import connect
            from server.server_core import decode_media_frame, encode_media_frame
            from termica.protocol import encode_csv

            async def thermal_frame(viewer):
                while True:
                    packet = await asyncio.wait_for(viewer.recv(), 3)
                    if isinstance(packet, bytes):
                        header, payload = decode_media_frame(packet)
                        if header.get("stream") == "thermal":
                            return header, payload

            async with connect(self.ws_url + "/ws/live?token=test-viewer") as viewer:
                hello = json.loads(await asyncio.wait_for(viewer.recv(), 3))
                self.assertEqual(hello["type"], "hello")
                async with connect(self.ws_url + "/ws/edge-media/test_robot?token=test-edge") as edge:
                    frame = np.full((120, 160), 22, dtype=np.float32)
                    frame[30:70, 60:90] = 34
                    for seq in range(1, 4):
                        await edge.send(encode_media_frame({
                            "stream": "thermal_csv", "format": "csv_zlib", "unit": "celsius",
                            "width": 160, "height": 120, "ts": time.time(),
                            "session_id": "integration", "seq": seq,
                            "robot_id": "untrusted-id",
                        }, encode_csv(frame)))
                        header, jpeg = await thermal_frame(viewer)
                        self.assertEqual(header["robot_id"], "test_robot")
                        self.assertEqual(header["seq"], seq)
                        self.assertEqual(header["detection"]["person_present"], seq == 3)
                        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                        self.assertEqual(image.shape, (480, 360, 3))
                        self.assertEqual(header["rotation_deg"], 270)
                    # Replacing an edge connection must reset temporal detection.
                    async with connect(self.ws_url + "/ws/edge-media/test_robot?token=test-edge") as replacement:
                        await replacement.send(encode_media_frame({
                            "stream": "thermal_csv", "format": "csv_zlib", "unit": "celsius",
                            "width": 160, "height": 120, "ts": time.time(),
                            "session_id": "integration", "seq": 4,
                        }, encode_csv(frame)))
                        header, _ = await thermal_frame(viewer)
                        self.assertFalse(header["detection"]["person_present"])
        asyncio.run(scenario())

    def test_binary_media_lidar_and_map_persistence(self):
        async def scenario():
            import numpy as np
            from websockets.asyncio.client import connect
            from websockets.exceptions import InvalidStatus
            from server.server_core import decode_media_frame, encode_cloud_payload, encode_media_frame

            async def next_binary(viewer):
                while True:
                    packet = await asyncio.wait_for(viewer.recv(), 3)
                    if isinstance(packet, bytes):
                        return packet

            with self.assertRaises(InvalidStatus) as rejected:
                async with connect(self.ws_url + "/ws/edge-media/test_robot"):
                    pass
            self.assertEqual(rejected.exception.response.status_code, 403)

            async with connect(self.ws_url + "/ws/live?token=test-operator") as viewer:
                hello = json.loads(await asyncio.wait_for(viewer.recv(), 3))
                self.assertEqual(hello["type"], "hello")
                async with connect(self.ws_url + "/ws/edge-media/test_robot?token=test-edge") as edge:
                    await edge.send(encode_media_frame({"stream": "video", "image_format": "webp"}, b"test-video"))
                    header, payload = decode_media_frame(await next_binary(viewer))
                    self.assertEqual(header["robot_id"], "test_robot")
                    self.assertEqual(payload, b"test-video")
                    points = np.array([[0, 0, 0], [1, 1, 0], [2, 1, 1]], dtype=np.float32)
                    blob, fmt, scale, offset, count = encode_cloud_payload(points, None, 2)
                    await edge.send(encode_media_frame({
                        "stream": "lidar", "fmt": fmt, "scale": scale, "offset": offset, "count": count,
                    }, blob))
                    header, _ = decode_media_frame(await next_binary(viewer))
                    self.assertEqual(header["stream"], "lidar")
                    self.assertEqual(header["count"], 3)

        asyncio.run(scenario())
        status, raw = self.request("/api/robots/test_robot/maps/snapshot", "test-operator", "POST")
        self.assertEqual(status, 200, raw)
        snapshot = json.loads(raw)["map"]
        self.assertEqual(snapshot["point_count"], 3)
        self.assertTrue((self.data / "maps/test_robot/latest.npz").exists())
        status, raw = self.request("/health")
        self.assertEqual(json.loads(raw)["stored_maps"], 2)
        status, raw = self.request(f"/api/maps/test_robot/{snapshot['map_id']}", "test-operator")
        self.assertEqual(status, 200, raw)
        self.assertEqual(json.loads(raw)["point_count"], 3)


if __name__ == "__main__":
    unittest.main()
