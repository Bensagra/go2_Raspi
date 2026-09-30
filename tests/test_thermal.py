"""Thermal protocol, USB recovery, detection and nonblocking media pipeline."""
import asyncio
import tempfile
import time
import unittest
import zlib
from pathlib import Path

import cv2
import numpy as np

from edge.thermal_camera import ThermalCamera
from server.thermal import ThermalProcessor
from server.server_core import CoreRuntime, decode_media_frame, encode_media_frame, parse_args
from termica.protocol import MAX_CSV_BYTES, decode_csv, encode_csv


def fixture(hot=True, seq=1, session="test"):
    frame = np.full((120, 160), 22.0, dtype=np.float32)
    if hot:
        frame[30:65, 60:85] = 34.0
    return {"stream": "thermal_csv", "format": "csv_zlib", "unit": "celsius",
            "width": 160, "height": 120, "seq": seq, "session_id": session,
            "ts": time.time()}, encode_csv(frame)


class ThermalProtocolTests(unittest.TestCase):
    def test_csv_temperatures_survive_transport(self):
        rng = np.random.default_rng(13)
        frame = rng.uniform(-10, 80, (120, 160)).astype(np.float32)
        header, _ = fixture()
        payload = encode_csv(frame)
        self.assertIn(b",", zlib.decompress(payload))
        np.testing.assert_allclose(decode_csv(header, payload), frame, atol=0.00006)

    def test_invalid_or_unbounded_input_rejected(self):
        header, payload = fixture()
        for change, raw in [
            ({"width": 159}, payload), ({"width": True}, payload),
            ({"unit": "kelvin"}, payload), ({"format": "jpeg"}, payload),
            ({}, payload[:-4]), ({}, payload + b"extra"),
            ({}, zlib.compress(b"x" * (MAX_CSV_BYTES + 1))),
            ({"width": 2, "height": 2}, zlib.compress(b"nan,22\n22,22\n")),
            ({"width": 2, "height": 3}, zlib.compress(b"#1,2\n22,22\n22,22\n")),
        ]:
            with self.subTest(change=change), self.assertRaises((ValueError, zlib.error)):
                decode_csv({**header, **change}, raw)

    def test_confirmation_clear_and_jpeg(self):
        detector = ThermalProcessor()
        for seq in range(1, 4):
            header, jpeg = detector.process(*fixture(seq=seq))
            self.assertEqual(header["detection"]["person_present"], seq == 3)
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape, (480, 360, 3))
        self.assertEqual((header["source_width"], header["source_height"]), (120, 160))
        self.assertEqual(header["rotation_deg"], 90)
        self.assertEqual(header["temperature"]["max_c"], 34)
        self.assertGreater(len(header["detection"]["regions"]), 0)
        for seq in range(4, 15):
            header, _ = detector.process(*fixture(hot=False, seq=seq))
        self.assertFalse(header["detection"]["person_present"])

    def test_rotation_keeps_hot_region_aligned_with_output(self):
        # Off-center rectangle makes direction observable (not just width/height).
        original, _ = ThermalProcessor(rotation_deg=0).process(*fixture())
        for angle in (0, 90, 180, 270):
            with self.subTest(angle=angle):
                detector = ThermalProcessor(rotation_deg=angle)
                header, jpeg = detector.process(*fixture())
                raw_header, raw_payload = fixture()
                expected = np.rot90(decode_csv(raw_header, raw_payload), -(angle // 90))
                np.testing.assert_array_equal(detector.history[-1], expected)
                region = header["detection"]["regions"][0]
                base = original["detection"]["regions"][0]
                x, y, w, h = (base[key] for key in ("x", "y", "width", "height"))
                bounds = {
                    0: (x, y, w, h),
                    90: (120 - y - h, x, h, w),
                    180: (160 - x - w, 120 - y - h, w, h),
                    270: (y, 160 - x - w, h, w),
                }
                self.assertEqual(tuple(region[key] for key in ("x", "y", "width", "height")), bounds[angle])
                image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(image.shape[:2], (header["height"], header["width"]))
                self.assertEqual(header["temperature"]["max_c"], original["temperature"]["max_c"])

    def test_server_rotation_default_and_override(self):
        self.assertEqual(parse_args([]).thermal_rotation_deg, 90)
        self.assertEqual(parse_args(["--thermal-rotation-deg", "270"]).thermal_rotation_deg, 270)

    def test_duplicate_rejected_and_new_session_resets_confirmation(self):
        detector = ThermalProcessor()
        for seq in range(1, 4):
            detector.process(*fixture(seq=seq))
        with self.assertRaises(ValueError):
            detector.process(*fixture(seq=3))
        header, _ = detector.process(*fixture(session="reconnected"))
        self.assertFalse(header["detection"]["person_present"])

    def test_gap_and_shape_change_reset_temporal_history(self):
        detector = ThermalProcessor()
        for seq in range(1, 4):
            detector.process(*fixture(seq=seq))
        detector.last_at -= 4
        header, _ = detector.process(*fixture(seq=4))
        self.assertFalse(header["detection"]["person_present"])
        incoming, _ = fixture(seq=5)
        header, _ = detector.process({**incoming, "width": 8, "height": 8},
                                     encode_csv(np.full((8, 8), 22)))
        self.assertEqual(header["temperature"]["max_c"], 22)


class ThermalCaptureTests(unittest.TestCase):
    def test_recovers_usb_and_keeps_only_latest_frame(self):
        class FakeCamera:
            def __init__(self):
                self.closed = False
                self.stopped = False
            def start_stream(self):
                pass
            def read(self, block=False):
                return None, np.full((12, 16), 30, np.float32)
            def stop_stream(self):
                self.stopped = True
            def close(self):
                self.closed = True
        dev = FakeCamera()
        attempts = []
        def opener(port):
            attempts.append(port)
            if len(attempts) == 1:
                raise OSError("USB disconnected")
            return dev
        camera = ThermalCamera(port="fake", fps=30, retry_s=0.01, opener=opener)
        camera.start()
        try:
            deadline = time.monotonic() + 2
            while camera.status()["frames"] < 3 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertGreaterEqual(camera.status()["frames"], 3)
            packet = camera.take_latest()
            self.assertGreaterEqual(packet["header"]["seq"], 3)
            np.testing.assert_equal(decode_csv(packet["header"], packet["payload"]), 30)
            self.assertEqual(camera.status()["error"], "")
        finally:
            camera.stop()
        self.assertTrue(dev.closed)
        self.assertTrue(dev.stopped)
        self.assertFalse(camera.thread.is_alive())

    def test_no_frames_reconnects_and_can_stop(self):
        class EmptyCamera:
            def start_stream(self):
                pass
            def read(self, block=False):
                return None, None
            def stop_stream(self):
                pass
            def close(self):
                pass
        calls = []
        def opener(port):
            calls.append(port)
            return EmptyCamera()
        camera = ThermalCamera(opener=opener, timeout_s=0.02, retry_s=0.01)
        camera.start()
        try:
            deadline = time.monotonic() + 1
            while len(calls) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertGreaterEqual(len(calls), 2)
            self.assertFalse(camera.status()["connected"])
        finally:
            camera.stop()


class ThermalPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        args = parse_args(["--disable-perception", "--map-storage-dir", self.temp.name,
                           "--audit-log", str(Path(self.temp.name) / "audit.jsonl")])
        self.runtime = CoreRuntime(args)

    async def asyncTearDown(self):
        for robot in list(self.runtime.thermal_tasks):
            await self.runtime._clear_thermal(robot)

    async def test_binary_pipeline_isolates_robots_and_recovers_bad_csv(self):
        runtime = self.runtime
        header, payload = fixture()
        runtime._handle_edge_binary_frame("a", encode_media_frame(header, b"bad zlib"))
        await runtime.thermal_tasks["a"]
        for seq in range(1, 4):
            runtime._handle_edge_binary_frame("a", encode_media_frame(*fixture(seq=seq)))
            await runtime.thermal_tasks["a"]
        runtime._handle_edge_binary_frame("b", encode_media_frame(*fixture(hot=False)))
        await runtime.thermal_tasks["b"]
        a, jpeg = decode_media_frame(runtime.latest_media_frames["a"]["thermal"])
        b, _ = decode_media_frame(runtime.latest_media_frames["b"]["thermal"])
        self.assertTrue(a["detection"]["person_present"])
        self.assertFalse(b["detection"]["person_present"])
        self.assertEqual(a["robot_id"], "a")
        self.assertEqual(a["rotation_deg"], 90)
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))
        await runtime._clear_thermal("a")
        self.assertNotIn("a", runtime.thermal_processors)
        self.assertNotIn("thermal", runtime.latest_media_frames["a"])

    async def test_burst_is_coalesced_to_newest_frame(self):
        for seq in range(1, 21):
            self.runtime._handle_edge_binary_frame("a", encode_media_frame(*fixture(seq=seq)))
        await self.runtime.thermal_tasks["a"]
        header, _ = decode_media_frame(self.runtime.latest_media_frames["a"]["thermal"])
        self.assertEqual(header["seq"], 20)
        self.assertEqual(len(self.runtime.thermal_processors["a"].history), 1)
