"""Real MP4 decoding, per-mission maps, recovery and bounded recording failures."""
import json
import tempfile
import time
import unittest
import zipfile
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import av
import cv2
import numpy as np

from server.missions import MissionConflict, MissionStore


def jpeg(value=80):
    image = np.full((120, 160, 3), value, np.uint8)
    return cv2.imencode(".jpg", image)[1].tobytes()


class MissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = MissionStore(self.temp.name, min_free_mb=0)
        self.addCleanup(self.store.close)

    def test_records_playable_videos_detection_log_and_colored_map(self):
        mission = self.store.start("robot", "Recorrido", "operator")
        for i in range(3):
            self.store.ingest("robot", "camera", {"image_format": "jpeg", "ts": 123 + i}, jpeg())
            self.store.ingest("robot", "thermal", {"image_format": "jpeg", "ts": 123 + i,
                "detection": {"person_present": i == 2, "regions": []},
                "temperature": {"max_c": 34}, "source_width": 160, "source_height": 120}, jpeg())
            time.sleep(0.02)
        points = np.array([[0, 0, 0], [1, 2, 3]], np.float32)
        colors = np.array([[255, 0, 0], [0, 255, 0]], np.uint8)
        self.store.ingest("robot", "lidar", {}, points, colors)
        result = self.store.stop(mission["mission_id"])
        self.assertEqual(result["status"], "completed", result["error"])
        self.assertEqual(result["missing_streams"], [])
        decoded = {}
        for stream in ("camera", "thermal"):
            with av.open(str(self.store.file(mission["mission_id"], stream + ".mp4"))) as video:
                frames = list(video.decode(video=0))
                self.assertEqual(len(frames), 3)
                self.assertLess(frames[0].time, frames[-1].time)
                decoded[stream] = frames[-1].to_ndarray(format="bgr24")
        # The recording has a burned-in detection banner, not just metadata.
        self.assertGreater(np.abs(decoded["thermal"][-28:].astype(float) - decoded["camera"][-28:]).mean(), 10)
        log = self.store.file(mission["mission_id"], "thermal_detections.jsonl")
        detections = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertTrue(detections[-1]["detection"]["person_present"])
        self.assertEqual(detections[-1]["frame"], 3)
        self.assertIn("video_time_s", detections[-1])
        with np.load(self.store.file(mission["mission_id"], "lidar_map.npz")) as data:
            np.testing.assert_equal(data["points"], points)
            np.testing.assert_equal(data["colors"], colors)
        with zipfile.ZipFile(self.store.file(mission["mission_id"], "mission.zip")) as archive:
            self.assertIn("camera.mp4", archive.namelist())
            self.assertIn("lidar_map.ply", archive.namelist())
            self.assertEqual(json.loads(archive.read("mission.json"))["status"], "completed")
        reloaded = MissionStore(self.temp.name, min_free_mb=0)
        reloaded.recover()
        self.assertEqual(reloaded.list("robot")[0]["name"], "Recorrido")

    def test_h264_starts_at_keyframe_and_recovers_after_gap(self):
        encoder = av.CodecContext.create("libx264", "w")
        encoder.width, encoder.height = 160, 120
        encoder.pix_fmt = "yuv420p"
        encoder.time_base = Fraction(1, 30)
        encoder.options = {"preset": "ultrafast", "tune": "zerolatency", "x264-params": "keyint=3:min-keyint=3:scenecut=0:repeat-headers=1"}
        packets = []
        for i in range(9):
            frame = av.VideoFrame.from_ndarray(np.full((120, 160, 3), 30 + i, np.uint8), format="bgr24")
            frame.pts = i
            packets.extend(encoder.encode(frame))
        packets.extend(encoder.encode())
        mission = self.store.start("robot", "H264", "operator")
        # Start mid-GOP. Lose one packet after sync, then recover at the next keyframe.
        for i, packet in enumerate(packets):
            if i in (0, 4):
                continue
            self.store.ingest("robot", "camera", {"image_format": "h264", "key": packet.is_keyframe,
                              "frame_index": i, "ts": 123 + i}, bytes(packet))
        result = self.store.stop(mission["mission_id"])
        self.assertEqual(result["status"], "completed", result["error"])
        self.assertGreater(result["skipped_camera_packets"], 0)
        with av.open(str(self.store.file(mission["mission_id"], "camera.mp4"))) as video:
            self.assertGreaterEqual(len(list(video.decode(video=0))), 3)

    def test_lifecycle_conflicts_missing_streams_and_mission_map_isolation(self):
        first = self.store.start("robot", "First", "operator")
        with self.assertRaises(MissionConflict):
            self.store.start("robot", "Duplicate", "operator")
        with self.assertRaises(MissionConflict):
            self.store.file(first["mission_id"], "mission.zip")
        self.store.ingest("another_robot", "camera", {"image_format": "jpeg"}, jpeg())
        self.store.ingest("robot", "lidar", {}, np.array([[1, 2, 3]], np.float32))
        self.store.stop(first["mission_id"])
        second = self.store.start("robot", "Second", "operator")
        result = self.store.stop(second["mission_id"])
        self.assertEqual(result["lidar_points"], 0)
        self.assertEqual(set(result["missing_streams"]), {"camera", "thermal", "lidar"})
        with self.assertRaises(FileNotFoundError):
            self.store.file(second["mission_id"], "camera.mp4")
        self.assertEqual(self.store.stop(second["mission_id"])["status"], "completed")

    def test_low_disk_and_queue_overflow_are_visible_failures(self):
        with patch("server.missions.shutil.disk_usage") as usage:
            usage.return_value.free = -1
            with self.assertRaises(OSError):
                self.store.start("robot", "No space", "operator")
        mission = self.store.start("robot", "Overflow", "operator")
        recorder = self.store.active["robot"]
        recorder.max_queue_bytes = 1
        self.store.ingest("robot", "camera", {"image_format": "jpeg"}, jpeg())
        result = self.store.stop(mission["mission_id"])
        self.assertEqual(result["status"], "error")
        self.assertIn("queue full", result["error"])

    def test_path_and_ticket_scope_and_recovery(self):
        mission = self.store.start("robot", "Archive", "operator")
        self.store.stop(mission["mission_id"])
        ticket = self.store.ticket(mission["mission_id"], "mission.json")
        self.assertTrue(self.store.verify_ticket(mission["mission_id"], "mission.json", ticket))
        self.assertFalse(self.store.verify_ticket(mission["mission_id"], "camera.mp4", ticket))
        self.assertFalse(self.store.verify_ticket(mission["mission_id"], "mission.json", "1.fake"))
        with self.assertRaises(ValueError):
            self.store.file("../outside", "mission.json")
        with self.assertRaises(ValueError):
            self.store.file(mission["mission_id"], "../secret")
        path = Path(self.temp.name) / mission["mission_id"] / "mission.json"
        meta = json.loads(path.read_text())
        meta["status"] = "recording"
        path.write_text(json.dumps(meta))
        restarted = MissionStore(self.temp.name, min_free_mb=0)
        restarted.recover()
        self.assertEqual(restarted.get(mission["mission_id"])["status"], "interrupted")

    def test_shutdown_finalizes_active_mission(self):
        mission = self.store.start("robot", "Shutdown", "operator")
        self.store.ingest("robot", "camera", {"image_format": "jpeg"}, jpeg())
        self.store.close()
        self.assertEqual(self.store.get(mission["mission_id"])["status"], "completed")
