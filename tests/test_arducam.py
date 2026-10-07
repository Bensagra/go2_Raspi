"""CSI discovery, capture watchdog, JPEG validation and isolated server streams."""
import asyncio
import os
import queue
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from edge.arducam_camera import AdaptiveJpeg, ArducamCamera, discover, prepare, encode_packet, put_latest
from edge.edge_gateway_service import parse_args as edge_args
from server.arducam import ArducamProcessor
from server.server_core import CoreRuntime, parse_args, encode_media_frame, decode_media_frame


def fixture(seq=1, session='one'):
    return encode_packet(np.full((90, 160, 3), 110, np.uint8),
                         {'max_width': 160, 'quality': 75}, session, seq, time.time(), time.monotonic())


def stuck_worker(config, mailbox, stop):
    mailbox.put(fixture(session=str(os.getpid())))
    while True:
        time.sleep(1)  # Models cap.read() ignoring cancellation.


class CaptureTests(unittest.TestCase):
    def test_discovers_changed_nodes_and_prepares_exact_graph(self):
        calls = []
        def command(*args):
            calls.append(args)
            if '--print-topology' in args:
                return '- entity 7: arducam-pivariety 13-000c (1 pad, 1 link)'
            if '--entity' in args:
                return '/dev/video8'
            return ''
        with patch('edge.arducam_camera.command', side_effect=command):
            graph = discover('/dev/media4')
            self.assertEqual(graph, ('/dev/media4', 'arducam-pivariety 13-000c', '/dev/video8'))
            prepare(*graph)
        self.assertIn(('v4l2-ctl', '-d', '/dev/video8',
                       '--set-fmt-video=width=3840,height=2160,pixelformat=UYVY'), calls)
        self.assertTrue(any('"csi2":4 [fmt:UYVY8_1X16/3840x2160' in call[-1] for call in calls))
        with patch('edge.arducam_camera.command', return_value='no camera'), self.assertRaisesRegex(RuntimeError, 'kernel'):
            discover('/dev/media4')

    def test_bounded_mailbox_resize_and_expired_capture(self):
        mailbox = queue.Queue(maxsize=1)
        for seq in range(1, 20):
            put_latest(mailbox, fixture(seq))
        self.assertEqual(mailbox.get()['header']['seq'], 19)
        packet = encode_packet(np.zeros((216, 384, 3), np.uint8), {'max_width': 128, 'quality': 75},
                               'resize', 1, time.time(), time.monotonic())
        self.assertEqual(cv2.imdecode(np.frombuffer(packet['payload'], np.uint8), 1).shape, (72, 128, 3))
        camera = ArducamCamera()
        packet['captured_monotonic'] -= 4
        camera.latest = packet
        self.assertIsNone(camera.take_latest())

    def test_blocking_read_is_killed_restarted_and_stop_is_bounded(self):
        camera = ArducamCamera(worker=stuck_worker, timeout_s=0.2, retry_s=0.02)
        camera.start()
        sessions = set()
        try:
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline and len(sessions) < 2:
                packet = camera.take_latest()
                if packet:
                    sessions.add(packet['header']['session_id'])
                time.sleep(0.02)
            self.assertEqual(len(sessions), 2, camera.status())
        finally:
            before = time.monotonic()
            camera.stop()
        self.assertFalse(camera.thread.is_alive())
        self.assertLess(time.monotonic() - before, 4)

    def test_adaptive_jpeg_fits_budget_and_recovers(self):
        image = np.random.default_rng(0).integers(0, 255, (2160, 3840, 3), np.uint8)
        adaptive, target = AdaptiveJpeg(1280, 75), 30_000
        for _ in range(20):
            size = len(encode_packet(image, adaptive.settings(), 's', 1, time.time(), 0)['payload'])
            adaptive.update(size, target)
        self.assertLessEqual(len(encode_packet(image, adaptive.settings(), 's', 1, time.time(), 0)['payload']),
                             target * 1.5)
        self.assertGreaterEqual(adaptive.width, 480)
        for _ in range(20):
            adaptive.update(1_000, target)
        self.assertEqual(adaptive.settings(), {'max_width': 1280, 'quality': 75})
        adaptive.update(10**9, 0)  # Uncapped uplink keeps the configured profile.
        self.assertEqual(adaptive.settings(), {'max_width': 1280, 'quality': 75})
        camera = ArducamCamera()
        camera.set_target_bytes(1234.9)
        self.assertEqual(camera.config['target_bytes'].value, 1234)

    def test_gateway_defaults_and_invalid_options(self):
        with patch('sys.argv', ['edge']):
            self.assertTrue(edge_args().enable_arducam)
        with patch('sys.argv', ['edge', '--disable-arducam']):
            self.assertFalse(edge_args().enable_arducam)
        for flag, value in [('--arducam-fps', 'nan'), ('--arducam-max-width', '4000'), ('--arducam-quality', '100')]:
            with patch('sys.argv', ['edge', flag, value]), patch('sys.stderr'), self.assertRaises(SystemExit):
                edge_args()


class ProcessorTests(unittest.TestCase):
    def test_jpeg_validation_sequence_and_session_recovery(self):
        processor = ArducamProcessor()
        packet = fixture()
        self.assertEqual(processor.process(packet['header'], packet['payload'])['width'], 160)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            processor.process(packet['header'], packet['payload'])
        new = fixture(session='two')
        processor.process(new['header'], new['payload'])
        for changes, payload in [({}, b'bad'), ({'width': 3840}, packet['payload']),
                                 ({'ts': float('nan')}, packet['payload']),
                                 ({'seq': True}, packet['payload']),
                                 ({'image_format': 'h264'}, packet['payload']),
                                 ({}, b'\xff\xd8' + bytes(4 * 1024 * 1024) + b'\xff\xd9')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                processor.process({**packet['header'], **changes}, payload)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = CoreRuntime(parse_args(['--disable-perception', '--map-storage-dir', self.temp.name,
            '--mission-storage-dir', str(Path(self.temp.name) / 'missions'),
            '--audit-log', str(Path(self.temp.name) / 'audit.jsonl')]))

    async def asyncTearDown(self):
        for robot in list(self.runtime.arducam_tasks):
            await self.runtime._clear_arducam(robot)
        self.runtime.missions.close()

    async def send(self, robot, packet):
        self.runtime._handle_edge_binary_frame(robot, encode_media_frame(packet['header'], packet['payload']))
        await self.runtime.arducam_tasks[robot]

    async def test_isolates_cameras_robots_and_invalid_frames(self):
        rt = self.runtime
        packet = fixture()
        packet['header']['robot_id'] = 'forged'
        mission = rt.missions.start('a', 'Arducam', 'operator')
        await self.send('a', {**packet, 'payload': b'invalid'})
        self.assertNotIn('arducam', rt.latest_media_frames['a'])
        await self.send('a', packet)
        await self.send('b', fixture())
        header, jpeg = decode_media_frame(rt.latest_media_frames['a']['arducam'])
        self.assertEqual(header['robot_id'], 'a')
        self.assertEqual(jpeg, packet['payload'])
        self.assertNotIn('video', rt.latest_media_frames['a'])
        result = await asyncio.to_thread(rt.missions.stop, mission['mission_id'])
        self.assertEqual(result['status'], 'completed', result['error'])
        self.assertEqual(result['streams']['arducam']['frames'], 1)
        self.assertEqual(result['streams']['camera']['frames'], 0)
        await rt._clear_arducam('a')
        self.assertNotIn('arducam', rt.latest_media_frames['a'])
        self.assertIn('arducam', rt.latest_media_frames['b'])
        await self.send('a', fixture())  # Same sequence accepted on a new connection.

    async def test_burst_keeps_only_latest(self):
        for seq in range(1, 20):
            packet = fixture(seq)
            self.runtime._handle_edge_binary_frame('a', encode_media_frame(packet['header'], packet['payload']))
        await self.runtime.arducam_tasks['a']
        header, _ = decode_media_frame(self.runtime.latest_media_frames['a']['arducam'])
        self.assertEqual(header['seq'], 19)
