"""A new mission cannot inherit cached, persisted or in-flight LiDAR geometry."""
import asyncio
import contextlib
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException

from server.mission_routes import MissionStart
from server.server_core import CoreRuntime, decode_media_frame, encode_media_frame, encode_cloud_payload, parse_args


class MissionLidarResetTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.args = parse_args(['--disable-perception', '--map-storage-dir', str(root / 'maps'),
                               '--mission-storage-dir', str(root / 'missions'), '--mission-min-free-mb', '0',
                               '--audit-log', str(root / 'audit.jsonl')])
        self.rt = CoreRuntime(self.args)
        self.start = next(r.endpoint for r in self.rt.app.routes if getattr(r, 'path', '') ==
                          '/api/robots/{robot_id}/missions' and 'POST' in getattr(r, 'methods', set()))
        self.auth = {'role': 'operator', 'user_id': 'tester'}

    async def asyncTearDown(self):
        self.rt.stop_event.set()
        for task in self.rt.lidar_tasks.values():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await asyncio.to_thread(self.rt.missions.close)

    async def wait_for(self, predicate):
        async with asyncio.timeout(3):
            while not predicate():
                await asyncio.sleep(0.01)

    async def scan(self, points, ts=None, robot='dog'):
        blob, fmt, scale, offset, count = encode_cloud_payload(points, None, 2)
        self.rt._handle_edge_binary_frame(robot, encode_media_frame({
            'stream': 'lidar', 'fmt': fmt, 'scale': scale, 'offset': offset, 'count': count,
            'ts': time.time() if ts is None else ts,
        }, blob))

    async def test_two_locations_reset_live_map_cache_and_recordings(self):
        old = np.array([[1, 2, 0], [2, 2, 0]], np.float32)
        new = np.array([[40, 50, 0], [41, 50, 0]], np.float32)
        first = await self.start('dog', MissionStart(name='Room one'), self.auth)
        await self.scan(old)
        await self.wait_for(lambda: len(self.rt.lidar_voxels['dog']) == 2)
        await asyncio.to_thread(self.rt.missions.stop, first['mission_id'])
        self.rt.robot_paths['dog'].append((1, 2))
        self.rt.lidar_voxel_colors['dog'][(1, 2, 0)] = (1, 2, 3)
        self.rt._save_latest_map_sync('dog')
        snapshot = self.rt._create_map_snapshot_sync('dog')
        mesh = self.rt._mesh_dir('dog', create=True)
        (mesh / 'latest.bin').write_bytes(b'old-room')
        (mesh / 'latest.json').write_text('{"robot_id":"dog"}')
        old_frame = self.rt.latest_media_frames['dog']['lidar']
        socket = object()
        self.rt.frontend_queues[socket] = asyncio.Queue()
        self.rt.frontend_latest_media[socket] = {'dog:lidar': old_frame}
        self.rt.frontend_ready[socket] = asyncio.Event()
        self.rt.frontend_queues[socket].put_nowait(old_frame)
        self.rt.frontend_queues[socket].put_nowait('unrelated-message')
        self.rt._update_lidar_voxels('other', old)
        second = await self.start('dog', MissionStart(name='Room two'), self.auth)
        self.assertEqual(len(self.rt.lidar_voxels['dog']), 0)
        self.assertFalse(self.rt.robot_paths['dog'])
        self.assertFalse(self.rt.lidar_voxel_colors['dog'])
        self.assertEqual(len(self.rt.lidar_voxels['other']), 2)
        empty, _ = decode_media_frame(self.rt.latest_media_frames['dog']['lidar'])
        self.assertEqual(empty['count'], 0)
        self.assertEqual(empty['map_generation'], second['mission_id'])
        self.assertFalse((mesh / 'latest.bin').exists())
        self.assertFalse((mesh / 'latest.json').exists())
        queue = self.rt.frontend_queues[socket]
        queued = [queue.get_nowait() for _ in range(queue.qsize())]
        self.assertNotIn(old_frame, queued)
        self.assertIn('unrelated-message', queued)
        self.assertEqual(self.rt._load_map_files('dog', 'latest')[1].shape, (0, 3))
        self.assertEqual(len(self.rt._load_map_files('dog', snapshot['map_id'])[1]), 2)
        # A Raspi clock that lags the Core must not silently drop the new mission's scans.
        await self.scan(new, ts=second['started_at'] - 60)
        await self.wait_for(lambda: len(self.rt.lidar_voxels['dog']) == 2)
        np.testing.assert_allclose(self.rt._copy_map_arrays('dog')[0], new)
        await asyncio.to_thread(self.rt.missions.stop, second['mission_id'])
        for mission, points in ((first, old), (second, new)):
            with np.load(self.rt.missions.file(mission['mission_id'], 'lidar_map.npz')) as stored:
                np.testing.assert_allclose(stored['points'], points)

    async def test_restart_before_any_scan_does_not_restore_previous_room(self):
        self.rt._update_lidar_voxels('dog', np.array([[1, 2, 0]], np.float32))
        self.rt._save_latest_map_sync('dog')
        mission = await self.start('dog', MissionStart(), self.auth)
        restarted = CoreRuntime(self.args)
        restarted._restore_persisted_maps_sync()
        self.assertFalse(restarted.lidar_voxels['dog'])
        self.assertEqual(restarted.lidar_generations['dog'], mission['mission_id'])

    async def test_conflicting_start_does_not_clear_running_mission(self):
        first = await self.start('dog', MissionStart(), self.auth)
        await self.scan(np.array([[1, 2, 0]], np.float32))
        await self.wait_for(lambda: bool(self.rt.lidar_voxels['dog']))
        with self.assertRaises(HTTPException) as conflict:
            await self.start('dog', MissionStart(), self.auth)
        self.assertEqual(conflict.exception.status_code, 409)
        self.assertEqual(self.rt.lidar_generations['dog'], first['mission_id'])
        self.assertEqual(len(self.rt.lidar_voxels['dog']), 1)
        self.assertNotIn('dog', self.rt.lidar_resetting)

    async def test_inflight_old_worker_cannot_repopulate_or_broadcast_after_reset(self):
        entered, release = threading.Event(), threading.Event()
        original = self.rt._process_lidar_packet_sync
        def delayed(*args):
            entered.set()
            release.wait(3)
            return original(*args)
        with patch.object(self.rt, '_process_lidar_packet_sync', side_effect=delayed):
            await self.scan(np.array([[1, 2, 0]], np.float32))
            await self.wait_for(entered.is_set)
            mission = await self.start('dog', MissionStart(), self.auth)
            release.set()
            await asyncio.sleep(0.05)
        self.assertFalse(self.rt.lidar_voxels['dog'])
        header, _ = decode_media_frame(self.rt.latest_media_frames['dog']['lidar'])
        self.assertEqual(header['count'], 0)
        self.assertEqual(header['map_generation'], mission['mission_id'])

    async def test_start_waits_for_old_autosave_and_overwrites_it_with_empty_map(self):
        self.rt._update_lidar_voxels('dog', np.array([[1, 2, 0]], np.float32))
        entered, release = threading.Event(), threading.Event()
        original = self.rt._write_map_files
        def delayed(*args, **kwargs):
            if len(args[2]):
                entered.set()
                release.wait(3)
            return original(*args, **kwargs)
        with patch.object(self.rt, '_write_map_files', side_effect=delayed):
            save = asyncio.create_task(asyncio.to_thread(self.rt._save_latest_map_sync, 'dog'))
            await self.wait_for(entered.is_set)
            start = asyncio.create_task(self.start('dog', MissionStart(), self.auth))
            await self.wait_for(lambda: 'dog' in self.rt.lidar_resetting)
            release.set()
            await save
            await start
        self.assertEqual(self.rt._load_map_files('dog', 'latest')[1].shape, (0, 3))

    async def test_cancelled_http_start_keeps_ingest_gate_until_worker_finishes(self):
        entered, release = threading.Event(), threading.Event()
        original = self.rt._start_mission_with_fresh_map
        def delayed(*args):
            entered.set()
            release.wait(3)
            return original(*args)
        with patch.object(self.rt, '_start_mission_with_fresh_map', side_effect=delayed):
            task = asyncio.create_task(self.start('dog', MissionStart(), self.auth))
            await self.wait_for(entered.is_set)
            task.cancel()
            await asyncio.sleep(0)
            self.assertIn('dog', self.rt.lidar_resetting)
            await self.scan(np.array([[1, 2, 0]], np.float32))
            self.assertNotIn('dog', self.rt.lidar_latest_packets)
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertNotIn('dog', self.rt.lidar_resetting)
        self.assertFalse(self.rt.lidar_voxels['dog'])

    async def test_failed_reset_does_not_leave_a_recording_running(self):
        with patch.object(self.rt, '_write_map_files', side_effect=OSError('disk full')):
            with self.assertRaises(HTTPException) as failure:
                await self.start('dog', MissionStart(), self.auth)
        self.assertEqual(failure.exception.status_code, 507)
        self.assertNotIn('dog', self.rt.lidar_resetting)
        recorder = self.rt.missions.active['dog']
        self.assertFalse(recorder.thread.is_alive())
        self.assertEqual(recorder.snapshot()['status'], 'error')
