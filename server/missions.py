"""Server-owned mission recording. No browser or MQTT connection is required to persist media."""
import copy
import hashlib
import hmac
import json
import re
import secrets
import shutil
import threading
import time
import uuid
import zipfile
from collections import OrderedDict, deque
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

ARTIFACTS = {"mission.json", "camera.mp4", "thermal.mp4", "frames.jsonl",
             "thermal_detections.jsonl", "lidar_map.npz", "lidar_map.ply", "mission.zip"}
BUSY = {"recording", "finalizing"}


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class MissionConflict(ValueError):
    pass


class VideoWriter:
    def __init__(self, path, image):
        self.container = av.open(str(path), "w", format="mp4", options={
            "movflags": "frag_keyframe+empty_moov+default_base_moof",
            "frag_duration": "1000000",
        })
        self.stream = self.container.add_stream("libx264", rate=30)
        self.stream.width = max(2, image.shape[1] // 2 * 2)
        self.stream.height = max(2, image.shape[0] // 2 * 2)
        self.stream.pix_fmt = "yuv420p"
        self.stream.time_base = Fraction(1, 1000)
        self.stream.codec_context.time_base = Fraction(1, 1000)
        self.stream.codec_context.max_b_frames = 0
        self.stream.codec_context.thread_count = 1
        self.stream.options = {"preset": "veryfast", "crf": "23", "tune": "zerolatency"}
        self.first_at = None
        self.last_pts = -1

    def write(self, image, at):
        if self.first_at is None:
            self.first_at = at
        pts = max(self.last_pts + 1, round((at - self.first_at) * 1000))
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        frame = frame.reformat(width=self.stream.width, height=self.stream.height, format="yuv420p")
        frame.pts, frame.time_base = pts, Fraction(1, 1000)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.last_pts = pts
        return pts / 1000.0

    def close(self):
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()


class MissionRecorder:
    """One bounded queue and one disk/codec worker per mission; never block the live loop."""
    def __init__(self, directory, robot_id, name, user_id, voxel_size, max_voxels, min_free_bytes):
        self.directory = directory
        self.min_free_bytes = min_free_bytes
        self.voxel_size, self.max_voxels = voxel_size, max_voxels
        self.condition = threading.Condition()
        self.inbox = deque()
        self.queued_bytes = 0
        self.max_queue_bytes = 64 * 1024 * 1024
        self.stopping = False
        self.started_monotonic = time.monotonic()
        self.meta = {
            "mission_id": directory.name, "robot_id": robot_id, "name": name,
            "started_by": user_id, "started_at": time.time(), "ended_at": None,
            "status": "recording", "error": "", "duration_s": 0,
            "streams": {s: {"frames": 0, "first_at_s": None, "last_at_s": None}
                        for s in ("camera", "thermal", "lidar")},
            "lidar_points": 0, "voxel_size_m": voxel_size,
            "skipped_camera_packets": 0, "artifacts": [], "missing_streams": [],
        }
        self.writers = {}
        self.voxels = OrderedDict()
        self.decoder = None
        self.camera_format = None
        self.camera_index = None
        self.frames_log = self.detections_log = None
        self.directory.mkdir(parents=True)
        self._checkpoint()
        self.thread = threading.Thread(target=self._run, name=f"mission-{directory.name[:8]}", daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.condition:
            result = copy.deepcopy(self.meta)
        if result["status"] in BUSY:
            result["duration_s"] = round(time.monotonic() - self.started_monotonic, 3)
        return result

    def ingest(self, stream, header, payload, colors=None):
        size = payload.nbytes + (colors.nbytes if colors is not None else 0) if isinstance(payload, np.ndarray) else len(payload)
        with self.condition:
            if self.stopping:
                return
            if len(self.inbox) >= 256 or self.queued_bytes + size > self.max_queue_bytes:
                self.meta["error"] = "Recording stopped: disk/encoder could not keep up (queue full)"
                self.meta["status"] = "finalizing"
                self.stopping = True
            else:
                self.inbox.append((stream, dict(header), payload, colors,
                                   time.monotonic() - self.started_monotonic, size))
                self.queued_bytes += size
            self.condition.notify()

    def request_stop(self):
        with self.condition:
            self.stopping = True
            if self.meta["status"] == "recording":
                self.meta["status"] = "finalizing"
            self.condition.notify()

    def _checkpoint(self):
        atomic_json(self.directory / "mission.json", self.snapshot())

    def _check_space(self):
        if shutil.disk_usage(self.directory).free < self.min_free_bytes:
            raise OSError("Recording stopped: not enough free disk space")

    def _decode(self, header, payload):
        fmt = header.get("image_format")
        if fmt != self.camera_format:
            self.decoder = None
            self.camera_index = None
            self.camera_format = fmt
        if fmt != "h264":
            if fmt not in {"jpg", "jpeg", "webp"}:
                raise ValueError("Unsupported camera image format")
            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Invalid camera image")
            return [image]
        index = header.get("frame_index")
        if isinstance(index, int) and self.camera_index is not None and index != self.camera_index + 1:
            self.decoder = None
        self.camera_index = index if isinstance(index, int) else None
        if self.decoder is None:
            if not header.get("key"):
                with self.condition:
                    self.meta["skipped_camera_packets"] += 1
                return []
            self.decoder = av.CodecContext.create("h264", "r")
        try:
            return [frame.to_ndarray(format="bgr24") for frame in self.decoder.decode(av.Packet(payload))]
        except av.error.FFmpegError:
            self.decoder = None
            with self.condition:
                self.meta["skipped_camera_packets"] += 1
            return []

    def _record_image(self, stream, image, header, at):
        if stream == "thermal":
            detection = header.get("detection", {})
            label = "Posible presencia humana" if detection.get("person_present") is True else "Sin presencia confirmada"
            if not isinstance(detection.get("person_present"), bool):
                label = "Deteccion no disponible"
            cv2.rectangle(image, (0, max(0, image.shape[0] - 28)), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
            cv2.putText(image, label, (7, image.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 220, 255) if detection.get("person_present") else (255, 255, 255), 1, cv2.LINE_AA)
        if stream not in self.writers:
            self.writers[stream] = VideoWriter(self.directory / f"{stream}.mp4", image)
        video_time = self.writers[stream].write(image, at)
        with self.condition:
            stats = self.meta["streams"][stream]
            stats["frames"] += 1
            if stats["first_at_s"] is None:
                stats["first_at_s"] = round(at, 3)
            stats["last_at_s"] = round(at, 3)
            frame_number = stats["frames"]
        record = {"stream": stream, "frame": frame_number, "mission_time_s": round(at, 3),
                  "video_time_s": video_time, "source_ts": header.get("ts"),
                  "source_frame_index": header.get("frame_index", header.get("seq"))}
        self.frames_log.write(json.dumps(record, allow_nan=False) + "\n")
        if stream == "thermal":
            record.update(temperature=header.get("temperature", {}), detection=header.get("detection", {}),
                          source_width=header.get("source_width"), source_height=header.get("source_height"))
            self.detections_log.write(json.dumps(record, allow_nan=False) + "\n")

    def _record_lidar(self, points, colors, at):
        valid = np.isfinite(points[:, :3]).all(axis=1)
        points = points[valid, :3]
        colors = colors[valid, :3] if colors is not None else np.full((len(points), 3), 190, np.uint8)
        keys = np.floor(points / self.voxel_size).astype(np.int64)
        _, indices = np.unique(keys, axis=0, return_index=True)
        for i in indices:
            key = tuple(keys[i])
            self.voxels[key] = (*map(float, points[i]), *map(int, colors[i]))
            self.voxels.move_to_end(key)
            if len(self.voxels) > self.max_voxels:
                self.voxels.popitem(last=False)
        with self.condition:
            stats = self.meta["streams"]["lidar"]
            stats["frames"] += 1
            if stats["first_at_s"] is None:
                stats["first_at_s"] = round(at, 3)
            stats["last_at_s"] = round(at, 3)
            self.meta["lidar_points"] = len(self.voxels)

    def _save_map(self):
        if not self.voxels:
            return
        data = np.asarray(list(self.voxels.values()), dtype=np.float32)
        points, colors = data[:, :3], data[:, 3:6].astype(np.uint8)
        path = self.directory / "lidar_map.npz"
        with path.with_suffix(".tmp").open("wb") as output:
            np.savez_compressed(output, points=points, colors=colors)
        path.with_suffix(".tmp").replace(path)
        vertices = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                               ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        for i, key in enumerate(("x", "y", "z")):
            vertices[key] = points[:, i]
        for i, key in enumerate(("red", "green", "blue")):
            vertices[key] = colors[:, i]
        path = self.directory / "lidar_map.ply"
        with path.with_suffix(".tmp").open("wb") as output:
            output.write(("ply\nformat binary_little_endian 1.0\n" + f"element vertex {len(points)}\n"
                          + "property float x\nproperty float y\nproperty float z\n"
                          + "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
            output.write(vertices.tobytes())
        path.with_suffix(".tmp").replace(path)

    def _run(self):
        checkpoint_at = map_at = time.monotonic()
        try:
            self.frames_log = (self.directory / "frames.jsonl").open("a", encoding="utf-8", buffering=1)
            self.detections_log = (self.directory / "thermal_detections.jsonl").open("a", encoding="utf-8", buffering=1)
            while True:
                with self.condition:
                    if not self.inbox and not self.stopping:
                        self.condition.wait(timeout=1)
                    if not self.inbox and self.stopping:
                        break
                    item = self.inbox.popleft() if self.inbox else None
                    if item is not None:
                        self.queued_bytes -= item[-1]
                now = time.monotonic()
                if now - checkpoint_at >= 2:
                    self._check_space()
                    self._checkpoint()
                    checkpoint_at = now
                if item is None:
                    continue
                stream, header, payload, colors, at, _ = item
                if stream == "reset":
                    self.decoder = None
                    self.camera_index = None
                elif stream == "lidar":
                    self._record_lidar(payload, colors, at)
                else:
                    images = self._decode(header, payload) if stream == "camera" else [cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)]
                    for image in images:
                        if image is None:
                            raise ValueError("Invalid recorded image")
                        self._record_image(stream, image, header, at)
                if self.voxels and (now - map_at >= 10 or not (self.directory / "lidar_map.npz").exists()):
                    self._save_map()
                    map_at = now
        except Exception as exc:
            with self.condition:
                self.meta["error"] = str(exc)
        finally:
            with self.condition:
                self.stopping = True
                self.meta["status"] = "finalizing"
                self.inbox.clear()
                self.queued_bytes = 0
            for writer in self.writers.values():
                try:
                    writer.close()
                except Exception as exc:
                    self.meta["error"] = self.meta["error"] or str(exc)
            for handle in (self.frames_log, self.detections_log):
                if handle:
                    try:
                        handle.close()
                    except Exception as exc:
                        self.meta["error"] = self.meta["error"] or str(exc)
            try:
                self._save_map()
            except Exception as exc:
                self.meta["error"] = self.meta["error"] or str(exc)
            with self.condition:
                self.meta.update(status="error" if self.meta["error"] else "completed", ended_at=time.time(),
                                 duration_s=round(time.monotonic() - self.started_monotonic, 3))
                self.meta["missing_streams"] = [s for s, stats in self.meta["streams"].items() if not stats["frames"]]
                self.meta["artifacts"] = sorted(p.name for p in self.directory.iterdir() if p.name in ARTIFACTS)
            try:
                self._checkpoint()
            except OSError as exc:
                # The last checkpoint remains recoverable even when the filesystem fails.
                with self.condition:
                    self.meta["status"] = "error"
                    self.meta["error"] = self.meta["error"] or f"Could not save mission manifest: {exc}"
            self.writers.clear()
            self.voxels.clear()
            self.decoder = None


class MissionStore:
    def __init__(self, root, voxel_size=0.08, max_voxels=120000, min_free_mb=256):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.voxel_size, self.max_voxels = voxel_size, max_voxels
        self.min_free_bytes = int(min_free_mb * 1024 * 1024)
        self.lock = threading.RLock()
        self.archive_lock = threading.Lock()
        self.active = {}
        self.secret = secrets.token_bytes(32)

    def recover(self):
        for path in self.root.glob("*/mission.json"):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
                if meta.get("status") in BUSY:
                    meta.update(status="interrupted", error="Server stopped before mission finalization", ended_at=None)
                    meta["artifacts"] = sorted(p.name for p in path.parent.iterdir() if p.name in ARTIFACTS)
                    atomic_json(path, meta)
            except (OSError, ValueError):
                continue

    def start(self, robot_id, name, user_id):
        with self.lock:
            current = self.active.get(robot_id)
            if current is not None and current.thread.is_alive():
                raise MissionConflict("Ya hay una misión activa para este robot")
            if shutil.disk_usage(self.root).free < self.min_free_bytes:
                raise OSError("No hay espacio libre suficiente para iniciar la misión")
            recorder = MissionRecorder(self.root / uuid.uuid4().hex, robot_id,
                name.strip() or time.strftime("Misión %Y-%m-%d %H:%M:%S"), user_id,
                self.voxel_size, self.max_voxels, self.min_free_bytes)
            self.active[robot_id] = recorder
            return recorder.snapshot()

    def ingest(self, robot_id, stream, header, payload, colors=None):
        with self.lock:
            recorder = self.active.get(robot_id)
        if recorder:
            recorder.ingest(stream, header, payload, colors)

    def stop(self, mission_id):
        self.directory(mission_id)
        with self.lock:
            recorder = next((r for r in self.active.values() if r.directory.name == mission_id), None)
            if recorder:
                recorder.request_stop()
        if recorder:
            recorder.thread.join()
        return self.get(mission_id)

    def close(self):
        with self.lock:
            recorders = list(self.active.values())
            for recorder in recorders:
                recorder.request_stop()
        for recorder in recorders:
            recorder.thread.join()

    def directory(self, mission_id):
        if not re.fullmatch(r"[0-9a-f]{32}", mission_id):
            raise ValueError("Identificador de misión inválido")
        path = self.root / mission_id
        if path.is_symlink() or not path.is_dir():
            raise FileNotFoundError("Misión no encontrada")
        return path

    def get(self, mission_id):
        directory = self.directory(mission_id)
        with self.lock:
            recorder = next((r for r in self.active.values() if r.directory.name == mission_id), None)
        if recorder:
            return recorder.snapshot()
        return json.loads((directory / "mission.json").read_text(encoding="utf-8"))

    def list(self, robot_id):
        result = []
        for path in self.root.glob("*/mission.json"):
            try:
                meta = self.get(path.parent.name)
                if meta.get("robot_id") == robot_id:
                    result.append(meta)
            except (OSError, ValueError):
                continue
        return sorted(result, key=lambda m: m["started_at"], reverse=True)

    def file(self, mission_id, filename):
        if filename not in ARTIFACTS:
            raise ValueError("Archivo de misión inválido")
        directory = self.directory(mission_id)
        with self.lock:
            if any(r.directory.name == mission_id and r.thread.is_alive() for r in self.active.values()):
                raise MissionConflict("La misión todavía se está grabando o guardando")
        if self.get(mission_id)["status"] in BUSY:
            raise MissionConflict("Finalizá la misión antes de descargarla")
        path = directory / filename
        if filename == "mission.zip" and not path.exists():
            with self.archive_lock:
                if not path.exists():
                    sources = [directory / name for name in sorted(ARTIFACTS - {"mission.zip"})
                               if (directory / name).is_file() and not (directory / name).is_symlink()]
                    if shutil.disk_usage(directory).free < sum(p.stat().st_size for p in sources) + self.min_free_bytes:
                        raise OSError("No hay espacio libre para preparar el ZIP; descargá los archivos individuales")
                    temporary = directory / "archive.tmp"
                    try:
                        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
                            for source in sources:
                                archive.write(source, source.name)
                        temporary.replace(path)
                    finally:
                        temporary.unlink(missing_ok=True)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError("Este archivo no tiene datos grabados")
        return path

    def ticket(self, mission_id, filename):
        expires = int(time.time()) + 600
        message = f"{mission_id}/{filename}/{expires}".encode()
        return f"{expires}.{hmac.new(self.secret, message, hashlib.sha256).hexdigest()}"

    def verify_ticket(self, mission_id, filename, ticket):
        try:
            expires, signature = ticket.split(".", 1)
            if not time.time() < int(expires) <= time.time() + 601:
                return False
            message = f"{mission_id}/{filename}/{expires}".encode()
            return hmac.compare_digest(signature, hmac.new(self.secret, message, hashlib.sha256).hexdigest())
        except (ValueError, AttributeError):
            return False
