"""Identificación de personas en la Arducam, en el servidor.

Pipeline por cuadro (todo con OpenCV DNN, sin PyTorch ni CUDA):

1. YOLOX detecta personas (cuerpos).
2. YuNet busca la cara dentro de la parte superior de cada persona.
3. SFace calcula un embedding de 128 valores y lo compara con la galería.
4. Un tracker por IoU mantiene el número de cada persona entre cuadros,
   aunque deje de verse la cara (se da vuelta, se agacha).

Cada persona nueva recibe un número correlativo ("Persona 3") que se guarda en
disco junto con sus embeddings, así la reconoce también otro día. Los modelos
(~75 MB, Apache-2.0, de opencv_zoo) se descargan solos la primera vez.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "person": ("object_detection_yolox_2022nov.onnx",
               f"{_ZOO}/object_detection_yolox/object_detection_yolox_2022nov.onnx",
               "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"),
    "face_det": ("face_detection_yunet_2023mar.onnx",
                 f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                 "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "face_rec": ("face_recognition_sface_2021dec.onnx",
                 f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
                 "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}

# SFace cosine similarity: >= MATCH is the same person (OpenCV recommends 0.363);
# < NEW is clearly someone else. In between is ambiguous and never creates a person.
DEFAULT_MATCH_THRESHOLD = 0.40
NEW_PERSON_THRESHOLD = 0.30
MIN_FACE_PX = 40            # Smaller faces are too blurry to enroll a new person.
MIN_FACE_SCORE = 0.85
MAX_EMBEDDINGS = 24         # Per person: diverse angles/lighting, oldest dropped.
TRACK_TTL_S = 2.5           # Keep a track (and its number) this long without detections.


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_models(models_dir: Path) -> Dict[str, Path]:
    """Return local paths, downloading and verifying any missing model."""
    models_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (name, url, sha) in MODELS.items():
        path = models_dir / name
        if not path.exists() or _sha256(path) != sha:
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            if _sha256(tmp) != sha:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"checksum inválido al descargar {name}")
            tmp.replace(path)
        paths[key] = path
    return paths


def _dnn_target(target: str) -> Tuple[int, int]:
    if target == "opencl" and cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_OPENCL
    return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU


class YoloxPersonDetector:
    SIZE = 640

    def __init__(self, path: Path, target: str = "cpu", conf: float = 0.45, nms: float = 0.45):
        self.net = cv2.dnn.readNetFromONNX(str(path))
        backend, tgt = _dnn_target(target)
        self.net.setPreferableBackend(backend)
        self.net.setPreferableTarget(tgt)
        self.conf, self.nms = conf, nms
        grids, strides = [], []
        for stride in (8, 16, 32):
            n = self.SIZE // stride
            yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), stride))
        self.grids = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    def detect(self, bgr: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        h, w = bgr.shape[:2]
        r = min(self.SIZE / h, self.SIZE / w)
        rw, rh = int(w * r), int(h * r)
        padded = np.full((self.SIZE, self.SIZE, 3), 114, np.uint8)
        padded[:rh, :rw] = cv2.resize(bgr, (rw, rh))
        self.net.setInput(padded.transpose(2, 0, 1)[None].astype(np.float32))
        out = self.net.forward()[0].copy()
        out[:, :2] = (out[:, :2] + self.grids) * self.strides
        out[:, 2:4] = np.exp(out[:, 2:4]) * self.strides
        scores = out[:, 4] * out[:, 5]  # objectness x class 0 ("person")
        keep = scores >= self.conf
        if not keep.any():
            return []
        boxes = out[keep, :4] / r
        scores = scores[keep]
        xywh = np.c_[boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 2], boxes[:, 3]]
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), self.conf, self.nms)
        result = []
        for i in np.array(idx).flatten():
            x, y, bw, bh = xywh[i]
            result.append((max(0.0, float(x)), max(0.0, float(y)), min(float(w), float(x + bw)),
                           min(float(h), float(y + bh)), float(scores[i])))
        return result


@dataclass
class Face:
    box: Tuple[float, float, float, float]   # full-frame pixels
    score: float
    embedding: np.ndarray                     # L2-normalised, 128 floats
    crop: np.ndarray                          # aligned 112x112 BGR
    frontal: bool

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def quality(self) -> float:
        return float(self.score * min(1.0, self.width / 112.0) * (1.0 if self.frontal else 0.6))

    @property
    def enrollable(self) -> bool:
        return self.width >= MIN_FACE_PX and self.score >= MIN_FACE_SCORE and self.frontal


class FaceEngine:
    MAX_SIDE = 640

    def __init__(self, det_path: Path, rec_path: Path, target: str = "cpu", score: float = 0.75):
        backend, tgt = _dnn_target(target)
        self.det = cv2.FaceDetectorYN.create(str(det_path), "", (320, 320), score, 0.3, 50, backend, tgt)
        self.rec = cv2.FaceRecognizerSF.create(str(rec_path), "", backend, tgt)

    def faces(self, bgr: np.ndarray, region: Tuple[int, int, int, int]) -> List[Face]:
        x1, y1, x2, y2 = region
        crop = bgr[y1:y2, x1:x2]
        if crop.shape[0] < 24 or crop.shape[1] < 24:
            return []
        scale = min(1.0, self.MAX_SIDE / max(crop.shape[:2]))
        small = cv2.resize(crop, None, fx=scale, fy=scale) if scale < 1.0 else crop
        self.det.setInputSize((small.shape[1], small.shape[0]))
        _, found = self.det.detect(small)
        result = []
        for row in (found if found is not None else []):
            row = row.copy()
            row[:14] /= scale  # back to crop pixels for alignment
            aligned = self.rec.alignCrop(crop, row)
            emb = self.rec.feature(aligned).flatten().astype(np.float32)
            emb /= max(float(np.linalg.norm(emb)), 1e-6)
            # Landmarks: right eye, left eye, nose. Nose centred between the eyes = frontal.
            ex1, ex2, nx = row[4], row[6], row[8]
            span = ex2 - ex1
            frontal = abs(span) > 1e-3 and 0.2 <= (nx - ex1) / span <= 0.8
            fx, fy, fw, fh = (float(v) for v in row[:4])
            result.append(Face((x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh), float(row[14]),
                               emb, aligned, bool(frontal)))
        return result


class PeopleGallery:
    """Numbered people with their face embeddings, persisted per robot."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.lock = threading.RLock()
        self.data: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        self.dirty: set = set()

    def _dir(self, robot_id: str) -> Path:
        safe = "".join(c for c in robot_id if c.isalnum() or c in "-_") or "robot"
        path = self.base_dir / safe
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load(self, robot_id: str) -> Dict[str, Any]:
        if robot_id not in self.data:
            path = self._dir(robot_id) / "people.json"
            data = {"next_number": 1, "people": {}}
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                except (OSError, ValueError):
                    pass
            self.data[robot_id] = data
            self.embeddings[robot_id] = {
                pid: np.asarray(p.get("embeddings", []), np.float32).reshape(-1, 128)
                for pid, p in data["people"].items()}
        return self.data[robot_id]

    def save(self, robot_id: str) -> None:
        with self.lock:
            data = self._load(robot_id)
            for pid, emb in self.embeddings[robot_id].items():
                data["people"][pid]["embeddings"] = np.round(emb, 5).tolist()
            path = self._dir(robot_id) / "people.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False))
            tmp.replace(path)
            self.dirty.discard(robot_id)

    def flush(self) -> None:
        for robot_id in list(self.dirty):
            self.save(robot_id)

    def match(self, robot_id: str, emb: np.ndarray) -> Tuple[Optional[str], float]:
        with self.lock:
            self._load(robot_id)
            best_pid, best = None, -1.0
            for pid, stored in self.embeddings[robot_id].items():
                if len(stored):
                    sim = float(np.max(stored @ emb))
                    if sim > best:
                        best_pid, best = pid, sim
            return best_pid, best

    def create(self, robot_id: str, face: Face) -> Dict[str, Any]:
        with self.lock:
            data = self._load(robot_id)
            number = int(data["next_number"])
            data["next_number"] = number + 1
            pid = f"p{number:04d}"
            now = time.time()
            data["people"][pid] = {"person_id": pid, "number": number, "label": "",
                                   "first_seen": now, "last_seen": now, "sightings": 1,
                                   "best_quality": 0.0, "embeddings": []}
            self.embeddings[robot_id][pid] = face.embedding[None].copy()
            self._store_crop(robot_id, pid, face)
            self.save(robot_id)
            return self.public(robot_id, pid)

    def add_sample(self, robot_id: str, pid: str, face: Face, similarity: float) -> None:
        with self.lock:
            person = self._load(robot_id)["people"].get(pid)
            if person is None:
                return
            person["last_seen"] = time.time()
            person["sightings"] = int(person.get("sightings", 0)) + 1
            stored = self.embeddings[robot_id][pid]
            # Keep only samples that add a new angle/lighting (not near-duplicates).
            if face.enrollable and float(np.max(stored @ face.embedding)) < 0.85:
                stored = np.vstack([stored, face.embedding[None]])[-MAX_EMBEDDINGS:]
                self.embeddings[robot_id][pid] = stored
            self._store_crop(robot_id, pid, face)
            self.dirty.add(robot_id)

    def _store_crop(self, robot_id: str, pid: str, face: Face) -> None:
        person = self.data[robot_id]["people"][pid]
        if face.quality > float(person.get("best_quality", 0.0)):
            cv2.imwrite(str(self._dir(robot_id) / f"{pid}.jpg"), face.crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            person["best_quality"] = round(face.quality, 4)

    def public(self, robot_id: str, pid: str) -> Dict[str, Any]:
        p = self._load(robot_id)["people"][pid]
        return {"person_id": pid, "number": p["number"], "label": p.get("label", ""),
                "name": p.get("label") or f"Persona {p['number']}",
                "first_seen": p["first_seen"], "last_seen": p["last_seen"],
                "sightings": p.get("sightings", 0),
                "samples": int(len(self.embeddings[robot_id].get(pid, ())))}

    def list(self, robot_id: str) -> List[Dict[str, Any]]:
        with self.lock:
            people = self._load(robot_id)["people"]
            return sorted((self.public(robot_id, pid) for pid in people), key=lambda p: p["number"])

    def crop_path(self, robot_id: str, pid: str) -> Optional[Path]:
        with self.lock:
            if pid not in self._load(robot_id)["people"]:
                return None
            path = self._dir(robot_id) / f"{pid}.jpg"
            return path if path.exists() else None

    def rename(self, robot_id: str, pid: str, label: str) -> bool:
        with self.lock:
            person = self._load(robot_id)["people"].get(pid)
            if person is None:
                return False
            person["label"] = label.strip()[:60]
            self.save(robot_id)
            return True

    def delete(self, robot_id: str, pid: Optional[str] = None) -> int:
        """Delete one person, or everyone (numbering restarts at 1)."""
        with self.lock:
            data = self._load(robot_id)
            targets = [pid] if pid else list(data["people"])
            removed = 0
            for target in targets:
                if data["people"].pop(target, None) is not None:
                    removed += 1
                self.embeddings[robot_id].pop(target, None)
                (self._dir(robot_id) / f"{target}.jpg").unlink(missing_ok=True)
            if pid is None:
                data["next_number"] = 1
            self.save(robot_id)
            return removed


@dataclass
class Track:
    track_id: int
    box: Tuple[float, float, float, float]
    last_seen: float
    votes: Dict[str, float] = field(default_factory=dict)
    person_id: Optional[str] = None
    similarity: float = 0.0
    unknown_hits: int = 0
    face_box: Optional[Tuple[float, float, float, float]] = None


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Tracker:
    def __init__(self, iou_threshold: float = 0.25):
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def update(self, boxes: List[Tuple[float, float, float, float]], now: float) -> List[Track]:
        """Return one track per box, in the same order as ``boxes``."""
        pairs = sorted(((iou(t.box, b), tid, i) for tid, t in self.tracks.items() for i, b in enumerate(boxes)),
                       reverse=True)
        used_tracks: set = set()
        result: List[Optional[Track]] = [None] * len(boxes)
        for score, tid, i in pairs:
            if score < self.iou_threshold:
                break
            if tid in used_tracks or result[i] is not None:
                continue
            used_tracks.add(tid)
            track = self.tracks[tid]
            track.box, track.last_seen, track.face_box = boxes[i], now, None
            result[i] = track
        for i, box in enumerate(boxes):
            if result[i] is None:
                result[i] = self.tracks[self.next_id] = Track(self.next_id, box, now)
                self.next_id += 1
        for tid in [tid for tid, t in self.tracks.items() if now - t.last_seen > TRACK_TTL_S]:
            del self.tracks[tid]
        return result


class IdentityResolver:
    """Turns per-frame face matches into a stable person number per track."""

    def __init__(self, gallery: PeopleGallery, match_threshold: float = DEFAULT_MATCH_THRESHOLD):
        self.gallery = gallery
        self.match_threshold = match_threshold

    def observe(self, robot_id: str, track: Track, face: Face) -> Optional[Dict[str, Any]]:
        """Feed one face seen inside ``track``. Returns the new person if one was created."""
        track.face_box = face.box
        pid, sim = self.gallery.match(robot_id, face.embedding)
        created = None
        if pid is not None and sim >= self.match_threshold:
            track.votes[pid] = track.votes.get(pid, 0.0) + sim
            track.unknown_hits = 0
            self.gallery.add_sample(robot_id, pid, face, sim)
        elif (pid is None or sim < NEW_PERSON_THRESHOLD) and face.enrollable and track.person_id is None:
            # Require the unknown face on two frames before enrolling (avoids
            # creating people out of a single blurry or partial detection).
            track.unknown_hits += 1
            if track.unknown_hits >= 2:
                created = self.gallery.create(robot_id, face)
                pid, sim = created["person_id"], 1.0
                track.votes[pid] = track.votes.get(pid, 0.0) + 1.0
                track.unknown_hits = 0
        if track.votes:
            best = max(track.votes, key=track.votes.get)
            if track.votes[best] >= 0.8 or (best == pid and sim >= 0.55):
                track.person_id = best
                if best == pid:
                    track.similarity = sim
        return created

    @staticmethod
    def dedupe(tracks: List[Track]) -> None:
        """One person number per frame: keep it on the track with the most evidence."""
        owners: Dict[str, Track] = {}
        for track in sorted(tracks, key=lambda t: t.votes.get(t.person_id or "", 0.0), reverse=True):
            if track.person_id is None:
                continue
            if track.person_id in owners:
                track.person_id = None
            else:
                owners[track.person_id] = track


class PeopleIdentifier:
    """Background worker: latest-only queue per robot, results via callbacks."""

    def __init__(self, models_dir: Path, people_dir: Path, target: str = "cpu",
                 match_threshold: float = DEFAULT_MATCH_THRESHOLD, max_fps: float = 5.0,
                 on_result: Optional[Callable[[str, Dict[str, Any]], None]] = None,
                 on_new_person: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        self.models_dir, self.target = models_dir, target
        self.gallery = PeopleGallery(people_dir)
        self.resolver = IdentityResolver(self.gallery, match_threshold)
        self.min_interval = 1.0 / max(max_fps, 0.1)
        self.on_result, self.on_new_person = on_result, on_new_person
        self.cond = threading.Condition()
        self.pending: Dict[str, Tuple[Dict[str, Any], bytes]] = {}
        self.trackers: Dict[str, Tracker] = {}
        self.last_run: Dict[str, float] = {}
        self.state = {"status": "idle", "error": "", "fps": 0.0, "ms": 0.0, "frames": 0}
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.detector: Optional[YoloxPersonDetector] = None
        self.faces: Optional[FaceEngine] = None

    def status(self) -> Dict[str, Any]:
        with self.cond:
            return {k: v for k, v in dict(self.state, target=self.target).items() if not k.startswith("_")}

    def submit(self, robot_id: str, header: Dict[str, Any], payload: bytes) -> None:
        with self.cond:
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, name="people-id", daemon=True)
                self.thread.start()
            self.pending[robot_id] = (header, payload)  # Drop older frames: latest only.
            self.cond.notify()

    def reset(self, robot_id: str) -> None:
        with self.cond:
            self.pending.pop(robot_id, None)
            self.trackers.pop(robot_id, None)

    def stop(self) -> None:
        self.stop_event.set()
        with self.cond:
            self.cond.notify_all()
        self.gallery.flush()

    def _load(self) -> None:
        with self.cond:
            self.state["status"] = "loading"
        paths = ensure_models(self.models_dir)
        self.detector = YoloxPersonDetector(paths["person"], self.target)
        self.faces = FaceEngine(paths["face_det"], paths["face_rec"], self.target)
        with self.cond:
            self.state["status"] = "ready"

    def _run(self) -> None:
        try:
            self._load()
        except Exception as exc:
            with self.cond:
                self.state.update(status="error", error=f"no se pudieron cargar los modelos: {exc}")
            return
        last_flush = time.monotonic()
        while not self.stop_event.is_set():
            with self.cond:
                while not self.pending and not self.stop_event.is_set():
                    self.cond.wait(timeout=1.0)
                if self.stop_event.is_set():
                    break
                robot_id = next(iter(self.pending))
                header, payload = self.pending.pop(robot_id)
            wait = self.min_interval - (time.monotonic() - self.last_run.get(robot_id, 0.0))
            if wait > 0:
                time.sleep(wait)
                with self.cond:  # A newer frame may have arrived while waiting.
                    header, payload = self.pending.pop(robot_id, (header, payload))
            self.last_run[robot_id] = time.monotonic()
            try:
                self._process(robot_id, header, payload)
            except Exception as exc:
                with self.cond:
                    self.state["error"] = f"{type(exc).__name__}: {exc}"
            if time.monotonic() - last_flush > 5.0:
                self.gallery.flush()
                last_flush = time.monotonic()

    def _process(self, robot_id: str, header: Dict[str, Any], payload: bytes) -> None:
        started = time.monotonic()
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        h, w = image.shape[:2]
        now = time.monotonic()
        bodies = self.detector.detect(image)
        tracker = self.trackers.setdefault(robot_id, Tracker())
        candidates: List[Tuple[Tuple[float, float, float, float], List[Face]]] = []
        if bodies:
            for x1, y1, x2, y2, _ in bodies:
                # Faces live in the upper part of the body box.
                top = (int(x1), int(y1), int(x2), int(y1 + max(0.55 * (y2 - y1), (x2 - x1))))
                found = self.faces.faces(image, (top[0], top[1], top[2], min(h, top[3])))
                candidates.append(((x1, y1, x2, y2), found[:1]))
        else:
            # Close-ups where the body detector misses: track the face itself.
            for face in self.faces.faces(image, (0, 0, w, h)):
                candidates.append((face.box, [face]))
        tracks = tracker.update([box for box, _ in candidates], now)
        new_people = []
        for track, (_, found) in zip(tracks, candidates):
            for face in found:
                created = self.resolver.observe(robot_id, track, face)
                if created:
                    new_people.append(created)
        IdentityResolver.dedupe(tracks)
        people = []
        for track in tracks:
            info = self.gallery.public(robot_id, track.person_id) if track.person_id else None
            x1, y1, x2, y2 = (float(v) for v in track.box)  # numpy -> JSON-safe
            entry = {"track_id": track.track_id,
                     "person_id": track.person_id,
                     "number": info["number"] if info else None,
                     "name": info["name"] if info else None,
                     "similarity": round(float(track.similarity), 3) if info else None,
                     "box": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]}
            if track.face_box:
                fx1, fy1, fx2, fy2 = (float(v) for v in track.face_box)
                entry["face_box"] = [round(fx1 / w, 4), round(fy1 / h, 4), round(fx2 / w, 4), round(fy2 / h, 4)]
            people.append(entry)
        elapsed = time.monotonic() - started
        with self.cond:
            previous = self.state.get("_done_at")
            done_at = time.monotonic()
            if previous:
                self.state["fps"] = round(0.8 * self.state["fps"] + 0.2 / max(done_at - previous, 1e-3), 2)
            self.state.update(_done_at=done_at, frames=self.state["frames"] + 1,
                              ms=round(elapsed * 1000, 1), error="")
        if self.on_result:
            self.on_result(robot_id, {"type": "people", "robot_id": robot_id, "stream": "arducam",
                                      "seq": header.get("seq"), "ts": header.get("ts"),
                                      "width": w, "height": h, "people": people,
                                      "processing_ms": round(elapsed * 1000, 1)})
        for person in new_people:
            if self.on_new_person:
                self.on_new_person(robot_id, person)
