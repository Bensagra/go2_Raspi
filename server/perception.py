"""Server-side perception: decode the robot camera, detect people, and run face
detection + recognition. All the heavy analysis lives here (the edge only ships
pixels), matching the "do all analysis on the server" requirement.

Design notes
------------
* The ML stack (ultralytics YOLO for people, insightface for face embeddings) is
  *optional and lazily imported*. The module imports cleanly on a machine without
  torch/CUDA so the server keeps running in mapping-only mode; perception simply
  reports ``available=False`` until the libs are installed on the GPU box::

      pip install ultralytics insightface onnxruntime-gpu

* Frame decoding uses PyAV for H.264 and OpenCV for webp/jpg (both already
  dependencies of this project), so it works regardless of the operator's chosen
  camera format.
* Face recognition keeps a small persistent gallery (embeddings + best crops) so
  the same person is not captured many times and known/unknown identities are
  tracked. Retention is configurable for privacy.

Frames are handled as BGR uint8 ndarrays (OpenCV convention).
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- decode
class FrameDecoder:
    """Decode incoming media frames to BGR ndarrays. Keeps one H.264 decoder per
    robot (stateful) and falls back to OpenCV for still formats."""

    def __init__(self) -> None:
        self._av = None
        self._cv2 = None
        self._h264: Dict[str, Any] = {}
        try:
            import av  # noqa: F401

            self._av = av
        except Exception:
            self._av = None
        try:
            import cv2  # noqa: F401

            self._cv2 = cv2
        except Exception:
            self._cv2 = None

    def reset(self, robot_id: str) -> None:
        self._h264.pop(robot_id, None)

    def decode(self, robot_id: str, header: Dict[str, Any], payload: bytes) -> Optional[np.ndarray]:
        fmt = str(header.get("image_format", "")).lower()
        if fmt == "h264":
            return self._decode_h264(robot_id, payload)
        if fmt in {"webp", "jpg", "jpeg"} and self._cv2 is not None:
            import numpy as _np

            arr = _np.frombuffer(payload, dtype=_np.uint8)
            img = self._cv2.imdecode(arr, self._cv2.IMREAD_COLOR)
            return img
        return None

    def _decode_h264(self, robot_id: str, payload: bytes) -> Optional[np.ndarray]:
        if self._av is None:
            return None
        ctx = self._h264.get(robot_id)
        if ctx is None:
            ctx = self._av.codec.CodecContext.create("h264", "r")
            self._h264[robot_id] = ctx
        last = None
        try:
            packet = self._av.packet.Packet(payload)
            for frame in ctx.decode(packet):
                last = frame.to_ndarray(format="bgr24")
        except Exception:
            # A mid-stream decoder hiccup: drop the context so the next keyframe
            # rebuilds it.
            self._h264.pop(robot_id, None)
            return None
        return last


# ------------------------------------------------------------------------ detection
@dataclass
class PersonBox:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class PersonDetector:
    """YOLO person detector (ultralytics). Lazily loaded; degrades to no-op."""

    def __init__(self, model: str = "yolov8n.pt", device: str = "cuda", conf: float = 0.45) -> None:
        self.model_name = model
        self.device = device
        self.conf = conf
        self._model = None
        self._available: Optional[bool] = None
        self._reason = ""

    @property
    def available(self) -> bool:
        if self._available is None:
            self._try_load()
        return bool(self._available)

    def _try_load(self) -> None:
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
            # Warm move to device (ignored if CPU-only torch).
            try:
                self._model.to(self.device)
            except Exception:
                self.device = "cpu"
            self._available = True
            self._reason = f"ultralytics {self.model_name} on {self.device}"
        except Exception as exc:  # torch / ultralytics not installed
            self._available = False
            self._reason = f"unavailable: {exc}"

    def detect(self, bgr: np.ndarray) -> List[PersonBox]:
        if not self.available or bgr is None:
            return []
        try:
            res = self._model.predict(
                bgr, classes=[0], conf=self.conf, device=self.device, verbose=False
            )
        except Exception:
            return []
        boxes: List[PersonBox] = []
        for r in res:
            for b in getattr(r, "boxes", []):
                xyxy = b.xyxy[0].tolist()
                score = float(b.conf[0]) if b.conf is not None else 0.0
                boxes.append(PersonBox(xyxy[0], xyxy[1], xyxy[2], xyxy[3], score))
        return boxes


@dataclass
class FaceResult:
    bbox: Tuple[float, float, float, float]
    det_score: float
    quality: float
    embedding: Optional[np.ndarray] = None
    crop: Optional[np.ndarray] = None


class FaceAnalyzer:
    """Face detection + embedding via insightface (buffalo_l). Lazily loaded.
    Falls back to OpenCV Haar detection (crop only, no embedding) when insightface
    is missing, so captures still work without recognition."""

    def __init__(self, device: str = "cuda", det_size: int = 640) -> None:
        self.device = device
        self.det_size = det_size
        self._app = None
        self._cv2 = None
        self._haar = None
        self._available: Optional[bool] = None
        self._recognition = False
        self._reason = ""

    @property
    def available(self) -> bool:
        if self._available is None:
            self._try_load()
        return bool(self._available)

    @property
    def has_recognition(self) -> bool:
        if self._available is None:
            self._try_load()
        return self._recognition

    def _try_load(self) -> None:
        try:
            from insightface.app import FaceAnalysis

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device == "cuda"
                else ["CPUExecutionProvider"]
            )
            app = FaceAnalysis(name="buffalo_l", providers=providers)
            app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(self.det_size, self.det_size))
            self._app = app
            self._available = True
            self._recognition = True
            self._reason = f"insightface buffalo_l ({self.device})"
            return
        except Exception as exc:
            self._reason = f"insightface unavailable: {exc}"
        # Fallback: Haar cascade (detection only).
        try:
            import cv2

            self._cv2 = cv2
            self._haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self._available = True
            self._recognition = False
            self._reason += " | fallback: opencv haar (no embeddings)"
        except Exception as exc:
            self._available = False
            self._reason += f" | opencv unavailable: {exc}"

    @staticmethod
    def _sharpness(crop: np.ndarray) -> float:
        try:
            import cv2

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            return 0.0

    def analyze(self, bgr: np.ndarray) -> List[FaceResult]:
        if not self.available or bgr is None:
            return []
        if self._app is not None:
            return self._analyze_insightface(bgr)
        return self._analyze_haar(bgr)

    def _analyze_insightface(self, bgr: np.ndarray) -> List[FaceResult]:
        try:
            faces = self._app.get(bgr)
        except Exception:
            return []
        out: List[FaceResult] = []
        h, w = bgr.shape[:2]
        for f in faces:
            x1, y1, x2, y2 = [float(v) for v in f.bbox]
            xi1, yi1 = max(0, int(x1)), max(0, int(y1))
            xi2, yi2 = min(w, int(x2)), min(h, int(y2))
            crop = bgr[yi1:yi2, xi1:xi2].copy() if xi2 > xi1 and yi2 > yi1 else None
            emb = getattr(f, "normed_embedding", None)
            if emb is not None:
                emb = np.asarray(emb, dtype=np.float32)
            face_h = y2 - y1
            sharp = self._sharpness(crop) if crop is not None else 0.0
            quality = self._quality(face_h, h, float(f.det_score), sharp)
            out.append(
                FaceResult(
                    bbox=(x1, y1, x2, y2),
                    det_score=float(f.det_score),
                    quality=quality,
                    embedding=emb,
                    crop=crop,
                )
            )
        return out

    def _analyze_haar(self, bgr: np.ndarray) -> List[FaceResult]:
        gray = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2GRAY)
        rects = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
        out: List[FaceResult] = []
        h = bgr.shape[0]
        for (x, y, fw, fh) in rects:
            crop = bgr[y : y + fh, x : x + fw].copy()
            sharp = self._sharpness(crop)
            out.append(
                FaceResult(
                    bbox=(float(x), float(y), float(x + fw), float(y + fh)),
                    det_score=1.0,
                    quality=self._quality(float(fh), h, 1.0, sharp),
                    embedding=None,
                    crop=crop,
                )
            )
        return out

    @staticmethod
    def _quality(face_h: float, img_h: int, det_score: float, sharpness: float) -> float:
        """Heuristic 0..1 capture quality: bigger, sharper, higher-confidence
        faces score higher. Used to keep the single best shot per person."""
        size_term = min(1.0, (face_h / max(img_h, 1)) / 0.30)  # ~30% of height = full
        sharp_term = min(1.0, sharpness / 120.0)
        return round(float(0.5 * size_term + 0.3 * sharp_term + 0.2 * det_score), 4)


# -------------------------------------------------------------------------- gallery
@dataclass
class Person:
    person_id: str
    label: str
    known: bool
    embeddings: List[np.ndarray] = field(default_factory=list)
    best_quality: float = 0.0
    best_crop_path: str = ""
    first_seen: float = 0.0
    last_seen: float = 0.0
    captures: int = 0

    def mean_embedding(self) -> Optional[np.ndarray]:
        if not self.embeddings:
            return None
        m = np.mean(np.stack(self.embeddings), axis=0)
        n = np.linalg.norm(m)
        return (m / n).astype(np.float32) if n > 0 else m.astype(np.float32)


class FaceGallery:
    """Persistent per-robot gallery of recognised people. Thread-safe."""

    def __init__(self, base_dir: Path, match_threshold: float = 0.35, retention_days: float = 0.0) -> None:
        self.base_dir = Path(base_dir)
        self.match_threshold = match_threshold
        self.retention_days = retention_days
        self._lock = threading.RLock()
        self._people: Dict[str, Dict[str, Person]] = {}  # robot -> person_id -> Person
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _robot_dir(self, robot_id: str) -> Path:
        safe = "".join(c for c in robot_id if c.isalnum() or c in ("_", "-")) or "robot"
        d = self.base_dir / safe
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load(self) -> None:
        for robot_dir in self.base_dir.glob("*"):
            if not robot_dir.is_dir():
                continue
            index = robot_dir / "index.json"
            if not index.exists():
                continue
            try:
                data = json.loads(index.read_text())
            except Exception:
                continue
            people: Dict[str, Person] = {}
            for pid, meta in data.get("people", {}).items():
                emb_path = robot_dir / f"{pid}.npy"
                embs: List[np.ndarray] = []
                if emb_path.exists():
                    try:
                        arr = np.load(emb_path)
                        embs = [row.astype(np.float32) for row in np.atleast_2d(arr)]
                    except Exception:
                        embs = []
                people[pid] = Person(
                    person_id=pid,
                    label=meta.get("label", pid),
                    known=bool(meta.get("known", False)),
                    embeddings=embs,
                    best_quality=float(meta.get("best_quality", 0.0)),
                    best_crop_path=meta.get("best_crop_path", ""),
                    first_seen=float(meta.get("first_seen", 0.0)),
                    last_seen=float(meta.get("last_seen", 0.0)),
                    captures=int(meta.get("captures", 0)),
                )
            self._people[robot_dir.name] = people

    def _save(self, robot_id: str) -> None:
        robot_dir = self._robot_dir(robot_id)
        people = self._people.get(robot_id, {})
        meta = {
            "people": {
                pid: {
                    "label": p.label,
                    "known": p.known,
                    "best_quality": p.best_quality,
                    "best_crop_path": p.best_crop_path,
                    "first_seen": p.first_seen,
                    "last_seen": p.last_seen,
                    "captures": p.captures,
                }
                for pid, p in people.items()
            }
        }
        (robot_dir / "index.json").write_text(json.dumps(meta, indent=2))
        for pid, p in people.items():
            if p.embeddings:
                np.save(robot_dir / f"{pid}.npy", np.stack(p.embeddings))

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return -1.0
        return float(np.dot(a, b) / (na * nb))

    def match_or_create(
        self,
        robot_id: str,
        face: FaceResult,
        cv2_module: Any,
        recognition: bool,
    ) -> Tuple[Person, bool, float]:
        """Return (person, is_new, score). With embeddings, match against the
        gallery; otherwise always create a new entry (no identity)."""
        now = time.time()
        with self._lock:
            people = self._people.setdefault(robot_id, {})
            best_pid, best_score = None, -1.0
            if recognition and face.embedding is not None:
                for pid, p in people.items():
                    me = p.mean_embedding()
                    if me is None:
                        continue
                    s = self._cosine(face.embedding, me)
                    if s > best_score:
                        best_score, best_pid = s, pid

            is_new = not (best_pid is not None and best_score >= self.match_threshold)
            if is_new:
                pid = f"person_{len(people) + 1:04d}_{int(now)}"
                person = Person(
                    person_id=pid,
                    label=pid,
                    known=False,
                    first_seen=now,
                    last_seen=now,
                )
                people[pid] = person
            else:
                person = people[best_pid]
                person.last_seen = now

            person.captures += 1
            if recognition and face.embedding is not None and len(person.embeddings) < 12:
                person.embeddings.append(face.embedding)

            # Keep the best crop on disk for the gallery thumbnail.
            if face.crop is not None and face.quality >= person.best_quality and cv2_module is not None:
                robot_dir = self._robot_dir(robot_id)
                crop_path = robot_dir / f"{person.person_id}.jpg"
                try:
                    cv2_module.imwrite(str(crop_path), face.crop)
                    person.best_crop_path = crop_path.name
                    person.best_quality = face.quality
                except Exception:
                    pass

            self._save(robot_id)
            return person, is_new, (best_score if not is_new else 0.0)

    def list_people(self, robot_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            people = self._people.get(robot_id, {})
            return [
                {
                    "person_id": p.person_id,
                    "label": p.label,
                    "known": p.known,
                    "captures": p.captures,
                    "best_quality": p.best_quality,
                    "first_seen": p.first_seen,
                    "last_seen": p.last_seen,
                    "crop": p.best_crop_path,
                }
                for p in sorted(people.values(), key=lambda q: q.last_seen, reverse=True)
            ]

    def crop_path(self, robot_id: str, person_id: str) -> Optional[Path]:
        with self._lock:
            p = self._people.get(robot_id, {}).get(person_id)
            if not p or not p.best_crop_path:
                return None
            path = self._robot_dir(robot_id) / p.best_crop_path
            return path if path.exists() else None

    def set_label(self, robot_id: str, person_id: str, label: str, known: bool) -> bool:
        with self._lock:
            p = self._people.get(robot_id, {}).get(person_id)
            if not p:
                return False
            p.label = label
            p.known = known
            self._save(robot_id)
            return True

    def purge(self, robot_id: str, person_id: Optional[str] = None) -> int:
        """Delete one person or the whole robot gallery (privacy)."""
        with self._lock:
            people = self._people.get(robot_id, {})
            robot_dir = self._robot_dir(robot_id)
            removed = 0
            targets = [person_id] if person_id else list(people.keys())
            for pid in targets:
                if people.pop(pid, None) is not None:
                    removed += 1
                for suffix in (".npy", ".jpg"):
                    fp = robot_dir / f"{pid}{suffix}"
                    if fp.exists():
                        fp.unlink()
            self._save(robot_id)
            return removed

    def enforce_retention(self) -> int:
        if self.retention_days <= 0:
            return 0
        cutoff = time.time() - self.retention_days * 86400.0
        removed = 0
        with self._lock:
            for robot_id, people in list(self._people.items()):
                for pid, p in list(people.items()):
                    if not p.known and p.last_seen < cutoff:
                        removed += self.purge(robot_id, pid)
        return removed


# ------------------------------------------------------------------------- facade
class Perception:
    """Facade owned by the server runtime: decode frames, detect people, analyse
    faces, and persist the gallery. Safe to construct without the ML stack."""

    def __init__(
        self,
        *,
        faces_dir: Path,
        device: str = "cuda",
        person_model: str = "yolov8n.pt",
        person_conf: float = 0.45,
        match_threshold: float = 0.35,
        retention_days: float = 0.0,
        enable_person: bool = True,
        enable_face: bool = True,
    ) -> None:
        self.decoder = FrameDecoder()
        self.person = PersonDetector(person_model, device, person_conf) if enable_person else None
        self.face = FaceAnalyzer(device) if enable_face else None
        self.gallery = FaceGallery(faces_dir, match_threshold, retention_days)
        self._frames: Dict[str, Tuple[np.ndarray, float]] = {}
        self._lock = threading.RLock()
        try:
            import cv2

            self._cv2 = cv2
        except Exception:
            self._cv2 = None

    def ingest(self, robot_id: str, header: Dict[str, Any], payload: bytes) -> None:
        frame = self.decoder.decode(robot_id, header, payload)
        if frame is not None and frame.size:
            with self._lock:
                self._frames[robot_id] = (frame, time.time())

    def latest_frame(self, robot_id: str, max_age_s: float = 2.0) -> Optional[np.ndarray]:
        with self._lock:
            item = self._frames.get(robot_id)
        if not item:
            return None
        frame, ts = item
        if time.time() - ts > max_age_s:
            return None
        return frame

    def detect_people(self, robot_id: str) -> Tuple[List[PersonBox], Optional[Tuple[int, int]]]:
        frame = self.latest_frame(robot_id)
        if frame is None or self.person is None:
            return [], None
        h, w = frame.shape[:2]
        return self.person.detect(frame), (w, h)

    def capture_face(self, robot_id: str) -> Optional[Dict[str, Any]]:
        """Analyse the latest frame, keep the best face, recognise + persist it."""
        if self.face is None:
            return None
        frame = self.latest_frame(robot_id)
        if frame is None:
            return None
        faces = self.face.analyze(frame)
        if not faces:
            return None
        best = max(faces, key=lambda f: f.quality)
        person, is_new, score = self.gallery.match_or_create(
            robot_id, best, self._cv2, self.face.has_recognition
        )
        return {
            "person_id": person.person_id,
            "label": person.label,
            "known": person.known,
            "is_new": is_new,
            "match_score": round(score, 3),
            "quality": best.quality,
            "recognition": self.face.has_recognition,
            "bbox": [round(v, 1) for v in best.bbox],
            "captures": person.captures,
        }

    def capabilities(self) -> Dict[str, Any]:
        return {
            "person_detector": {
                "enabled": self.person is not None,
                "available": bool(self.person and self.person.available),
                "info": (self.person._reason if self.person else "disabled"),
            },
            "face": {
                "enabled": self.face is not None,
                "available": bool(self.face and self.face.available),
                "recognition": bool(self.face and self.face.has_recognition),
                "info": (self.face._reason if self.face else "disabled"),
            },
        }
