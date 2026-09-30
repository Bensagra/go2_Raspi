"""Detección térmica y render JPEG sin ventana ni dependencia del hardware."""
import math
import time
from collections import deque

import cv2
import numpy as np

from termica.detector_csv import (
    AREA_MIN_PERSONA, FRAMES_PARA_CONFIRMAR, FRAMES_PARA_DESCARTAR,
    TEMP_PIEL_MAX, TEMP_PIEL_MIN, detectar_personas,
)
from termica.protocol import decode_csv


class ThermalProcessor:
    def __init__(self, rotation_deg=270):
        if type(rotation_deg) is not int or rotation_deg not in (0, 90, 180, 270):
            raise ValueError("La rotacion termica debe ser 0, 90, 180 o 270 grados")
        self.rotation_deg = rotation_deg
        self.history = deque(maxlen=3)
        self.present = False
        self.with_blob = self.without_blob = 0
        self.last_at = 0.0
        self.session = None
        self.last_seq = -1

    def process(self, header, payload):
        frame = decode_csv(header, payload)
        # Rotate temperatures before detection/rendering so regions and labels
        # share the displayed orientation, including recordings and other UIs.
        frame = np.ascontiguousarray(np.rot90(frame, k=-(self.rotation_deg // 90)))
        now = time.monotonic()
        seq = header.get("seq")
        session = header.get("session_id")
        ts = header.get("ts")
        if (type(seq) is not int or seq < 0 or not isinstance(session, str)
                or not session or len(session) > 128
                or type(ts) not in (int, float) or not math.isfinite(ts)):
            raise ValueError("Metadatos termicos invalidos")
        reset = (session != self.session or now - self.last_at > 3.0
                 or (self.history and frame.shape != self.history[-1].shape))
        if not reset and seq <= self.last_seq:
            raise ValueError("Cuadro termico duplicado o fuera de orden")
        if reset:
            self.history.clear()
            self.present = False
            self.with_blob = self.without_blob = 0
        self.session, self.last_seq, self.last_at = session, seq, now
        self.history.append(frame)
        celsius = cv2.medianBlur(np.mean(self.history, axis=0).astype(np.float32), 3)
        boxes, _ = detectar_personas(celsius, TEMP_PIEL_MIN, TEMP_PIEL_MAX, AREA_MIN_PERSONA)
        if boxes:
            self.with_blob += 1
            self.without_blob = 0
        else:
            self.without_blob += 1
            self.with_blob = 0
        if self.with_blob >= FRAMES_PARA_CONFIRMAR:
            self.present = True
        elif self.without_blob >= FRAMES_PARA_DESCARTAR:
            self.present = False

        low, high = float(celsius.min()), float(celsius.max())
        gray = np.clip((celsius - low) / max(high - low, 0.5) * 255, 0, 255).astype(np.uint8)
        height, width = celsius.shape
        scale = min(4.0, 640.0 / width, 480.0 / height)
        out_width, out_height = round(width * scale), round(height * scale)
        image = cv2.resize(cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO),
                           (out_width, out_height), interpolation=cv2.INTER_CUBIC)
        detections = []
        for x, y, w, h, area, peak in boxes:
            x, y, w, h = map(int, (x, y, w, h))
            detections.append({"x": x, "y": y, "width": w, "height": h,
                               "area": area, "max_c": peak})
            p1 = (round(x * scale), round(y * scale))
            p2 = (round((x + w) * scale), round((y + h) * scale))
            cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
        cv2.putText(image, f"{low:.1f} - {high:.1f} C", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise ValueError("No se pudo generar el JPEG termico")
        return {
            "type": "media", "stream": "thermal", "image_format": "jpeg",
            "width": out_width, "height": out_height,
            "source_width": width, "source_height": height,
            "rotation_deg": self.rotation_deg,
            "ts": ts, "seq": seq, "session_id": session,
            "temperature": {"min_c": low, "max_c": high,
                            "center_c": float(celsius[height // 2, width // 2])},
            "detection": {"person_present": self.present, "regions": detections,
                          "method": "temperature_area_heuristic",
                          "temp_min_c": TEMP_PIEL_MIN, "temp_max_c": TEMP_PIEL_MAX,
                          "area_min_pixels": AREA_MIN_PERSONA},
        }, encoded.tobytes()
