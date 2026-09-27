"""Lectura USB independiente del robot y de la red; buzón de un solo cuadro."""
import contextlib
import threading
import time
import uuid

from termica.camara_csv import abrir
from termica.protocol import encode_csv, validate_frame


class ThermalCamera:
    def __init__(self, port=None, fps=8.0, emissivity=None, retry_s=3.0, timeout_s=5.0,
                 opener=abrir):
        self.port = port
        self.fps = fps
        self.emissivity = emissivity
        self.retry_s = retry_s
        self.timeout_s = timeout_s
        self.opener = opener
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.state = {"connected": False, "frames": 0, "last_frame_ts": 0, "error": ""}
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, name="thermal-camera", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    def status(self):
        with self.lock:
            return dict(self.state)

    def take_latest(self):
        with self.lock:
            packet, self.latest = self.latest, None
        if packet is not None and time.monotonic() - packet.pop("captured_monotonic") > self.timeout_s:
            return None
        return packet

    def _run(self):
        while not self.stop_event.is_set():
            dev = None
            try:
                dev = self.opener(self.port)
                if self.emissivity is not None:
                    dev.set_emissivity(self.emissivity)
                dev.start_stream()
                session = uuid.uuid4().hex
                seq = 0
                last_valid = time.monotonic()
                last_emit = float("-inf")
                while not self.stop_event.is_set():
                    _, frame = dev.read(block=False)
                    now = time.monotonic()
                    if frame is None:
                        if now - last_valid > self.timeout_s:
                            raise TimeoutError("La camara termica no entrega cuadros")
                        self.stop_event.wait(0.01)
                        continue
                    frame = validate_frame(frame)
                    last_valid = now
                    if now - last_emit < 1.0 / self.fps:
                        continue
                    captured_ts = time.time()
                    payload = encode_csv(frame)
                    seq += 1
                    packet = {
                        "binary": True, "stream": "thermal_csv", "payload": payload,
                        "captured_monotonic": now,
                        "header": {"stream": "thermal_csv", "format": "csv_zlib",
                                   "unit": "celsius", "width": frame.shape[1],
                                   "height": frame.shape[0], "ts": captured_ts,
                                   "session_id": session, "seq": seq},
                    }
                    with self.lock:
                        self.latest = packet
                        self.state.update(connected=True, error="", last_frame_ts=captured_ts)
                        self.state["frames"] += 1
                    last_emit = now
            except Exception as exc:
                with self.lock:
                    self.latest = None
                    self.state.update(connected=False, error=str(exc))
            finally:
                if dev is not None:
                    with contextlib.suppress(Exception):
                        dev.stop_stream()
                    with contextlib.suppress(Exception):
                        dev.close()
                with self.lock:
                    self.state["connected"] = False
            self.stop_event.wait(self.retry_s)
