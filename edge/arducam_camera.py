"""B0541 on Pi 5: direct V4L2, bounded JPEG mailbox and killable capture worker.

The existing four-lane overlay must already be installed. No boot files or drivers
are changed here. A separate process contains blocking V4L2 reads.
"""
import contextlib
import multiprocessing as mp
import queue
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

import cv2

WIDTH, HEIGHT = 3840, 2160


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=5)
    if result.returncode:
        raise RuntimeError(f"{args[0]}: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def discover(media=None):
    matches = []
    for candidate in [media] if media else sorted(Path('/dev').glob('media*')):
        try:
            topology = command('media-ctl', '-d', str(candidate), '--print-topology')
        except RuntimeError:
            continue
        sensors = re.findall(r'entity \d+: (arducam-pivariety \d+-[0-9a-f]+) \(', topology)
        if sensors:
            if len(sensors) != 1:
                raise RuntimeError('Más de una Arducam en el mismo grafo multimedia')
            video = command('media-ctl', '-d', str(candidate), '--entity', 'rp1-cfe-csi2_ch0')
            if not re.fullmatch(r'/dev/video\d+', video):
                raise RuntimeError(f'Nodo de captura inesperado: {video}')
            matches.append((str(candidate), sensors[0], video))
    if not matches:
        raise RuntimeError('El kernel no detectó la Arducam; revisar CSI1 y el overlay de cuatro líneas')
    if len(matches) != 1:
        raise RuntimeError('Hay varias Arducam; seleccionar una con --arducam-media')
    return matches[0]


def prepare(media, sensor, video):
    command('media-ctl', '-d', media, '--links', '"csi2":4 -> "rp1-cfe-csi2_ch0":0 [1]')
    command('media-ctl', '-d', media, '--set-v4l2',
            f'"{sensor}":0 [fmt:UYVY8_1X16/{WIDTH}x{HEIGHT} field:none]')
    for pad in (0, 4):
        command('media-ctl', '-d', media, '--set-v4l2',
                f'"csi2":{pad} [fmt:UYVY8_1X16/{WIDTH}x{HEIGHT} field:none '
                'colorspace:raw xfer:none ycbcr:601 quantization:full-range]')
    command('v4l2-ctl', '-d', video,
            f'--set-fmt-video=width={WIDTH},height={HEIGHT},pixelformat=UYVY')


class V4L2Camera:
    def __init__(self, media=None, prepare_graph=True):
        import fcntl  # Linux only; importing the gateway on other hosts remains possible.
        self.cap = None
        self.owner = None
        try:
            media, sensor, self.video = discover(media)
            # Cooperative lock across gateway instances, held BEFORE graph changes.
            self.owner = open(media, 'rb')
            fcntl.flock(self.owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if prepare_graph:
                prepare(media, sensor, self.video)
            self.cap = cv2.VideoCapture(self.video, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise RuntimeError(f'No se pudo abrir {self.video}')
            for prop, value in (
                (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'UYVY')),
                (cv2.CAP_PROP_FRAME_WIDTH, WIDTH), (cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT),
                (cv2.CAP_PROP_BUFFERSIZE, 2), (cv2.CAP_PROP_CONVERT_RGB, 1),
            ):
                self.cap.set(prop, value)
            actual = tuple(int(self.cap.get(p)) for p in (
                cv2.CAP_PROP_FOURCC, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
            if actual != (cv2.VideoWriter_fourcc(*'UYVY'), WIDTH, HEIGHT):
                raise RuntimeError(f'Formato V4L2 inesperado: {actual}; se requiere UYVY 3840x2160')
        except BaseException:
            self.close()
            raise

    def read(self):
        ok, image = self.cap.read()
        if not ok or image is None:
            raise RuntimeError('La Arducam no entrega cuadros')
        if image.shape != (HEIGHT, WIDTH, 3) or str(image.dtype) != 'uint8':
            raise RuntimeError(f'Imagen BGR inesperada: {image.shape} {image.dtype}')
        return image

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.owner is not None:
            self.owner.close()
            self.owner = None


def encode_packet(image, config, session, seq, captured_at, captured_monotonic):
    source_h, source_w = image.shape[:2]
    width = min(config['max_width'], source_w)
    height = max(2, round(source_h * width / source_w))
    if width != source_w:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, config['quality']])
    if not ok:
        raise RuntimeError('No se pudo codificar el cuadro Arducam')
    payload = encoded.tobytes()
    return {'binary': True, 'stream': 'arducam', 'payload': payload,
            'captured_monotonic': captured_monotonic,
            'header': {'stream': 'arducam', 'image_format': 'jpg', 'source': 'arducam_b0541',
                       'width': width, 'height': height, 'source_width': source_w,
                       'source_height': source_h, 'session_id': session, 'seq': seq,
                       'ts': captured_at, 'encoded_bytes': len(payload)}}


def put_latest(mailbox, value):
    try:
        mailbox.put_nowait(value)
    except queue.Full:
        with contextlib.suppress(queue.Empty):
            mailbox.get_nowait()
        with contextlib.suppress(queue.Full):
            mailbox.put_nowait(value)


def capture_worker(config, mailbox, stop):
    camera = None
    try:
        cv2.setNumThreads(1)
        camera = V4L2Camera(config['media'], config['prepare_graph'])
        session, seq, last_emit = uuid.uuid4().hex, 0, float('-inf')
        while not stop.is_set():
            image = camera.read()  # Always drain the sensor, even between uplink frames.
            now, captured_at = time.monotonic(), time.time()
            if now - last_emit < 1 / config['fps']:
                continue
            seq += 1
            packet = encode_packet(image, config, session, seq, captured_at, now)
            packet['header']['device'] = camera.video
            put_latest(mailbox, packet)
            last_emit = now
    except Exception as exc:
        put_latest(mailbox, {'error': str(exc)})
    finally:
        if camera is not None:
            camera.close()
        # Flush diagnostics; the supervisor bounds shutdown even if IPC stalls.
        mailbox.close()


class ArducamCamera:
    def __init__(self, media=None, fps=5.0, max_width=1280, quality=75, prepare_graph=True,
                 retry_s=3.0, timeout_s=8.0, startup_timeout_s=40.0,
                 worker=capture_worker):
        self.config = dict(media=media, fps=fps, max_width=max_width, quality=quality,
                           prepare_graph=prepare_graph)
        self.retry_s, self.timeout_s, self.startup_timeout_s = retry_s, timeout_s, startup_timeout_s
        self.worker = worker
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.state = dict(connected=False, frames=0, last_frame_ts=0, error='')
        self.thread = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name='arducam-supervisor', daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def status(self):
        with self.lock:
            return dict(self.state)

    def take_latest(self):
        with self.lock:
            packet, self.latest = self.latest, None
        if packet and time.monotonic() - packet.pop('captured_monotonic') < 3:
            return packet
        return None

    def _run(self):
        ctx = mp.get_context('spawn')  # No fork of WebRTC/OpenCV threads.
        while not self.stop_event.is_set():
            mailbox, stop = ctx.Queue(maxsize=1), ctx.Event()
            process = ctx.Process(target=self.worker, args=(self.config, mailbox, stop), daemon=True)
            try:
                process.start()
                deadline = time.monotonic() + self.startup_timeout_s
                while not self.stop_event.is_set():
                    try:
                        packet = mailbox.get(timeout=0.1)
                    except queue.Empty:
                        if not process.is_alive():
                            raise RuntimeError(f'Captura Arducam finalizó (código {process.exitcode})')
                        if time.monotonic() > deadline:
                            raise TimeoutError('Captura Arducam bloqueada; reiniciando V4L2')
                        continue
                    if 'error' in packet:
                        raise RuntimeError(packet['error'])
                    if time.monotonic() - packet['captured_monotonic'] >= 3:
                        continue
                    deadline = time.monotonic() + self.timeout_s
                    with self.lock:
                        self.latest = packet
                        self.state.update(connected=True, error='', last_frame_ts=packet['header']['ts'],
                                          device=packet['header'].get('device'))
                        self.state['frames'] += 1
            except Exception as exc:
                with self.lock:
                    self.state['error'] = str(exc)
            finally:
                stop.set()
                if process.pid is not None:
                    process.join(timeout=0.5)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=1)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=1)
                    if process.is_alive():
                        # Never open a second owner if the kernel cannot release the first.
                        self.stop_event.set()
                    else:
                        process.close()
                mailbox.close()
                with self.lock:
                    self.latest = None
                    self.state['connected'] = False
            self.stop_event.wait(self.retry_s)


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Capturar un JPEG de diagnóstico sin abrir ventanas')
    parser.add_argument('--media', default=None)
    parser.add_argument('--output', default='arducam-check.jpg')
    args = parser.parse_args()
    camera = ArducamCamera(media=args.media)
    camera.start()
    try:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            packet = camera.take_latest()
            if packet:
                Path(args.output).write_bytes(packet['payload'])
                print(json.dumps(packet['header'], indent=2))
                print(f'Imagen guardada: {args.output}')
                return
            if camera.status()['error']:
                raise SystemExit(camera.status()['error'])
            time.sleep(0.1)
        raise SystemExit('Tiempo de espera agotado')
    finally:
        camera.stop()


if __name__ == '__main__':
    main()
