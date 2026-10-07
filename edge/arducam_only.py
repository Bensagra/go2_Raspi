#!/usr/bin/env python3
"""Prueba aislada: Arducam -> servidor -> front, sin Go2, MQTT ni WebRTC.

Usa la misma captura (edge.arducam_camera) y el mismo protocolo binario A7/v1
que el gateway, pero imprime en consola el estado de la cámara y del envío.

    python -m edge.arducam_only --server ws://IP_DEL_SERVIDOR:8000
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import websockets

from edge.arducam_camera import ArducamCamera

MEDIA_FRAME_MAGIC = 0xA7
MEDIA_FRAME_VERSION = 1


def encode_media_frame(header, payload):
    header_bytes = json.dumps(header, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return (bytes([MEDIA_FRAME_MAGIC, MEDIA_FRAME_VERSION]) + len(header_bytes).to_bytes(4, "little")
            + header_bytes + payload)


async def run(args):
    url = f"{args.server.rstrip('/')}/ws/edge-media/{args.robot_id}?token={args.token}"
    camera = ArducamCamera(media=args.media, fps=args.fps, max_width=args.max_width, quality=args.quality)
    camera.start()
    sent, last_report, last_error = 0, time.monotonic(), None
    try:
        while True:
            try:
                async with websockets.connect(url, compression=None, open_timeout=10,
                                              max_size=8 * 1024 * 1024) as ws:
                    print(f"[ws] conectado a {url}", flush=True)
                    while True:
                        packet = camera.take_latest()
                        if packet is not None:
                            header = dict(packet["header"], robot_id=args.robot_id)
                            await asyncio.wait_for(ws.send(encode_media_frame(header, packet["payload"])), 5)
                            sent += 1
                        status = camera.status()
                        if status["error"] != last_error:
                            print(f"[cam] error: {status['error']}" if status["error"] else "[cam] OK", flush=True)
                            last_error = status["error"]
                        if time.monotonic() - last_report >= 2:
                            print(f"[estado] conectada={status['connected']} capturados={status['frames']} "
                                  f"enviados={sent} dispositivo={status.get('device')}", flush=True)
                            last_report = time.monotonic()
                        await asyncio.sleep(0.02)
            except Exception as exc:
                print(f"[ws] error: {type(exc).__name__}: {exc} — reintento en 2 s", flush=True)
                await asyncio.sleep(2)
    finally:
        await asyncio.to_thread(camera.stop)


def main():
    parser = argparse.ArgumentParser(description="Enviar solo la Arducam al servidor (sin Go2)")
    parser.add_argument("--server", required=True, help="ws://HOST:8000 (o wss://... con prefijo)")
    parser.add_argument("--robot-id", default="go2_01")
    parser.add_argument("--token", default="edge-media-dev-token")
    parser.add_argument("--media", default=None, help="/dev/mediaN si hay varias cámaras")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--quality", type=int, default=75)
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
