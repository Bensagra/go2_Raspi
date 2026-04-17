#!/usr/bin/env python3
import argparse
import asyncio
import contextlib
import json
import struct
import sys
import time
from typing import BinaryIO, Optional

import cv2

from unitree_webrtc_connect import UnitreeWebRTCConnection, WebRTCConnectionMethod


FIXED_ROBOT_IP = "192.168.123.161"
MAGIC = b"G2FM"
LENGTHS_STRUCT = struct.Struct(">II")


class CameraSSHSender:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.conn: Optional[UnitreeWebRTCConnection] = None
        self.stop_event = asyncio.Event()
        self.track_event = asyncio.Event()
        self.video_task: Optional[asyncio.Task[None]] = None
        self.frame_count = 0
        self.started_at = 0.0
        self.output: BinaryIO = sys.stdout.buffer

    def _emit_packet(self, metadata: dict, image_bytes: bytes) -> None:
        metadata_bytes = json.dumps(metadata, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        self.output.write(MAGIC)
        self.output.write(LENGTHS_STRUCT.pack(len(metadata_bytes), len(image_bytes)))
        self.output.write(metadata_bytes)
        self.output.write(image_bytes)
        self.output.flush()

    def _encode_frame(self, image):
        if self.args.image_format == "jpg":
            return cv2.imencode(
                ".jpg",
                image,
                [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality],
            )

        return cv2.imencode(
            ".png",
            image,
            [cv2.IMWRITE_PNG_COMPRESSION, self.args.png_compression],
        )

    async def _consume_video_track(self, track) -> None:
        self.started_at = time.monotonic()

        while not self.stop_event.is_set():
            frame = await track.recv()
            self.frame_count += 1
            image = frame.to_ndarray(format="bgr24")

            should_emit = self.frame_count == 1 or self.frame_count % self.args.emit_every == 0
            if should_emit:
                ok, encoded = self._encode_frame(image)
                if ok and encoded is not None:
                    frame_format = frame.format.name if frame.format else "unknown"
                    metadata = {
                        "kind": "frame",
                        "frame_index": self.frame_count,
                        "pts": frame.pts,
                        "time_base": str(frame.time_base),
                        "width": frame.width,
                        "height": frame.height,
                        "format": frame_format,
                        "dtype": str(image.dtype),
                        "shape": list(image.shape),
                        "image_format": self.args.image_format,
                        "unix_ts": time.time(),
                    }
                    self._emit_packet(metadata, encoded.tobytes())
                else:
                    print("Failed to encode frame", file=sys.stderr, flush=True)

            should_print = self.frame_count == 1 or self.frame_count % self.args.print_every == 0
            if should_print:
                elapsed = max(time.monotonic() - self.started_at, 1e-6)
                fps_estimate = self.frame_count / elapsed
                print(
                    f"[sender frame {self.frame_count}] "
                    f"shape={image.shape} "
                    f"dtype={image.dtype} "
                    f"fps_est={fps_estimate:.2f}",
                    file=sys.stderr,
                    flush=True,
                )

            if self.args.max_frames > 0 and self.frame_count >= self.args.max_frames:
                print(f"Reached max_frames={self.args.max_frames}. Stopping sender.", file=sys.stderr, flush=True)
                self.stop_event.set()

    async def _on_video_track(self, track) -> None:
        print("Video track received. Streaming binary packets to stdout...", file=sys.stderr, flush=True)

        if self.video_task and not self.video_task.done():
            self.video_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.video_task

        self.video_task = asyncio.create_task(self._consume_video_track(track))
        self.track_event.set()

    async def run(self) -> None:
        self.conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.args.ip)

        try:
            # Keep stdout clean for binary frame payloads.
            with contextlib.redirect_stdout(sys.stderr):
                print(f"Connecting to robot in STA mode at {self.args.ip}...", file=sys.stderr, flush=True)
                await self.conn.connect()
                self.conn.video.add_track_callback(self._on_video_track)
                self.conn.video.switchVideoChannel(True)
                print("Video channel enabled.", file=sys.stderr, flush=True)

                try:
                    await asyncio.wait_for(self.track_event.wait(), timeout=self.args.track_timeout)
                except asyncio.TimeoutError:
                    print(
                        "No video track received within timeout. "
                        "Verify robot mode/network and camera availability.",
                        file=sys.stderr,
                        flush=True,
                    )
                    self.stop_event.set()

                while not self.stop_event.is_set():
                    await asyncio.sleep(0.1)

                if self.video_task:
                    await self.video_task
        finally:
            if self.video_task and not self.video_task.done():
                self.video_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.video_task

            if self.conn:
                with contextlib.suppress(Exception):
                    self.conn.video.switchVideoChannel(False)
                with contextlib.suppress(Exception):
                    await self.conn.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture Go2 camera frames and stream them as binary packets to stdout. "
            "Designed to be piped over SSH."
        )
    )
    parser.add_argument("--ip", default=FIXED_ROBOT_IP, help="Robot IP in STA mode")
    parser.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="Print sender stats to stderr every N frames",
    )
    parser.add_argument(
        "--emit-every",
        type=int,
        default=1,
        help="Emit one packet every N frames",
    )
    parser.add_argument(
        "--image-format",
        choices=["jpg", "png"],
        default="jpg",
        help="Image encoding used in stream packets",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        help="JPEG quality 1-100 (only used for --image-format jpg)",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=3,
        help="PNG compression level 0-9 (only used for --image-format png)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames (0 means run until Ctrl+C)",
    )
    parser.add_argument(
        "--track-timeout",
        type=float,
        default=12.0,
        help="Seconds to wait for first video track",
    )

    args = parser.parse_args()

    if args.print_every <= 0:
        parser.error("--print-every must be > 0")

    if args.emit_every <= 0:
        parser.error("--emit-every must be > 0")

    if args.jpeg_quality < 1 or args.jpeg_quality > 100:
        parser.error("--jpeg-quality must be between 1 and 100")

    if args.png_compression < 0 or args.png_compression > 9:
        parser.error("--png-compression must be between 0 and 9")

    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")

    if args.track_timeout <= 0:
        parser.error("--track-timeout must be > 0")

    return args


def main() -> None:
    args = parse_args()
    app = CameraSSHSender(args)

    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
