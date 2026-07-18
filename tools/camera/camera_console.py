#!/usr/bin/env python3
import argparse
import asyncio
import base64
import contextlib
import json
import sys
import time
from typing import Optional

import cv2

from unitree_webrtc_connect import UnitreeWebRTCConnection, WebRTCConnectionMethod


FIXED_ROBOT_IP = "192.168.123.161"


class CameraConsoleApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.conn: Optional[UnitreeWebRTCConnection] = None
        self.stop_event = asyncio.Event()
        self.track_event = asyncio.Event()
        self.video_task: Optional[asyncio.Task[None]] = None
        self.frame_count = 0
        self.started_at = 0.0
        self.output_stream = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout

    def _emit_frame_payload(self, payload: dict) -> None:
        self.output_stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.output_stream.flush()

    def _build_connection(self) -> UnitreeWebRTCConnection:
        return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=FIXED_ROBOT_IP)

    async def _consume_video_track(self, track) -> None:
        self.started_at = time.monotonic()

        while not self.stop_event.is_set():
            frame = await track.recv()
            self.frame_count += 1
            image = frame.to_ndarray(format="bgr24")

            emit_frame = self.frame_count == 1 or self.frame_count % self.args.emit_every == 0
            if emit_frame:
                encode_ok = False
                encoded = None
                if self.args.image_format == "jpg":
                    encode_ok, encoded = cv2.imencode(
                        ".jpg",
                        image,
                        [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality],
                    )
                else:
                    encode_ok, encoded = cv2.imencode(".png", image)

                if encode_ok and encoded is not None:
                    frame_format = frame.format.name if frame.format else "unknown"
                    payload = {
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
                        "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
                    }
                    self._emit_frame_payload(payload)
                else:
                    print("Failed to encode frame.", file=sys.stderr, flush=True)

            should_print_stats = self.frame_count == 1 or self.frame_count % self.args.print_every == 0
            if should_print_stats:
                elapsed = max(time.monotonic() - self.started_at, 1e-6)
                fps_estimate = self.frame_count / elapsed
                frame_format = frame.format.name if frame.format else "unknown"

                print(
                    f"[frame {self.frame_count}] "
                    f"frame_type={type(frame).__name__} "
                    f"pts={frame.pts} "
                    f"time_base={frame.time_base} "
                    f"size={frame.width}x{frame.height} "
                    f"format={frame_format} "
                    f"ndarray_type={type(image).__name__} "
                    f"dtype={image.dtype} "
                    f"shape={image.shape} "
                    f"fps_est={fps_estimate:.2f}",
                    file=sys.stderr,
                    flush=True,
                )

            if self.args.max_frames > 0 and self.frame_count >= self.args.max_frames:
                print(f"Reached max_frames={self.args.max_frames}. Stopping stream.", file=sys.stderr, flush=True)
                self.stop_event.set()

    async def _on_video_track(self, track) -> None:
        print("Video track received. Streaming JSON frames to stdout...", file=sys.stderr, flush=True)

        if self.video_task and not self.video_task.done():
            self.video_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.video_task

        self.video_task = asyncio.create_task(self._consume_video_track(track))
        self.track_event.set()

    async def run(self) -> None:
        self.conn = self._build_connection()

        try:
            # Redirect library prints to stderr so stdout contains only frame payloads for piping.
            with contextlib.redirect_stdout(sys.stderr):
                print(f"Connecting to robot in STA mode at {FIXED_ROBOT_IP}...", file=sys.stderr, flush=True)
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
                    await asyncio.sleep(0.2)

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
            "Connect to Unitree camera via unitree_webrtc_connect and print "
            "JSON frame values to stdout for piping into camera_viewer.py. "
            f"Mode is fixed to LocalSTA ({FIXED_ROBOT_IP})."
        )
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="Print debug stats to stderr every N frames (also prints frame 1)",
    )
    parser.add_argument(
        "--emit-every",
        type=int,
        default=1,
        help="Emit one JSON image payload every N frames (default 1 = live)",
    )
    parser.add_argument(
        "--image-format",
        choices=["jpg", "png"],
        default="jpg",
        help="Image encoding used inside JSON payload",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        help="JPEG quality 1-100 (used only when --image-format jpg)",
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

    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")

    if args.track_timeout <= 0:
        parser.error("--track-timeout must be > 0")

    return args


def main() -> None:
    args = parse_args()
    app = CameraConsoleApp(args)

    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
