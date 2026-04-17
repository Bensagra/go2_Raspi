#!/usr/bin/env python3
import argparse
import contextlib
import json
import struct
import sys
import time
from pathlib import Path
from typing import BinaryIO, Optional, Tuple

import cv2
import numpy as np


MAGIC = b"G2FM"
LENGTHS_STRUCT = struct.Struct(">II")
MAX_METADATA_BYTES = 256 * 1024
MAX_IMAGE_BYTES = 40 * 1024 * 1024


def _read_exact(stream: BinaryIO, size: int) -> Optional[bytes]:
    if size == 0:
        return b""

    chunks = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            if remaining == size:
                return None
            raise EOFError("Unexpected EOF while reading stream")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_until_magic(stream: BinaryIO) -> bool:
    matched = 0

    while True:
        chunk = stream.read(1)
        if not chunk:
            return False

        byte = chunk[0]
        if byte == MAGIC[matched]:
            matched += 1
            if matched == len(MAGIC):
                return True
        else:
            matched = 1 if byte == MAGIC[0] else 0


def _read_packet(stream: BinaryIO) -> Optional[Tuple[dict, bytes]]:
    if not _read_until_magic(stream):
        return None

    lengths = _read_exact(stream, LENGTHS_STRUCT.size)
    if lengths is None:
        return None

    metadata_len, image_len = LENGTHS_STRUCT.unpack(lengths)

    if metadata_len <= 0 or metadata_len > MAX_METADATA_BYTES:
        raise ValueError(f"Invalid metadata length: {metadata_len}")

    if image_len <= 0 or image_len > MAX_IMAGE_BYTES:
        raise ValueError(f"Invalid image length: {image_len}")

    metadata_bytes = _read_exact(stream, metadata_len)
    image_bytes = _read_exact(stream, image_len)
    if metadata_bytes is None or image_bytes is None:
        return None

    metadata = json.loads(metadata_bytes.decode("utf-8"))
    return metadata, image_bytes


def run_receiver(args: argparse.Namespace) -> None:
    input_stream = sys.stdin.buffer
    save_dir: Optional[Path] = None
    if args.save_dir:
        save_dir = Path(args.save_dir).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    started_at = time.monotonic()

    print("Waiting for binary frame packets on stdin...", file=sys.stderr, flush=True)

    while True:
        packet = _read_packet(input_stream)
        if packet is None:
            print("Stream ended.", file=sys.stderr, flush=True)
            break

        metadata, image_bytes = packet

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            print("Failed to decode image payload", file=sys.stderr, flush=True)
            continue

        frame_count += 1

        if args.stdout_json:
            sys.stdout.write(json.dumps(metadata, separators=(",", ":"), ensure_ascii=True) + "\n")
            sys.stdout.flush()

        if save_dir and args.save_every > 0 and frame_count % args.save_every == 0:
            output = save_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(output), image)

        if frame_count == 1 or frame_count % args.stats_every == 0:
            elapsed = max(time.monotonic() - started_at, 1e-6)
            fps_estimate = frame_count / elapsed
            source_index = metadata.get("frame_index", "?")
            print(
                f"[receiver frame {frame_count}] src_index={source_index} "
                f"shape={image.shape} fps_est={fps_estimate:.2f}",
                file=sys.stderr,
                flush=True,
            )

        if args.show:
            cv2.imshow(args.window_name, image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Pressed q. Stopping receiver.", file=sys.stderr, flush=True)
                break

            with contextlib.suppress(Exception):
                visible = cv2.getWindowProperty(args.window_name, cv2.WND_PROP_VISIBLE)
                if visible < 1:
                    print("Viewer window was closed. Stopping receiver.", file=sys.stderr, flush=True)
                    break

        if args.max_frames > 0 and frame_count >= args.max_frames:
            print(f"Reached max_frames={args.max_frames}. Stopping receiver.", file=sys.stderr, flush=True)
            break

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read binary camera packets from stdin (for example, piped over SSH) "
            "and decode them on the server."
        )
    )
    parser.add_argument("--show", action="store_true", help="Display frames in an OpenCV window")
    parser.add_argument("--window-name", default="Go2 Remote", help="Viewer window title")
    parser.add_argument(
        "--save-dir",
        help="Optional directory to save frames as JPEG",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save one frame every N frames (0 disables saving)",
    )
    parser.add_argument(
        "--stats-every",
        type=int,
        default=30,
        help="Print receiver stats every N frames",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames (0 means run until EOF/Ctrl+C)",
    )
    parser.add_argument(
        "--stdout-json",
        action="store_true",
        help="Emit metadata as JSON lines on stdout",
    )

    args = parser.parse_args()

    if args.save_every < 0:
        parser.error("--save-every must be >= 0")

    if args.stats_every <= 0:
        parser.error("--stats-every must be > 0")

    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")

    return args


def main() -> None:
    args = parse_args()

    try:
        run_receiver(args)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
