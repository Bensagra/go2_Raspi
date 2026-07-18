#!/usr/bin/env python3
import argparse
import base64
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

def _decode_base64_image(data: str) -> Optional[np.ndarray]:
    text = data.strip()
    if not text:
        return None

    # Accept data URI format: data:image/jpeg;base64,...
    if text.startswith("data:image") and "," in text:
        text = text.split(",", 1)[1]

    try:
        raw = base64.b64decode(text, validate=True)
    except Exception:
        return None

    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return image


def _decode_path_image(value: str) -> Optional[np.ndarray]:
    path = Path(value).expanduser()
    if not path.is_file():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _normalize_array_to_bgr(image_array: np.ndarray) -> Optional[np.ndarray]:
    if image_array.size == 0:
        return None

    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    if image_array.ndim == 2:
        return cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    if image_array.ndim == 3 and image_array.shape[2] == 1:
        return cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    if image_array.ndim == 3 and image_array.shape[2] == 3:
        return image_array
    if image_array.ndim == 3 and image_array.shape[2] == 4:
        return cv2.cvtColor(image_array, cv2.COLOR_BGRA2BGR)
    return None


def _decode_json_image(payload: str) -> Optional[np.ndarray]:
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None

    if isinstance(obj, dict):
        # Common key aliases for base64 strings.
        for key in ("image_base64", "frame_base64", "jpeg_base64", "png_base64", "image", "frame"):
            value = obj.get(key)
            if isinstance(value, str):
                image = _decode_base64_image(value)
                if image is None:
                    image = _decode_path_image(value)
                if image is not None:
                    return image

        # Optional format for numeric array payloads.
        # Example: {"data": [...], "shape": [480, 640, 3], "dtype": "uint8"}
        data = obj.get("data")
        shape = obj.get("shape")
        dtype = obj.get("dtype", "uint8")
        if isinstance(data, list) and isinstance(shape, list):
            with contextlib.suppress(Exception):
                arr = np.asarray(data, dtype=np.dtype(dtype)).reshape(tuple(shape))
                return _normalize_array_to_bgr(arr)

    return None


def decode_line_to_image(line: str) -> Optional[np.ndarray]:
    payload = line.strip()
    if not payload:
        return None

    image = _decode_json_image(payload)
    if image is not None:
        return image

    image = _decode_path_image(payload)
    if image is not None:
        return image

    return _decode_base64_image(payload)


def run_viewer(args: argparse.Namespace) -> None:
    frame_count = 0
    skipped_count = 0
    started_at = time.monotonic()

    save_dir: Optional[Path] = None
    if args.save_dir:
        save_dir = Path(args.save_dir).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)

    print("Esperando lineas por stdin para visualizar.", flush=True)
    print("Formatos aceptados: base64 JPEG/PNG, data URI, ruta de archivo, o JSON.", flush=True)
    print("Controles ventana: q=cerrar, s=guardar snapshot.", flush=True)

    for raw_line in sys.stdin:
        image = decode_line_to_image(raw_line)
        if image is None:
            skipped_count += 1
            if args.warn_every > 0 and (skipped_count == 1 or skipped_count % args.warn_every == 0):
                print(f"Linea omitida (no es imagen decodificable). total_omitidas={skipped_count}", flush=True)
            continue

        frame_count += 1
        cv2.imshow(args.window_name, image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Pressed q. Stopping viewer.", flush=True)
            break

        if key == ord("s"):
            if save_dir is None:
                print("save_dir no configurado. Usa --save-dir para habilitar snapshots.", flush=True)
            else:
                output = save_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(output), image)
                print(f"Saved snapshot: {output}", flush=True)

        if frame_count == 1 or frame_count % args.print_every == 0:
            elapsed = max(time.monotonic() - started_at, 1e-6)
            fps_estimate = frame_count / elapsed
            print(
                f"[frame {frame_count}] "
                f"dtype={image.dtype} "
                f"shape={image.shape} "
                f"fps_est={fps_estimate:.2f}",
                flush=True,
            )

        if args.max_frames > 0 and frame_count >= args.max_frames:
            print(f"Reached max_frames={args.max_frames}. Stopping viewer.", flush=True)
            break

        with contextlib.suppress(Exception):
            visible = cv2.getWindowProperty(args.window_name, cv2.WND_PROP_VISIBLE)
            if visible < 1:
                print("Viewer window was closed. Stopping.", flush=True)
                break

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read image payloads from stdin and visualize them in an OpenCV window."
        )
    )
    parser.add_argument("--window-name", default="Go2 Camera", help="Viewer window title")
    parser.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="Print frame stats every N frames (also prints frame 1)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames (0 means run until q/close window/EOF)",
    )
    parser.add_argument(
        "--save-dir",
        help="Optional directory to save snapshots when pressing s",
    )
    parser.add_argument(
        "--warn-every",
        type=int,
        default=0,
        help="Print warning every N non-decodable lines (0 disables warnings)",
    )

    args = parser.parse_args()

    if args.print_every <= 0:
        parser.error("--print-every must be > 0")

    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")

    if args.warn_every < 0:
        parser.error("--warn-every must be >= 0")

    return args


def main() -> None:
    args = parse_args()

    try:
        run_viewer(args)
    except KeyboardInterrupt:
        print("Interrupted by user.", flush=True)


if __name__ == "__main__":
    main()
