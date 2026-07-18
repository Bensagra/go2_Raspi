"""Compact, bounded wire format shared by the Raspberry Pi and the server.

The outer frame intentionally matches the binary media envelope already used by
``edge_gateway_service.py`` and ``server_core.py``::

    magic:u8 | version:u8 | json_header_length:u32-le | json_header | payload

Point coordinates are centered, quantized to signed int16 and zlib compressed.
The JSON header carries the scale and offset needed to restore float32 XYZ.
"""

from __future__ import annotations

import json
import math
import zlib
from typing import Any, Dict, Mapping, Tuple

import numpy as np


MAGIC = 0xA7
VERSION = 1
PREFIX_BYTES = 6
MAX_HEADER_BYTES = 64 * 1024
POINT_FORMAT = "i16_xyz_zlib"


class ProtocolError(ValueError):
    """The received frame is malformed, unsupported, or exceeds a limit."""


def encode_wire_frame(header: Mapping[str, Any], payload: bytes) -> bytes:
    header_bytes = json.dumps(
        dict(header), ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise ProtocolError("frame header is too large")
    out = bytearray(PREFIX_BYTES + len(header_bytes) + len(payload))
    out[0] = MAGIC
    out[1] = VERSION
    out[2:6] = len(header_bytes).to_bytes(4, "little")
    out[6 : 6 + len(header_bytes)] = header_bytes
    out[6 + len(header_bytes) :] = payload
    return bytes(out)


def decode_wire_frame(
    wire: bytes,
    *,
    max_payload_bytes: int = 32 * 1024 * 1024,
) -> Tuple[Dict[str, Any], bytes]:
    if not isinstance(wire, (bytes, bytearray, memoryview)):
        raise ProtocolError("binary WebSocket frame required")
    view = memoryview(wire)
    if len(view) < PREFIX_BYTES or view[0] != MAGIC or view[1] != VERSION:
        raise ProtocolError("invalid frame magic or version")
    header_len = int.from_bytes(view[2:6], "little")
    if header_len <= 0 or header_len > MAX_HEADER_BYTES:
        raise ProtocolError("invalid frame header length")
    payload_start = PREFIX_BYTES + header_len
    if payload_start > len(view):
        raise ProtocolError("truncated frame header")
    payload_len = len(view) - payload_start
    if max_payload_bytes <= 0 or payload_len > max_payload_bytes:
        raise ProtocolError("compressed payload exceeds configured limit")
    try:
        header = json.loads(bytes(view[PREFIX_BYTES:payload_start]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid JSON frame header") from exc
    if not isinstance(header, dict):
        raise ProtocolError("frame header must be a JSON object")
    return header, bytes(view[payload_start:])


def _inflate_limited(compressed: bytes, limit: int) -> bytes:
    if limit <= 0:
        raise ProtocolError("invalid decompression limit")
    inflater = zlib.decompressobj()
    try:
        raw = inflater.decompress(compressed, limit + 1)
        if inflater.unconsumed_tail or len(raw) > limit:
            raise ProtocolError("decompressed point cloud exceeds configured limit")
        raw += inflater.flush()
    except zlib.error as exc:
        raise ProtocolError("invalid zlib point payload") from exc
    if len(raw) > limit or not inflater.eof:
        raise ProtocolError("truncated or oversized zlib point payload")
    return raw


def encode_cloud_frame(
    points: np.ndarray,
    metadata: Mapping[str, Any] | None = None,
    *,
    quantization_cm: float = 1.0,
    compression_level: int = 6,
) -> bytes:
    cloud = np.asarray(points, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ProtocolError("point cloud must have shape Nx3")
    cloud = np.ascontiguousarray(cloud[:, :3])
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if cloud.shape[0] == 0:
        raise ProtocolError("point cloud is empty")
    if not math.isfinite(quantization_cm) or quantization_cm <= 0:
        raise ProtocolError("quantization_cm must be finite and positive")
    if compression_level < 0 or compression_level > 9:
        raise ProtocolError("compression_level must be between 0 and 9")

    bounds_min = cloud.min(axis=0)
    bounds_max = cloud.max(axis=0)
    offset = (bounds_min + bounds_max) / 2.0
    scale = max(float(quantization_cm) / 100.0, 0.0001)
    max_delta = float(np.max(np.abs(cloud - offset)))
    if max_delta > 0:
        scale = max(scale, max_delta / 32760.0)
    quantized = np.clip(np.rint((cloud - offset) / scale), -32768, 32767).astype("<i2")
    payload = zlib.compress(quantized.tobytes(order="C"), compression_level)

    header: Dict[str, Any] = dict(metadata or {})
    header.update(
        {
            "schema": "go2-lidar-3d-v1",
            "stream": "lidar",
            "fmt": POINT_FORMAT,
            "count": int(cloud.shape[0]),
            "scale": float(scale),
            "offset": [float(value) for value in offset],
        }
    )
    return encode_wire_frame(header, payload)


def decode_cloud_frame(
    wire: bytes,
    *,
    max_payload_bytes: int = 32 * 1024 * 1024,
    max_points: int = 1_000_000,
) -> Tuple[Dict[str, Any], np.ndarray]:
    header, payload = decode_wire_frame(wire, max_payload_bytes=max_payload_bytes)
    if header.get("schema") != "go2-lidar-3d-v1":
        raise ProtocolError("unsupported point-cloud schema")
    if header.get("stream") != "lidar" or header.get("fmt") != POINT_FORMAT:
        raise ProtocolError("unsupported point-cloud format")

    try:
        count = int(header["count"])
        scale = float(header["scale"])
        offset = np.asarray(header["offset"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("invalid point-cloud metadata") from exc
    if count <= 0 or max_points <= 0 or count > max_points:
        raise ProtocolError("point count exceeds configured limit")
    if not math.isfinite(scale) or scale <= 0:
        raise ProtocolError("invalid point-cloud scale")
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ProtocolError("invalid point-cloud offset")

    expected_bytes = count * 3 * np.dtype("<i2").itemsize
    raw = _inflate_limited(payload, expected_bytes)
    if len(raw) != expected_bytes:
        raise ProtocolError("point count does not match point payload")
    quantized = np.frombuffer(raw, dtype="<i2").reshape(count, 3)
    points = quantized.astype(np.float32) * scale + offset
    if not np.isfinite(points).all():
        raise ProtocolError("decoded point cloud contains non-finite coordinates")
    return header, np.ascontiguousarray(points, dtype=np.float32)
