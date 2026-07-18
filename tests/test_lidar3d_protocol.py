import zlib

import numpy as np
import pytest

from lidar3d.protocol import (
    ProtocolError,
    decode_cloud_frame,
    decode_wire_frame,
    encode_cloud_frame,
    encode_wire_frame,
)


def test_cloud_roundtrip_respects_quantization_error():
    rng = np.random.default_rng(7)
    points = rng.normal(size=(4096, 3)).astype(np.float32) * np.array(
        [4.0, 3.0, 1.0], dtype=np.float32
    )
    wire = encode_cloud_frame(
        points,
        {"robot_id": "go2_01", "coordinate_frame": "utlidar_odom"},
        quantization_cm=1.0,
    )
    header, restored = decode_cloud_frame(wire)

    assert header["robot_id"] == "go2_01"
    assert restored.shape == points.shape
    assert np.max(np.abs(restored - points)) <= header["scale"] / 2 + 1e-5


def test_nonfinite_points_are_removed_before_encoding():
    points = np.array([[1, 2, 3], [np.nan, 0, 0], [4, 5, 6]], dtype=np.float32)
    wire = encode_cloud_frame(points)
    header, restored = decode_cloud_frame(wire)

    assert header["count"] == 2
    assert restored.shape == (2, 3)
    assert np.isfinite(restored).all()


def test_rejects_header_count_that_does_not_match_payload():
    points = np.arange(30, dtype=np.float32).reshape(10, 3)
    wire = encode_cloud_frame(points)
    header, payload = decode_wire_frame(wire)
    header["count"] = 11
    tampered = encode_wire_frame(header, payload)

    with pytest.raises(ProtocolError, match="point count"):
        decode_cloud_frame(tampered)


def test_rejects_decompression_beyond_declared_point_count():
    header = {
        "schema": "go2-lidar-3d-v1",
        "stream": "lidar",
        "fmt": "i16_xyz_zlib",
        "count": 1,
        "scale": 0.01,
        "offset": [0, 0, 0],
    }
    payload = zlib.compress(np.zeros((100, 3), dtype="<i2").tobytes())
    wire = encode_wire_frame(header, payload)

    with pytest.raises(ProtocolError, match="exceeds configured limit"):
        decode_cloud_frame(wire)


def test_rejects_non_json_header():
    prefix = bytes([0xA7, 1]) + (1).to_bytes(4, "little")
    with pytest.raises(ProtocolError, match="JSON"):
        decode_wire_frame(prefix + b"{")
