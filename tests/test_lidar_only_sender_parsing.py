import importlib
import sys
import types

import numpy as np
import pytest


@pytest.fixture()
def sender_module(monkeypatch):
    fake = types.ModuleType("unitree_webrtc_connect")
    fake.RTC_TOPIC = {
        "ULIDAR_ARRAY": "rt/utlidar/voxel_map_compressed",
        "ROBOTODOM": "rt/utlidar/robot_pose",
        "ULIDAR_SWITCH": "rt/utlidar/switch",
    }

    class ConnectionMethod:
        LocalAP = 1
        LocalSTA = 2

    class Connection:
        pass

    fake.WebRTCConnectionMethod = ConnectionMethod
    fake.UnitreeWebRTCConnection = Connection
    monkeypatch.setitem(sys.modules, "unitree_webrtc_connect", fake)
    monkeypatch.delitem(sys.modules, "edge.lidar_only_sender", raising=False)
    return importlib.import_module("edge.lidar_only_sender")


def test_extracts_native_decoder_points(sender_module):
    expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    message = {"data": {"data": {"points": expected}}}

    actual = sender_module.extract_points(message)

    np.testing.assert_allclose(actual, expected.astype(np.float32))


def test_extracts_libvoxel_packed_float_positions(sender_module):
    expected = np.array([[1.25, -2.5, 0.5], [4, 5, 6]], dtype="<f4")
    packed = np.frombuffer(expected.tobytes(), dtype=np.uint8)
    message = {"data": {"data": {"positions": packed}}}

    actual = sender_module.extract_points(message)

    np.testing.assert_array_equal(actual, expected)


def test_extracts_pose_and_quaternion_yaw(sender_module):
    message = {
        "data": {
            "pose": {
                "position": {"x": 2, "y": -1, "z": 0.3},
                "orientation": {"x": 0, "y": 0, "z": 2**-0.5, "w": 2**-0.5},
            }
        }
    }

    pose = sender_module.extract_pose(message)

    assert pose is not None
    assert pose["x"] == 2
    assert pose["y"] == -1
    assert pose["z"] == 0.3
    assert pose["yaw"] == pytest.approx(np.pi / 2)


def test_builds_server_url_from_runtime_ipv4(sender_module):
    args = types.SimpleNamespace(
        server_ws_url="",
        server_ip="192.168.1.57",
        server_port=8765,
        robot_id="go2_01",
        token="secret with spaces",
    )
    sender = sender_module.LidarOnlySender.__new__(sender_module.LidarOnlySender)
    sender.args = args

    assert sender._server_url() == (
        "ws://192.168.1.57:8765/lidar/go2_01?token=secret%20with%20spaces"
    )


def test_builds_server_url_from_runtime_ipv6(sender_module):
    args = types.SimpleNamespace(
        server_ws_url="",
        server_ip="2001:db8::57",
        server_port=8765,
        robot_id="go2_01",
        token="",
    )
    sender = sender_module.LidarOnlySender.__new__(sender_module.LidarOnlySender)
    sender.args = args

    assert sender._server_url() == "ws://[2001:db8::57]:8765/lidar/go2_01"
