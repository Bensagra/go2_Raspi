"""Command delivery must not claim success while MQTT is disconnected."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import HTTPException
import paho.mqtt.client as mqtt

from server.server_core import CommandIn, CoreRuntime, parse_args


class ControlConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = CoreRuntime(parse_args([
            "--disable-perception", "--map-storage-dir", self.temp.name,
            "--audit-log", str(Path(self.temp.name) / "audit.jsonl"),
        ]))
        self.client = Mock()
        self.client.is_connected.return_value = False
        self.client.publish.return_value = SimpleNamespace(rc=mqtt.MQTT_ERR_SUCCESS)
        self.runtime.mqtt_client = self.client
        self.command_endpoint = next(route.endpoint for route in self.runtime.app.routes
                                     if getattr(route, "path", "") == "/api/robots/{robot_id}/commands")
        self.auth = {"role": "operator", "user_id": "test"}

    async def test_offline_http_command_rejected_without_pending_or_publish(self):
        with self.assertRaises(HTTPException) as caught:
            await self.command_endpoint("go2_01", CommandIn(type="move"), self.auth)
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(self.runtime.pending_commands, {})
        self.client.publish.assert_not_called()

    async def test_disconnect_during_publish_returns_error_and_removes_pending(self):
        self.client.is_connected.return_value = True
        self.client.publish.return_value.rc = mqtt.MQTT_ERR_NO_CONN
        with self.assertRaises(HTTPException) as caught:
            await self.command_endpoint("go2_01", CommandIn(type="move"), self.auth)
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(self.runtime.pending_commands, {})

    async def test_recovery_publishes_move_with_existing_limits_and_ttl(self):
        self.client.is_connected.return_value = True
        result = await self.command_endpoint(
            "go2_01", CommandIn(type="move", payload={"linear_x": 0.1}, ttl_ms=700), self.auth)
        self.assertTrue(result["ok"])
        topic, payload = self.client.publish.call_args.args
        wire = json.loads(payload)
        self.assertEqual(topic, "go2/go2_01/commands/in")
        self.assertEqual(wire["payload"]["linear_x"], 0.1)
        self.assertEqual(wire["ttl_ms"], 700)

    async def test_streaming_and_autonomy_do_not_queue_offline_commands(self):
        self.assertFalse(self.runtime._publish_realtime_drive("go2_01", "test", {}, 1))
        self.assertFalse(self.runtime._publish_realtime_stop("go2_01", "test"))
        self.assertIsNone(self.runtime._publish_speed_profile_command("go2_01", "test", "normal"))
        self.assertFalse(self.runtime.autonomy_drive("go2_01", 0.1, 0, 0))
        self.assertIsNone(self.runtime.autonomy_command("go2_01", "set_autonomy", {"enabled": True}))
        self.client.publish.assert_not_called()

    async def test_diagnostics_distinguish_broker_edge_and_robot(self):
        runtime = self.runtime
        self.assertFalse(runtime._control_link_status("go2_01")["mqtt_connected"])
        self.client.is_connected.return_value = True
        self.assertFalse(runtime._control_link_status("go2_01")["edge_telemetry_fresh"])
        await runtime._process_mqtt_payload("go2_01", "telemetry", {"robot_link": {"connected": False}})
        self.assertTrue(runtime._control_link_status("go2_01")["edge_telemetry_fresh"])
        self.assertFalse(runtime._control_link_status("go2_01")["robot_connected"])
        await runtime._process_mqtt_payload("go2_01", "telemetry", {"robot_link": {"connected": True}})
        self.assertTrue(runtime._control_link_status("go2_01")["ok"])
        runtime.telemetry_received_at["go2_01"] = time.monotonic() - 6
        self.assertFalse(runtime._control_link_status("go2_01")["ok"])
