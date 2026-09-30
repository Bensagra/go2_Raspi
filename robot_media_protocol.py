"""Wire constants and strict validation shared by the server and Raspberry Pi."""
import math

TALK_RATE = 16000
TALK_MAX_SECONDS = 20
TALK_MAX_PACKET = 3200  # at most 100 ms of mono PCM16
TALK_IDLE_SECONDS = 2.0


def flashlight_payload(payload):
    value = payload.get("brightness")
    if value is None:
        enabled = payload.get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError("provide brightness (integer 0..10) or enabled (boolean)")
        value = 10 if enabled else 0
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10:
        raise ValueError("brightness must be an integer from 0 to 10")
    return {"brightness": value}


def require_robot_success(response):
    """A transport reply is not proof that the robot accepted a VUI/AudioHub RPC."""
    try:
        code = response["data"]["header"]["status"]["code"]
    except (TypeError, KeyError) as exc:
        raise RuntimeError("robot response has no status code") from exc
    if code != 0:
        raise RuntimeError(f"robot rejected request (code={code})")
    return response


def fresh_packet(message, now):
    try:
        ts = float(message["ts"])
        return math.isfinite(ts) and -2 <= now - ts <= 1
    except (KeyError, TypeError, ValueError):
        return False
