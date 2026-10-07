"""Environment adapter for the existing Server Core on HosTICer."""

import json
import os
import re
from pathlib import Path
from typing import Mapping, Optional


class ForwardedPrefixMiddleware:
    """TIC strips the public prefix and forwards it in X-Forwarded-Prefix."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in {"http", "websocket"}:
            prefix = dict(scope.get("headers", [])).get(b"x-forwarded-prefix", b"").decode("latin1").rstrip("/")
            if (prefix and re.fullmatch(r"/[A-Za-z0-9_./-]+", prefix)
                    and ".." not in prefix and "//" not in prefix):
                scope = {**scope, "root_path": prefix}
        await self.app(scope, receive, send)


def hosting_args(env: Optional[Mapping[str, str]] = None):
    from server.server_core import parse_args

    env = os.environ if env is None else env
    data_dir = Path(env.get("DATA_DIR") or "./data").expanduser().resolve()
    argv = [
        "--map-storage-dir", str(data_dir / "maps"),
        "--faces-dir", str(data_dir / "faces"),
        "--people-dir", str(data_dir / "people"),
        "--people-models-dir", str(data_dir / "models"),
        "--audit-log", str(data_dir / "audit" / "server_audit.jsonl"),
        "--perception-device", env.get("PERCEPTION_DEVICE", "cpu"),
    ]
    for key, flag in (
        ("MQTT_HOST", "--mqtt-host"),
        ("MQTT_PORT", "--mqtt-port"),
        ("MQTT_USERNAME", "--mqtt-username"),
        ("MQTT_PASSWORD", "--mqtt-password"),
        ("MQTT_TOPIC_PREFIX", "--mqtt-topic-prefix"),
        ("MQTT_CLIENT_ID", "--mqtt-client-id"),
    ):
        if key in env:
            argv.append(f"{flag}={env[key]}")
    tls = env.get("MQTT_TLS", "false").strip().lower()
    if tls not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError("MQTT_TLS must be true or false")
    if tls in {"true", "1", "yes"}:
        argv.append("--mqtt-tls")

    # All existing tuning flags stay available, without parsing Uvicorn's argv
    # or interpolating a shell command. Explicit hosting env values win below.
    try:
        extra = json.loads(env.get("SERVER_ARGS", "[]"))
    except ValueError as exc:
        raise ValueError("SERVER_ARGS must be a JSON array of CLI arguments") from exc
    if not isinstance(extra, list) or any(not isinstance(item, str) for item in extra):
        raise ValueError("SERVER_ARGS must be a JSON array of strings")
    args = parse_args(argv + extra)

    # Do not carry the local development credentials into a public deployment.
    # Without tokens the server can pass /health, but control/media are locked.
    try:
        tokens = json.loads(env.get("API_TOKENS", "[]"))
    except ValueError as exc:
        raise ValueError("API_TOKENS must be a JSON array of token:role:user entries") from exc
    if not isinstance(tokens, list):
        raise ValueError("API_TOKENS must be a JSON array")
    seen = set()
    for entry in tokens:
        parts = entry.split(":") if isinstance(entry, str) else []
        if (len(parts) != 3 or not all(part.strip() for part in parts)
                or parts[1].strip() not in {"viewer", "operator", "admin"}):
            raise ValueError("Each API_TOKENS entry must be token:viewer|operator|admin:user")
        token = parts[0].strip()
        if token in seen or any(char.isspace() for char in token):
            raise ValueError("API_TOKENS must contain unique tokens without whitespace")
        seen.add(token)
    args.api_token = tokens
    args.edge_media_token = env.get("EDGE_MEDIA_TOKEN", "").strip()
    args.robot_id = [s.strip() for s in env.get("ROBOT_IDS", "go2_01").split(",") if s.strip()]
    args.cors_origin = [s.strip() for s in env.get("CORS_ORIGINS", "*").split(",") if s.strip()]

    # HosTICer's application files are read-only. Always persist under DATA_DIR,
    # including when SERVER_ARGS contains paths from a previous local launch.
    args.map_storage_dir = str(data_dir / "maps")
    args.mission_storage_dir = str(data_dir / "missions")
    args.faces_dir = str(data_dir / "faces")
    args.people_dir = str(data_dir / "people")
    args.people_models_dir = str(data_dir / "models")
    args.audit_log = str(data_dir / "audit" / "server_audit.jsonl")
    return args


def create_app():
    data_dir = Path(os.environ.get("DATA_DIR") or "./data").expanduser().resolve()
    # Optional ML/plotting libraries sometimes initialize caches on import.
    # Their caches are disposable, so they belong in the writable /tmp area.
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/go2-cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/go2-matplotlib")
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/go2-ultralytics")

    from server.server_core import CoreRuntime

    runtime = CoreRuntime(hosting_args())
    app = runtime.app
    app.state.runtime = runtime
    app.add_middleware(ForwardedPrefixMiddleware)

    @app.get("/")
    async def root():
        return {"service": "Go2 Server Core", "health": "health", "docs": "docs"}

    if not runtime.api_tokens or not runtime.args.edge_media_token:
        print(
            "[server] Configure API_TOKENS and EDGE_MEDIA_TOKEN in TIC to enable access; "
            "missing credentials are disabled.",
            flush=True,
        )
    print(f"[server] Persistent data: {data_dir}", flush=True)
    return app
