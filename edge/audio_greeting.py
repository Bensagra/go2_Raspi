"""Small, dependency-free helpers for Go2 greeting audio playback."""

import contextlib
import json
import os
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional


def normalize_audio_name(value: str) -> str:
    name = os.path.splitext(os.path.basename(value.strip()))[0]
    return unicodedata.normalize("NFKC", name).casefold()


def resolve_audio_file(value: str) -> str:
    """Resolve the bundled greeting independently of cwd and Unicode form."""
    if not value:
        return value

    requested = Path(value).expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        candidates.extend((Path.cwd() / requested, repo_root / requested))

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())

        parent = candidate.parent
        if not parent.is_dir():
            continue
        target = normalize_audio_name(candidate.name)
        for entry in parent.iterdir():
            if entry.is_file() and normalize_audio_name(entry.name) == target:
                return str(entry.resolve())

    # Keep the most useful absolute path in diagnostics.
    return str(candidates[-1].resolve())


def find_audio_uuid(response: Any, name: str) -> Optional[str]:
    """Extract a UUID from Unitree's nested, JSON-encoded audio-list response."""
    target = normalize_audio_name(name)
    found: Dict[str, str] = {}

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            fields = {str(k).casefold(): v for k, v in obj.items()}
            audio_name = next(
                (
                    fields[key]
                    for key in ("custom_name", "name", "file_name", "title")
                    if isinstance(fields.get(key), str)
                ),
                None,
            )
            unique_id = next(
                (
                    fields[key]
                    for key in ("unique_id", "uuid", "id")
                    if isinstance(fields.get(key), (str, int))
                ),
                None,
            )
            if audio_name and unique_id is not None:
                found[normalize_audio_name(audio_name)] = str(unique_id)
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)
        elif isinstance(obj, str):
            text = obj.strip()
            if text.startswith(("{", "[")):
                with contextlib.suppress(json.JSONDecodeError):
                    walk(json.loads(text))

    walk(response)
    if target in found:
        return found[target]
    for stored_name, unique_id in found.items():
        if target and (target in stored_name or stored_name in target):
            return unique_id
    return None
