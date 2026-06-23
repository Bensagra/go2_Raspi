"""Small, dependency-free helpers for Go2 greeting audio playback."""

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import wave
from array import array
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple


def normalize_audio_name(value: str) -> str:
    name = os.path.splitext(os.path.basename(value.strip()))[0]
    decomposed = unicodedata.normalize("NFKD", name).casefold()
    without_marks = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    return " ".join(
        "".join(char if char.isalnum() else " " for char in without_marks).split()
    )


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
        matches = [
            entry
            for entry in parent.iterdir()
            if entry.is_file() and normalize_audio_name(entry.name) == target
        ]
        requested_suffix = candidate.suffix.casefold()
        matches.sort(
            key=lambda entry: (
                entry.suffix.casefold() != requested_suffix,
                entry.name.casefold(),
            )
        )
        if matches:
            return str(matches[0].resolve())

    # Keep the most useful absolute path in diagnostics.
    return str(candidates[-1].resolve())


def prepare_go2_wav(path: str) -> Optional[str]:
    """Normalize an audio asset to mono, PCM16, 44.1 kHz without altering it."""
    if not path or not os.path.isfile(path):
        return None

    source = Path(path).resolve()
    stat = source.stat()
    cache_key = hashlib.sha256(
        f"{source}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "go2-audio-cache" / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"{source.stem}.wav"
    temporary_output = cache_dir / f"{source.stem}.tmp.wav"
    if output.is_file():
        return str(output)
    with contextlib.suppress(OSError):
        temporary_output.unlink()

    if source.suffix.casefold() == ".wav":
        try:
            with wave.open(str(source), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                compression = wav_file.getcomptype()
                raw_frames = wav_file.readframes(wav_file.getnframes())

            if (
                channels == 1
                and sample_width == 2
                and sample_rate == 44100
                and compression == "NONE"
            ):
                return str(source)

            # Pure-Python conversion covers normal PCM16 WAVs, including the
            # common 48 kHz files produced by phones and audio editors.
            if sample_width == 2 and compression == "NONE" and channels > 0:
                samples = array("h")
                samples.frombytes(raw_frames)
                if sys.byteorder != "little":
                    samples.byteswap()
                frame_count = len(samples) // channels
                if frame_count <= 0:
                    return None

                if channels == 1:
                    mono = list(samples)
                else:
                    mono = [
                        sum(
                            samples[index + channel]
                            for channel in range(channels)
                        )
                        / channels
                        for index in range(0, frame_count * channels, channels)
                    ]
                target_count = max(
                    1, int(round(frame_count * 44100.0 / max(sample_rate, 1)))
                )
                converted = array("h")
                if frame_count == 1 or target_count == 1:
                    converted.append(int(max(-32768, min(32767, round(mono[0])))))
                else:
                    scale = (frame_count - 1) / (target_count - 1)
                    for target_index in range(target_count):
                        position = target_index * scale
                        left = int(position)
                        right = min(left + 1, frame_count - 1)
                        fraction = position - left
                        value = mono[left] + (mono[right] - mono[left]) * fraction
                        converted.append(
                            int(max(-32768, min(32767, round(value))))
                        )
                if sys.byteorder != "little":
                    converted.byteswap()

                with wave.open(str(temporary_output), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(44100)
                    wav_file.writeframes(converted.tobytes())
                os.replace(temporary_output, output)
                return str(output)
        except (OSError, EOFError, wave.Error, ValueError):
            pass

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(source),
                    "-vn",
                    "-c:a",
                    "pcm_s16le",
                    "-ar",
                    "44100",
                    "-ac",
                    "1",
                    str(temporary_output),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if temporary_output.is_file():
                os.replace(temporary_output, output)
                return str(output)
        except (OSError, subprocess.CalledProcessError):
            pass

    try:
        from pydub import AudioSegment

        segment = (
            AudioSegment.from_file(str(source))
            .set_frame_rate(44100)
            .set_channels(1)
            .set_sample_width(2)
        )
        segment.export(
            str(temporary_output),
            format="wav",
            parameters=["-ar", "44100", "-ac", "1"],
        )
        if temporary_output.is_file():
            os.replace(temporary_output, output)
            return str(output)
        return None
    except Exception:
        with contextlib.suppress(OSError):
            temporary_output.unlink()
        return None


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


class AudioGreetingError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


async def resolve_audio_uuid_on_hub(
    hub: Any,
    source_path: str,
    prepare_wav: Callable[[str], Optional[str]],
    *,
    timeout_s: float = 6.0,
    poll_attempts: int = 8,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Tuple[str, bool]:
    """Find an existing audio UUID or upload the WAV and wait for indexing."""

    source_name = Path(source_path).stem

    try:
        response = await asyncio.wait_for(hub.get_audio_list(), timeout=timeout_s)
    except Exception:
        response = None

    existing_uuid = find_audio_uuid(response, source_name)
    if existing_uuid:
        return existing_uuid, False

    try:
        prepared_path = await asyncio.to_thread(prepare_wav, source_path)
    except Exception as exc:
        raise AudioGreetingError("prepare_wav", str(exc)) from exc
    if not prepared_path:
        raise AudioGreetingError("prepare_wav", f"audio unavailable: {source_path}")

    try:
        await hub.upload_audio_file(prepared_path)
    except Exception as exc:
        raise AudioGreetingError("upload", str(exc)) from exc

    uploaded_name = Path(prepared_path).stem
    last_list_error: Optional[Exception] = None
    attempts = max(1, poll_attempts)
    for attempt in range(attempts):
        await sleep(0.4 if attempt == 0 else 0.6)
        try:
            response = await asyncio.wait_for(hub.get_audio_list(), timeout=timeout_s)
        except Exception as exc:
            last_list_error = exc
            continue

        uploaded_uuid = find_audio_uuid(response, uploaded_name)
        if not uploaded_uuid and uploaded_name != source_name:
            uploaded_uuid = find_audio_uuid(response, source_name)
        if uploaded_uuid:
            return uploaded_uuid, True

    if last_list_error is not None:
        raise AudioGreetingError("list_after_upload", str(last_list_error))
    raise AudioGreetingError(
        "resolve_uuid",
        f"uploaded audio was not indexed: {uploaded_name}",
    )
