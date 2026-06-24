"""Small, dependency-free helpers for Go2 greeting audio playback."""

import asyncio
import base64
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import wave
from array import array
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple


TARGET_PCM16_PEAK = 30000


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


def _normalize_pcm16_samples(samples: array) -> Tuple[array, bool]:
    peak = max((abs(int(sample)) for sample in samples), default=0)
    if peak <= 0 or peak >= TARGET_PCM16_PEAK:
        return samples, False

    gain = TARGET_PCM16_PEAK / float(peak)
    normalized = array("h")
    for sample in samples:
        value = int(round(int(sample) * gain))
        normalized.append(max(-32768, min(32767, value)))
    return normalized, True


def _write_pcm16_mono_wav(path: Path, samples: array, sample_rate: int = 44100) -> None:
    out = array("h", samples)
    if sys.byteorder != "little":
        out.byteswap()
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(out.tobytes())


def _normalize_wav_file(path: Path) -> None:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        compression = wav_file.getcomptype()
        raw_frames = wav_file.readframes(wav_file.getnframes())
    if channels != 1 or sample_width != 2 or sample_rate != 44100 or compression != "NONE":
        return

    samples = array("h")
    samples.frombytes(raw_frames)
    if sys.byteorder != "little":
        samples.byteswap()
    normalized, changed = _normalize_pcm16_samples(samples)
    if not changed:
        return

    tmp = path.with_suffix(".normalize.tmp.wav")
    _write_pcm16_mono_wav(tmp, normalized)
    os.replace(tmp, path)


def prepare_go2_wav(path: str) -> Optional[str]:
    """Normalize an audio asset to mono, PCM16, 44.1 kHz without altering it."""
    if not path or not os.path.isfile(path):
        return None

    source = Path(path).resolve()
    stat = source.stat()
    cache_key = hashlib.sha256(
        f"v2-normalized:{source}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
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
                samples = array("h")
                samples.frombytes(raw_frames)
                if sys.byteorder != "little":
                    samples.byteswap()
                normalized, changed = _normalize_pcm16_samples(samples)
                if not changed:
                    return str(source)
                _write_pcm16_mono_wav(temporary_output, normalized)
                os.replace(temporary_output, output)
                return str(output)

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
                normalized, _ = _normalize_pcm16_samples(converted)

                _write_pcm16_mono_wav(temporary_output, normalized)
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
                with contextlib.suppress(Exception):
                    _normalize_wav_file(output)
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
            with contextlib.suppress(Exception):
                _normalize_wav_file(output)
            return str(output)
        return None
    except Exception:
        with contextlib.suppress(OSError):
            temporary_output.unlink()
        return None


def find_audio_uuid(
    response: Any,
    name: str,
    *,
    allow_newest_fallback: bool = True,
) -> Optional[str]:
    """Extract a UUID from Unitree's nested, JSON-encoded audio-list response.

    Tries (1) exact normalized name, (2) fuzzy name, (3) optionally the most
    recently created entry. The newest fallback is only safe immediately after
    uploading, when the robot may index the file under an unexpected name."""
    target = normalize_audio_name(name)
    found: Dict[str, str] = {}
    newest: list = [-1.0, None]  # [create_time, uuid]

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
            if unique_id is not None:
                if audio_name:
                    found[normalize_audio_name(audio_name)] = str(unique_id)
                ctime = next(
                    (fields[key] for key in ("create_time", "created_at", "ctime", "timestamp")
                     if isinstance(fields.get(key), (str, int, float))),
                    None,
                )
                with contextlib.suppress(TypeError, ValueError):
                    if ctime is not None and float(ctime) > newest[0]:
                        newest[0] = float(ctime)
                        newest[1] = str(unique_id)
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
    return newest[1] if allow_newest_fallback else None


class AudioGreetingError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


class DirectAudioHub:
    """Dependency-light AudioHub client matching unitree_webrtc_connect.

    The library's WebRTCAudioHub imports pydub at module import time; on some
    Python builds that fails even though the robot request protocol is available.
    This class sends the same AUDIO_API requests and upload chunk payloads while
    relying on prepare_go2_wav for any transcoding/normalization.
    """

    def __init__(
        self,
        request: Callable[[int, Optional[Any]], Awaitable[Any]],
        audio_api: Dict[str, int],
    ) -> None:
        self._request = request
        self._api = audio_api

    async def get_audio_list(self):
        return await self._request(int(self._api["GET_AUDIO_LIST"]), {})

    async def play_by_uuid(self, unique_id: str):
        return await self._request(
            int(self._api["SELECT_START_PLAY"]),
            {"unique_id": str(unique_id)},
        )

    async def pause(self):
        return await self._request(int(self._api["PAUSE"]), {})

    async def resume(self):
        return await self._request(int(self._api["UNSUSPEND"]), {})

    async def set_play_mode(self, play_mode: str):
        return await self._request(
            int(self._api["SET_PLAY_MODE"]),
            {"play_mode": str(play_mode)},
        )

    async def get_play_mode(self):
        return await self._request(int(self._api["GET_PLAY_MODE"]), {})

    async def upload_audio_file(self, audiofile_path: str):
        with open(audiofile_path, "rb") as audio_file:
            audio_data = audio_file.read()

        b64_data = base64.b64encode(audio_data).decode("utf-8")
        chunk_size = 4096
        chunks = [
            b64_data[index : index + chunk_size]
            for index in range(0, len(b64_data), chunk_size)
        ] or [""]
        file_name = os.path.splitext(os.path.basename(audiofile_path))[0]
        file_md5 = hashlib.md5(audio_data).hexdigest()
        create_time = int(time.time() * 1000)
        last_response = None

        for index, chunk in enumerate(chunks, 1):
            last_response = await self._request(
                int(self._api["UPLOAD_AUDIO_FILE"]),
                {
                    "file_name": file_name,
                    "file_type": "wav",
                    "file_size": len(audio_data),
                    "current_block_index": index,
                    "total_block_number": len(chunks),
                    "block_content": chunk,
                    "current_block_size": len(chunk),
                    "file_md5": file_md5,
                    "create_time": create_time,
                },
            )
            await asyncio.sleep(0.1)

        return last_response


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

    existing_uuid = find_audio_uuid(
        response,
        source_name,
        allow_newest_fallback=False,
    )
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

        uploaded_uuid = find_audio_uuid(
            response,
            uploaded_name,
            allow_newest_fallback=True,
        )
        if not uploaded_uuid and uploaded_name != source_name:
            uploaded_uuid = find_audio_uuid(
                response,
                source_name,
                allow_newest_fallback=True,
            )
        if uploaded_uuid:
            return uploaded_uuid, True

    if last_list_error is not None:
        raise AudioGreetingError("list_after_upload", str(last_list_error))
    raise AudioGreetingError(
        "resolve_uuid",
        f"uploaded audio was not indexed: {uploaded_name}",
    )
