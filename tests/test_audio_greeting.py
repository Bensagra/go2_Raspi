import asyncio
import json
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

from edge.audio_greeting import (
    find_audio_uuid,
    prepare_go2_wav,
    resolve_audio_file,
    resolve_audio_uuid_on_hub,
)


class AudioGreetingTests(unittest.TestCase):
    def test_finds_uuid_in_unitree_nested_json_response(self) -> None:
        response = {
            "data": {
                "data": json.dumps(
                    {
                        "audio_list": [
                            {
                                "CUSTOM_NAME": "Escuela Técnica Ort 3",
                                "UNIQUE_ID": "audio-123",
                            }
                        ]
                    }
                )
            }
        }

        self.assertEqual(
            find_audio_uuid(response, "Escuela Técnica Ort 3.m4a"),
            "audio-123",
        )

    def test_audio_name_matching_handles_unicode_and_extension(self) -> None:
        response = {
            "audio_list": [
                {
                    "custom_name": "Escuela Te\u0301cnica Ort 3.wav",
                    "unique_id": "audio-456",
                }
            ]
        }

        self.assertEqual(
            find_audio_uuid(response, "Escuela Técnica Ort 3"),
            "audio-456",
        )

    def test_resolves_hyphenated_unicode_wav_from_spaced_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            actual = Path(temp_dir) / "Escuela-Te\u0301cnica-Ort-3.wav"
            actual.write_bytes(b"RIFF-test")
            (Path(temp_dir) / "Escuela Técnica Ort 3.m4a").write_bytes(b"m4a")

            resolved = resolve_audio_file(
                str(Path(temp_dir) / "Escuela Técnica Ort 3.wav")
            )

            self.assertEqual(Path(resolved), actual.resolve())

    def test_normalizes_48khz_pcm_wav_for_go2(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "greeting.wav"
            samples = array("h", [0, 1000, -1000, 500] * 12000)
            with wave.open(str(source), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(48000)
                wav_file.writeframes(samples.tobytes())

            prepared = prepare_go2_wav(str(source))

            self.assertIsNotNone(prepared)
            self.assertNotEqual(Path(prepared), source)
            with wave.open(str(prepared), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getframerate(), 44100)
                self.assertEqual(wav_file.getcomptype(), "NONE")

    def test_boosts_quiet_compatible_wav_for_go2_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "quiet.wav"
            samples = array("h", [0, 1000, -1000, 500] * 2000)
            with wave.open(str(source), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(44100)
                wav_file.writeframes(samples.tobytes())

            prepared = prepare_go2_wav(str(source))

            self.assertIsNotNone(prepared)
            self.assertNotEqual(Path(prepared), source)
            with wave.open(str(prepared), "rb") as wav_file:
                out = array("h")
                out.frombytes(wav_file.readframes(wav_file.getnframes()))
            self.assertGreaterEqual(max(abs(int(v)) for v in out), 25000)


class FakeAudioHub:
    def __init__(self) -> None:
        self.uploaded_paths = []
        self.list_calls = 0

    async def get_audio_list(self):
        self.list_calls += 1
        audio_list = []
        if self.uploaded_paths:
            audio_list.append(
                {
                    "CUSTOM_NAME": Path(self.uploaded_paths[-1]).stem,
                    "UNIQUE_ID": "uploaded-uuid",
                }
            )
        return {"data": {"data": json.dumps({"audio_list": audio_list})}}

    async def upload_audio_file(self, path: str) -> None:
        self.uploaded_paths.append(path)


class AudioGreetingFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_uploads_then_resolves_uuid(self) -> None:
        hub = FakeAudioHub()

        async def no_sleep(_: float) -> None:
            await asyncio.sleep(0)

        uuid, uploaded = await resolve_audio_uuid_on_hub(
            hub,
            "/audio/Escuela-Técnica-Ort-3.wav",
            lambda _: "/cache/Escuela-Técnica-Ort-3.wav",
            sleep=no_sleep,
        )

        self.assertEqual(uuid, "uploaded-uuid")
        self.assertTrue(uploaded)
        self.assertEqual(
            hub.uploaded_paths, ["/cache/Escuela-Técnica-Ort-3.wav"]
        )
        self.assertGreaterEqual(hub.list_calls, 2)

    async def test_does_not_play_unrelated_existing_audio_before_upload(self) -> None:
        class HubWithUnrelatedExisting(FakeAudioHub):
            async def get_audio_list(self):
                self.list_calls += 1
                audio_list = [
                    {
                        "CUSTOM_NAME": "silence",
                        "UNIQUE_ID": "wrong-uuid",
                        "CREATE_TIME": 999,
                    }
                ]
                if self.uploaded_paths:
                    audio_list.append(
                        {
                            "CUSTOM_NAME": Path(self.uploaded_paths[-1]).stem,
                            "UNIQUE_ID": "uploaded-uuid",
                            "CREATE_TIME": 1000,
                        }
                    )
                return {"data": {"data": json.dumps({"audio_list": audio_list})}}

        hub = HubWithUnrelatedExisting()

        async def no_sleep(_: float) -> None:
            await asyncio.sleep(0)

        uuid, uploaded = await resolve_audio_uuid_on_hub(
            hub,
            "/audio/Escuela-Técnica-Ort-3.wav",
            lambda _: "/cache/Escuela-Técnica-Ort-3.wav",
            sleep=no_sleep,
        )

        self.assertEqual(uuid, "uploaded-uuid")
        self.assertTrue(uploaded)
        self.assertEqual(
            hub.uploaded_paths, ["/cache/Escuela-Técnica-Ort-3.wav"]
        )


if __name__ == "__main__":
    unittest.main()
