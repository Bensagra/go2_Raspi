import json
import unittest

from edge.audio_greeting import find_audio_uuid, resolve_audio_file


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

    def test_bundled_audio_resolves_from_any_working_directory(self) -> None:
        path = resolve_audio_file("Escuela Técnica Ort 3.m4a")
        self.assertTrue(path.endswith(".m4a"))
        with open(path, "rb") as audio_file:
            self.assertTrue(audio_file.read(8))


if __name__ == "__main__":
    unittest.main()
