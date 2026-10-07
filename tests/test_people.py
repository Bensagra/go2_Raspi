"""People numbering: tracker, identity votes and the persistent gallery (no ML models needed)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from server.people import Face, IdentityResolver, PeopleGallery, Tracker, TRACK_TTL_S


def embedding(seed, noise=0.0, base=None):
    rng = np.random.default_rng(seed)
    v = (base if base is not None else rng.normal(size=128)) + noise * rng.normal(size=128)
    return (v / np.linalg.norm(v)).astype(np.float32)


def face(emb, width=80, frontal=True):
    return Face((10, 10, 10 + width, 10 + width), 0.95, emb, np.zeros((112, 112, 3), np.uint8), frontal)


class TrackerTests(unittest.TestCase):
    def test_keeps_ids_in_box_order_and_expires(self):
        tracker = Tracker()
        a, b = (0, 0, 100, 200), (300, 0, 400, 200)
        first = tracker.update([a, b], 0.0)
        # Reversed order and slight motion: ids follow the boxes, not the list position.
        second = tracker.update([(305, 0, 405, 200), (5, 0, 105, 200)], 0.2)
        self.assertEqual([t.track_id for t in second], [first[1].track_id, first[0].track_id])
        third = tracker.update([], TRACK_TTL_S + 1)
        self.assertEqual(third, [])
        self.assertEqual(tracker.tracks, {})


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.gallery = PeopleGallery(Path(self.temp.name))
        self.resolver = IdentityResolver(self.gallery)

    def tearDown(self):
        self.temp.cleanup()

    def test_enrolls_after_two_frames_then_recognises_new_track(self):
        alice = embedding(1)
        track = Tracker().update([(0, 0, 100, 200)], 0)[0]
        self.assertIsNone(self.resolver.observe("r", track, face(alice)))
        created = self.resolver.observe("r", track, face(embedding(1, 0.2, alice)))
        self.assertEqual((created["number"], created["name"]), (1, "Persona 1"))
        self.assertEqual(track.person_id, created["person_id"])

        # Another day: new track, slightly different face, same number.
        later = Tracker().update([(0, 0, 100, 200)], 0)[0]
        self.resolver.observe("r", later, face(embedding(1, 0.3, alice)))
        self.assertEqual(later.person_id, created["person_id"])

    def test_different_person_gets_next_number_and_bad_faces_never_enroll(self):
        alice, bob = embedding(1), embedding(2)
        t1 = Tracker().update([(0, 0, 100, 200)], 0)[0]
        for _ in range(2):
            self.resolver.observe("r", t1, face(alice))
        t2 = Tracker().update([(0, 0, 100, 200)], 0)[0]
        for _ in range(3):
            self.resolver.observe("r", t2, face(bob, width=20))          # too small
            self.resolver.observe("r", t2, face(bob, frontal=False))    # profile
        self.assertIsNone(t2.person_id)
        for _ in range(2):
            self.resolver.observe("r", t2, face(bob))
        self.assertEqual([p["name"] for p in self.gallery.list("r")], ["Persona 1", "Persona 2"])

    def test_one_number_per_frame_and_persistence(self):
        alice = embedding(1)
        tracks = Tracker().update([(0, 0, 100, 200), (300, 0, 400, 200)], 0)
        for _ in range(2):
            self.resolver.observe("r", tracks[0], face(alice))
        self.resolver.observe("r", tracks[1], face(alice))
        IdentityResolver.dedupe(tracks)
        self.assertEqual([t.person_id is not None for t in tracks], [True, False])

        pid = tracks[0].person_id
        self.assertTrue(self.gallery.rename("r", pid, "Juan"))
        self.gallery.flush()
        reloaded = PeopleGallery(Path(self.temp.name))
        self.assertEqual(reloaded.list("r")[0]["name"], "Juan")
        self.assertGreater(reloaded.match("r", alice)[1], 0.99)
        self.assertEqual(reloaded.delete("r"), 1)
        track = Tracker().update([(0, 0, 100, 200)], 0)[0]
        resolver = IdentityResolver(reloaded)
        for _ in range(2):
            resolver.observe("r", track, face(embedding(3)))
        self.assertEqual(reloaded.list("r")[0]["number"], 1)  # Purging everyone restarts numbering.


class IdentifierMessageTests(unittest.TestCase):
    def test_result_message_is_json_with_stable_number(self):
        import json
        import cv2
        from server.people import PeopleIdentifier

        class FakeDetector:  # Real models return numpy float32 coordinates.
            def detect(self, image):
                return [(np.float32(10), np.float32(20), np.float32(110), np.float32(220), np.float32(0.9))]

        alice = embedding(1)

        class FakeFaces:
            def faces(self, image, region):
                return [Face((np.float32(30), np.float32(30), np.float32(110), np.float32(110)), 0.95,
                             alice, np.zeros((112, 112, 3), np.uint8), True)]

        with tempfile.TemporaryDirectory() as temp:
            messages, new = [], []
            ident = PeopleIdentifier(Path(temp), Path(temp), on_result=lambda r, m: messages.append(m),
                                     on_new_person=lambda r, p: new.append(p))
            ident.detector, ident.faces = FakeDetector(), FakeFaces()
            jpg = cv2.imencode(".jpg", np.zeros((240, 320, 3), np.uint8))[1].tobytes()
            for seq in (1, 2, 3):
                ident._process("r", {"seq": seq, "ts": 1.0}, jpg)
            decoded = json.loads(json.dumps(messages[-1]))
            self.assertEqual(decoded["type"], "people")
            self.assertEqual(decoded["people"][0]["name"], "Persona 1")
            self.assertEqual(decoded["people"][0]["box"], [round(10 / 320, 4), round(20 / 240, 4),
                                                           round(110 / 320, 4), round(220 / 240, 4)])
            self.assertEqual([p["name"] for p in new], ["Persona 1"])
            json.dumps(new)


if __name__ == "__main__":
    unittest.main()
