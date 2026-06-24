import unittest

import numpy as np

from server.perception import FaceAnalyzer


class FaceQualityTests(unittest.TestCase):
    def test_padded_crop_keeps_context_around_face(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        crop = FaceAnalyzer._crop_with_padding(frame, (120, 70, 180, 150))

        self.assertIsNotNone(crop)
        self.assertGreater(crop.shape[0], 80)
        self.assertGreater(crop.shape[1], 60)

    def test_face_touching_edge_scores_lower_than_centered_face(self) -> None:
        centered = FaceAnalyzer._quality(
            (110, 60, 190, 160),
            (320, 240),
            0.95,
            220.0,
        )
        edge = FaceAnalyzer._quality(
            (0, 60, 80, 160),
            (320, 240),
            0.95,
            220.0,
        )

        self.assertGreater(centered, edge)


if __name__ == "__main__":
    unittest.main()
