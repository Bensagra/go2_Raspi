import unittest

import numpy as np

from edge.safety_guard import SafetyGuard


class SafetyGuardTests(unittest.TestCase):
    def test_floor_at_zero_height_is_not_a_front_wall(self) -> None:
        guard = SafetyGuard(
            ground_z_default_m=-0.30,
            ground_z_tolerance_m=0.45,
            min_consider_range_m=0.20,
        )
        xs = np.linspace(0.25, 1.40, 16)
        ys = np.linspace(-0.45, 0.45, 9)
        floor = np.array(
            [[x, y, 0.02] for x in xs for y in ys],
            dtype=np.float32,
        )

        guard.update(floor, now=1.0)
        result = guard.filter_velocity(0.3, 0.0, 0.0, now=1.1)

        self.assertFalse(result["blocked"])
        self.assertIsNone(result["front_clearance_m"])
        self.assertNotIn("obstacle_front", result["reasons"])

    def test_near_body_returns_are_ignored(self) -> None:
        guard = SafetyGuard(min_consider_range_m=0.20)
        self_returns = np.array(
            [[0.08, -0.03, -0.05], [0.09, 0.0, -0.04], [0.10, 0.03, -0.05]],
            dtype=np.float32,
        )

        guard.update(self_returns, now=1.0)
        result = guard.filter_velocity(0.3, 0.0, 0.0, now=1.1)

        self.assertFalse(result["blocked"])
        self.assertIsNone(result["front_clearance_m"])

    def test_real_front_cluster_blocks_forward_motion(self) -> None:
        guard = SafetyGuard(
            ground_z_default_m=-0.30,
            ground_z_tolerance_m=0.45,
            min_consider_range_m=0.20,
            min_cluster_points=3,
            obstacle_cluster_radius_m=0.20,
        )
        floor = np.array(
            [[x, y, -0.30] for x in np.linspace(0.30, 1.50, 8) for y in (-0.4, 0.4)],
            dtype=np.float32,
        )
        obstacle = np.array(
            [[0.35, -0.04, -0.08], [0.36, 0.00, -0.07], [0.34, 0.04, -0.09]],
            dtype=np.float32,
        )

        guard.update(np.vstack((floor, obstacle)), now=1.0)
        result = guard.filter_velocity(0.3, 0.0, 0.0, now=1.1)

        self.assertTrue(result["blocked"])
        self.assertIn("obstacle_front", result["reasons"])
        self.assertLessEqual(result["front_clearance_m"], 0.37)


if __name__ == "__main__":
    unittest.main()
