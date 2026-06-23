"""Reactive LiDAR safety guard for the Go2 edge gateway.

This is the *hard* "never collide / never fall" layer. It runs on the edge
(Raspberry Pi) directly on the raw LiDAR scan — before any compression/uplink —
so the guarantee survives network latency or a dropped link: even if the server
brain or the operator commands a velocity that would drive into an obstacle (or
off a ledge), this layer clamps or vetoes it locally.

Frame convention (Go2 body frame): x forward, y left, z up; angular z is yaw
(positive = turn left). Velocities are the same units the SPORT `Move` command
expects.

The module is pure numpy with no edge/robot imports so it can be unit tested in
isolation. Thresholds are intentionally exposed (and tunable live, like the
color calibration) because the right values depend on the LiDAR mount height,
the L1 blind cone and the floor — they need a quick field calibration.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import numpy as np


class SafetyGuard:
    def __init__(
        self,
        *,
        # Obstacles are points whose height (z, relative to the estimated ground)
        # falls inside this band. Below it is floor; above it is ceiling/overhang
        # the robot passes under.
        obstacle_min_height_m: float = 0.06,
        obstacle_max_height_m: float = 1.20,
        # Robot footprint half width (m): how wide a corridor we need to keep clear
        # when translating forward/back.
        robot_half_width_m: float = 0.26,
        # Clearance thresholds for translation (m): below stop_distance the motion
        # in that direction is vetoed; between stop and slow it is ramped down.
        stop_distance_m: float = 0.35,
        slow_distance_m: float = 0.90,
        # Half angle (deg) of the cone scanned around the direction of motion.
        motion_cone_half_deg: float = 45.0,
        # Cliff / negative-obstacle detection. We look at a strip in front of the
        # robot; if the floor there drops below ground - cliff_drop, or the strip
        # is unexpectedly empty (a void), forward motion is vetoed.
        cliff_enabled: bool = True,
        cliff_lookahead_m: float = 0.55,
        cliff_min_range_m: float = 0.12,
        cliff_drop_m: float = 0.12,
        cliff_strip_half_width_m: float = 0.22,
        # Ground level estimation. The ground z is taken as a low percentile of the
        # nearby point heights, clamped around ground_z_default to stay robust when
        # the L1 barely sees the floor.
        ground_z_default_m: float = -0.30,
        ground_z_tolerance_m: float = 0.25,
        # Points beyond this radius are ignored by the guard (keeps it cheap and
        # focused on imminent collisions).
        max_consider_radius_m: float = 4.0,
        # A scan older than this many seconds is treated as missing. With no fresh
        # LiDAR we fail safe: translation is blocked (set fail_safe_block=True).
        scan_timeout_s: float = 1.0,
        fail_safe_block: bool = True,
        # Minimum obstacle points inside a cone to count as a real obstacle (reject
        # isolated noise returns).
        min_cluster_points: int = 3,
    ) -> None:
        self.obstacle_min_height_m = obstacle_min_height_m
        self.obstacle_max_height_m = obstacle_max_height_m
        self.robot_half_width_m = robot_half_width_m
        self.stop_distance_m = stop_distance_m
        self.slow_distance_m = max(slow_distance_m, stop_distance_m + 1e-3)
        self.motion_cone_half_rad = math.radians(motion_cone_half_deg)
        self.cliff_enabled = cliff_enabled
        self.cliff_lookahead_m = cliff_lookahead_m
        self.cliff_min_range_m = cliff_min_range_m
        self.cliff_drop_m = cliff_drop_m
        self.cliff_strip_half_width_m = cliff_strip_half_width_m
        self.ground_z_default_m = ground_z_default_m
        self.ground_z_tolerance_m = ground_z_tolerance_m
        self.max_consider_radius_m = max_consider_radius_m
        self.scan_timeout_s = scan_timeout_s
        self.fail_safe_block = fail_safe_block
        self.min_cluster_points = min_cluster_points

        # Latest obstacle scan in body frame (x fwd, y left), polar precomputed.
        self._obst_xy: Optional[np.ndarray] = None  # (N, 2)
        self._obst_range: Optional[np.ndarray] = None  # (N,)
        self._obst_bearing: Optional[np.ndarray] = None  # (N,)
        self._ground_z: float = ground_z_default_m
        self._cliff_flag: bool = False
        self._last_update_monotonic: float = 0.0
        self._last_point_count: int = 0

    # ------------------------------------------------------------------ update
    def update(self, points_body: Optional[np.ndarray], now: Optional[float] = None) -> None:
        """Ingest a LiDAR scan already expressed in the robot body frame.

        points_body: (N, 3) float array, x forward, y left, z up (meters).
        """
        now = time.monotonic() if now is None else now
        self._last_update_monotonic = now

        if points_body is None:
            return
        pts = np.asarray(points_body, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
            self._obst_xy = None
            self._obst_range = None
            self._obst_bearing = None
            self._last_point_count = 0
            return

        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        self._last_point_count = int(pts.shape[0])
        if pts.shape[0] == 0:
            self._obst_xy = None
            return

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        planar = np.hypot(x, y)
        near = planar <= self.max_consider_radius_m
        x, y, z, planar = x[near], y[near], z[near], planar[near]
        if x.shape[0] == 0:
            self._obst_xy = None
            return

        # Estimate ground height from a low percentile of nearby returns, clamped
        # so a few stray low points cannot make everything look like floor.
        ground = float(np.percentile(z, 8.0))
        self._ground_z = float(
            np.clip(
                ground,
                self.ground_z_default_m - self.ground_z_tolerance_m,
                self.ground_z_default_m + self.ground_z_tolerance_m,
            )
        )

        rel_h = z - self._ground_z
        is_obstacle = (rel_h >= self.obstacle_min_height_m) & (
            rel_h <= self.obstacle_max_height_m
        )
        ox, oy = x[is_obstacle], y[is_obstacle]
        if ox.shape[0] == 0:
            self._obst_xy = np.empty((0, 2), dtype=np.float32)
            self._obst_range = np.empty((0,), dtype=np.float32)
            self._obst_bearing = np.empty((0,), dtype=np.float32)
        else:
            self._obst_xy = np.column_stack((ox, oy)).astype(np.float32)
            self._obst_range = np.hypot(ox, oy).astype(np.float32)
            self._obst_bearing = np.arctan2(oy, ox).astype(np.float32)

        self._cliff_flag = self._detect_cliff(x, y, z)

    def _detect_cliff(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> bool:
        """Heuristic negative-obstacle (ledge/stair) detector for the frontal strip.

        Returns True when the floor in front drops away. With a 3D LiDAR a down
        step shows up as floor returns sitting well below the local ground, or as
        a hole with no returns where floor was expected. Conservative by design.
        """
        if not self.cliff_enabled:
            return False
        strip = (
            (x >= self.cliff_min_range_m)
            & (x <= self.cliff_lookahead_m)
            & (np.abs(y) <= self.cliff_strip_half_width_m)
        )
        zs = z[strip]
        # Reference against the *least negative* of the adaptive estimate and the
        # calibrated default: a deep drop right in front pulls the adaptive ground
        # down and would otherwise hide itself. The floor the robot stands on is
        # the higher one.
        ground_ref = max(self._ground_z, self.ground_z_default_m)
        if zs.shape[0] >= self.min_cluster_points:
            # A real drop: floor in the strip sits below ground - cliff_drop.
            if float(np.percentile(zs, 20.0)) < ground_ref - self.cliff_drop_m:
                return True
            return False
        # Few/no returns in a strip where the robot is about to step: only treat
        # as a void if the scan otherwise has plenty of nearby points (so it is a
        # genuine hole, not just a sparse scan).
        if self._last_point_count >= 200:
            return True
        return False

    # ------------------------------------------------------------------ filter
    def _min_clearance(self, direction_rad: float) -> float:
        """Closest obstacle range within the motion cone around `direction_rad`,
        accounting for the robot half width. Returns inf when clear."""
        if self._obst_range is None or self._obst_range.shape[0] == 0:
            return math.inf
        bearing = self._obst_bearing
        rng = self._obst_range
        # Angular difference wrapped to [-pi, pi].
        diff = np.arctan2(
            np.sin(bearing - direction_rad), np.cos(bearing - direction_rad)
        )
        in_cone = np.abs(diff) <= self.motion_cone_half_rad
        # Also include obstacles whose lateral offset from the motion axis is
        # within the footprint, even if just outside the cone (close & wide).
        lateral = rng * np.sin(diff)
        forward = rng * np.cos(diff)
        in_corridor = (
            (forward > 0)
            & (np.abs(lateral) <= self.robot_half_width_m)
            & (rng <= self.slow_distance_m + 0.5)
        )
        mask = in_cone | in_corridor
        if not np.any(mask):
            return math.inf
        if int(np.count_nonzero(mask)) < self.min_cluster_points:
            return math.inf
        return float(np.min(rng[mask]))

    def _scale_for_clearance(self, clearance: float) -> float:
        if clearance <= self.stop_distance_m:
            return 0.0
        if clearance >= self.slow_distance_m:
            return 1.0
        span = self.slow_distance_m - self.stop_distance_m
        return float((clearance - self.stop_distance_m) / span)

    def filter_velocity(
        self, vx: float, vy: float, wz: float, now: Optional[float] = None
    ) -> Dict[str, Any]:
        """Clamp a requested body velocity so it cannot drive into an obstacle or
        off a ledge. Pure rotation (wz) is always allowed. Returns a dict with the
        safe velocity and rich diagnostics for telemetry."""
        now = time.monotonic() if now is None else now
        vx_out, vy_out = float(vx), float(vy)
        reasons = []

        scan_age = now - self._last_update_monotonic if self._last_update_monotonic else math.inf
        scan_stale = scan_age > self.scan_timeout_s

        if scan_stale and self.fail_safe_block and (abs(vx_out) > 1e-3 or abs(vy_out) > 1e-3):
            # No trustworthy scan: refuse to translate (rotation still allowed so
            # the robot can look around and recover a scan).
            reasons.append("scan_stale")
            vx_out = 0.0
            vy_out = 0.0

        front = math.inf
        cliff = self._cliff_flag if not scan_stale else False

        if not scan_stale:
            # Forward / backward translation along x.
            if vx_out > 0:
                front = self._min_clearance(0.0)
                if cliff:
                    vx_out = 0.0
                    reasons.append("cliff_front")
                else:
                    scale = self._scale_for_clearance(front)
                    if scale < 1.0:
                        reasons.append("obstacle_front" if scale == 0.0 else "slow_front")
                    vx_out *= scale
            elif vx_out < 0:
                back = self._min_clearance(math.pi)
                scale = self._scale_for_clearance(back)
                if scale < 1.0:
                    reasons.append("obstacle_back")
                vx_out *= scale

            # Lateral translation along y (left positive).
            if vy_out > 0:
                left = self._min_clearance(math.pi / 2.0)
                scale = self._scale_for_clearance(left)
                if scale < 1.0:
                    reasons.append("obstacle_left")
                vy_out *= scale
            elif vy_out < 0:
                right = self._min_clearance(-math.pi / 2.0)
                scale = self._scale_for_clearance(right)
                if scale < 1.0:
                    reasons.append("obstacle_right")
                vy_out *= scale

        blocked = abs(vx_out - vx) > 1e-3 or abs(vy_out - vy) > 1e-3
        return {
            "vx": vx_out,
            "vy": vy_out,
            "wz": float(wz),
            "blocked": bool(blocked),
            "reasons": reasons,
            "front_clearance_m": None if math.isinf(front) else round(front, 3),
            "cliff": bool(cliff),
            "scan_stale": bool(scan_stale),
            "scan_age_s": None if math.isinf(scan_age) else round(scan_age, 3),
            "ground_z_m": round(self._ground_z, 3),
            "obstacle_points": 0 if self._obst_range is None else int(self._obst_range.shape[0]),
        }

    def status(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Lightweight summary for telemetry (no velocity request)."""
        now = time.monotonic() if now is None else now
        scan_age = now - self._last_update_monotonic if self._last_update_monotonic else math.inf
        sectors = {}
        if self._obst_range is not None and self._obst_range.shape[0] > 0:
            for name, direction in (
                ("front", 0.0),
                ("left", math.pi / 2.0),
                ("right", -math.pi / 2.0),
                ("back", math.pi),
            ):
                clr = self._min_clearance(direction)
                sectors[name] = None if math.isinf(clr) else round(clr, 3)
        return {
            "sectors_m": sectors,
            "cliff": bool(self._cliff_flag),
            "ground_z_m": round(self._ground_z, 3),
            "scan_age_s": None if math.isinf(scan_age) else round(scan_age, 3),
            "scan_stale": bool(scan_age > self.scan_timeout_s),
            "obstacle_points": 0 if self._obst_range is None else int(self._obst_range.shape[0]),
        }

    def apply_settings(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Update tunable thresholds live (used by the set_safety command)."""
        mapping = {
            "stop_distance_m": "stop_distance_m",
            "slow_distance_m": "slow_distance_m",
            "robot_half_width_m": "robot_half_width_m",
            "obstacle_min_height_m": "obstacle_min_height_m",
            "obstacle_max_height_m": "obstacle_max_height_m",
            "cliff_enabled": "cliff_enabled",
            "cliff_lookahead_m": "cliff_lookahead_m",
            "cliff_drop_m": "cliff_drop_m",
            "ground_z_default_m": "ground_z_default_m",
            "max_consider_radius_m": "max_consider_radius_m",
            "fail_safe_block": "fail_safe_block",
        }
        for key, attr in mapping.items():
            if key in payload:
                value = payload[key]
                if key in {"cliff_enabled", "fail_safe_block"}:
                    setattr(self, attr, bool(value))
                else:
                    setattr(self, attr, float(value))
        self.slow_distance_m = max(self.slow_distance_m, self.stop_distance_m + 1e-3)
        return self.config()

    def config(self) -> Dict[str, Any]:
        return {
            "stop_distance_m": self.stop_distance_m,
            "slow_distance_m": self.slow_distance_m,
            "robot_half_width_m": self.robot_half_width_m,
            "obstacle_min_height_m": self.obstacle_min_height_m,
            "obstacle_max_height_m": self.obstacle_max_height_m,
            "cliff_enabled": self.cliff_enabled,
            "cliff_lookahead_m": self.cliff_lookahead_m,
            "cliff_drop_m": self.cliff_drop_m,
            "ground_z_default_m": self.ground_z_default_m,
            "max_consider_radius_m": self.max_consider_radius_m,
            "fail_safe_block": self.fail_safe_block,
        }
