"""Autonomous exploration + people-capture state machine (server side).

One AutonomyController is spawned per active robot. It glues together:

* the FrontierExplorer (where to go to keep mapping),
* the Perception facade (find people, capture + recognise faces),
* the robot command channel (drive_velocity goals to the edge).

The edge owns the hard collision/cliff avoidance, so this brain stays high level
and optimistic: it just steers toward goals and toward people, and the edge guard
guarantees it never drives into anything.

State machine
-------------
    EXPLORE  --person seen--> APPROACH --close enough--> CAPTURE --done--> EXPLORE
       |                          |  (lost / timeout)        |
   (no frontier)              EXPLORE <--------------------- EXPLORE
       v
     DONE

E-stop / stop transition to STOPPED from anywhere and end the session.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from exploration import FrontierExplorer
except ImportError:  # package import
    from server.exploration import FrontierExplorer  # type: ignore


@dataclass
class AutonomyConfig:
    control_hz: float = 5.0
    v_max: float = 0.35
    w_max: float = 0.8
    grid_res: float = 0.15
    robot_radius_m: float = 0.30
    obstacle_min_height_m: float = 0.10
    obstacle_max_height_m: float = 1.60
    # Perception cadence + triggers.
    detect_interval_s: float = 0.4
    person_trigger_score: float = 0.55
    person_trigger_area_frac: float = 0.03  # bbox area fraction of frame to engage
    # APPROACH: stop and capture when the person bbox fills this fraction of the
    # frame height (a proxy for "close enough for a good face"). The edge guard
    # keeps the actual safe distance.
    approach_fill_frac: float = 0.50
    approach_timeout_s: float = 18.0
    person_lost_timeout_s: float = 3.0
    # CAPTURE window: collect frames and keep the best face.
    capture_window_s: float = 2.4
    # After capturing, ignore people for this long / within this radius so we
    # don't re-capture the same person on the spot.
    capture_cooldown_s: float = 15.0
    capture_cooldown_radius_m: float = 1.5
    # Stuck recovery: if the robot barely moves while commanded to, rotate to
    # find a new way.
    stuck_timeout_s: float = 6.0
    stuck_progress_m: float = 0.12


class AutonomyController:
    def __init__(self, runtime: Any, robot_id: str, config: AutonomyConfig) -> None:
        self.rt = runtime
        self.robot_id = robot_id
        self.cfg = config
        self.explorer = FrontierExplorer(
            grid_res=config.grid_res,
            robot_radius_m=config.robot_radius_m,
            obstacle_min_height_m=config.obstacle_min_height_m,
            obstacle_max_height_m=config.obstacle_max_height_m,
        )
        self.state = "idle"
        self.task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self.started_at = 0.0

        # Runtime working memory.
        self._last_detect_at = 0.0
        self._last_plan: Dict[str, Any] = {}
        self._target_person: Optional[Dict[str, Any]] = None
        self._approach_started_at = 0.0
        self._person_last_seen_at = 0.0
        self._capture_started_at = 0.0
        self._best_capture: Optional[Any] = None
        self._capture_positions: List[Tuple[float, float, float]] = []  # x,y,ts
        self._last_progress_pose: Optional[Tuple[float, float]] = None
        self._last_progress_at = 0.0
        self._recovery_until = 0.0
        self.captures_count = 0

    # ----------------------------------------------------------------- control
    def start(self) -> None:
        if self.task and not self.task.done():
            return
        self._stop.clear()
        self.task = asyncio.create_task(self.run())

    async def stop(self, reason: str = "stopped") -> None:
        self._stop.set()
        if self.task:
            try:
                await asyncio.wait_for(self.task, timeout=3.0)
            except Exception:
                self.task.cancel()
        self.state = reason

    def snapshot(self) -> Dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "state": self.state,
            "running": bool(self.task and not self.task.done()),
            "captures": self.captures_count,
            "plan": {
                k: self._last_plan.get(k)
                for k in ("done", "goal_xy", "coverage", "explored_area_m2",
                          "frontier_cells", "reachable_frontier_clusters")
            },
            "target_person": self._target_person,
            "started_at": self.started_at,
        }

    # -------------------------------------------------------------------- loop
    async def run(self) -> None:
        self.started_at = time.time()
        # Make sure the robot is streaming what we need and accepts autonomy.
        self._cmd("set_lidar", {"enabled": True, "subscribe": True})
        self._cmd("set_camera_stream", {
            "enabled": True,
            "format": "h264",
            "target_fps": 24,
            "max_width": 960,
            "bitrate_kbps": 2200,
            "gop": 24,
        })
        self._cmd("set_autonomy", {"enabled": True})
        self._set_state("explore")
        interval = 1.0 / max(self.cfg.control_hz, 1.0)
        try:
            while not self._stop.is_set():
                await asyncio.sleep(interval)
                try:
                    await self._tick()
                except Exception as exc:  # never let one bad tick kill autonomy
                    await self.rt.autonomy_event(self.robot_id, "autonomy_error", {"error": str(exc)})
                    self._drive(0.0, 0.0, 0.0)
        finally:
            self._drive(0.0, 0.0, 0.0)
            self._cmd("set_autonomy", {"enabled": False})
            if self.state not in {"done", "estopped", "stopped"}:
                self._set_state("stopped")

    async def _tick(self) -> None:
        pose = self._pose()
        if pose is None:
            self._drive(0.0, 0.0, 0.0)
            return
        rx, ry, ryaw = pose

        if self.state == "explore":
            await self._tick_explore(rx, ry, ryaw)
        elif self.state == "approach":
            await self._tick_approach(rx, ry, ryaw)
        elif self.state == "capture":
            await self._tick_capture(rx, ry, ryaw)
        else:
            self._drive(0.0, 0.0, 0.0)

    async def _tick_explore(self, rx: float, ry: float, ryaw: float) -> None:
        # 1) Look for people to engage.
        if self._maybe_engage_person(rx, ry):
            return

        # 2) Recovery rotation if we were stuck.
        now = time.monotonic()
        if now < self._recovery_until:
            self._drive(0.0, 0.0, self.cfg.w_max * 0.8)
            return

        # 3) Frontier exploration.
        points, path = self.rt._copy_map_arrays(self.robot_id)
        plan = self.explorer.compute(points, path, (rx, ry))
        self._last_plan = plan
        if plan.get("done"):
            self._drive(0.0, 0.0, 0.0)
            self._set_state("done")
            await self.rt.autonomy_event(
                self.robot_id, "autonomy_done",
                {"coverage": plan.get("coverage"), "area_m2": plan.get("explored_area_m2")},
            )
            self._stop.set()
            return
        goal = plan.get("goal_xy")
        if goal is None:
            self._drive(0.0, 0.0, self.cfg.w_max * 0.6)  # scan for frontiers
            return

        cmd = self.explorer.velocity_to_goal(
            (rx, ry), ryaw, (goal[0], goal[1]), v_max=self.cfg.v_max, w_max=self.cfg.w_max
        )
        self._update_progress(rx, ry, moving=cmd["vx"] > 0.02 or abs(cmd["wz"]) > 0.05)
        self._drive(cmd["vx"], cmd["vy"], cmd["wz"])

    async def _tick_approach(self, rx: float, ry: float, ryaw: float) -> None:
        now = time.monotonic()
        if now - self._approach_started_at > self.cfg.approach_timeout_s:
            self._target_person = None
            self._set_state("explore")
            return

        boxes, dims = self.rt.perception.detect_people(self.robot_id) if self.rt.perception else ([], None)
        if not boxes or dims is None:
            if now - self._person_last_seen_at > self.cfg.person_lost_timeout_s:
                self._target_person = None
                self._set_state("explore")
            else:
                self._drive(0.0, 0.0, 0.0)
            return

        w, h = dims
        # Engage the largest (nearest) person.
        box = max(boxes, key=lambda b: b.area)
        self._person_last_seen_at = now
        self._target_person = {
            "cx_frac": round(box.cx / w, 3),
            "fill_frac": round(box.height / h, 3),
            "score": round(box.score, 3),
        }

        fill = box.height / h
        if fill >= self.cfg.approach_fill_frac:
            self._drive(0.0, 0.0, 0.0)
            self._capture_started_at = now
            self._best_capture = None
            self._capture_positions.append((rx, ry, time.time()))
            self._set_state("capture")
            return

        # Steer: center the person, advance while roughly centered. The edge guard
        # keeps us from actually colliding (the person is an obstacle).
        offset = (box.cx - w / 2.0) / (w / 2.0)  # -1 .. 1
        wz = float(np.clip(-self.cfg.w_max * offset, -self.cfg.w_max, self.cfg.w_max))
        vx = self.cfg.v_max * 0.6 if abs(offset) < 0.35 else 0.0
        self._drive(vx, 0.0, wz)

    async def _tick_capture(self, rx: float, ry: float, ryaw: float) -> None:
        self._drive(0.0, 0.0, 0.0)
        now = time.monotonic()
        if self.rt.perception is not None:
            candidate = self.rt.perception.best_face_candidate(self.robot_id)
            if candidate and (
                self._best_capture is None
                or candidate.quality > self._best_capture.quality
            ):
                self._best_capture = candidate

        if now - self._capture_started_at < self.cfg.capture_window_s:
            return

        if self._best_capture is not None and self.rt.perception is not None:
            capture = self.rt.perception.commit_face_capture(
                self.robot_id,
                self._best_capture,
            )
        else:
            capture = None

        if capture is not None:
            self.captures_count += 1
            await self.rt.autonomy_event(
                self.robot_id, "person_captured",
                {**capture, "robot_pose": [round(rx, 2), round(ry, 2)]},
            )
        else:
            quality = (
                None
                if self._best_capture is None
                else round(float(self._best_capture.quality), 4)
            )
            await self.rt.autonomy_event(
                self.robot_id,
                "capture_empty",
                {"best_quality": quality},
            )
        self._target_person = None
        self._best_capture = None
        self._set_state("explore")

    # ----------------------------------------------------------------- helpers
    def _maybe_engage_person(self, rx: float, ry: float) -> bool:
        if self.rt.perception is None:
            return False
        now = time.monotonic()
        if now - self._last_detect_at < self.cfg.detect_interval_s:
            return False
        self._last_detect_at = now
        boxes, dims = self.rt.perception.detect_people(self.robot_id)
        if not boxes or dims is None:
            return False
        w, h = dims
        box = max(boxes, key=lambda b: b.area)
        area_frac = box.area / float(w * h)
        if box.score < self.cfg.person_trigger_score or area_frac < self.cfg.person_trigger_area_frac:
            return False
        # Cooldown: skip if we just captured near here.
        tnow = time.time()
        for (cx, cy, cts) in self._capture_positions[-10:]:
            if tnow - cts < self.cfg.capture_cooldown_s and math.hypot(rx - cx, ry - cy) < self.cfg.capture_cooldown_radius_m:
                return False
        self._approach_started_at = now
        self._person_last_seen_at = now
        self._set_state("approach")
        return True

    def _update_progress(self, rx: float, ry: float, moving: bool) -> None:
        now = time.monotonic()
        if self._last_progress_pose is None:
            self._last_progress_pose = (rx, ry)
            self._last_progress_at = now
            return
        moved = math.hypot(rx - self._last_progress_pose[0], ry - self._last_progress_pose[1])
        if moved >= self.cfg.stuck_progress_m:
            self._last_progress_pose = (rx, ry)
            self._last_progress_at = now
            return
        if moving and (now - self._last_progress_at) > self.cfg.stuck_timeout_s:
            # Stuck: trigger a recovery rotation and reset the timer.
            self._recovery_until = now + 1.2
            self._last_progress_pose = (rx, ry)
            self._last_progress_at = now

    def _pose(self) -> Optional[Tuple[float, float, float]]:
        tel = self.rt.latest_telemetry.get(self.robot_id, {})
        # Use the LiDAR-frame pose (rt/utlidar/robot_pose): it is in the SAME frame
        # as the accumulated voxel map the explorer plans on. Mixing the sport-leg
        # odometry pose with map-frame walls puts the robot in the wrong grid cell,
        # so walls/distances (and the heading toward goals) come out wrong. Fall
        # back to the sport pose only when the LiDAR pose is unavailable.
        candidates = []
        if isinstance(tel, dict):
            mp = tel.get("map_pose")
            if isinstance(mp, dict):
                candidates.append(mp)
            sp = tel.get("pose")
            if isinstance(sp, dict):
                candidates.append(sp)
        for pose in candidates:
            try:
                x = float(pose.get("x"))
                y = float(pose.get("y"))
                yaw = float(pose.get("yaw", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw):
                return x, y, yaw
        return None

    def _drive(self, vx: float, vy: float, wz: float) -> None:
        self.rt.autonomy_drive(self.robot_id, vx, vy, wz)

    def _cmd(self, cmd_type: str, payload: Dict[str, Any]) -> None:
        self.rt.autonomy_command(self.robot_id, cmd_type, payload)

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            asyncio.create_task(
                self.rt.autonomy_event(self.robot_id, "autonomy_state", self.snapshot())
            )
