"""Frontier-based exploration planner (server side).

Consumes the accumulated LiDAR voxel map + the robot's travelled path and picks
the next place to go so the robot keeps discovering unmapped space, then reports
"done" when no reachable frontier remains. It only produces a *goal point* and a
suggested body velocity; the hard collision/cliff avoidance is the edge's job, so
this planner can stay simple and optimistic.

Everything is pure numpy so it can be unit tested without the robot.

World frame: x, y in meters (odom), z up. Cells are square (grid_res).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

UNKNOWN = 0
FREE = 1
OCCUPIED = 2


class FrontierExplorer:
    def __init__(
        self,
        *,
        grid_res: float = 0.15,
        # Obstacle band: voxels whose height above the map floor is in this range
        # mark a cell occupied. Below it is floor (free); above passes overhead.
        obstacle_min_height_m: float = 0.10,
        obstacle_max_height_m: float = 1.60,
        # Robot footprint radius (m): obstacles are inflated by this so goals and
        # the BFS keep clearance from walls.
        robot_radius_m: float = 0.30,
        # A frontier cluster must have at least this many cells to be a real goal
        # (rejects single-cell noise frontiers).
        frontier_min_cells: int = 4,
        # Path cells are dilated by this radius (m) to seed the free space the
        # robot has actually driven through.
        path_clearance_m: float = 0.25,
        # Stop exploring once the nearest reachable frontier is closer than this
        # (nothing meaningful left) — handled by the caller via `done`.
        max_grid_cells: int = 600,
    ) -> None:
        self.grid_res = grid_res
        self.obstacle_min_height_m = obstacle_min_height_m
        self.obstacle_max_height_m = obstacle_max_height_m
        self.robot_radius_m = robot_radius_m
        self.frontier_min_cells = frontier_min_cells
        self.path_clearance_m = path_clearance_m
        self.max_grid_cells = max_grid_cells

    # ------------------------------------------------------------------ public
    def compute(
        self,
        points: np.ndarray,
        path_xy: np.ndarray,
        robot_xy: Tuple[float, float],
    ) -> Dict[str, Any]:
        """points: (N,3) world voxel centers. path_xy: (M,2) travelled path.
        Returns a plan dict: goal_xy (or None), done, coverage, frontier counts,
        and the occupancy grid metadata (for the dashboard overlay)."""
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3) if points is not None else np.empty((0, 3), np.float32)
        path_xy = np.asarray(path_xy, dtype=np.float32).reshape(-1, 2) if path_xy is not None else np.empty((0, 2), np.float32)

        if points.shape[0] == 0 and path_xy.shape[0] == 0:
            return self._empty_plan("no_map")

        # Bounds from everything we know about, padded by a ring of unknown so
        # frontiers can form at the edges.
        xs = np.concatenate([points[:, 0], path_xy[:, 0]]) if path_xy.size else points[:, 0]
        ys = np.concatenate([points[:, 1], path_xy[:, 1]]) if path_xy.size else points[:, 1]
        pad = max(self.robot_radius_m * 2.0, 0.6)
        min_x, max_x = float(xs.min()) - pad, float(xs.max()) + pad
        min_y, max_y = float(ys.min()) - pad, float(ys.max()) + pad

        w = int(math.ceil((max_x - min_x) / self.grid_res)) + 1
        h = int(math.ceil((max_y - min_y) / self.grid_res)) + 1
        # Guard against an unbounded grid (downsample resolution if needed).
        res = self.grid_res
        while max(w, h) > self.max_grid_cells:
            res *= 1.5
            w = int(math.ceil((max_x - min_x) / res)) + 1
            h = int(math.ceil((max_y - min_y) / res)) + 1

        origin = np.array([min_x, min_y], dtype=np.float32)

        def to_cell(px: np.ndarray, py: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            cx = np.clip(((px - origin[0]) / res).astype(np.int32), 0, w - 1)
            cy = np.clip(((py - origin[1]) / res).astype(np.int32), 0, h - 1)
            return cx, cy

        grid = np.full((h, w), UNKNOWN, dtype=np.uint8)

        # Classify voxels into floor (free evidence) vs obstacle.
        if points.shape[0]:
            floor_z = float(np.percentile(points[:, 2], 5.0))
            rel = points[:, 2] - floor_z
            obst = (rel >= self.obstacle_min_height_m) & (rel <= self.obstacle_max_height_m)
            floor = rel < self.obstacle_min_height_m
            fcx, fcy = to_cell(points[floor, 0], points[floor, 1])
            grid[fcy, fcx] = FREE
            ocx, ocy = to_cell(points[obst, 0], points[obst, 1])
            grid[ocy, ocx] = OCCUPIED  # obstacle overrides free

        # The travelled path is definitely free; stamp a small disc around it.
        if path_xy.shape[0]:
            pcx, pcy = to_cell(path_xy[:, 0], path_xy[:, 1])
            rad = max(int(round(self.path_clearance_m / res)), 0)
            for cx, cy in zip(pcx.tolist(), pcy.tolist()):
                x0, x1 = max(0, cx - rad), min(w, cx + rad + 1)
                y0, y1 = max(0, cy - rad), min(h, cy + rad + 1)
                block = grid[y0:y1, x0:x1]
                block[block != OCCUPIED] = FREE

        # Inflate obstacles by the robot radius for planning clearance.
        inflate = max(int(round(self.robot_radius_m / res)), 0)
        occ_inflated = self._inflate(grid == OCCUPIED, inflate)
        traversable = (grid == FREE) & (~occ_inflated)

        robot_cx = int(np.clip(round((robot_xy[0] - origin[0]) / res), 0, w - 1))
        robot_cy = int(np.clip(round((robot_xy[1] - origin[1]) / res), 0, h - 1))

        # Frontiers: free cells bordering unknown, away from obstacles.
        frontier_mask = self._frontier_cells(grid, traversable)

        # Reachability BFS from the robot over traversable cells; nearest frontier.
        goal_cell, dist_field, reachable_frontiers = self._nearest_frontier(
            traversable, frontier_mask, (robot_cy, robot_cx)
        )

        free_count = int(np.count_nonzero(grid == FREE))
        coverage = self._coverage(grid)
        plan: Dict[str, Any] = {
            "done": goal_cell is None,
            "reason": "no_frontier" if goal_cell is None else "",
            "goal_xy": None,
            "frontier_cells": int(np.count_nonzero(frontier_mask)),
            "reachable_frontier_clusters": reachable_frontiers,
            "free_cells": free_count,
            "explored_area_m2": round(free_count * res * res, 2),
            "coverage": coverage,
            "grid": {
                "origin": [float(origin[0]), float(origin[1])],
                "res": float(res),
                "w": w,
                "h": h,
            },
        }
        if goal_cell is not None:
            gy, gx = goal_cell
            plan["goal_xy"] = [
                float(origin[0] + (gx + 0.5) * res),
                float(origin[1] + (gy + 0.5) * res),
            ]
            plan["goal_distance_m"] = round(
                float(dist_field[gy, gx]) * res, 2
            ) if dist_field is not None else None
        return plan

    def velocity_to_goal(
        self,
        robot_xy: Tuple[float, float],
        robot_yaw: float,
        goal_xy: Tuple[float, float],
        *,
        v_max: float = 0.35,
        w_max: float = 0.8,
        heading_tol_deg: float = 25.0,
        arrive_radius_m: float = 0.25,
    ) -> Dict[str, Any]:
        """Simple proportional controller: face the goal, then advance. Obstacle
        avoidance is the edge guard's responsibility, so this just steers."""
        dx = goal_xy[0] - robot_xy[0]
        dy = goal_xy[1] - robot_xy[1]
        dist = math.hypot(dx, dy)
        if dist <= arrive_radius_m:
            return {"vx": 0.0, "vy": 0.0, "wz": 0.0, "arrived": True, "distance_m": dist}

        desired = math.atan2(dy, dx)
        err = math.atan2(math.sin(desired - robot_yaw), math.cos(desired - robot_yaw))
        wz = float(np.clip(1.5 * err, -w_max, w_max))
        if abs(err) <= math.radians(heading_tol_deg):
            # Slow down as heading error grows; full speed when aligned.
            vx = v_max * max(0.0, math.cos(err))
        else:
            vx = 0.0  # turn in place first
        return {
            "vx": float(vx),
            "vy": 0.0,
            "wz": wz,
            "arrived": False,
            "distance_m": dist,
            "heading_error_deg": round(math.degrees(err), 1),
        }

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _inflate(mask: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0 or not mask.any():
            return mask.copy()
        out = mask.copy()
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
        return out

    @staticmethod
    def _frontier_cells(grid: np.ndarray, traversable: np.ndarray) -> np.ndarray:
        unknown = grid == UNKNOWN
        neigh_unknown = np.zeros_like(unknown)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neigh_unknown |= np.roll(np.roll(unknown, dy, axis=0), dx, axis=1)
        return traversable & neigh_unknown

    def _nearest_frontier(
        self,
        traversable: np.ndarray,
        frontier_mask: np.ndarray,
        start: Tuple[int, int],
    ) -> Tuple[Optional[Tuple[int, int]], Optional[np.ndarray], int]:
        h, w = traversable.shape
        sy, sx = start
        # If the robot cell isn't traversable (e.g. just spawned), snap to the
        # nearest traversable cell so BFS can start.
        if not traversable[sy, sx]:
            ys, xs = np.where(traversable)
            if ys.size == 0:
                return None, None, 0
            k = int(np.argmin((ys - sy) ** 2 + (xs - sx) ** 2))
            sy, sx = int(ys[k]), int(xs[k])

        dist = np.full((h, w), -1, dtype=np.int32)
        dist[sy, sx] = 0
        q = deque([(sy, sx)])
        reached: List[Tuple[int, int, int]] = []  # (dist, y, x) on frontier
        while q:
            cy, cx = q.popleft()
            d = dist[cy, cx]
            if frontier_mask[cy, cx]:
                reached.append((d, cy, cx))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and dist[ny, nx] < 0 and traversable[ny, nx]:
                    dist[ny, nx] = d + 1
                    q.append((ny, nx))

        if not reached:
            return None, dist, 0

        # Cluster reached frontier cells by connected components to count clusters
        # and pick the nearest cluster's representative (closest cell in it).
        clusters = self._cluster_frontiers(frontier_mask & (dist >= 0))
        valid = [c for c in clusters if len(c) >= self.frontier_min_cells]
        if not valid:
            # fall back to nearest single frontier cell if no big cluster
            reached.sort(key=lambda r: r[0])
            d, gy, gx = reached[0]
            return (gy, gx), dist, len(clusters)

        best_cell: Optional[Tuple[int, int]] = None
        best_d = None
        for cluster in valid:
            for (cy, cx) in cluster:
                d = dist[cy, cx]
                if d < 0:
                    continue
                if best_d is None or d < best_d:
                    best_d = d
                    best_cell = (cy, cx)
        return best_cell, dist, len(valid)

    @staticmethod
    def _cluster_frontiers(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
        h, w = mask.shape
        seen = np.zeros_like(mask)
        clusters: List[List[Tuple[int, int]]] = []
        ys, xs = np.where(mask)
        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            if seen[y0, x0]:
                continue
            comp: List[Tuple[int, int]] = []
            stack = [(y0, x0)]
            seen[y0, x0] = True
            while stack:
                cy, cx = stack.pop()
                comp.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            clusters.append(comp)
        return clusters

    @staticmethod
    def _coverage(grid: np.ndarray) -> float:
        known = int(np.count_nonzero(grid != UNKNOWN))
        total = grid.size
        return round(known / total, 3) if total else 0.0

    def _empty_plan(self, reason: str) -> Dict[str, Any]:
        return {
            "done": False,
            "reason": reason,
            "goal_xy": None,
            "frontier_cells": 0,
            "reachable_frontier_clusters": 0,
            "free_cells": 0,
            "explored_area_m2": 0.0,
            "coverage": 0.0,
            "grid": None,
        }
