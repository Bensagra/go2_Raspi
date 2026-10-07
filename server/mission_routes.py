"""Authenticated mission lifecycle, map viewing and streamed artifact downloads."""
import asyncio
import base64
import contextlib
import zlib

import numpy as np
from fastapi import Depends, Header, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from server.missions import MissionConflict


class MissionStart(BaseModel):
    name: str = Field(default="", max_length=120)


async def mission_call(function, *args):
    try:
        return await asyncio.to_thread(function, *args)
    except MissionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=507, detail=f"Error de almacenamiento: {exc}") from exc


def register_mission_routes(app, runtime):
    def operator(auth):
        if auth["role"] not in {"operator", "admin"}:
            raise HTTPException(status_code=403, detail="operator role required")

    @app.get("/api/robots/{robot_id}/missions")
    async def list_missions(robot_id: str, limit: int = Query(default=200, ge=1, le=1000),
                            auth=Depends(runtime._auth_dependency)):
        items = await mission_call(runtime.missions.list, robot_id)
        return {"robot_id": robot_id, "missions": items[:limit], "total": len(items)}

    @app.post("/api/robots/{robot_id}/missions", status_code=201)
    async def start_mission(robot_id: str, body: MissionStart, auth=Depends(runtime._auth_dependency)):
        operator(auth)
        try:
            runtime._validate_storage_component(robot_id, "robot id")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if runtime.stop_event.is_set():
            raise HTTPException(status_code=503, detail="Server is shutting down")
        async with runtime.mission_start_locks[robot_id]:
            previous_generation = runtime.lidar_generations[robot_id]
            runtime.lidar_resetting.add(robot_id)
            start_task = asyncio.create_task(mission_call(runtime._start_mission_with_fresh_map,
                                                         robot_id, body.name, auth["user_id"]))
            try:
                mission = await asyncio.shield(start_task)
            except asyncio.CancelledError:
                # Cancelling to_thread does not stop its worker. Keep the ingest
                # gate closed until the reset actually finishes.
                with contextlib.suppress(Exception):
                    await start_task
                raise
            finally:
                try:
                    await runtime._finish_mission_map_reset(robot_id, previous_generation)
                finally:
                    runtime.lidar_resetting.discard(robot_id)
        runtime._audit("mission_started", {"mission_id": mission["mission_id"], "robot_id": robot_id})
        await runtime._broadcast({"type": "mission", "robot_id": robot_id, "data": mission})
        return mission

    @app.get("/api/missions/{mission_id}")
    async def get_mission(mission_id: str, auth=Depends(runtime._auth_dependency)):
        return await mission_call(runtime.missions.get, mission_id)

    @app.post("/api/missions/{mission_id}/stop")
    async def stop_mission(mission_id: str, auth=Depends(runtime._auth_dependency)):
        operator(auth)
        mission = await mission_call(runtime.missions.stop, mission_id)
        runtime._audit("mission_stopped", {"mission_id": mission_id, "status": mission["status"]})
        await runtime._broadcast({"type": "mission", "robot_id": mission["robot_id"], "data": mission})
        return mission

    @app.post("/api/missions/{mission_id}/download/{filename}")
    async def download_link(mission_id: str, filename: str, auth=Depends(runtime._auth_dependency)):
        await mission_call(runtime.missions.file, mission_id, filename)
        ticket = runtime.missions.ticket(mission_id, filename)
        return {"path": f"/api/missions/{mission_id}/files/{filename}?ticket={ticket}", "expires_in_s": 600}

    def playback_payload(mission_id):
        mission = runtime.missions.get(mission_id)
        if mission["status"] in {"recording", "finalizing"}:
            raise MissionConflict("Finalizá la misión antes de reproducirla")
        videos = {}
        for stream in ("camera", "arducam", "thermal"):
            filename = f"{stream}.mp4"
            if filename not in mission["artifacts"]:
                continue
            runtime.missions.file(mission_id, filename)
            stats = mission["streams"][stream]
            ticket = runtime.missions.ticket(mission_id, filename)
            videos[stream] = {
                "path": f"/api/missions/{mission_id}/files/{filename}?inline=true&ticket={ticket}",
                "offset_s": stats["first_at_s"] or 0,
                "last_at_s": stats["last_at_s"],
            }
        return {"mission": mission, "videos": videos, "expires_in_s": 600,
                "map_path": f"/api/missions/{mission_id}/map" if "lidar_map.npz" in mission["artifacts"] else None}

    @app.post("/api/missions/{mission_id}/playback")
    async def playback(mission_id: str, auth=Depends(runtime._auth_dependency)):
        return await mission_call(playback_payload, mission_id)

    @app.api_route("/api/missions/{mission_id}/files/{filename}", methods=["GET", "HEAD"])
    async def download_file(mission_id: str, filename: str, ticket: str = Query(default=""),
                            inline: bool = Query(default=False),
                            authorization: str = Header(default="")):
        if authorization:
            await runtime._auth_dependency(authorization)
        elif not runtime.missions.verify_ticket(mission_id, filename, ticket):
            raise HTTPException(status_code=401, detail="Invalid or expired download link")
        path = await mission_call(runtime.missions.file, mission_id, filename)
        media_type = "video/mp4" if filename.endswith(".mp4") else "application/octet-stream"
        return FileResponse(path, media_type=media_type, filename=f"{mission_id}_{filename}",
                            content_disposition_type="inline" if inline and media_type == "video/mp4" else "attachment",
                            headers={"Cache-Control": "private, no-store", "Referrer-Policy": "no-referrer"})

    def map_payload(mission_id):
        path = runtime.missions.file(mission_id, "lidar_map.npz")
        mission = runtime.missions.get(mission_id)
        with np.load(path, allow_pickle=False) as stored:
            points = np.asarray(stored["points"], dtype="<f4")
        return {"metadata": {"map_id": mission_id, "robot_id": mission["robot_id"],
                              "title": mission["name"], "point_count": len(points),
                              "created_at": mission["started_at"], "is_latest": False},
                "point_format": "f32_xyz_zlib", "point_count": len(points),
                "points_base64": base64.b64encode(zlib.compress(points.tobytes())).decode("ascii"),
                "path_format": "f32_xy", "path_point_count": 0, "path_base64": ""}

    @app.get("/api/missions/{mission_id}/map")
    async def mission_map(mission_id: str, auth=Depends(runtime._auth_dependency)):
        return await mission_call(map_payload, mission_id)
