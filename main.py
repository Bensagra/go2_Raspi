"""HosTICer / ASGI entrypoint: uvicorn main:app.

The robot server remains in server/server_core.py. Uvicorn owns the port and
process lifecycle; importing this module never starts a second HTTP server.
"""

from server.hosting import create_app

app = create_app()
