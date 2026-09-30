"""Live, bounded PTT audio. Voice never shares the robot's movement queue."""
import asyncio
import base64
import time
from collections import deque
from fractions import Fraction

from aiortc import AudioStreamTrack
from av import AudioFrame

from robot_media_protocol import (
    TALK_RATE, TALK_MAX_SECONDS, TALK_MAX_PACKET, TALK_IDLE_SECONDS,
    fresh_packet,
)


class TalkTrack(AudioStreamTrack):
    """Paced 20 ms frames; silence when idle; never replay an old backlog."""
    def __init__(self):
        super().__init__()
        self.buffer = bytearray()
        self.active = False
        self.pts = 0
        self.next_at = None

    def feed(self, pcm):
        if self.active:
            self.buffer.extend(pcm)
            # Keep only the last 200 ms if the RTP sender falls behind.
            if len(self.buffer) > 6400:
                del self.buffer[:-6400]

    def clear(self):
        self.active = False
        self.buffer.clear()

    async def recv(self):
        now = time.monotonic()
        self.next_at = max(self.next_at or now, now - 0.02)
        await asyncio.sleep(max(0, self.next_at - now))
        self.next_at += 0.02
        raw = bytes(self.buffer[:640]) if self.active else b""
        del self.buffer[:640]
        frame = AudioFrame(format="s16", layout="mono", samples=320)
        frame.planes[0].update(raw.ljust(640, b"\x00"))
        frame.sample_rate = TALK_RATE
        frame.pts = self.pts
        frame.time_base = Fraction(1, TALK_RATE)
        self.pts += 320
        return frame


class EdgeTalk:
    def __init__(self, gateway):
        self.gateway = gateway
        self.track = None
        self.session = None
        self.closed = deque(maxlen=64)
        self.queue = asyncio.Queue(maxsize=32)
        self.lock = asyncio.Lock()
        self.deadline = self.last_packet = 0.0
        self.sequence = -1

    def attach(self, connection):
        self.track = TalkTrack()
        sender = next((t.sender for t in connection.pc.getTransceivers() if t.kind == "audio"), None)
        if sender is None:
            raise RuntimeError("robot did not negotiate an outgoing audio channel")
        sender.replaceTrack(self.track)

    def status(self, session, status, **extra):
        self.gateway._mqtt_publish("talk/status", {"session_id": session, "status": status, **extra}, qos=1)

    def enqueue(self, message):
        if not isinstance(message, dict) or not fresh_packet(message, time.time()):
            return
        try:
            self.queue.put_nowait(message)
        except asyncio.QueueFull:
            # Fail closed instead of building seconds of delayed speech.
            if self.track:
                self.track.clear()
            self.deadline = 0.0

    async def stop(self, reason="stopped"):
        if self.track:
            self.track.clear()
        async with self.lock:
            await self._stop(reason)

    async def _stop(self, reason):
        session = self.session
        if self.track:
            self.track.clear()
        self.session = None
        if not session:
            return
        self.closed.append(session)
        self.status(session, "stopped", reason=reason)

    async def handle(self, message):
        session = message.get("session_id")
        if not isinstance(session, str) or not 1 <= len(session) <= 64:
            return
        kind = message.get("type")
        async with self.lock:
            if kind == "stop":
                if session == self.session:
                    await self._stop("stopped")
                elif session not in self.closed:
                    self.closed.append(session)
                return
            if not fresh_packet(message, time.time()) or session in self.closed:
                return
            if kind == "start":
                if self.session == session:
                    return  # QoS 1 duplicate must not reset the deadline.
                if self.session:
                    self.status(session, "error", error="speaker busy")
                    return
                if (self.gateway.conn is None or self.track is None or
                        self.gateway.conn.pc.connectionState != "connected"):
                    self.status(session, "error", error="robot audio is disconnected")
                    return
                self.session = session
                try:
                    self.track.clear()
                    self.track.active = True
                    self.sequence = -1
                    self.last_packet = time.monotonic()
                    self.deadline = self.last_packet + TALK_MAX_SECONDS
                    self.status(session, "ready", sample_rate=TALK_RATE, max_seconds=TALK_MAX_SECONDS)
                except Exception as exc:
                    self.status(session, "error", error=str(exc))
                    await self._stop("start_failed")
                return
            if kind != "pcm" or session != self.session:
                return
            if time.monotonic() >= self.deadline:
                await self._stop("time_limit")
                return
            seq = message.get("sequence")
            encoded = message.get("pcm", "")
            if type(seq) is not int or seq <= self.sequence or not isinstance(encoded, str) or len(encoded) > 4268:
                return
            try:
                pcm = base64.b64decode(encoded, validate=True)
            except ValueError:
                return
            if not pcm or len(pcm) % 2 or len(pcm) > TALK_MAX_PACKET:
                return
            self.sequence = seq
            self.last_packet = time.monotonic()
            self.track.feed(pcm)

    async def run(self):
        while True:
            try:
                message = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                await self.handle(message)
            except asyncio.TimeoutError:
                pass
            except Exception as exc:
                self.gateway._publish_event("talk_error", {"error": str(exc)})
                await self.stop("error")
            if self.session and (time.monotonic() >= self.deadline or
                                 time.monotonic() - self.last_packet >= TALK_IDLE_SECONDS):
                await self.stop("timeout")
