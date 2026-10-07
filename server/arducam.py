"""Bound and validate independent Arducam JPEGs before recording/broadcasting."""
import math

import cv2
import numpy as np


def jpeg_size(data):
    if not 4 <= len(data) <= 4 * 1024 * 1024 or not data.startswith(b'\xff\xd8') or not data.endswith(b'\xff\xd9'):
        raise ValueError('Invalid or oversized Arducam JPEG')
    pos = 2
    while pos < len(data):
        if data[pos] != 255:
            break
        while pos < len(data) and data[pos] == 255:
            pos += 1
        if pos >= len(data):
            break
        marker = data[pos]
        pos += 1
        if marker in (0xda, 0xd9):
            break
        if marker == 0x01 or 0xd0 <= marker <= 0xd7:
            continue
        length = int.from_bytes(data[pos:pos + 2], 'big')
        if length < 2 or pos + length > len(data):
            break
        if marker in (0xc0, 0xc1, 0xc2):
            if length < 8:
                break
            height = int.from_bytes(data[pos + 3:pos + 5], 'big')
            width = int.from_bytes(data[pos + 5:pos + 7], 'big')
            if not 1 <= width <= 3840 or not 1 <= height <= 2160:
                raise ValueError('Arducam JPEG dimensions exceed capture bounds')
            return width, height
        pos += length
    raise ValueError('Arducam JPEG has no valid size')


class ArducamProcessor:
    def __init__(self):
        self.session = None
        self.seq = 0

    def process(self, header, payload):
        if header.get('image_format') not in {'jpg', 'jpeg'}:
            raise ValueError('Arducam requires JPEG')
        width, height = jpeg_size(payload)
        if (header.get('width'), header.get('height')) != (width, height):
            raise ValueError('Arducam JPEG/header dimensions differ')
        ts, seq, session = header.get('ts'), header.get('seq'), header.get('session_id')
        if isinstance(ts, bool) or not isinstance(ts, (float, int)) or not math.isfinite(ts):
            raise ValueError('Invalid Arducam timestamp')
        if type(seq) is not int or seq < 1 or not isinstance(session, str) or not 1 <= len(session) <= 128:
            raise ValueError('Invalid Arducam sequence/session')
        if session == self.session and seq <= self.seq:
            raise ValueError('Duplicate/out-of-order Arducam frame')
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (height, width):
            raise ValueError('Invalid Arducam JPEG image')
        self.session, self.seq = session, seq
        # Whitelist metadata: edge identity always comes from the authenticated URL.
        return {'type': 'media', 'stream': 'arducam', 'source': 'arducam_b0541',
                'image_format': 'jpg', 'width': width, 'height': height,
                'source_width': 3840, 'source_height': 2160,
                'ts': ts, 'seq': seq, 'session_id': session, 'encoded_bytes': len(payload)}
