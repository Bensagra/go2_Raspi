"""CSV Celsius comprimido para el canal binario de medios existente."""
import io
import zlib

import numpy as np

MAX_DIMENSION = 640
MAX_PIXELS = 320 * 240
MAX_CSV_BYTES = MAX_PIXELS * 20


def validate_frame(frame):
    frame = np.asarray(frame, dtype=np.float32)
    if (frame.ndim != 2 or min(frame.shape) < 2
            or max(frame.shape) > MAX_DIMENSION or frame.size > MAX_PIXELS
            or not np.isfinite(frame).all()):
        raise ValueError("Se esperaba una matriz termica 2D finita de hasta 76800 pixeles")
    return frame


def encode_csv(frame):
    frame = validate_frame(frame)
    text = io.StringIO()
    np.savetxt(text, frame, fmt="%.4f", delimiter=",")
    raw = text.getvalue().encode("ascii")
    if len(raw) > MAX_CSV_BYTES:
        raise ValueError("CSV termico demasiado grande")
    return zlib.compress(raw, 1)


def decode_csv(header, payload):
    if header.get("format") != "csv_zlib" or header.get("unit") != "celsius":
        raise ValueError("Formato termico no soportado")
    width, height = header.get("width"), header.get("height")
    if (type(width) is not int or type(height) is not int
            or not 2 <= width <= MAX_DIMENSION or not 2 <= height <= MAX_DIMENSION
            or width * height > MAX_PIXELS or len(payload) > MAX_CSV_BYTES):
        raise ValueError("Dimensiones o longitud termica invalidas")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(payload, MAX_CSV_BYTES + 1)
    if (len(raw) > MAX_CSV_BYTES or decoder.unconsumed_tail
            or not decoder.eof or decoder.unused_data):
        raise ValueError("CSV comprimido incompleto o demasiado grande")
    # Limit rows/columns before numpy allocates a potentially huge matrix.
    rows = raw.decode("ascii").splitlines()
    if len(rows) != height or any(row.count(",") != width - 1 for row in rows):
        raise ValueError("El CSV no coincide con sus dimensiones")
    frame = np.loadtxt(io.StringIO("\n".join(rows)), delimiter=",", dtype=np.float32, ndmin=2)
    if frame.shape != (height, width):
        raise ValueError("El CSV no coincide con sus dimensiones")
    return validate_frame(frame)
