"""Lee la camara SenXor y publica un CSV completo por captura, sin deteccion."""
import argparse
import os
from pathlib import Path
import time
import uuid

import numpy as np


def abrir(puerto=None):
    # Importacion diferida: el servidor y las pruebas no necesitan el SDK USB.
    from senxor import connect, list_senxor

    dispositivos = list_senxor("serial")
    if puerto:
        if os.path.exists(puerto):
            puerto = os.path.realpath(puerto)
        dispositivos = [d for d in dispositivos if puerto.upper() in str(d).upper()]
    if not dispositivos:
        raise RuntimeError("No se encontro la camara. Comprueba puerto, cable USB y permisos.")
    return connect(dispositivos[0])


def publicar_csv(carpeta, frame, identificador):
    """El detector solo ve el .csv cuando la escritura ya termino."""
    destino = carpeta / f"termica_{time.time_ns():020d}_{identificador}.csv"
    temporal = destino.with_suffix(".tmp")
    try:
        np.savetxt(temporal, frame, fmt="%.4f", delimiter=",")
        os.replace(temporal, destino)
    finally:
        temporal.unlink(missing_ok=True)
    return destino


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("puerto", nargs="?", help="Ej.: COM5 o /dev/ttyACM0")
    parser.add_argument("--salida", type=Path, default=Path("capturas"))
    parser.add_argument("--intervalo", type=float, default=0.0,
                        help="Segundos minimos entre CSV; 0 guarda cada frame")
    parser.add_argument("--emisividad", type=float)
    args = parser.parse_args()
    if not np.isfinite(args.intervalo) or args.intervalo < 0:
        parser.error("--intervalo debe ser finito y >= 0")
    if args.emisividad is not None and not 0.01 <= args.emisividad <= 1:
        parser.error("--emisividad debe estar entre 0.01 y 1")
    args.salida.mkdir(parents=True, exist_ok=True)
    dev = abrir(args.puerto)
    identificador = uuid.uuid4().hex
    cantidad = 0
    ultima_captura = None
    ultimo_mensaje = 0.0
    try:
        print(f"Conectada: {dev.name}")
        print(f"CSV en: {args.salida.resolve()} | Ctrl+C para salir", flush=True)
        if args.emisividad is not None:
            dev.set_emissivity(args.emisividad)
        dev.start_stream()
        while True:
            _, frame = dev.read()
            if frame is None:
                continue
            ahora = time.monotonic()
            if ultima_captura is not None and ahora - ultima_captura < args.intervalo:
                continue
            celsius = np.asarray(frame, dtype=np.float32)
            if celsius.ndim != 2 or celsius.size == 0 or not np.isfinite(celsius).all():
                print("Frame invalido: omitido", flush=True)
                continue
            destino = publicar_csv(args.salida, celsius, identificador)
            ultima_captura = ahora
            cantidad += 1
            if ahora - ultimo_mensaje >= 2:
                print(f"{cantidad} capturas guardadas. Ultima: {destino.name}", flush=True)
                ultimo_mensaje = ahora
    except KeyboardInterrupt:
        print("\nCaptura detenida.")
    finally:
        try:
            dev.stop_stream()
        except Exception:
            pass
        dev.close()


if __name__ == "__main__":
    main()

