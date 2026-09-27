"""Visor y detector independiente: solo lee CSV, no necesita senxor ni camara.

Cada CSV contiene una matriz de temperaturas Celsius sin encabezado.
La deteccion es una heuristica de temperatura y area, no reconocimiento de personas.
"""
import argparse
import time
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np


try:
    import winsound  # solo Windows; el aviso sonoro es opcional
except ImportError:
    winsound = None

PALETAS = [
    ("Ironbow", cv2.COLORMAP_INFERNO),
    ("Caliente", cv2.COLORMAP_HOT),
    ("Arcoiris", cv2.COLORMAP_JET),
    ("Turbo", cv2.COLORMAP_TURBO),
    ("Gris", None),
    ("Gris invertido", "inv"),
]

VENTANA = "Camara termica - Thermal-45 (C)"

# --- Deteccion de personas -------------------------------------------------
# La piel expuesta (cara, cuello, manos) suele verse entre ~28 y ~38 C.
# Por encima de 39 C casi siempre es un objeto caliente (taza, portatil, luz),
# asi que ese techo ayuda a no confundirlos con una persona.
TEMP_PIEL_MIN = 29.0        # C  (ajustable en marcha con k / l)
TEMP_PIEL_MAX = 39.0        # C
AREA_MIN_PERSONA = 55       # pixeles (sobre el frame de 160x120) que debe ocupar
FRAMES_PARA_CONFIRMAR = 3   # frames seguidos con blob valido -> "hay persona"
FRAMES_PARA_DESCARTAR = 8   # frames seguidos sin blob      -> "ya no hay persona"
REPETIR_AVISO_SEG = 20      # si sigue presente, recuerda el aviso cada X s


class EntradaCSV:
    """Consume archivos completos una sola vez durante esta ejecucion."""
    def __init__(self, ruta):
        self.ruta = ruta
        self.archivo_unico = ruta.is_file()
        self.procesados = set()
        self.pendientes = deque()

    def leer(self):
        if not self.pendientes:
            candidatos = [self.ruta] if self.archivo_unico else sorted(self.ruta.glob("*.csv"))
            self.pendientes.extend(p for p in candidatos if p not in self.procesados)
        while self.pendientes:
            ruta = self.pendientes.popleft()
            self.procesados.add(ruta)
            try:
                frame = np.loadtxt(ruta, delimiter=",", dtype=np.float32, ndmin=2)
                if frame.size == 0 or min(frame.shape) < 2 or not np.isfinite(frame).all():
                    raise ValueError("se esperaba una matriz 2D finita de temperaturas")
            except (OSError, ValueError) as err:
                print(f"CSV omitido: {ruta}: {err}", flush=True)
                continue
            return frame
        return None


def aplicar_paleta(gris, paleta):
    _, mapa = paleta
    if mapa is None:
        return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
    if mapa == "inv":
        return cv2.cvtColor(255 - gris, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(gris, mapa)


def marca(img, x, y, texto, color):
    cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 16, 2)
    for grosor, col in ((3, (0, 0, 0)), (1, color)):
        cv2.putText(img, texto, (x + 9, y - 9), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, col, grosor, cv2.LINE_AA)


def texto_hud(img, texto, y, color=(255, 255, 255)):
    for grosor, col in ((3, (0, 0, 0)), (1, color)):
        cv2.putText(img, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, col, grosor, cv2.LINE_AA)


def guardar(img_color, celsius):
    sello = time.strftime("%Y%m%d_%H%M%S")
    png = Path(f"termica_{sello}.png")
    csv = Path(f"termica_{sello}.csv")
    cv2.imwrite(str(png), img_color)
    np.savetxt(csv, celsius, fmt="%.2f", delimiter=",")
    print(f"Guardado:\n  {png.resolve()}\n  {csv.resolve()}  (grados C por pixel)")


def pitar():
    """Pitido corto en un hilo aparte para no frenar el video."""
    if winsound is None:
        print("\a", end="", flush=True)  # campana del terminal como respaldo
        return
    threading.Thread(
        target=lambda: winsound.Beep(880, 180) or winsound.Beep(1175, 180),
        daemon=True,
    ).start()


def detectar_personas(celsius, temp_min, temp_max, area_min):
    """Devuelve una lista de recuadros (x, y, w, h, area, t_max) con aspecto
    de persona: zonas contiguas dentro del rango de temperatura de la piel y
    con area suficiente."""
    mascara = ((celsius >= temp_min) & (celsius <= temp_max)).astype(np.uint8)

    # Limpia puntos sueltos y cierra huecos pequenos dentro de la silueta.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

    n, _etiquetas, stats, _cent = cv2.connectedComponentsWithStats(mascara, 8)

    recuadros = []
    for i in range(1, n):  # 0 es el fondo
        x, y, w, h, area = stats[i]
        if area < area_min:
            continue
        region = celsius[y:y + h, x:x + w]
        recuadros.append((x, y, w, h, int(area), float(region.max())))
    return recuadros, mascara


def main():
    parser = argparse.ArgumentParser(description="Detecta zonas termicas desde un CSV o una carpeta.")
    parser.add_argument("entrada", type=Path, nargs="?", default=Path("capturas"))
    parser.add_argument("--sin-ventana", action="store_true", help="Solo mensajes de consola")
    parser.add_argument("--sin-sonido", action="store_true")
    args = parser.parse_args()
    if not args.entrada.exists():
        if args.entrada.suffix.lower() == ".csv":
            parser.error(f"No existe el CSV: {args.entrada}")
        args.entrada.mkdir(parents=True, exist_ok=True)
    entrada = EntradaCSV(args.entrada)
    confirmar = 1 if entrada.archivo_unico else FRAMES_PARA_CONFIRMAR
    print(f"Leyendo: {args.entrada.resolve()}", flush=True)
    print("CSV unico: evaluacion sin confirmacion temporal." if entrada.archivo_unico
          else "Esperando CSV nuevos; tambien se procesan los existentes. Ctrl+C para salir.", flush=True)
    emisividad = None  # Se configura en camara_csv.py, no en el detector.

    idx_paleta = 0
    zoom = 4
    congelado = False
    ultimo = None
    escala_fija = None  # (min, max) para estabilizar los colores
    SUAVIZADOS = [1, 3, 6, 10]   # cuantos frames se promedian
    idx_suave = 1
    historial = deque(maxlen=SUAVIZADOS[idx_suave])

    # Estado de la deteccion de personas
    aviso_sonoro = not args.sin_sonido
    temp_piel_min = TEMP_PIEL_MIN
    persona_presente = False
    cuenta_con = 0
    cuenta_sin = 0
    ultimo_aviso = 0.0

    if not args.sin_ventana:
        cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)

    try:
        while True:
            nuevo_frame = False
            if not congelado:
                frame = entrada.leer()
                if frame is None:
                    if entrada.archivo_unico and (args.sin_ventana or ultimo is None):
                        break
                    if args.sin_ventana:
                        time.sleep(0.05)
                        continue
                else:
                    nuevo_frame = True
                    crudo = np.asarray(frame, dtype=np.float32)
                    if historial and historial[-1].shape != crudo.shape:
                        historial.clear()
                        cuenta_con = cuenta_sin = 0
                        persona_presente = False
                    historial.append(crudo)
                    promedio = np.mean(historial, axis=0).astype(np.float32)
                    if SUAVIZADOS[idx_suave] > 1:
                        promedio = cv2.medianBlur(promedio, 3)
                    ultimo = promedio
            if ultimo is None:
                if not args.sin_ventana and cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                    break
                time.sleep(0.05)
                continue

            celsius = ultimo
            alto, ancho = celsius.shape

            if escala_fija is None:
                t_lo, t_hi = float(celsius.min()), float(celsius.max())
            else:
                t_lo, t_hi = escala_fija
            rango = max(t_hi - t_lo, 0.5)
            gris = np.clip((celsius - t_lo) / rango * 255, 0, 255).astype(np.uint8)

            img = aplicar_paleta(gris, PALETAS[idx_paleta])
            img = cv2.resize(img, (ancho * zoom, alto * zoom),
                             interpolation=cv2.INTER_CUBIC)

            # ----- Deteccion de personas -----
            recuadros, _mascara = detectar_personas(
                celsius, temp_piel_min, TEMP_PIEL_MAX, AREA_MIN_PERSONA)
            hay_persona_ahora = len(recuadros) > 0

            if nuevo_frame:
                if hay_persona_ahora:
                    cuenta_con += 1
                    cuenta_sin = 0
                else:
                    cuenta_sin += 1
                    cuenta_con = 0

                if not persona_presente and cuenta_con >= confirmar:
                    persona_presente = True
                    ultimo_aviso = time.time()
                    t_pico = max(r[5] for r in recuadros)
                    print(f"[{time.strftime('%H:%M:%S')}] PERSONA DETECTADA "
                          f"({len(recuadros)} zona/s, pico {t_pico:.1f} C)")
                    if aviso_sonoro:
                        pitar()
                elif persona_presente and cuenta_sin >= FRAMES_PARA_DESCARTAR:
                    persona_presente = False
                    print(f"[{time.strftime('%H:%M:%S')}] la persona ha salido de cuadro")
                elif (persona_presente and hay_persona_ahora
                      and time.time() - ultimo_aviso >= REPETIR_AVISO_SEG):
                    ultimo_aviso = time.time()
                    print(f"[{time.strftime('%H:%M:%S')}] la persona sigue presente")
                    if aviso_sonoro:
                        pitar()

                if entrada.archivo_unico and not hay_persona_ahora:
                    print("Sin zonas compatibles con el umbral de deteccion.", flush=True)

            # Dibuja los recuadros de las zonas tipo persona
            for x, y, w, h, area, t_max in recuadros:
                p1 = (x * zoom, y * zoom)
                p2 = ((x + w) * zoom, (y + h) * zoom)
                cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
                etiqueta = f"persona {t_max:.1f}C"
                for grosor, col in ((3, (0, 0, 0)), (1, (0, 255, 0))):
                    cv2.putText(img, etiqueta, (p1[0], max(p1[1] - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, grosor,
                                cv2.LINE_AA)

            # ----- Marcas de temperatura -----
            y_max, x_max = np.unravel_index(celsius.argmax(), celsius.shape)
            y_min, x_min = np.unravel_index(celsius.argmin(), celsius.shape)
            centro = float(celsius[alto // 2, ancho // 2])

            marca(img, x_max * zoom, y_max * zoom,
                  f"{celsius.max():.1f}C", (0, 0, 255))
            marca(img, x_min * zoom, y_min * zoom,
                  f"{celsius.min():.1f}C", (255, 200, 0))
            marca(img, ancho // 2 * zoom, alto // 2 * zoom,
                  f"{centro:.1f}C", (255, 255, 255))

            # ----- HUD -----
            hud = f"{PALETAS[idx_paleta][0]} | zoom {zoom}x | suav {SUAVIZADOS[idx_suave]}f"
            hud += f" | piel {temp_piel_min:.0f}-{TEMP_PIEL_MAX:.0f}C"
            hud += f" | aviso {'ON' if aviso_sonoro else 'OFF'}"
            if emisividad is not None:
                hud += f" | emis {emisividad:.2f}"
            if escala_fija is not None:
                hud += f" | escala fija {t_lo:.1f}-{t_hi:.1f}C"
            if congelado:
                hud += " | CONGELADO"
            texto_hud(img, hud, 22)
            texto_hud(img, "c paleta  z zoom  f suavizado  p aviso  k/l umbral  "
                           "g congelar  s guardar  r escala  q salir",
                      alto * zoom - 12)

            if persona_presente:
                aviso = ">>> PERSONA DETECTADA <<<"
                (tw, th), _ = cv2.getTextSize(aviso, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cx = (ancho * zoom - tw) // 2
                cv2.rectangle(img, (cx - 12, 34), (cx + tw + 12, 44 + th),
                              (0, 0, 200), -1)
                cv2.putText(img, aviso, (cx, 40 + th), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 255, 255), 2, cv2.LINE_AA)

            if args.sin_ventana:
                continue
            cv2.imshow(VENTANA, img)
            tecla = cv2.waitKey(0 if entrada.archivo_unico else 30) & 0xFF
            if tecla in (ord("q"), 27):
                break
            elif tecla == ord("c"):
                idx_paleta = (idx_paleta + 1) % len(PALETAS)
            elif tecla == ord("z"):
                zoom = zoom + 1 if zoom < 6 else 2
            elif tecla == ord("g") and not entrada.archivo_unico:
                congelado = not congelado
            elif tecla == ord("f"):
                idx_suave = (idx_suave + 1) % len(SUAVIZADOS)
                historial = deque(historial, maxlen=SUAVIZADOS[idx_suave])
            elif tecla == ord("p"):
                aviso_sonoro = not aviso_sonoro
            elif tecla == ord("k"):
                temp_piel_min = max(20.0, temp_piel_min - 1.0)
            elif tecla == ord("l"):
                temp_piel_min = min(TEMP_PIEL_MAX - 1.0, temp_piel_min + 1.0)
            elif tecla == ord("s"):
                guardar(img, celsius)
            elif tecla == ord("r"):
                escala_fija = None if escala_fija else (
                    float(celsius.min()), float(celsius.max()))
    except KeyboardInterrupt:
        print("\nDeteccion detenida.")
    finally:
        if not args.sin_ventana:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

