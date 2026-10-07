# Identificación de personas (Arducam)

El servidor detecta personas en la imagen de la Arducam, les asigna un número
estable ("Persona 1", "Persona 2"…) y las reconoce cuando vuelven, también otro día.
El dashboard dibuja un recuadro con el número sobre la imagen en vivo y muestra una
galería donde se le puede poner nombre a cada persona.

Código: `server/people.py` (pipeline), `server/server_core.py` (integración y API),
`frontend/frontend_dashboard.html` (recuadros y galería).

## Cómo funciona

Por cada cuadro de la Arducam (hasta 5 por segundo):

1. **YOLOX** detecta los cuerpos.
2. **YuNet** busca la cara en la parte superior de cada cuerpo.
3. **SFace** convierte la cara en un vector de 128 números y lo compara con la galería.
4. Un **seguimiento por cuadros** mantiene el número de cada persona aunque se dé vuelta
   o deje de verse la cara (hasta 2,5 s sin detectarla).

Reglas para no confundir personas:

- Una persona **nueva** solo se registra con una cara de frente, de al menos 40 px,
  vista en **dos cuadros seguidos**. De perfil o muy lejos queda como "Sin identificar".
- Se reconoce a alguien conocido si la similitud es ≥ 0,40 (`--people-match-threshold`).
  Entre 0,30 y 0,40 es dudoso: no se asigna ni se crea a nadie.
- Un mismo número no puede estar en dos recuadros a la vez.
- Cada persona guarda hasta 24 muestras de su cara (ángulos e iluminaciones distintas),
  así la reconoce mejor con el tiempo.

Todo corre con **OpenCV**, que el servidor ya tiene. No hace falta PyTorch, CUDA ni
insightface. En CPU tarda unos 100–130 ms por cuadro.

## Puesta en marcha

Solo el **servidor** cambia. La Raspberry sigue igual; conviene usar la Arducam en
1080p (`--arducam-max-width 1920 --arducam-min-width 1920`) porque con caras más
grandes reconoce mejor.

1. Verificá que el OpenCV del servidor tenga los módulos necesarios:
   ```bash
   python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'FaceDetectorYN'), hasattr(cv2, 'FaceRecognizerSF'))"
   ```
   Tiene que decir `4.x True True`. Si no:
   `pip install --force-reinstall opencv-python-headless==4.13.0.92`.
2. Reiniciá el servidor. La identificación está **activada por defecto**.
3. La primera vez descarga solo los modelos (~75 MB, de
   [opencv_zoo](https://github.com/opencv/opencv_zoo), licencia Apache-2.0) en
   `server/models/` y verifica su checksum. El servidor necesita salida a Internet ese
   primer arranque; si no tiene, copiá los tres `.onnx` a esa carpeta a mano.
4. En el dashboard, la sección **🧑 Personas identificadas (Arducam)** muestra el estado:
   "descargando/cargando modelos…" y después "activa · N fps · M ms por cuadro".

## Opciones del servidor

| Flag | Valor inicial | Función |
|---|---|---|
| `--disable-people-id` | (activada) | Apaga la identificación. `--disable-perception` también la apaga |
| `--people-target` | `cpu` | `opencl` usa la GPU por OpenCL (por ejemplo AMD RX 570). Si no hay OpenCL, sigue en CPU |
| `--people-match-threshold` | `0.40` | Más alto = menos confusiones pero más "Sin identificar" |
| `--people-max-fps` | `5` | Cuadros analizados por segundo como máximo |
| `--people-dir` | `server/people` | Galería: números, nombres, vectores de cara y la mejor foto |
| `--people-models-dir` | `server/models` | Modelos ONNX |

En hosting (`uvicorn main:app`) las carpetas van a `DATA_DIR/people` y `DATA_DIR/models`.

Con la RX 570 se puede probar `--people-target opencl`. Compará los ms por cuadro que
muestra el dashboard contra `cpu` y quedate con el más rápido. No está probado en esa placa.

## API

Mismo token de API que el resto. Renombrar y borrar requieren rol `operator` o `admin`.

| Método y ruta | Uso |
|---|---|
| `GET /api/people/status` | Estado: `idle`, `loading`, `ready` o `error`, con fps y ms por cuadro |
| `GET /api/robots/{robot}/people` | Lista: `person_id`, `number`, `name`, `label`, `first_seen`, `last_seen`, `sightings`, `samples` |
| `GET /api/robots/{robot}/people/{id}/image` | Mejor foto de la cara (JPEG) |
| `POST /api/robots/{robot}/people/{id}` | Ponerle nombre: `{"label": "Juan"}`. Vacío vuelve a "Persona N" |
| `DELETE /api/robots/{robot}/people/{id}` | Borrar una persona |
| `DELETE /api/robots/{robot}/people` | Borrar todas; la numeración vuelve a 1 |

Por `/ws/live` llegan dos mensajes nuevos (JSON):

```json
{"type": "people", "robot_id": "go2_01", "stream": "arducam", "seq": 42, "ts": 1791370800.0,
 "width": 1920, "height": 1080, "processing_ms": 110.8,
 "people": [{"track_id": 3, "person_id": "p0001", "number": 1, "name": "Persona 1",
             "similarity": 0.93, "box": [0.58, 0.05, 0.90, 0.99], "face_box": [0.71, 0.14, 0.82, 0.39]}]}
```

`box` y `face_box` están normalizados (0–1): multiplicar por el ancho y alto del canvas.
Si alguien no está identificado, `person_id`, `number` y `name` vienen en `null`.
Descartar los recuadros con más de ~1,5 s de antigüedad.

```json
{"type": "event", "robot_id": "go2_01", "data": {"event": "person_new", "data": {"person_id": "p0003", "number": 3, "name": "Persona 3"}}}
```

## Privacidad

La galería guarda **datos biométricos** (fotos y vectores de caras). Está fuera de Git
(`.gitignore`). Usá "Borrar todas" o los endpoints `DELETE` cuando corresponda, e informá
a las personas que el robot las registra. Los borrados quedan en el registro de auditoría.
