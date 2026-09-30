# Cámara térmica en vivo

El gateway de la Raspy lee la SenXor USB continuamente, aunque el Go2 esté desconectado.
Cada captura enviada contiene la matriz **CSV en grados Celsius**, sin encabezado,
comprimida con zlib. Viaja por el mismo `/ws/edge-media/{robot_id}` que ya usan cámara,
LiDAR y audio. No se escriben miles de CSV al disco: se generan y consumen en memoria.

El servidor descomprime el CSV, ejecuta `detectar_personas()` de
`termica/detector_csv.py`, genera el mapa de calor con recuadros y lo envía como cuadros
JPEG binarios por **el mismo `/ws/live` del frontend**, con `stream: "thermal"`.
Es video por secuencia de imágenes (tipo MJPEG sobre WebSocket), independiente del
video H.264 del Go2. No requiere otro puerto, otra conexión ni WebCodecs.

## Orientación de la cámara

La térmica se gira **270° en sentido horario (90° a la izquierda) por defecto**
en el servidor. Esto agrega 180° al giro anterior de 90° para corregir la imagen
que quedaba patas arriba. La matriz de temperaturas se gira antes de detectar y
dibujar: imagen, recuadros y coordenadas quedan alineados, y los textos se leen
derechos. Se aplica a la vista normal, ampliada, otros frontends y las nuevas
grabaciones de misiones. No modifica grabaciones anteriores ni el CSV crudo de
la Raspy; no hace falta agregar una rotación CSS en el frontend.

Actualizá el servidor y reinicialo con su comando habitual para aplicar el cambio.
Si el montaje necesita otro ángulo, configurá el **Server Core**:

```bash
python server/server_core.py --thermal-rotation-deg 270
```

Agregá ese argumento al resto de los que ya usás. Los valores son `0`, `90`,
`180` o `270` (predeterminado), siempre en sentido horario; `270` equivale a 90°
a la izquierda. En hosting con `uvicorn main:app`, agregalo a `SERVER_ARGS`:

```text
SERVER_ARGS=["--thermal-rotation-deg","270"]
```

Conservá los demás argumentos que ya tengas en ese array. El ángulo se aplica
a todos los robots atendidos por ese Core. Si tenías `--thermal-rotation-deg 90`
explícito, cambialo a `270` o retiralo para usar el nuevo valor predeterminado.

## Puesta en marcha

1. Subir la versión actualizada del repositorio tanto al server como a la Raspy,
   incluyendo **`termica/`**, `edge/thermal_camera.py` y `server/thermal.py`.
2. En la Raspy, dentro del entorno Python que corre el gateway:

   ```bash
   python -m pip install -r requirements_3layer.txt
   ```

   La dependencia USB agregada es `pysenxor-lite>=3.1.2,<4`. El SDK usa
   `from senxor import connect, list_senxor`; no instalar un paquete de nombre
   parecido en su lugar. [SDK oficial](https://github.com/MeridianInnovation/pysenxor-lite).
   En Linux, el usuario del servicio necesita acceso al puerto serie (grupo `dialout`).
3. Mantener el comando actual de `edge/edge_gateway_service.py`, incluyendo su
   `--media-ws-url` y `--media-ws-token`. La térmica **está activada por defecto**.
   Opcionalmente agregar:

   ```text
   --thermal-port /dev/ttyACM0 --thermal-fps 8 --thermal-emissivity 0.95
   ```

   Si no se indica puerto, se busca la primera SenXor. Para una instalación fija,
   se puede usar una ruta estable `/dev/serial/by-id/...` si el dispositivo la ofrece.
   La emisividad se conserva como está configurada en el dispositivo cuando no se
   pasa el argumento. `--disable-thermal` desactiva esta lectura.
4. Reiniciar el servidor con su comando habitual (`uvicorn main:app` en hosting).
   `requirements.txt` ya incluye NumPy y OpenCV headless; **no requiere el SDK USB**.
   La detección térmica se ejecuta cuando llegan CSV, incluso si la percepción
   YOLO/rostros está desactivada.
5. Abrir la versión actualizada de `frontend/frontend_dashboard.html` y conectar
   como siempre. El panel **Cámara térmica** aparece debajo del video RGB.
   Muestra imagen, FPS, mínima/máxima/centro en °C y posible presencia humana.
   Clic o Enter amplía la imagen. No hay que agregar nada más a este dashboard.

No ejecutar simultáneamente `termica/camara_csv.py` y el gateway contra la misma
cámara USB. Los dos scripts originales siguen disponibles para captura CSV y
análisis local por separado.

## Contrato para otro frontend

Usar el WebSocket autenticado existente (`/ws/live?token=...`) con
`binaryType = "arraybuffer"`. Se mantiene el contenedor actual:

| Bytes | Contenido |
| --- | --- |
| 0 | Magic `0xA7` |
| 1 | Versión `1` |
| 2–5 | Longitud del header JSON, uint32 little-endian |
| 6… | Header JSON UTF-8 |
| Resto | JPEG cuando `stream === "thermal"` |

El header incluye:

```json
{
  "type": "media",
  "stream": "thermal",
  "robot_id": "go2_01",
  "image_format": "jpeg",
  "width": 360,
  "height": 480,
  "source_width": 120,
  "source_height": 160,
  "rotation_deg": 270,
  "ts": 1790000000.0,
  "server_received_ts": 1790000000.1,
  "session_id": "identificador-de-captura",
  "seq": 42,
  "temperature": {"min_c": 22.0, "max_c": 34.0, "center_c": 27.5},
  "detection": {
    "person_present": true,
    "method": "temperature_area_heuristic",
    "temp_min_c": 27.0,
    "temp_max_c": 42.0,
    "area_min_pixels": 55,
    "regions": [{"x": 30, "y": 75, "width": 35, "height": 25, "area": 871, "max_c": 34.0}]
  }
}
```

Ejemplo ilustrativo para un sensor de 160×120 girado 270°. `source_width` y
`source_height`, y las coordenadas de los recuadros, corresponden a la matriz
**ya girada**, no al CSV crudo ni al JPEG ampliado. `rotation_deg` informa el giro
ya aplicado: el frontend no debe volver a rotar. Las temperaturas mostradas corresponden a la matriz suavizada usada por
el detector. `ts` es la hora de captura en la Raspy; `server_received_ts` es la hora
de recepción en el servidor, ambas en segundos Unix.

En un frontend que ya usa los helpers del dashboard, agregar un canvas independiente,
un renderer en `state.mediaRender.thermal`, su caso en `streamCanvas()` y esta rama
en `handleBinaryFrame()` **después de filtrar `robot_id`**:

```js
if (header.stream === "thermal") {
  renderMediaImageBytes(payload, header.image_format, "thermal", header);
  return;
}
```

`payload` es un `Uint8Array` con el JPEG, no CSV ni base64. El renderer crea un
`Blob` de tipo `image/jpeg`, lo decodifica con `createImageBitmap()` y dibuja en el
canvas. El código completo ya está integrado en el dashboard, incluyendo la cola
de un solo cuadro, liberación de bitmaps, estadísticas y estados de desconexión.
Mostrar los metadatos junto con el cuadro decodificado. Si pasan 3 segundos sin un
cuadro nuevo, ocultar la detección y marcar señal interrumpida; también invalidar
los cuadros pendientes al cambiar de robot o cerrar la conexión. El servidor no
reproduce capturas térmicas antiguas a un frontend recién conectado.

## Comportamiento y ajustes

- Lectura USB en un hilo independiente, reintento cada 3 s si falla y reconexión si
  pasan 5 s sin capturas. El SDK se importa sólo en la Raspy y sólo al abrir la cámara.
- Se lee continuamente y se limita el envío a `--thermal-fps` (8 por defecto,
  rango 1–30). Ante congestión se conserva el cuadro más reciente. No es un archivo
  histórico de todas las capturas; se prioriza la vista en vivo.
- Comparte `--media-max-kbps` con los otros medios. Si hay muchos descartes en
  `telemetry.media.uplink_budget_drops.thermal_csv`, bajar `--thermal-fps` o subir
  el presupuesto según la capacidad real del enlace.
- Estado de cámara en `telemetry.media.thermal`: `connected`, `frames`,
  `last_frame_ts`, `error`. Eventos MQTT `thermal_camera_error` y
  `thermal_camera_connected` para diagnóstico.
- El procesamiento CSV y JPEG corre fuera del bucle asíncrono del servidor. La
  memoria pendiente está limitada a un cuadro por robot y un cuadro por visor.
- Detección: promedio de 3 cuadros, filtro mediana, umbral 27–42 °C, área mínima
  55 píxeles; confirma tras 3 cuadros procesados con zona válida y descarta tras
  8 sin zona. Un cambio de sesión, tamaño o una pausa de más de 3 s reinicia la
  confirmación. Parámetros reutilizados de `termica/detector_csv.py`.
- Es una **heurística térmica**, no reconocimiento de personas. El área mínima
  está pensada para 160×120; otra resolución requiere calibración. No dispara
  movimiento ni saludos del robot.

## Verificación

```bash
python -m unittest discover -s tests -v
```

Incluye CSV inválido/truncado, límite de descompresión, precisión de temperaturas,
confirmación y descarte, recuperación USB simulada, aislamiento entre robots,
descarte de cola y CSV → JPEG por los WebSockets reales del servidor de hosting.
La cámara física debe verificarse en la Raspy con su cable, permisos y firmware.
