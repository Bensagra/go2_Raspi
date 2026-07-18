# Metodologia de conexion 3 capas (recomendada)

Para el arranque cotidiano usa `./go2 server`, `./go2 edge` y
`./go2 dashboard` como explica el [`README`](../README.md). Los comandos largos
de este documento quedan como referencia de todas las opciones disponibles.

Esta implementacion reemplaza la conexion SSH directa como flujo principal.

Arquitectura:

- Capa 1 Edge (Raspi): `edge/edge_gateway_service.py`
- Capa 2 Server Core: `server/server_core.py`
- Capa 3 Frontend operador: `frontend/frontend_dashboard.html`

## Objetivo

Separar canales por tipo de dato:

- Canal operativo:
  - MQTT para telemetria, eventos, heartbeat y ACK de comandos.
  - API/WebSocket del servidor para front y autorizacion de control.
- Canal pesado:
  - Uplink de media independiente Edge -> Server por WebSocket (`/ws/edge-media/{robot_id}`)
  - Video y audio no se mezclan con telemetria critica.
  - **Frames binarios** (sin base64): cabecera JSON compacta + payload crudo
    (`magic | version | header_len(u32 LE) | header | payload`). ~33% mas liviano
    en la red y sin costo de base64 en la Raspi. El audio sigue como JSON (paquetes chicos).

## Flujo

Subida:

1. Go2 -> Edge Raspi
2. Edge publica telemetria/eventos a MQTT
3. Edge envia media pesada al Server por WebSocket dedicado
4. Server distribuye estado/media al Front por WebSocket

Bajada:

1. Front envia intenciones de comando al Server
2. Server autentica, valida ACL y rate limit
3. Server publica comando en MQTT al Edge
4. Edge valida TTL/heartbeat y ejecuta en Go2
5. Edge responde ACK en MQTT

## Seguridad y robustez implementada

- Comandos de alto nivel (`move`, `turn`, `stop`, `enter_mode`)
- Validacion por rol de usuario en server (`viewer/operator/admin`)
- Rate limit por usuario+robot
- Heartbeat server -> edge
- Failsafe local en edge: si se pierde heartbeat y hay movimiento, `StopMove`
- Auditoria en JSONL de comandos y eventos relevantes

## Dependencias

Instala en entorno Python compartido:

```bash
pip install -r requirements/full.txt
```

## Arranque por capas

### 1) Broker MQTT

Ejemplo con Mosquitto local:

```bash
mosquitto -p 1883
```

### 2) Server Core

```bash
python server/server_core.py \
  --host 0.0.0.0 \
  --port 8000 \
  --cors-origin "*" \
  --mqtt-host 127.0.0.1 \
  --mqtt-port 1883 \
  --edge-media-token edge-media-dev-token \
  --api-token dev-operator-token:operator:operator_01 \
  --api-token dev-viewer-token:viewer:viewer_01 \
  --robot-id go2_01
```

### 3) Edge Raspi

```bash
python edge/edge_gateway_service.py \
  --robot-id go2_01 \
  --go2-ip 192.168.123.161 \
  --mqtt-host 192.168.1.10 \
  --mqtt-port 1883 \
  --enable-camera \
  --enable-audio \
  --enable-lidar \
  --subscribe-profile core,lidar,audio \
  --disable-traffic-saving \
  --media-ws-url ws://192.168.1.10:8000/ws/edge-media/{robot_id} \
  --media-ws-token edge-media-dev-token
```

Los valores por defecto están preparados para una red débil:

- Cámara: **H.264 inter-cuadro** (compresión temporal real), 640 px, 30 FPS y techo de bitrate 700 kbps.
  Reemplaza al MJPEG (WebP por cuadro): mismo cuadro, ~7-15x menos ancho de banda, así entran 30-40 FPS
  en ~2 Mbps sin ahogar el control ni el LiDAR. Se decodifica en el navegador con WebCodecs (Chrome/Edge).
  El bitstream H.264 se entrega de forma confiable y en orden (nunca se descartan deltas); si no hay WebCodecs
  el dashboard cae automáticamente a MJPEG (WebP).
- LiDAR: 0.7 actualizaciones/s, hasta 2500 puntos, cuantización de 2 cm y zlib nivel 6.
  El **server ya no rasteriza un JPEG**: acumula la nube en voxels y la envía como
  nube de puntos binaria (`i16 xyz + zlib`). Manda un **keyframe** (mapa completo, cada
  `--lidar-keyframe-interval-s`, 3 s por defecto) y **deltas** entre medio (solo los voxels
  nuevos), así cuando el robot está quieto casi no consume ancho de banda. El dashboard
  la dibuja en **WebGL 3D en vivo** (órbita/zoom/pan, color por altura, robot + trayectoria,
  LOD y mapa acumulado), con grabación/replay de sesiones y panel de métricas + auto-calidad.
- Reconstrucción **coloreada con la cámara**: el edge proyecta cada punto LiDAR sobre el
  cuadro de cámara (modelo fisheye aproximado, **sin calibración**) y le toma el color del
  píxel; manda nube `xyz+rgb` (formato `i16_xyz_rgb_zlib`, con bit de "tiene color"). El
  server acumula el último color por voxel; el dashboard muestra el espacio con color real
  (toggle **Color: cámara/altura**). Como no hay calibración, los extrínsecos (FOV, pitch,
  altura, offset) se ajustan **en vivo** con el comando `set_color` desde el panel
  "⚙ Calibrar color de cámara". Se activa con `--enable-colorization` en el edge o tildando
  "Colorizar" en el dashboard. Flags del edge: `--color-cam-fov-deg`, `--color-cam-pitch-deg`,
  `--color-cam-height-m`, `--color-cam-forward-m`, `--color-max-distance-m`.
- **Modelo 3D sólido (automático en el server):** en segundo plano el server reconstruye una
  **malla sólida iluminada** (superficie + color de cámara) desde el mapa acumulado — en numpy
  puro (caras visibles del shell + suavizado Laplaciano + normales por vértice), sin dependencias
  extra. La guarda en `server/maps/<robot>/mesh/latest.bin` (+ `.json`) y la sirve lista por
  `GET /api/meshes/{robot}/latest`. El dashboard tiene un visor de malla iluminada (panel
  "Modelo 3D sólido"): "Cargar modelo", "Reconstruir ahora" (`POST /api/meshes/{robot}/rebuild`),
  y auto-carga cuando el server avisa `mesh_ready`. Corre cada `--mesh-interval-s` (60 s; 0 lo
  desactiva). Flags: `--mesh-voxel-size`, `--mesh-smooth-iters`, `--mesh-min-voxels`,
  `--mesh-max-vertices`. *Nota:* es un sólido coloreado a la resolución del voxel (no fotogrametría);
  con LiDAR L1 + cámara 640p no calibrada es lo máximo realista, pero queda navegable y reconocible.
- Audio: 16 kHz en el edge y apagado por defecto en el dashboard.
- Uplink de media: techo de 2200 kbps; el video H.264 tiene prioridad (su propio control de bitrate) y los
  streams descartables (LiDAR/audio) ceden ancho de banda al superarse el presupuesto.

Encoder del edge: `--camera-h264-encoder auto` prueba el encoder por hardware del Pi (`h264_v4l2m2m`) y, si la
build de PyAV/ffmpeg no lo trae, cae a `libx264` (`ultrafast`/`zerolatency`), suficiente para 640×480@30 en un Pi 4.
Ajustables: `--camera-format {h264,webp,jpg}`, `--camera-bitrate-kbps`, `--camera-gop` (0 = keyframe cada 2 s).

El dashboard ofrece perfiles `Débil`, `Balanceada` y `Alta calidad`. Para priorizar la respuesta de
los comandos y evitar tener que apagar la cámara, dejar seleccionado `Débil`.
El perfil `Alta calidad` solicita hasta 40 FPS (limitado por los FPS que entregue físicamente el Go2).

### 4) Frontend

Abre en navegador:

- `frontend/frontend_dashboard.html`

Configura:

- API: `http://<IP_SERVIDOR>:8000`
- Token: `dev-operator-token`
- Robot: `go2_01`

## Navegación autónoma + anti-choque + reconocimiento de personas

Sistema de tres capas para que el robot **mapee solo el entorno sin chocar nunca**
y, mientras pasea, **busque personas y capture/reconozca su cara**. Toda la
inteligencia pesada (planeamiento, detección, reconocimiento) vive en el server;
el anti-choque vive en el edge para que sea una garantía dura.

### Capa de seguridad (edge) — "nunca choca / nunca se cae"

`edge/safety_guard.py` es un guard reactivo que corre en la Raspi sobre el LiDAR
**crudo**, antes de cualquier compresión/uplink, así la garantía sobrevive a la
latencia o a una caída de red. Filtra **todo** movimiento (manual *y* autónomo):

- Sectoriza la nube en body frame y, según la dirección del movimiento pedido,
  **veta** (clearance < `stop_distance`) o **ralentiza** (entre `stop` y `slow`)
  la traslación. La rotación en el lugar siempre se permite.
- **Cliff/borde:** detecta escalones/desniveles (obstáculo negativo) en la franja
  frontal y veta el avance hacia el vacío. *Requiere calibrar el piso.*
- Si el scan está viejo (`scan_timeout`), bloquea la traslación (fail-safe) y deja
  rotar para recuperar visión.
- Solo se arma cuando el LiDAR está activo; sin LiDAR no rompe el manejo manual
  (queda "sin LiDAR" en telemetría).
- Emite `safety_intervention` (throttle 0.5 s) y publica el alert `obstacle_front`/
  `cliff_front`. Telemetría: bloque `safety` (sectores, cliff, ground_z, última
  intervención) + bloque `autonomy`.

Flags del edge (todos calibrables en vivo con el comando `set_safety`):
`--safety-stop-distance-m`, `--safety-slow-distance-m`, `--safety-robot-half-width-m`,
`--safety-ground-z-m` (piso), `--safety-cliff-drop-m`, `--safety-max-radius-m`,
`--disable-safety-guard`, `--safety-cliff-disabled`.

### Capa de navegación (server) — exploración por fronteras

`server/exploration.py` arma una grilla de ocupación 2D desde el voxel-map
acumulado + el camino recorrido, busca **fronteras** (libre lindando con
desconocido), elige la más cercana alcanzable por BFS y produce una velocidad
hacia ella (el anti-choque del edge se encarga de no chocar). **Termina cuando no
quedan fronteras** ("mapear todo y parar").

`server/autonomy.py` es la **máquina de estados** (una por robot):
`EXPLORE → APPROACH(persona) → CAPTURE → EXPLORE … → DONE`, con recuperación de
atasco y cooldown para no recapturar a la misma persona. E-stop/stop la cortan.

### Capa de percepción (server, GPU) — personas + caras

`server/perception.py` decodifica la cámara (H.264 con PyAV / webp-jpg con OpenCV),
corre **YOLO** para personas y **insightface** para detección + embedding +
**reconocimiento** de caras, con gating de calidad (tamaño/nitidez/score) para
quedarse con la mejor toma. La galería se persiste en `--faces-dir` (recorte JPG +
embeddings + identidad conocido/desconocido).

**Las dependencias ML son opcionales:** el sistema mapea igual sin ellas; la
percepción se activa al instalar `ultralytics insightface onnxruntime-gpu` en el
server CUDA (ver `GET /api/perception/capabilities`).

**Privacidad:** se guarda biometría. Definí retención con `--face-retention-days`
(auto-borra desconocidos viejos) y purgá con el botón "Borrar todas" o
`DELETE /api/robots/{id}/faces`.

### Comandos nuevos (server → edge)

- `set_autonomy {enabled}` — habilita el loop autónomo en el edge.
- `drive_velocity {vx, vy, wz}` — meta de velocidad continua (la repite el edge a
  `--autonomy-drive-hz`, filtrada por el guard, con TTL local).
- `e_stop` — parada de emergencia: StopMove + apaga autonomía.
- `set_safety {…}` — ajusta umbrales del guard en vivo.

### Dashboard

Panel "🤖 Navegación autónoma": Iniciar/Detener/**PARADA EMERGENCIA**, estado de
la FSM, % cobertura, meta y capturas. Lectura "🛡 Seguridad" (guard armado,
distancias por sector, borde, última intervención). Galería "👤 Caras capturadas"
con miniatura, identidad, etiquetado (clic) y borrado por privacidad.

### Arranque

Edge: agregá `--enable-lidar` (ya recomendado) — el guard se arma solo. Calibrá el
piso con `--safety-ground-z-m` según el montaje del L1.

Server (en la máquina con GPU):

```bash
pip install ultralytics insightface onnxruntime-gpu   # opcional, activa percepción
python server/server_core.py ... --perception-device cuda --faces-dir ./server/faces
```

Desde el dashboard: **Iniciar exploración**. El robot mapea solo evitando
obstáculos; al ver una persona se acerca, captura y reconoce la cara; al cubrir
todo, se detiene.

## API principal

- `POST /api/robots/{robot_id}/commands`
- `GET /api/robots/{robot_id}/state`
- `GET /api/robots/{robot_id}/replay`
- `GET|POST /api/robots/{robot_id}/autonomy` (`{action: start|stop|estop|status}`)
- `GET /api/perception/capabilities`
- `GET /api/robots/{robot_id}/faces` · `GET .../faces/{person_id}/image`
- `POST .../faces/{person_id}` (`{label, known}`) · `DELETE .../faces[/{person_id}]`
- `WS /ws/live?token=...`
- `WS /ws/edge-media/{robot_id}?token=...`

## Formato de comando (server -> edge)

```json
{
  "command_id": "cmd_123",
  "robot_id": "go2_01",
  "type": "move",
  "payload": {
    "linear_x": 0.2,
    "angular_z": 0.1,
    "duration_ms": 800
  },
  "issued_by": "operator_01",
  "ts": 1776592200,
  "ttl_ms": 1200
}
```

## Formato de ACK (edge -> server)

```json
{
  "command_id": "cmd_123",
  "robot_id": "go2_01",
  "status": "executed",
  "reason": "",
  "edge_ts": 1776592201
}
```

## Nota sobre WebRTC

Este MVP ya separa media en un canal dedicado y permite evolucionar a WebRTC/SFU sin tocar el canal operativo MQTT+control.
La migracion natural es reemplazar el uplink WebSocket de media por ingest WHIP/WebRTC en `edge/edge_gateway_service.py` y un media server en la capa Server.
