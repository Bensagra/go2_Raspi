# Arducam B0541: Raspberry → servidor → front

La Arducam se suma a la cámara del Go2 y a la térmica. El gateway la habilita por defecto, envía JPEG por el WebSocket de media existente y el servidor la graba **solo durante las misiones**. El front muestra una vista independiente y permite ampliar, reproducir y descargar `arducam.mp4`.

## Probar en la Raspberry

Actualizar el código **en Raspberry, servidor y front**, y reiniciar gateway/servidor. No alcanza con actualizar solamente la Raspberry: el servidor anterior ignora `stream: "arducam"`.

Se requiere la Pi 5 con la configuración que ya produjo imágenes: overlay `arducam-pivariety-4lane`, cuatro líneas CSI, sensor en CSI1 y captura UYVY 3840×2160. La integración **no modifica** `/boot/firmware/config.txt`, overlays, drivers ni firmware. Restaura los enlaces y formatos temporales cada vez que abre la cámara, incluido después de un reinicio.

Instalar las herramientas si faltan:

```bash
sudo apt install v4l-utils python3-opencv
```

El usuario que ejecuta el gateway necesita acceso a `/dev/media*`, `/dev/v4l-subdev*` y `/dev/video*` (habitualmente grupo `video`). Usar el Python/entorno del gateway que ya tenga sus dependencias y OpenCV con V4L2. Para un entorno **nuevo**, `python3 -m venv --system-site-packages .venv` permite reutilizar OpenCV de apt; no recrear un entorno existente.

Desde la raíz del proyecto, **con el gateway y cualquier visor de cámara detenidos**, hacer primero una captura independiente:

```bash
python -m edge.arducam_camera --output /tmp/arducam-check.jpg
```

Imprime el nodo detectado, resolución, sesión y timestamp; guarda un JPEG y libera la cámara. Sale con error si falla la detección/captura. No necesita escritorio gráfico. El diagnóstico termina después de aproximadamente 45 segundos como máximo, más el cierre del proceso.

Después arrancar el gateway con el comando habitual. La Arducam ya está habilitada; estos flags explicitan el perfil inicial:

```bash
python edge/edge_gateway_service.py \
  --go2-ip IP_DEL_GO2 \
  --mqtt-host HOST_DEL_BROKER \
  --robot-id go2_01 \
  --media-ws-url 'ws://HOST_DEL_SERVIDOR:8000/ws/edge-media/{robot_id}' \
  --media-ws-token TOKEN_EDGE_EXISTENTE \
  --enable-arducam \
  --arducam-fps 5 \
  --arducam-max-width 1280 \
  --arducam-quality 75 \
  --media-max-kbps 6000
```

Reemplazar los marcadores y conservar las demás opciones de MQTT/TLS/robot que ya usás. Para un servidor HTTPS usar su URL `wss://...`, manteniendo cualquier prefijo de ruta.

`--media-max-kbps 6000` da margen inicial a los tres streams; es un límite compartido, no un consumo garantizado. El perfil de red aplicado desde el dashboard puede cambiarlo: el perfil equilibrado fija 5000 kbps y el débil 2200. Si aumentan `media.media_budget_drops.arducam` o faltan imágenes, revisar ese límite y probar `--arducam-max-width 640 --arducam-fps 3 --arducam-quality 60`. El formato JPEG tiene bitrate variable según la escena. La cola descarta imágenes para evitar acumular retraso.

## Comprobar el flujo completo

1. Abrir `frontend/frontend_dashboard.html`, conectarlo al servidor y seleccionar el mismo robot.
2. La sección **Arducam · Raspberry** debe pasar a **En vivo**. Clic/Enter amplía la imagen; si no llegan cuadros durante 3 segundos muestra señal interrumpida.
3. Iniciar una misión y comprobar que aumenta el contador Arducam. La cámara del Go2 y la térmica mantienen sus canales separados.
4. Se puede cerrar el navegador: la grabación continúa en el servidor mientras llegan imágenes.
5. Finalizar y guardar. Abrir el reproductor: Arducam comparte la línea de tiempo de misión con los otros videos.
6. Descargar **Arducam · MP4** o el ZIP completo; el ZIP incluye `arducam.mp4` si llegaron imágenes. Sin imágenes, el manifiesto informa `arducam` en `missing_streams` y no crea un MP4 vacío.

El servidor utiliza su `--mission-storage-dir` habitual (en TIC, `DATA_DIR/missions`). No agrega otro puerto, token ni servicio. Las misiones anteriores sin Arducam siguen siendo reproducibles.

## Opciones y comportamiento

| Flag del gateway | Valor inicial | Función |
|---|---:|---|
| `--enable-arducam` / `--disable-arducam` | Habilitada | Iniciar o excluir la captura CSI |
| `--arducam-media /dev/mediaN` | Autodetección | Seleccionar el grafo si hay varias cámaras |
| `--arducam-fps` | 5 | Máximo de imágenes enviadas por segundo, 1–30 |
| `--arducam-max-width` | 1280 | Ancho del JPEG, 320–3840; conserva proporción |
| `--arducam-quality` | 75 | Calidad JPEG, 25–95 |

La captura del sensor sigue siendo 4K UYVY. El perfil inicial transmite **1280×720**, y esa resolución se guarda en el MP4. Para guardar 4K hay que transmitir con `--arducam-max-width 3840` y ajustar FPS/ancho de banda; el rendimiento 4K no está validado. El FPS configurado es un máximo, no una garantía del sensor o la red.

Un único proceso abre el dispositivo. Se detectan `/dev/mediaN`, el nombre I²C del sensor y `/dev/videoN`; no se asume `video0`. Un bloqueo cooperativo evita dos instancias de este gateway sobre el mismo grafo. Otros programas que no respeten ese bloqueo deben estar detenidos.

La captura funciona independientemente de la conexión al Go2. Mantiene un solo JPEG pendiente, descarta imágenes antiguas y vuelve a preparar/abrir ante errores. Si V4L2 se bloquea, el supervisor termina el proceso tras el timeout (40 s de inicio, 8 s sin cuadros después de conectar) y reintenta a los 3 s. Si el kernel no permite terminarlo, detiene los reintentos para evitar un segundo propietario; revisar el sistema/reiniciar en ese caso. No usa Picamera2/rpicam.

## Contrato para otro front

Mismo protocolo binario A7/v1 y `/ws/live?token=...`. El JPEG ocupa el payload; el encabezado es:

```json
{
  "type": "media",
  "robot_id": "go2_01",
  "stream": "arducam",
  "source": "arducam_b0541",
  "image_format": "jpg",
  "width": 1280,
  "height": 720,
  "source_width": 3840,
  "source_height": 2160,
  "session_id": "identificador-de-captura",
  "seq": 1,
  "ts": 1791370800.0,
  "server_received_ts": 1791370800.1,
  "encoded_bytes": 80000
}
```

Los números son ilustrativos. Renderizar el payload con `createImageBitmap(new Blob([payload], {type: 'image/jpeg'}))`. Usar un canvas y un buzón propios para `arducam`, filtrar `robot_id`, cerrar los bitmaps reemplazados e invalidar decodificaciones pendientes al desconectar/cambiar robot. No mezclarlo con `video` (Go2) ni `thermal`.

El servidor valida JPEG, tamaño real (hasta 3840×2160 y 4 MiB), secuencia y timestamps antes de grabar/enviar. Descarta duplicados por sesión y conserva solo el último cuadro pendiente por robot si el procesamiento se atrasa. No reenvía una imagen cacheada de Arducam al abrir un visor: espera una nueva.

- Telemetría: `media.arducam_enabled`, `media.arducam.{connected,frames,last_frame_ts,error,device}`; el contador es de captura, no prueba recepción en el servidor.
- Eventos: `arducam_camera_error`, `arducam_camera_connected`; errores de validación del servidor en auditoría `arducam_frame_error`.
- Manifiesto de misión: `streams.arducam.{frames,first_at_s,last_at_s}`.
- `frames.jsonl`: `stream: "arducam"`, tiempos de misión/video, timestamp, secuencia y sesión de origen.
- Playback existente: `POST /api/missions/{mission_id}/playback` agrega `videos.arducam` con `path`, `offset_s`, `last_at_s`.
- Descarga existente: `POST /api/missions/{mission_id}/download/arducam.mp4`. Misma autorización y tickets temporales; soporte de rangos HTTP para reproducción.

La Arducam no modifica la calibración/colorización LiDAR ni reemplaza la entrada de percepción del Go2: es una cámara adicional de observación y registro.

## Diagnóstico

Si el kernel no detecta la cámara, revisar el overlay/configuración persistente y conexiones; no reinstalar drivers indiscriminadamente. Para inspeccionar sin capturar:

```bash
v4l2-ctl --list-devices
media-ctl -d /dev/media0 --print-topology
sudo dmesg | grep -Ei 'arducam|pivariety|rp1-cfe|csi|lane'
```

Usar el `mediaN` correspondiente, no necesariamente `media0`. Si el diagnóstico funciona pero no llega al front, revisar URL/token del uplink, eventos `media_uplink_*`, presupuesto de red y que servidor/front también estén actualizados.

Validación local: pruebas con imágenes sintéticas, WebSockets reales, MP4 decodificados, ZIP, API de playback y supervisor con una lectura bloqueada simulada. La prueba física, color/FPS y estabilidad prolongada quedan para la Raspberry.

Referencias de API: [OpenCV VideoCapture](https://docs.opencv.org/3.4.20/d8/dfe/classcv_1_1VideoCapture.html), [flags V4L2 de OpenCV](https://docs.opencv.org/4.13.0/d4/d15/group__videoio__flags__base.html), [controlador RP1 CFE del kernel](https://www.kernel.org/doc/html/v6.13/admin-guide/media/raspberrypi-rp1-cfe.html). La topología y el modo B0541 implementados proceden de la guía de captura suministrada por el usuario.
