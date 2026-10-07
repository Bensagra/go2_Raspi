# Misiones: instalación, API e integración con otro frontend

El servidor guarda cada misión con video del Go2, video Arducam, video térmico con detección y un mapa LiDAR propio. El dashboard permite **Iniciar misión → Finalizar y guardar → Abrir reproductor**, sin descargar archivos manualmente. Go2, Arducam y térmica comparten una línea de tiempo; el mapa se abre en el visor 3D.

## 1. Conectar las tres partes

```text
Robot Go2 → Raspy (gateway + cámara térmica USB)
                  ├── MQTT ↔ broker ↔ servidor: control y telemetría
                  └── WebSocket → servidor: cámara, térmica y LiDAR
                                      ├── disco: misiones
                                      ├── /ws/live → frontend: datos en vivo
                                      └── API HTTP → frontend: biblioteca, reproductor y mapas
```

El frontend se conecta al servidor, no directamente a la Raspy. El broker MQTT es un proceso aparte: el comando de Python del servidor no lo arranca. La reproducción de misiones guardadas no necesita que el robot, la Raspy o MQTT estén conectados.

### Servidor Windows

Actualizá estos archivos del repositorio y reiniciá el servidor. Desde la raíz, en **CMD**, podés mantener tu comando y agregar una carpeta de almacenamiento:

```bat
.venv\Scripts\python.exe server\server_core.py ^
  --host 0.0.0.0 --port 8000 ^
  --cors-origin "*" ^
  --mqtt-host 192.168.1.88 --mqtt-port 1883 ^
  --edge-media-token edge-media-dev-token ^
  --api-token dev-operator-token:operator:operator_01 ^
  --api-token dev-viewer-token:viewer:viewer_01 ^
  --robot-id go2_01 ^
  --mission-storage-dir "D:\Go2\misiones"
```

Elegí una unidad existente y una carpeta donde el proceso tenga permiso para escribir. El flag es opcional: por defecto usa `server/missions`, relativo a la carpeta desde la que arrancás Python. No hace falta una base de datos ni instalar un ejecutable FFmpeg aparte: se usa PyAV, ya incluido en las dependencias. Las credenciales del ejemplo son las de desarrollo de esta instalación.

En hosting mediante `main:app`, se usa **`DATA_DIR/missions`** automáticamente; ver [HOSTING_TIC.md](HOSTING_TIC.md). Ejecutar **un solo proceso/worker** del servidor: el estado activo de misiones y del robot reside en ese proceso. No usar `--reload` durante una grabación.

### Raspy

Conservá la configuración de captura que ya funciona. Debe seguir enviando las fuentes a:

```text
ws://192.168.1.88:8000/ws/edge-media/go2_01?token=EDGE_MEDIA_TOKEN
```

En el gateway, los flags siguen siendo `--media-ws-url ws://192.168.1.88:8000/ws/edge-media/{robot_id}` y `--media-ws-token ...`, con cámara/LiDAR y la térmica habilitados. No hay otro puerto ni protocolo de subida de misiones. Ver [THREE_LAYER_CONNECTION.md](THREE_LAYER_CONNECTION.md) y [THERMAL_STREAMING.md](THERMAL_STREAMING.md) para los flags de captura térmica.

**Iniciar misión graba las fuentes que llegan; no activa sensores ni mueve el robot.** Si el dashboard deshabilita cámara o LiDAR en el gateway, esa fuente deja de grabarse hasta volver a habilitarla. Verificá los contadores antes de empezar el recorrido.

## 2. Qué queda guardado

Cada misión tiene un ID de 32 caracteres hexadecimales y su propia carpeta:

| Archivo | Contenido |
| --- | --- |
| `mission.json` | Nombre, robot, autor, fechas, duración, estado, estadísticas y archivos disponibles. |
| `camera.mp4` | Video H.264 de la cámara, sin pista de audio. |
| `thermal.mp4` | Video H.264 de la térmica con las regiones detectadas y el estado de presencia dibujados en la imagen. |
| `thermal_detections.jsonl` | Una línea JSON por cuadro térmico: detección, temperaturas y tiempos. |
| `frames.jsonl` | Índice temporal de cuadros de ambos videos. |
| `lidar_map.npz` | Arrays `points` (XYZ float32, metros) y `colors` (RGB uint8). |
| `lidar_map.ply` | Mismo mapa acumulado, con color, para herramientas 3D. |
| `mission.zip` | Copia de los archivos disponibles, creada cuando se solicita la descarga completa. |

Se crean videos y mapa solamente si llegaron datos. Los JSONL pueden quedar vacíos. La térmica guarda el resultado visual y los datos de detección; **no guarda todas las matrices CSV originales**. El detector térmico es una heurística de temperatura/área, no una identificación de personas.

El mapa empieza vacío en cada misión; usa el sistema de coordenadas del LiDAR recibido y la resolución/límite de voxeles del servidor. Por defecto conserva hasta 120.000 voxeles; al llegar al límite se descartan los menos recientes. Es un mapa acumulado, **no una secuencia temporal de nubes ni una grabación de trayectoria**. El visor existente colorea por altura; PLY y NPZ conservan los colores recibidos.

### LiDAR nuevo al iniciar cada misión

`POST /api/robots/{robot}/missions` también reinicia el **mapa en vivo del Core**:
vacía puntos, colores, trayectoria y caché de keyframes; invalida el modelo 3D
automático anterior. Persiste `latest` vacío para no recuperar el lugar anterior
si el servidor se reinicia antes del primer escaneo. Los archivos de misiones
anteriores y las instantáneas con nombre se conservan. El reinicio sólo afecta
al robot de esa misión; un inicio rechazado por misión activa (`409`) no borra
el mapa que está en uso.

Cada mapa nuevo lleva `map_generation`, cuyo valor es el `mission_id`. Los trabajos
de LiDAR, guardado y reconstrucción que estaban en curso no pueden volver a
insertar/publicar el mapa anterior después del cambio. El corte **no depende del
reloj de la Raspi**: un reloj atrasado respecto del Core no descarta escaneos.

El dashboard limpia la vista y espera puntos nuevos; no reaparece el LiDAR viejo
por una descompresión atrasada. Se detiene el replay/grabador local 3D si estaba
activo al cambiar de mapa, conservando sus cuadros para exportación. **Iniciar
misión no enciende el sensor**: habilitá LiDAR si está apagado. Tampoco reinicia
la odometría/SLAM del firmware; la nube nueva conserva las coordenadas que entrega
el robot.

Para desplegar, actualizá Core (`server_core.py` y `mission_routes.py`), el gateway
y el HTML del dashboard, y reiniciá los procesos Python. Los mapas descargados de
misiones antiguas seguirán mostrando sus recorridos originales.

#### Contrato para un frontend propio

Por `/ws/live`, después del cambio se emite:

```json
{"type":"map_reset","robot_id":"go2_01","map_generation":"id-de-la-nueva-mision"}
```

También se emite/cachea un **keyframe vacío válido** del protocolo binario de nube
(`stream:"lidar"`, `mode:"keyframe"`, `fmt:"f32_xyz_zlib"`, `count:0`, `path:[]` y
`map_generation`), cuyo payload es zlib de bytes vacíos. Los keyframes/deltas
siguientes incluyen la misma generación. El keyframe vacío permite limpiar
visores que sólo implementan el protocolo binario existente.

Al cambiar `map_generation`, vaciá puntos, trayectoria, pose y modelo derivado;
invalidá las descompresiones/descargas que empezaron con la generación anterior.
No vuelvas a limpiar si recibís `map_reset` de la generación que ya estás mostrando:
el keyframe puede llegar primero. Recordá las generaciones retiradas para ignorar
paquetes viejos que todavía estuvieran pendientes. Al cambiar robot/conexión,
invalidá también el trabajo de decodificación y el estado local de generación.
Ver `useLidarGeneration()` y `handleLidarFrame()` del dashboard como referencia.

La ruta de inicio, su respuesta `201` y las rutas de mapas de misiones no cambian;
no hace falta enviar un segundo comando de limpieza desde el frontend.

Cerrar o recargar el navegador no detiene la grabación. Se finaliza desde la API/dashboard o al apagar normalmente el servidor. Si se corta la luz o se mata el proceso, al reiniciar aparece como `interrupted`: los fragmentos de video y mapas ya escritos pueden recuperarse, pero no se garantiza el último tramo ni que un MP4 incompleto sea reproducible. No se reanuda automáticamente.

La cola de grabación es limitada y corre en un hilo separado. Si el codificador/disco no alcanza, termina con `error` en vez de acumular memoria indefinidamente. Por defecto exige 256 MB libres (`--mission-min-free-mb`); se comprueba al iniciar y periódicamente. Las misiones **no se borran automáticamente**: hay que disponer de espacio y respaldar/gestionar la carpeta.

## 3. Autenticación y base de URL

```js
const API = "http://192.168.1.88:8000"; // sin barra final
const ROBOT = "go2_01";
const TOKEN = "TU_TOKEN_DEL_USUARIO";

async function api(path, { method = "GET", body } = {}) {
  const response = await fetch(API + path, {
    method,
    headers: {
      Authorization: `Bearer ${TOKEN}`,
      ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
    },
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(`${response.status}: ${error.detail || response.statusText}`);
  }
  return response.json();
}
```

`viewer` puede listar, consultar, reproducir y descargar. `operator` y `admin` también pueden iniciar/finalizar. El modelo actual de roles no restringe las misiones por usuario creador.

Si el servidor está publicado bajo un prefijo, `API` debe incluirlo, por ejemplo `https://host/proyecto/server`. Concatená `API + path`: los `path` devueltos empiezan con `/api/` y son relativos a la aplicación. No uses `new URL(path, API)`, porque perdería ese prefijo. Un frontend HTTPS debe usar un servidor HTTPS/WSS para evitar el bloqueo de contenido mixto.

## 4. API de misiones

Todas las rutas requieren `Authorization: Bearer ...`, excepto el acceso a archivos con ticket válido.

| Método y ruta | Resultado |
| --- | --- |
| `POST /api/robots/{robot}/missions` con `{ "name": "Recorrido" }` | `201`, manifiesto de la nueva misión. Una activa por robot. |
| `GET /api/robots/{robot}/missions?limit=200` | `{ robot_id, missions: [...], total }`, nuevas primero; límite entre 1 y 1000. |
| `GET /api/missions/{id}` | Manifiesto actualizado. |
| `POST /api/missions/{id}/stop` | Espera el cierre de videos y mapa; devuelve manifiesto final. Se puede repetir. |
| `POST /api/missions/{id}/playback` | Manifiesto + enlaces temporales para los videos + ruta del mapa. |
| `GET /api/missions/{id}/map` | Mapa para el visor 3D, descrito más abajo. |
| `POST /api/missions/{id}/download/{filename}` | `{ path, expires_in_s: 600 }` para una descarga opcional. |
| `GET` o `HEAD /api/missions/{id}/files/{filename}?ticket=...` | Archivo; soporta HTTP Range y respuesta `206` para buscar dentro del video. |

Estados: `recording`, `finalizing`, `completed`, `error`, `interrupted`. No ofrecer reproducción/descarga mientras esté `recording` o `finalizing` (`409`). Para `error`/`interrupted`, mostrar que la grabación es parcial y ofrecer los archivos presentes en `artifacts`.

Otros errores: `400` ID/archivo inválido, `401` token/ticket inválido o vencido, `403` rol insuficiente, `404` misión/archivo ausente, `507` problema de almacenamiento. La finalización/ZIP puede tardar; usar un timeout apropiado, por ejemplo 120 segundos. Ante timeout, consultar el estado antes de repetir una operación.

```js
const mission = await api(`/api/robots/${encodeURIComponent(ROBOT)}/missions`, {
  method: "POST", body: { name: "Inspección planta baja" },
});
const id = mission.mission_id;
// Durante la grabación: consultar /api/missions/{id} cada 3 segundos.
const completed = await api(`/api/missions/${id}/stop`, { method: "POST" });
const library = await api(`/api/robots/${encodeURIComponent(ROBOT)}/missions`);
```

Campos útiles del manifiesto: `mission_id`, `name`, `robot_id`, `status`, `started_at`/`ended_at` (Unix, segundos), `duration_s`, `error`, `artifacts`, `missing_streams`, `lidar_points`, `skipped_camera_packets` y `streams.camera|arducam|thermal|lidar.{frames,first_at_s,last_at_s}`. Los tiempos de cada stream son segundos desde el comienzo de la misión. Si el contador no avanza o `duration_s - last_at_s` crece, la fuente no está enviando cuadros recientes.

## 5. Reproducir directamente en otro frontend

No hay que hacer `fetch` del MP4 completo, construir un Blob ni pedir el ZIP. Se obtiene un enlace y se asigna al `src` de un `<video>`; el navegador lee el archivo por HTTP y puede pedir rangos de bytes para avanzar/retroceder. Ver [el elemento video de HTML](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/video).

Ejemplo mínimo con controles nativos independientes:

```html
<video id="camera" controls playsinline preload="metadata" style="width:100%;max-width:640px"></video>
<video id="thermal" controls playsinline preload="metadata" style="width:100%;max-width:640px"></video>
<p id="playbackStatus" role="status"></p>
```

```js
let renewalTimer;
let playbackGeneration = 0;
const videoElements = {
  camera: document.getElementById("camera"),
  thermal: document.getElementById("thermal"),
};

async function openPlayback(id, renew = false) {
  const generation = ++playbackGeneration;
  clearTimeout(renewalTimer);
  const result = await api(`/api/missions/${id}/playback`, { method: "POST" });
  if (generation !== playbackGeneration) return; // ignorar respuesta anterior
  for (const [stream, video] of Object.entries(videoElements)) {
    const track = result.videos[stream];
    const position = renew ? video.currentTime : 0;
    const resume = renew && !video.paused;
    video.pause();
    video.hidden = !track;
    video.onloadedmetadata = null;
    if (!track) { video.removeAttribute("src"); video.load(); continue; }
    video.onloadedmetadata = () => {
      video.currentTime = Math.min(position, Math.max(0, video.duration - 0.001));
      if (resume) video.play().catch(() => {});
    };
    video.src = API + track.path;
    video.load();
  }
  document.getElementById("playbackStatus").textContent = result.mission.error
    ? `Grabación parcial: ${result.mission.error}` : result.mission.name;
  renewalTimer = setTimeout(() => {
    openPlayback(id, true).catch(error => {
      document.getElementById("playbackStatus").textContent = error.message;
    });
  }, (result.expires_in_s - 60) * 1000);
  return result;
}

function closePlayback() {
  playbackGeneration++;
  clearTimeout(renewalTimer);
  for (const video of Object.values(videoElements)) {
    video.pause(); video.removeAttribute("src"); video.load();
  }
}
// await openPlayback(id); manejar errores en la interfaz.
// Al salir de la pantalla/componente: closePlayback().
```

Respuesta de `/playback` (forma del contrato; valores ilustrativos):

```json
{
  "mission": { "mission_id": "...", "name": "Inspección", "duration_s": 120, "status": "completed" },
  "videos": {
    "camera": { "path": "/api/missions/.../files/camera.mp4?inline=true&ticket=...", "offset_s": 0.4, "last_at_s": 119.1 },
    "thermal": { "path": "/api/missions/.../files/thermal.mp4?inline=true&ticket=...", "offset_s": 1.2, "last_at_s": 118.9 }
  },
  "expires_in_s": 600,
  "map_path": "/api/missions/.../map"
}
```

Una fuente ausente se omite de `videos`; `map_path` es `null` cuando no hay mapa. El manifiesto real contiene también el resto de campos documentados arriba.

Los enlaces usan un ticket de **10 minutos**, válido solamente para ese archivo y misión, que permite al elemento video autenticarse sin cabecera Bearer. No llevan el token de la API. Renovar antes del vencimiento o al volver de una pestaña suspendida; el dashboard ya lo hace. Reiniciar el servidor invalida tickets anteriores: pedí otro `/playback`. No persistir los enlaces como si fueran permanentes.

### Una línea de tiempo común

El dashboard implementa el reproductor completo en `frontend/frontend_dashboard.html`, funciones `openMissionPlayer`, `updateMissionPlayer`, `closeMissionPlayer` y `openPlayerMap`. Incluye pausa, búsqueda, velocidades 0,5×/1×/2×, renovación de enlaces y apertura del mapa.

Para integrar sincronización en otro framework:

- Mantener `missionTime` entre 0 y `mission.duration_s`.
- Para cada video, `video.currentTime = missionTime - videos[stream].offset_s`, acotado a su duración.
- Antes de `offset_s`, pausar y mostrar que esa fuente todavía no había comenzado. Después de `last_at_s`, pausar y mostrar fin de grabación.
- Reproducir ambos videos a la misma velocidad; corregir deriva y frenar el reloj común cuando un video necesario esté buscando/cargando. El dashboard usa una tolerancia de 0,4 segundos.
- Al renovar `src`, conservar el tiempo de misión y el estado de pausa. Liberar videos/timers al desmontar el componente.

La sincronización se basa en el **momento de recepción en el servidor**, no en una calibración de relojes entre sensores. Una interrupción de cuadros dentro de un video mantiene la imagen anterior hasta el siguiente cuadro. El reloj común incluye esperas iniciales/finales aunque las fuentes tengan videos de distinta duración.

### Detección térmica

Para visualizarla, no hace falta procesar CSV ni volver a ejecutar el detector: el video térmico ya incluye regiones y estado. Para un panel de eventos propio, leer `thermal_detections.jsonl` mediante el endpoint de archivos y cabecera Bearer. Cada línea tiene `mission_time_s`, `video_time_s`, `source_ts`, `temperature`, `detection` y dimensiones fuente; `detection.person_present` indica el estado calculado. Usar `mission_time_s` para ubicar un evento en el reloj común. No cargar un JSONL enorme entero en memoria si la misión es larga.

## 6. Abrir el mapa sin descargarlo manualmente

`GET /api/missions/{id}/map` devuelve:

```text
metadata: { map_id, robot_id, title, point_count, created_at, is_latest }
point_format: "f32_xyz_zlib"
point_count: cantidad de puntos
points_base64: Base64 de zlib con float32 little-endian [x,y,z,x,y,z,...]
path_format: "f32_xy"
path_point_count: 0
path_base64: ""
```

Ejemplo de decodificación:

```js
const map = await api(`/api/missions/${id}/map`);
const compressed = Uint8Array.from(atob(map.points_base64), c => c.charCodeAt(0));
const inflated = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("deflate"));
const buffer = await new Response(inflated).arrayBuffer();
const view = new DataView(buffer);
const xyz = new Float32Array(buffer.byteLength / 4);
for (let i = 0; i < xyz.length; i++) xyz[i] = view.getFloat32(i * 4, true);
// Crear BufferGeometry/Points en Three.js, o subir xyz a un buffer WebGL.
```

Si el navegador no ofrece `DecompressionStream`, usar una biblioteca zlib. El dashboard reutiliza `decodeFloatPayload` y `uploadSavedMap` del visor existente. A diferencia del video por rangos, el mapa se recibe completo, sujeto al límite de voxeles del servidor.

## 7. Mantener también la vista en vivo

La reproducción grabada usa HTTP; el vivo sigue por el WebSocket existente:

```js
const liveUrl = new URL(API + "/ws/live");
liveUrl.protocol = liveUrl.protocol === "https:" ? "wss:" : "ws:";
liveUrl.searchParams.set("token", TOKEN);
const ws = new WebSocket(liveUrl);
ws.binaryType = "arraybuffer";
ws.onmessage = event => {
  if (typeof event.data === "string") {
    const message = JSON.parse(event.data);
    if (message.type === "mission" && message.robot_id === ROBOT) {
      // Inicio/finalización solicitados: refrescar biblioteca.
    }
    // telemetry, control_status, drive_status, command_ack, etc.
    return;
  }
  const bytes = new Uint8Array(event.data);
  if (bytes.length < 6 || bytes[0] !== 0xA7 || bytes[1] !== 1) return;
  const length = new DataView(event.data).getUint32(2, true);
  if (length > bytes.length - 6) return;
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(6, 6 + length)));
  if (header.robot_id !== ROBOT) return;
  const payload = bytes.subarray(6 + length);
  // header.stream: "video", "thermal", "lidar", "audio".
  // Térmica: JPEG; cámara: consultar header.image_format (H.264/JPEG/WebP).
};
```

El servidor envía datos de distintos robots: filtrar por `robot_id`. La cámara H.264 en vivo requiere el decodificador WebCodecs existente; no asignar un paquete H.264 bruto al `src` de video. El reproductor de misiones recibe MP4 estándar, por lo que no necesita esa lógica. Para vivo térmico y campos de detección ver [THERMAL_STREAMING.md](THERMAL_STREAMING.md).

El evento `mission` no es un contador periódico ni sustituye consultar la API: refrescar cada 3 segundos mientras haya una misión activa para detectar progreso/errores del hilo de grabación. Conectar el WebSocket no inicia una misión.

Los comandos de movimiento conservan su API y ACK existentes; ver [THREE_LAYER_CONNECTION.md](THREE_LAYER_CONNECTION.md). Que se vea video no confirma conexión MQTT para movimiento: revisar `control_status`, `drive_status` y `/api/robots/{robot}/state`.

## 8. Verificación al actualizar

1. Reiniciar el servidor actualizado y recargar el dashboard actualizado.
2. Iniciar una misión, comprobar que avanzan cámara/térmica/LiDAR, finalizarla.
3. Abrir reproductor, reproducir, pausar, buscar un instante y abrir el mapa.
4. Recargar la página y comprobar que sigue en la biblioteca.
5. En otro frontend, usar el mismo servidor/robot/token y las rutas de esta guía.

Pruebas automatizadas del proyecto:

```bash
python -m unittest discover -s tests
```

Cubren MP4 decodificables, detección persistida, aislamiento entre misiones, grabación sin visor, recuperación, roles, tickets y reproducción HTTP con Range/HEAD. Los cambios del repositorio no actualizan automáticamente el servidor Windows: hay que copiar/sincronizar el código y reiniciarlo allí.

## Cámara CSI adicional

Ver [ARDUCAM_STREAMING.md](ARDUCAM_STREAMING.md) para captura, arranque y contrato del stream `arducam`. Las nuevas misiones incluyen `streams.arducam` y, si recibieron cuadros, `arducam.mp4`; playback agrega `videos.arducam`. Los ejemplos anteriores que solo contienen Go2/térmica siguen siendo válidos para misiones sin la cámara adicional.
