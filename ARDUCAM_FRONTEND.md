# Arducam en el frontend: guía de conexión

Cómo mostrar en vivo la Arducam (cámara CSI de la Raspberry) en un frontend propio.
El servidor ya está implementado; esta guía describe **solo el lado del cliente**.
Implementación de referencia: `frontend/frontend_dashboard.html`
(`decodeMediaFrame`, `handleBinaryFrame`, `processMediaFrames`, `updateArducamStatus`).

## 1. Recorrido de los datos

```
Arducam ──► Raspberry (edge/edge_gateway_service.py o edge/arducam_only.py)
              │  JPEG por WebSocket binario: /ws/edge-media/{robot_id}
              ▼
          Server Core (server/server_core.py)  ── valida el JPEG y lo reenvía
              │  mismo JPEG por WebSocket binario: /ws/live
              ▼
          Frontend  ── decodifica el frame A7 y dibuja el JPEG en un <canvas>
```

- El frontend **solo** necesita `/ws/live`. No usa MQTT ni habla con la Raspberry.
- La Arducam **no depende del Go2**: se ve aunque el robot esté apagado.
- Es una secuencia de JPEG (tipo MJPEG), no H.264. No se necesita WebCodecs.
- No hay que suscribirse a nada: el servidor envía la media de **todos** los robots
  a cada cliente conectado y el cliente filtra por `robot_id`.

## 2. Conexión

| Dato | Valor |
|---|---|
| URL | `ws://HOST:8000/ws/live?token=TOKEN` (`wss://` si la API es HTTPS) |
| Token | Token de API de rol `viewer`, `operator` o `admin`. Desarrollo: `dev-operator-token` |
| Robot | `go2_01` por defecto |
| `binaryType` | **`"arraybuffer"`** (obligatorio para decodificar los frames) |

Cómo armar la URL a partir de la base de la API, respetando un prefijo de ruta si existe
(por ejemplo `https://dominio/app`):

```js
function liveUrl(apiBase, token) {
  const u = new URL(apiBase);
  u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
  u.pathname = u.pathname.replace(/\/$/, "") + "/ws/live";
  u.search = "?token=" + encodeURIComponent(token);
  return u.toString();
}
```

- Si el token es inválido, el servidor cierra con código **4401**.
- El primer mensaje es de texto: `{"type":"hello","user":{...},"known_robots":["go2_01"]}`.
- El socket mezcla **mensajes de texto (JSON)** y **mensajes binarios**:
  `typeof event.data === "string"` es JSON; un `ArrayBuffer` es un frame de media.

## 3. Formato de los frames binarios (A7/v1)

```
byte 0       0xA7            (magic)
byte 1       0x01            (versión)
bytes 2..5   uint32 little-endian = largo N del header
bytes 6..6+N header JSON en UTF-8
resto        payload (para la Arducam: el archivo JPEG completo)
```

```js
function decodeMediaFrame(buffer) {
  if (buffer.byteLength < 6) throw new Error("frame demasiado corto");
  const view = new DataView(buffer);
  if (view.getUint8(0) !== 0xA7 || view.getUint8(1) !== 1) throw new Error("frame inválido");
  const headerLen = view.getUint32(2, true);
  if (6 + headerLen > buffer.byteLength) throw new Error("header inválido");
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 6, headerLen)));
  const payload = new Uint8Array(buffer, 6 + headerLen);
  return { header, payload };
}
```

### Header de la Arducam

Por el mismo socket llegan otros streams (`video` del Go2, `thermal`, `lidar`, `audio`).
Un frame es de la Arducam cuando **`header.stream === "arducam"`**.

```json
{
  "type": "media",
  "stream": "arducam",
  "robot_id": "go2_01",
  "source": "arducam_b0541",
  "image_format": "jpg",
  "width": 1280,
  "height": 720,
  "source_width": 3840,
  "source_height": 2160,
  "ts": 1791370800.0,
  "server_received_ts": 1791370800.1,
  "seq": 42,
  "session_id": "a3f1…",
  "encoded_bytes": 80000
}
```

| Campo | Uso |
|---|---|
| `robot_id` | Filtrar: ignorar frames de otros robots |
| `width`, `height` | Tamaño real del JPEG. **Varía en vivo**: la Raspberry achica la imagen si falta ancho de banda (mínimo 480 de ancho). No fijar el tamaño del canvas a 1280×720 |
| `ts` | Momento de captura en la Raspberry (segundos Unix) |
| `server_received_ts` | Momento en que llegó al servidor |
| `seq`, `session_id` | Contador por sesión de captura. `seq` vuelve a 1 cuando cambia `session_id` (reinicio de la cámara) |

## 4. Mostrar la imagen

Reglas, en este orden de importancia:

1. **Buzón de un solo cuadro**: guardar solo el último frame pendiente y decodificar
   uno por vez. Si llegan varios mientras se decodifica, se descartan los viejos.
   Nunca hacer una cola, porque acumula retraso.
2. **Canvas propio** para la Arducam, separado del video del Go2 y de la térmica.
3. **Cerrar** los `ImageBitmap` reemplazados (`bitmap.close()`) para no perder memoria.
4. **Generación**: al desconectar o cambiar de robot, incrementar un contador y descartar
   cualquier decodificación que termine con el contador viejo.
5. **Frescura**: descartar cuadros que tarden ≥ 3 s en decodificarse.

Ejemplo mínimo completo (sin dependencias):

```html
<canvas id="arducam" style="width:100%;aspect-ratio:16/9;background:#000"></canvas>
<span id="arducamStatus">Sin conexión</span>
<script>
const ROBOT_ID = "go2_01";
const canvas = document.getElementById("arducam");
const statusEl = document.getElementById("arducamStatus");
const render = { busy: false, pending: null, bitmap: null, generation: 0 };
const frameTimes = [];
let lastFrameAt = 0;
let ws = null;

function connect(apiBase, token) {
  render.generation++;
  ws = new WebSocket(liveUrl(apiBase, token));   // liveUrl: sección 2
  ws.binaryType = "arraybuffer";
  ws.onmessage = ev => {
    if (typeof ev.data === "string") return;      // JSON: telemetría, eventos, etc.
    let frame;
    try { frame = decodeMediaFrame(ev.data); } catch { return; }   // sección 3
    const { header, payload } = frame;
    if (header.stream !== "arducam") return;
    if (header.robot_id && header.robot_id !== ROBOT_ID) return;
    render.pending = { payload, receivedAt: Date.now() };
    drain();
  };
  ws.onclose = ev => {
    render.generation++;
    statusEl.textContent = ev.code === 4401 ? "Token inválido" : "Sin conexión";
    // Reconectar con espera creciente si corresponde.
  };
}

async function drain() {
  if (render.busy) return;
  render.busy = true;
  try {
    while (render.pending) {
      const { payload, receivedAt } = render.pending;
      render.pending = null;
      const generation = render.generation;
      const bitmap = await createImageBitmap(new Blob([payload], { type: "image/jpeg" }));
      if (generation !== render.generation || Date.now() - receivedAt >= 3000) {
        bitmap.close();
        continue;
      }
      if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
      }
      canvas.getContext("2d").drawImage(bitmap, 0, 0);
      render.bitmap?.close();
      render.bitmap = bitmap;
      lastFrameAt = Date.now();
      frameTimes.push(lastFrameAt);
    }
  } catch {
    // JPEG corrupto: ignorar ese cuadro.
  } finally {
    render.busy = false;
  }
}

// Estado en vivo: "señal interrumpida" si pasan 3 s sin cuadros.
setInterval(() => {
  const now = Date.now();
  while (frameTimes.length && frameTimes[0] < now - 1000) frameTimes.shift();
  if (!ws || ws.readyState !== WebSocket.OPEN) statusEl.textContent = "Sin conexión";
  else if (!lastFrameAt) statusEl.textContent = "Esperando cámara";
  else if (now - lastFrameAt >= 3000) statusEl.textContent = "Señal interrumpida";
  else statusEl.textContent = `En vivo · ${frameTimes.length} fps`;
}, 500);
</script>
```

Si `createImageBitmap` no existe (navegadores viejos), usar un `Image` con
`URL.createObjectURL(blob)` y revocar la URL después; ver `decodeMediaBlob` en el dashboard.

## 5. Comportamientos del servidor a tener en cuenta

- **No hay cuadro inicial**: al conectarse, el servidor **no** reenvía una imagen vieja de
  la Arducam (sí lo hace con el video del Go2). La pantalla queda en "Esperando cámara"
  hasta que llega el próximo cuadro nuevo; esto es normal.
- El servidor descarta cuadros duplicados, desordenados o con más de 3 s de antigüedad,
  y si se atrasa se queda solo con el último. Del lado del cliente no hace falta reordenar.
- Los FPS los decide la Raspberry (`--arducam-fps`, 5 por defecto). El frontend no puede
  cambiarlos. El tamaño de cada imagen se ajusta solo al ancho de banda disponible.
- Si se cae `/ws/live`, reconectar con espera creciente (por ejemplo 1 s → 2 s → 5 s,
  máximo 10 s) e incrementar `render.generation`.

## 6. Estado de la cámara por telemetría (opcional)

Por el mismo socket llegan mensajes de texto
`{"type":"telemetry","robot_id":"go2_01","data":{...}}`. Dentro de `data.media`:

| Campo | Significado |
|---|---|
| `arducam_enabled` | La Raspberry arrancó con la Arducam habilitada |
| `arducam.connected` | La captura está entregando cuadros |
| `arducam.frames` | Cuadros capturados (en la Raspberry; no prueba que lleguen al servidor) |
| `arducam.error` | Último error de captura (texto), vacío si está bien |
| `arducam.device` | Nodo `/dev/videoN` en uso |
| `uplink_budget_drops.arducam` | Cuadros descartados por falta de ancho de banda |

**La telemetría viaja por MQTT**, separada de la imagen. Si la Raspberry no llega al
broker MQTT, la imagen de la Arducam puede verse igual pero no habrá telemetría.
Para saber si la cámara funciona, confiar en los frames recibidos, no en la telemetría.

## 7. Probar sin cámara real

Con el servidor corriendo, este script de Python simula la Raspberry y manda cuadros
de prueba a `/ws/edge-media` (token del edge, **no** el de la API):

```python
import asyncio, json, time, uuid
import cv2, numpy as np, websockets

async def main(host="127.0.0.1:8000", robot="go2_01", token="edge-media-dev-token"):
    url = f"ws://{host}/ws/edge-media/{robot}?token={token}"
    session = uuid.uuid4().hex
    async with websockets.connect(url) as ws:
        for seq in range(1, 1000):
            img = np.zeros((720, 1280, 3), np.uint8)
            cv2.putText(img, f"PRUEBA {seq}", (80, 380), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 8)
            jpg = cv2.imencode(".jpg", img)[1].tobytes()
            header = json.dumps({"stream": "arducam", "image_format": "jpg", "width": 1280, "height": 720,
                                 "session_id": session, "seq": seq, "ts": time.time()}).encode()
            await ws.send(bytes([0xA7, 1]) + len(header).to_bytes(4, "little") + header + jpg)
            await asyncio.sleep(0.2)   # 5 fps

asyncio.run(main())
```

`width` y `height` tienen que coincidir con el JPEG real; si no, el servidor descarta el cuadro.

## 8. Diagnóstico

| Síntoma | Causa probable |
|---|---|
| El socket cierra con 4401 | Token de API inválido (no usar el token del edge) |
| Conecta pero nunca llega `stream: "arducam"` | La Raspberry no está enviando: revisar que corra el gateway o `edge/arducam_only.py` con la URL y el token del edge correctos |
| Llegan frames pero no se dibujan | Falta `ws.binaryType = "arraybuffer"`, o se filtra mal `robot_id` |
| Imagen deformada | El canvas tiene un tamaño fijo; ajustarlo a `bitmap.width`/`bitmap.height` |
| Retraso que crece con el tiempo | Se encolan cuadros en vez de quedarse solo con el último (sección 4, regla 1) |
| "Señal interrumpida" intermitente | Ancho de banda de la Raspberry; ver `ARDUCAM_STREAMING.md` (`--media-max-kbps`) |

Grabación y reproducción de misiones (`arducam.mp4`): ver `MISIONES_FRONTEND.md`.
Detalle del lado de la Raspberry y del servidor: ver `ARDUCAM_STREAMING.md`.
