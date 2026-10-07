# Go2: antichoque, handy, linterna y captura

Guía de implementación e integración con otro frontend. Aplica a la arquitectura
**frontend → Server Core → MQTT → Raspberry Pi → Go2**, con
`server/server_core.py` (o `main:app`) y `edge/edge_gateway_service.py`.
Los programas anteriores `go2_ssh_gateway.py` / `go2_ssh_client.py` no implementan
este protocolo nuevo.

## 1. Qué quedó implementado

| Función | Comportamiento |
|---|---|
| Antichoque | Arranca **apagado** al iniciar el proceso del edge. Se puede activar explícitamente. |
| Handy | Voz en vivo del micrófono del navegador al parlante del Go2; mantener para hablar o abrir/cerrar con un clic. |
| Límite de voz | 20 segundos por turno; corte al soltar, cerrar, salir de la pestaña o perder el enlace. Un solo emisor por robot. |
| Linterna | Encendido/apagado y API de brillo de 0 a 10; estado confirmado por ACK del robot. |
| Captura | Descarga PNG del cuadro visible de la cámara RGB, con robot y fecha en el nombre. |
| Cliente reutilizable | `frontend/go2-media-controls.js`, JavaScript sin dependencias de frameworks. |

La captura es **del cuadro mostrado**, con la resolución del stream. No dispara
una fotografía de mayor resolución en el sensor, no captura el LiDAR y no guarda
una foto en el servidor. La latencia propia del video sigue existiendo: se rechazan
cuadros de más de 3 segundos. La voz se transmite mientras hablás; no se espera a
terminar una grabación para subirla como archivo.

Archivos principales:

- `robot_media_protocol.py`: límites y validaciones compartidas.
- `edge/talk.py`: PCM → pista WebRTC, buffer acotado, exclusividad y corte local.
- `server/talk.py`: autenticación y puente de voz WebSocket → MQTT.
- `edge/edge_gateway_service.py`: integración, linterna, telemetría y arranque.
- `server/server_core.py`: permisos, validación de linterna y registro del canal.
- `frontend/go2-media-controls.js`: handy, comandos confirmados y captura.
- `frontend/frontend_dashboard.html`: controles ya integrados.

## 2. Actualizar y arrancar

Actualizá **servidor, Raspi y frontend juntos**. Copiar sólo el HTML deja funciones
sin implementar. El HTML necesita `go2-media-controls.js` en el mismo directorio.
El módulo `robot_media_protocol.py` debe estar en la raíz del proyecto tanto en
la Raspi como en el servidor.

Con las dependencias de este repositorio:

```bash
python -m pip install -r requirements_3layer.txt
```

`aiortc` ya es dependencia del driver Unitree; se usan también PyAV y WebSockets
que forman parte del entorno actual. No hace falta un servidor de audio adicional
ni instalar FFmpeg para el handy; no se convierten archivos WebM/MP3.

Ejemplo de desarrollo (reemplazar IP y tokens por los de tu instalación):

```bash
# En el servidor, con el broker MQTT ya accesible:
python server/server_core.py \
  --host 0.0.0.0 --port 8000 \
  --mqtt-host 127.0.0.1 --mqtt-port 1883 \
  --robot-id go2_01 \
  --api-token TU_TOKEN:operator:operador_01 \
  --edge-media-token TU_TOKEN_EDGE
```

```bash
# En la Raspberry Pi:
python edge/edge_gateway_service.py \
  --robot-id go2_01 \
  --go2-ip 192.168.123.161 \
  --mqtt-host IP_SERVIDOR --mqtt-port 1883 \
  --enable-camera \
  --media-ws-url 'ws://IP_SERVIDOR:8000/ws/edge-media/{robot_id}' \
  --media-ws-token TU_TOKEN_EDGE
```

No agregues `--enable-safety-guard` si querés que arranque apagado.
Si un servicio systemd, script o supervisor ya contiene ese argumento, retiralo:
el argumento explícito tiene prioridad sobre el valor predeterminado.
`--disable-safety-guard` sigue disponible para declarar expresamente OFF.

`--enable-audio` habilita **escuchar el micrófono del perro**, y no es necesario
para hablar por el parlante. El handy usa una pista de salida independiente.
Para capturar sí hace falta recibir video. `--enable-lidar` sólo es necesario para
usar las funciones de LiDAR, incluido el antichoque si decidís activarlo.

Frontend local en la computadora del operador:

```bash
python -m http.server 8080 --bind 127.0.0.1 --directory frontend
```

Abrí `http://localhost:8080/frontend_dashboard.html` y completá la URL de la API,
el token de operador y el ID del robot. Los comandos arrancan al conectar el
dashboard. Para hosting seguí también [HOSTING_TIC.md](HOSTING_TIC.md): se mantienen
las variables `MQTT_HOST`, `API_TOKENS`, `EDGE_MEDIA_TOKEN`, `ROBOT_IDS` y
`CORS_ORIGINS`. El backend no sirve automáticamente estos archivos estáticos:
publicá HTML y JS en tu hosting de frontend.

### HTTPS, permisos y proxy

- El micrófono requiere **HTTPS** o `localhost`, permiso de micrófono y un
  navegador con `AudioContext` y `AudioWorklet`. HTTP por IP de LAN no habilita
  `getUserMedia`; linterna y antichoque sí pueden usarse por HTTP.
- Un frontend HTTPS necesita una API HTTPS y WebSockets WSS accesibles. Evitá
  mezclar una página HTTPS con una API HTTP.
- El proxy debe permitir `Upgrade: websocket` en `/ws/talk/{robot_id}`, además de
  los canales existentes `/ws/live` y `/ws/edge-media/{robot_id}`.
- `apiBase` puede incluir un prefijo, por ejemplo
  `https://dominio.example/mi-app`. El cliente conserva ese prefijo.
- Configurá CORS para el origen del frontend. Si embebés el frontend en un iframe,
  debe tener permiso de micrófono (`allow="microphone"`) y las políticas del
  documento padre deben permitirlo.
- Si usás Content Security Policy, el AudioWorklet se carga desde un Blob URL:
  permití ese origen en la directiva que use tu navegador para módulos/worklets
  (habitualmente `script-src blob:`), además de `connect-src` hacia API y WSS.
- Sincronizá la hora del servidor y Raspi (NTP). Los paquetes de voz se descartan
  si son viejos y las capturas usan el timestamp del edge. También mantené en hora
  el dispositivo del operador.
- El token de la API es distinto del token de subida de media del edge.
  Usá los tokens existentes de operador/admin; no incrustes uno de administrador
  en un frontend público. El WebSocket usa token en query; evitá registrar esa
  query en logs del proxy.

## 3. Uso en el dashboard existente

1. Conectá API, robot y token. Verificá que haya telemetría reciente.
2. **Antichoque**: la telemetría indica OFF al iniciar el edge. El botón permite
   encenderlo y apagarlo; espera confirmación. No se habilita al conectar el
   navegador ni al abrir una misión.
3. **Mantener para hablar**: sostené el botón; esperá “Transmitiendo” y hablá.
   Soltá para cortar. En el teclado, enfocá el botón y mantené Espacio o Enter.
4. **Abrir micrófono (20 s)**: alternativa cómoda para conceder el permiso la
   primera vez. El botón cambia a “Cerrar micrófono”. Se cierra automáticamente
   a los 20 s. Podés abrir un turno nuevo después.
5. **Encender linterna / Apagar linterna**: envían brillo 10 / 0. Se muestra
   “Esperando confirmación” hasta recibir ejecución o error.
6. **Capturar imagen**: descarga un PNG. Si no hay cuadro reciente, muestra un
   error en lugar de descargar una imagen vacía o vieja.

Mientras está abierto el micrófono, ese dashboard deja de reproducir el audio
que vuelve del perro, para reducir el eco. Los otros clientes no se silencian.
El edge omite nuevos saludos/archivos mientras hay una sesión de voz; un archivo
que ya estaba sonando no se detiene automáticamente. El handy conserva el volumen
que tenga el parlante: si está en cero, ajustalo en la app del robot.

## 4. Integración rápida en otro frontend

Copiá `frontend/go2-media-controls.js` a tus archivos públicos. No necesitás copiar
el dashboard completo. Cargalo antes del código que crea los controles:

```html
<script src="/go2-media-controls.js"></script>
<button id="talk" type="button" aria-pressed="false">Abrir micrófono</button>
<button id="lightOn" type="button">Encender linterna</button>
<button id="lightOff" type="button">Apagar linterna</button>
<button id="photo" type="button">Capturar imagen</button>
<button id="safetyOff" type="button">Apagar antichoque</button>
<p id="status" role="status"></p>
<canvas id="camera" width="640" height="360"></canvas>

<script>
const connection = {
  apiBase: 'https://TU_API', // sin /api al final; admite prefijo de aplicación
  token: 'TOKEN_DE_LA_SESION',
  robotId: 'go2_01',
};
const status = document.getElementById('status');
const talkButton = document.getElementById('talk');
const canvas = document.getElementById('camera');
let lastFrameAt = 0;

// Llamá esto desde TU decodificador de video, al mostrar un cuadro válido.
// frame: ImageBitmap, HTMLImageElement o VideoFrame ya decodificado.
// encodedTs: header.encoded_ts || header.ts, en segundos Unix, enviado por el edge.
function onCameraFrame(frame, encodedTs) {
  canvas.width = frame.displayWidth || frame.width;
  canvas.height = frame.displayHeight || frame.height;
  canvas.getContext('2d').drawImage(frame, 0, 0);
  lastFrameAt = Number(encodedTs) * 1000;
}

const talk = new Go2Media.TalkClient(message => {
  const active = ['opening', 'talking'].includes(message.status);
  talkButton.textContent = active ? 'Cerrar micrófono' : 'Abrir micrófono';
  talkButton.setAttribute('aria-pressed', String(active));
  status.textContent = message.status === 'talking'
    ? `Transmitiendo: ${message.remaining} s`
    : message.error || (message.status === 'opening' ? 'Abriendo…' : 'Micrófono cerrado');
});

async function showError(action) {
  try { await action(); }
  catch (error) { status.textContent = error.message; }
}

talkButton.onclick = () => showError(async () => {
  if (talk.active) talk.stop();
  else await talk.start(connection);
});

async function light(brightness) {
  status.textContent = 'Esperando al robot…';
  const result = await Go2Media.confirmedCommand(connection, 'set_flashlight', { brightness });
  status.textContent = result.enabled ? `Linterna: ${result.brightness}/10` : 'Linterna apagada';
}
document.getElementById('lightOn').onclick = () => showError(() => light(10));
document.getElementById('lightOff').onclick = () => showError(() => light(0));
document.getElementById('safetyOff').onclick = () => showError(async () => {
  const result = await Go2Media.confirmedCommand(connection, 'set_safety', {enabled:false});
  status.textContent = result.safety_enabled ? 'Antichoque encendido' : 'Antichoque apagado';
});
document.getElementById('photo').onclick = () => showError(async () => {
  const shot = await Go2Media.captureCanvas(canvas, {robotId: connection.robotId, lastFrameAt});
  const url = URL.createObjectURL(shot.blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = shot.filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
  status.textContent = `Captura lista: ${shot.width} × ${shot.height}`;
});

// Al desmontar tu vista: talk.dispose(). Al cambiar robot/token/API:
// talk.stop(); lastFrameAt = 0; luego actualizá connection y el stream.
</script>
```

Este ejemplo integra las funciones; **no implementa el decodificador del stream**.
Conectá `onCameraFrame` al reproductor que ya tenga tu frontend. Para hacerlo desde
cero, el dashboard contiene `decodeMediaFrame`, `processMediaFrames`,
`decodeH264Chunk` y `handleDecodedVideoFrame`. El sobre binario empieza con `0xA7`,
versión `1`, longitud de JSON u32 little-endian, JSON y payload. Las imágenes
WebP/JPEG pueden decodificarse con `createImageBitmap`; H.264 necesita WebCodecs y
esperar un keyframe. Filtrá siempre `header.robot_id` y `header.stream === 'video'`.
En H.264 conservá el timestamp del edge asociado a cada chunk hasta que se dibuje
su `VideoFrame` (el dashboard lo hace con `videoSnapshotTimes`). No pongas
`lastFrameAt = Date.now()` a un cuadro reenviado desde caché: lo haría parecer nuevo.

Para una app React/Vue/Svelte, la misma clase funciona sin modificaciones:
creala una vez al montar, reflejá `onStatus` en el estado del componente y llamá
`dispose()` al desmontar. Llamá `start()` directamente desde un gesto del usuario;
no desde un efecto de montaje. Si cambia la conexión, cerrá antes la sesión vieja.

### Variante mantener presionado

En vez del `onclick` anterior, podés usar:

```js
button.style.touchAction = 'none';
button.addEventListener('pointerdown', event => {
  if (event.button !== 0) return;
  event.preventDefault();
  button.setPointerCapture(event.pointerId);
  talk.start(connection).catch(error => { status.textContent = error.message; });
});
for (const eventName of ['pointerup', 'pointercancel', 'lostpointercapture']) {
  button.addEventListener(eventName, () => talk.stop());
}
button.addEventListener('blur', () => talk.stop());
button.addEventListener('keydown', event => {
  if (!['Space', 'Enter'].includes(event.code)) return;
  event.preventDefault();
  event.stopPropagation();
  if (!event.repeat) talk.start(connection).catch(error => { status.textContent = error.message; });
});
button.addEventListener('keyup', event => {
  if (!['Space', 'Enter'].includes(event.code)) return;
  event.preventDefault();
  event.stopPropagation();
  talk.stop();
});
```

`button` es tu botón de hablar. No uses a la vez `onclick` de apertura/cierre y
estos handlers sobre el mismo botón. Conservá un botón alternativo de apertura
si necesitás que la primera concesión del permiso sea fácil en dispositivos táctiles.

## 5. Contrato de comandos HTTP y ACK

### Roles

| Operación | viewer | operator / admin |
|---|---|---|
| Ver video / descargar captura local | Sí, si recibe el stream | Sí |
| Consultar telemetría y replay | Sí | Sí |
| Cambiar linterna / antichoque | No (403) | Sí |
| Abrir sesión de voz | No (4403) | Sí |

Los comandos usan:

```http
POST /api/robots/go2_01/commands
Authorization: Bearer TU_TOKEN
Content-Type: application/json
```

Linterna:

```json
{
  "command_id": "web-un-identificador-unico",
  "type": "set_flashlight",
  "payload": { "brightness": 10 },
  "ttl_ms": 3000
}
```

- `brightness`: entero **0..10**. `0` apaga, `10` máxima intensidad.
- Alternativa: `{"enabled":true}` equivale a 10; `false` equivale a 0.
- Si están ambos, prevalece `brightness`.
- Valores fuera de rango, strings (`"10"`), booleanos en `brightness` y números
  decimales se rechazan con HTTP 400. No se corrigen silenciosamente.
- Se conserva la ACL y el rate limit existentes del endpoint de comandos.

Respuesta HTTP:

```json
{"ok":true,"command_id":"web-un-identificador-unico","status":"queued"}
```

**`queued` no significa ejecutado.** Esperá el ACK final por `/ws/live?token=...`:

```json
{
  "type": "command_ack",
  "robot_id": "go2_01",
  "data": {
    "command_id": "web-un-identificador-unico",
    "command_type": "set_flashlight",
    "status": "executed",
    "result": { "executed": "set_flashlight", "brightness": 10, "enabled": true }
  }
}
```

También pueden llegar `accepted` (todavía no terminó), `error` o `rejected`, con
`reason`. Filtrá por **robot_id y command_id** y resolvé sólo al estado final.
Registrá tu espera antes de enviar, ya que el ACK puede llegar antes que el HTTP.

`Go2Media.confirmedCommand()` evita esa carrera consultando
`GET /api/robots/{robot_id}/replay?limit=200` cada 200 ms hasta encontrar el ACK;
por defecto espera hasta 8 segundos. Si ya tenés `/ws/live`, podés reemplazar ese
polling por tu correlación de ACKs. La función devuelve **`ack.result`**, no la
respuesta HTTP queued. Un timeout significa resultado desconocido, no prueba
que el comando no haya llegado: consultá la telemetría antes de reintentar.

El edge usa VUI API `1005`, parámetro `{"brightness":N}`. Verifica que el código
de respuesta sea cero antes de confirmar. La telemetría incluye:

```json
{
  "flashlight": { "brightness": 10 },
  "talk": { "active": false, "max_seconds": 20 },
  "safety": { "enabled": false, "armed": false }
}
```

`flashlight.brightness` es `null` hasta que este edge confirme un cambio, y se
reinicia a `null` si se desconecta del robot. Es el último valor confirmado por
este servicio; no detecta cambios realizados desde otra app. El arranque no
enciende ni apaga físicamente la linterna.

### Antichoque

Se mantiene el comando:

```json
{"type":"set_safety","payload":{"enabled":false},"ttl_ms":3000}
```

Para activar, cambiá a `true`. El resultado ejecutado incluye `safety_enabled`.
Su valor no se guarda entre reinicios del proceso: sin un flag explícito, cada
inicio del edge vuelve a OFF. Una reconexión de red dentro del mismo proceso
conserva el último valor elegido. Esto controla el guard LiDAR de **este proyecto**;
no cambia configuraciones independientes del firmware o de la app Unitree.

## 6. Contrato de voz en vivo

### Apertura

```text
wss://TU_API/ws/talk/go2_01?token=TOKEN_OPERADOR
```

El socket es exclusivo para la voz. No envíes PCM a `/ws/live` ni a
`/ws/edge-media`. No hace falta mandar un JSON `start`: **abrir este socket solicita
la sesión**. El servidor exige token válido, rol operator/admin, robot conocido,
MQTT conectado y telemetría del edge de menos de 5 segundos con robot conectado.

Antes de enviar audio esperá:

```json
{
  "session_id":"identificador-generado-por-servidor",
  "status":"ready",
  "sample_rate":16000,
  "max_seconds":20
}
```

`ready` confirma que el edge activó su pista de salida en una conexión WebRTC
establecida; no verifica acústicamente que el altavoz sea audible.

### Formato y envío

Cada mensaje WebSocket de audio es **binario**:

| Propiedad | Valor |
|---|---|
| Codificación | PCM signed 16 bit little-endian, sin cabecera WAV |
| Canales | 1 (mono) |
| Frecuencia | 16000 Hz |
| Paquete del cliente incluido | 640 muestras = 1280 bytes = 40 ms |
| Máximo aceptado | 3200 bytes = 100 ms; tamaño positivo y par |
| Ritmo | Tiempo real; hasta 500 ms de margen inicial de ráfaga |
| Turno máximo | 20 s, validado también en servidor y edge |
| Inactividad | 2 s sin audio cortan la sesión |

No sirve mandar los blobs de `MediaRecorder` directamente: suelen ser Opus/WebM,
AAC u otro contenedor. El cliente incluido usa AudioWorklet, mezcla a mono,
remuestrea desde la frecuencia real del dispositivo y empaqueta PCM16 LE.

La voz consume aproximadamente 256 kbit/s de PCM navegador → servidor.
Servidor → MQTT agrega Base64 (~341 kbit/s más JSON/cabeceras). El tramo
Raspi → Go2 usa el codec de la pista WebRTC negociada. El límite de subida de
video `--media-max-kbps` no limita este canal de voz.

### Cierre

Mandá un texto JSON y cerrá el socket:

```json
{"type":"stop"}
```

También basta con cerrar la conexión. Pueden recibirse:

```json
{"status":"stopped"}
```

```json
{"status":"error","error":"speaker busy"}
```

El cliente incluido siempre detiene los tracks de `getUserMedia`, desconecta los
nodos y cierra el AudioContext. También cancela una apertura si soltaste mientras
el navegador todavía pedía permiso. No se reabre automáticamente al reconectar.

El corte se ejecuta en varios niveles:

1. Navegador: soltar, botón Cerrar, 20 s, blur, pestaña oculta, `pagehide`, cambio de
   conexión o salida de la vista (`dispose`). El dashboard además corta si pierde
   `/ws/live`.
2. Servidor: máximo de turno, 2 s sin mensajes, errores de formato/rate, cierre del
   navegador o pérdida de MQTT; publica `stop` en el `finally`.
3. Edge: 20 s, 2 s sin PCM, desconexión del robot, cola llena o stop explícito;
   vacía el buffer y envía silencio. No depende de que llegue el stop del servidor.

La red y los buffers del robot pueden agregar un pequeño retardo audible al
corte. El buffer propio del edge guarda como máximo 200 ms y el cliente corta
si la cola WebSocket supera 6400 bytes, para no acumular segundos de voz vieja.

### Errores de apertura

| Código WS | Significado |
|---|---|
| 4401 | Token inválido o ausente |
| 4403 | Rol sin permiso para hablar |
| 4409 | Otro cliente ya usa el parlante |
| 4410 | Robot desconocido o enlace de control no disponible |

Si el servidor rechaza antes de aceptar el WebSocket, algunos navegadores
muestran un cierre genérico `1006` en vez del código específico. Mostrá el error
sin intentar transmitir de todos modos. Después de aceptar, también se informan
errores JSON como timeout, MQTT desconectado o formato de PCM incorrecto.

### Transporte interno MQTT

Para cada robot, con el prefijo configurado (default `go2`):

```text
go2/go2_01/talk/in       servidor → edge
go2/go2_01/talk/status   edge → servidor
```

Ajustá las ACL del broker si las limitaste a los topics anteriores. `start` y
`stop` usan QoS 1; PCM usa QoS 0; ningún mensaje usa retain. Los paquetes incluyen
`session_id` y `ts` Unix. Los PCM llevan además `sequence` creciente y `pcm` en
Base64. El edge descarta otra sesión, duplicados, secuencias viejas y paquetes
con antigüedad mayor a 1 s o más de 2 s en el futuro. No guardes estos topics
como retained ni los reinyectes desde un replay.

La exclusividad del servidor es en memoria: corré **un proceso/worker del Core
por este grupo de robots**, igual que el estado vivo existente. El edge además
rechaza una segunda sesión, pero múltiples workers no comparten la entrega de
estados y no son una configuración soportada para este canal.

La voz no se guarda en los archivos de audio del robot, en el replay del Core
ni en su auditoría JSONL. Puede existir en buffers transitorios de navegador,
MQTT, edge y WebRTC. El broker debe tener la configuración de confianza y acceso
correspondiente a tu instalación.

## 7. API del módulo JavaScript

```js
const client = new Go2Media.TalkClient(onStatus);
await client.start({apiBase, token, robotId});
client.active;  // true durante apertura y transmisión
client.stop();  // cierra este turno, permite iniciar otro
client.dispose(); // cierre + quitar listeners globales al desmontar

const result = await Go2Media.confirmedCommand(
  {apiBase, token, robotId}, 'set_flashlight', {brightness: 6}, 8000
);

const shot = await Go2Media.captureCanvas(canvas, {
  robotId, lastFrameAt, maxAgeMs: 3000
});
// shot: {blob, filename, frameAt, width, height}
```

`onStatus` recibe `{status:'opening'}`, `{status:'talking', remaining:N}`,
`{status:'stopped'}` o `{status:'error', error:'...'}`. `remaining` sirve para un
contador visual; no lo anuncies completo varias veces por segundo con un lector
de pantalla. `start()` puede fallar: capturá el rechazo y mostrale el motivo al
operador.

`captureCanvas` **no descarga por su cuenta**: devuelve el Blob. Podés descargarlo,
mostrar una previsualización con `URL.createObjectURL` o subirlo a tu servicio de
archivos. Revocá la URL cuando ya no la uses. El canvas debe estar libre de
contaminación por imágenes de otro origen sin CORS. El dashboard decodifica los
bytes del WebSocket y dibuja localmente.

## 8. Comprobación después del despliegue

1. Reiniciá el proceso edge sin `--enable-safety-guard`; confirmá
   `safety.enabled:false` en telemetría. Cerrá/abrí el dashboard y comprobá que no
   lo active. Probá luego la activación y desactivación explícitas.
2. Con un token viewer, intentá la linterna y la voz: debe denegarlos. Con un
   operador, encendé/apagá la luz y verificá el cambio físico y el ACK `executed`.
3. En HTTPS/localhost, concedé el micrófono y hablá primero 2–3 segundos. Verificá
   físicamente que tu voz salga por el parlante del Go2 y que soltar corte.
4. Probá abrir/cerrar con el botón alternativo, soltar durante el pedido de
   permiso y cambiar de pestaña. El indicador de micrófono del navegador debe
   apagarse. Probá una sesión de más de 20 s: debe terminar sola.
5. Abrí otro cliente y tratá de hablar simultáneamente: debe indicar ocupado.
6. Durante un turno, desconectá el cliente o el MQTT de pruebas. El edge debe
   cortar en hasta ~2 s sin nuevos paquetes, aunque no reciba stop.
7. Con video reciente, descargá una captura: verificá PNG, dimensiones, nombre y
   que coincida con el cuadro mostrado. Apagá el video y esperá más de 3 s: debe
   rechazar otra captura. Repetí con WebP/JPEG y H.264 si usás ambos.

Pruebas automáticas locales:

```bash
python -m unittest discover -s tests -q
node --test tests/test_media_client.cjs
node --check frontend/go2-media-controls.js
```

Las pruebas usan robot/MQTT/navegador simulados. Cubren permisos, bytes PCM,
remuestreo, exclusividad, cierre, inactividad, límite, captura y ACK de linterna.
No sustituyen comprobar el sonido y la luz con el Go2 real. En esta implementación
no se realizó una prueba física con el robot ni una inspección visual en navegador;
la herramienta de navegador no estaba disponible.

## 9. Diagnóstico rápido

| Síntoma | Revisar |
|---|---|
| No aparece pedido de micrófono | HTTPS/localhost, permisos del sitio/OS, iframe y soporte de AudioWorklet. |
| Micrófono dice “abriendo” y luego timeout | Actualización del edge y Core; ACL de `talk/in` y `talk/status`; telemetría; reloj/NTP. |
| `speaker busy` | Otro turno activo. Cerrar ese turno; una sesión abandonada vence por inactividad. |
| Dice transmitiendo pero no se oye | Volumen del robot, conexión WebRTC sendrecv, firmware compatible y ausencia de otro audio ya reproduciéndose. |
| Voz cortada / red lenta | Capacidad de subida; bajar video/LiDAR. No aumentar buffers para acumular voz. |
| HTTP queued pero linterna no cambia | Esperar ACK final. Error/timeout del VUI, edge antiguo o robot desconectado. |
| Linterna muestra “sin confirmar” | Normal al iniciar/reconectar; realizar una acción para obtener estado confirmado. |
| Captura rechazada aunque hay imagen | Puede ser una imagen congelada/caché; revisar timestamps, NTP y video activo. |
| `Go2Media is not defined` | Publicar/cargar `go2-media-controls.js` antes del código de controles. |
| Funciona local, falla detrás de proxy | Prefijo correcto de `apiBase`, Upgrade WebSocket, HTTPS/WSS, CORS y CSP. |
| Antichoque sigue arrancando ON | Revisar argumentos reales de systemd/supervisor: `--enable-safety-guard` fuerza ON. |

## 10. Referencias del protocolo del robot

El mapeo VUI (brillo `1005`, volumen `1003`) sigue el
[ejemplo VUI del driver Unitree WebRTC](https://github.com/legion1581/unitree_webrtc_connect/blob/master/examples/go2/data_channel/vui/vui.py).
La pista de audio saliente usa el transceiver `sendrecv` del
[canal de audio del driver](https://github.com/legion1581/unitree_webrtc_connect/blob/master/unitree_webrtc_connect/webrtc_audio.py)
y el mecanismo de envío de audio del
[ejemplo de reproducción por WebRTC](https://github.com/legion1581/unitree_webrtc_connect/blob/master/examples/go2/audio/mp3_player/play_mp3.py).
No se usa la subida de archivos `UPLOAD_MEGAPHONE` para este flujo en vivo.

También se corrigió el identificador VUI utilizado por el control de volumen
existente: ahora usa `1003`, de modo que los saludos no envíen parámetros de
volumen a la API de brillo.

## 11. Resultado de verificación de esta entrega

**Aprobado en revisión de código y contratos**, con validación visual y física
pendiente. La suite completa pasó: **52 pruebas Python y 7 pruebas JavaScript**.
También se verificó sintaxis JavaScript y compilación de los módulos Python.

| Hallazgo de revisión | Estado |
|---|---|
| Cierre normal de voz sin excepción ni evento de error | Resuelto y probado |
| Linterna/antichoque por HTTP sin depender de `randomUUID` | Resuelto y probado |
| PTT corta al cambiar foco del botón | Resuelto en código |
| Captura distingue cuadro reciente de caché/decodificación atrasada | Resuelto: timestamp del edge |
| Guía coincide con permisos, rutas y formatos implementados | Revisado |
| Apariencia en navegador y sonido/luz con Go2 real | Pendiente: no disponibles en esta sesión |

El detector de UI señaló una advertencia tipográfica preexistente en el dashboard
(tamaños muy próximos). No se rediseñó la interfaz fuera del alcance de estos controles.

## Arducam B0541 adicional

La cámara CSI de la Raspberry se integra como `stream: "arducam"`, separado del video Go2 y de la térmica. Se graba durante las misiones y se muestra/reproduce en el dashboard. Ver [ARDUCAM_STREAMING.md](ARDUCAM_STREAMING.md) para instalación, configuración y contrato completo.
