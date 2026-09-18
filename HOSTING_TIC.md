# Server Core en HosTICer (FastAPI)

Este deploy contiene el **servidor** existente en Python: API, WebSocket, MQTT,
telemetría, control, mapas, mallas, autonomía y los módulos de percepción.
No ejecuta el edge de la Raspi ni necesita Node.

TIC debe asignar el recipe `fastapi`. Desde la raíz del repositorio instala
`requirements.txt` y ejecuta **`uvicorn main:app`**. No hace falta Dockerfile,
leer `PORT` ni arrancar otro Uvicorn desde Python. Dependencias: Python 3.11+.

## Configuración del panel

| Variable | Valor |
| --- | --- |
| `MQTT_HOST` | Host/IP del broker que puedan alcanzar tanto TIC como el edge. |
| `MQTT_PORT` | `1883` sin TLS o el puerto del broker. |
| `MQTT_TLS` | `true` si el broker usa TLS; por defecto `false`. |
| `MQTT_USERNAME`, `MQTT_PASSWORD` | Credenciales del broker, si corresponde. |
| `MQTT_TOPIC_PREFIX` | `go2` por defecto; debe coincidir con el edge. |
| `API_TOKENS` | Array JSON, por ejemplo `["TOKEN_ALEATORIO:operator:operador_01"]`. Roles: `viewer`, `operator`, `admin`. |
| `EDGE_MEDIA_TOKEN` | Token que usa el edge para subir video/LiDAR/audio. |
| `ROBOT_IDS` | `go2_01` o varios IDs separados por comas. |
| `CORS_ORIGINS` | Orígenes del frontend separados por comas; por defecto `*`. |
| `PERCEPTION_DEVICE` | `cpu` por defecto para este hosting; `cuda` solo si TIC proporciona GPU. |
| `SERVER_ARGS` | Opcional: array JSON con flags existentes, por ejemplo `["--mesh-interval-s","120"]`. |

Los secretos se configuran en el panel, **no en Git**. `.env.example` documenta
los nombres, pero no se carga automáticamente. Sin tokens configurados,
`/health` funciona y el acceso autenticado queda cerrado; no se habilitan los
tokens de desarrollo del arranque local.

TIC ya define `DATA_DIR` y `PORT`: no los sobrescribas en el panel. Mapas,
caras y auditoría se guardan respectivamente en `DATA_DIR/maps`,
`DATA_DIR/faces` y `DATA_DIR/audit`. Los caches opcionales van a `/tmp`.
`SERVER_ARGS` no puede sacar los datos persistentes de `DATA_DIR` ni reemplazar
los tokens configurados en sus variables específicas.

El broker MQTT es externo a esta aplicación: no se instala Mosquitto dentro del
hosting. El servidor puede arrancar con el broker desconectado y reintentar la
conexión; para operar el robot, la conexión al broker debe funcionar.

## Qué queda disponible

- `/health`: respuesta 200 barata, sin conexión a MQTT ni lectura de mapas en cada pedido.
- `/docs` y `/openapi.json`: documentación de la API existente.
- `/api/...`, `/ws/live` y `/ws/edge-media/{robot_id}`: mismas rutas del Server Core.
- `/`: identificación del servidor; este deploy no sirve el dashboard.

Nginx quita el prefijo público `/<proyecto>/<app>/` antes de entregar el pedido.
Las rutas se mantienen en la raíz. `X-Forwarded-Prefix` permite que `/docs`
encuentre el esquema OpenAPI bajo la ruta pública correcta. Los clientes deben
usar el prefijo público completo también en la URL WebSocket.

La decodificación de video usa PyAV y OpenCV headless, sin escritorio ni micrófono.
Los modelos ML opcionales del proyecto no se incluyen en el requirements del
hosting: YOLO requiere Ultralytics/Torch y el reconocimiento por embeddings
requiere InsightFace/ONNX. `/api/perception/capabilities` informa qué está
disponible. La API, streaming, mapeo y exploración funcionan sin esos modelos,
igual que en el servidor original. Para habilitar la pila ML completa hay que
verificar recursos y wheels con TIC; no se da por disponible una GPU.

## Verificar localmente

```bash
python3 -m venv .venv-server
.venv-server/bin/pip install --only-binary=:all: -r requirements.txt
DATA_DIR=/tmp/go2-server-data .venv-server/bin/uvicorn main:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
```

Para probar autenticación, exportá `API_TOKENS` y `EDGE_MEDIA_TOKEN` antes del
arranque. Las pruebas del adaptador y del arranque real se ejecutan con:

```bash
.venv-server/bin/python -m unittest discover -s tests -p 'test_hosting.py' -v
```

## Publicar

Desde la red del colegio, copiá el bloque de deploy de **tu app** en el panel
<https://hosting.ort.edu.ar>. Usá su URL SSH exacta (usuario completo, proyecto,
app y puerto `2222`), sin adivinarla. Después de commitear los cambios:

```bash
git remote add tic <URL-SSH-QUE-MUESTRA-TU-APP>
git push tic main
```

Si el remoto existe, usá `git remote set-url tic <URL-SSH-QUE-MUESTRA-TU-APP>`.
Leé las líneas `remote:` del build y verificá el estado del panel y el
`/<proyecto>/<app>/health` público: el éxito de `git push` por sí solo no
confirma la publicación.
