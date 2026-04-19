# Metodologia de conexion 3 capas (recomendada)

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
pip install -r requirements_3layer.txt
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

### 4) Frontend

Abre en navegador:

- `frontend/frontend_dashboard.html`

Configura:

- API: `http://<IP_SERVIDOR>:8000`
- Token: `dev-operator-token`
- Robot: `go2_01`

## API principal

- `POST /api/robots/{robot_id}/commands`
- `GET /api/robots/{robot_id}/state`
- `GET /api/robots/{robot_id}/replay`
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
