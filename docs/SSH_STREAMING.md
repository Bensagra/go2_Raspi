# Streaming Go2

## Modo recomendado: arquitectura 3 capas con canales separados

La metodologia recomendada ahora es:

1. Edge en Raspi: `edge/edge_gateway_service.py`
2. Server Core: `server/server_core.py`
3. Frontend operador: `frontend/frontend_dashboard.html`

Documento principal:

- `THREE_LAYER_CONNECTION.md`

Canales:

- MQTT para telemetria, eventos, heartbeat, ACK
- API/WebSocket para control y distribucion al frontend
- Canal de media separado Edge -> Server (`/ws/edge-media/{robot_id}`)

Esto evita mezclar datos operativos con datos pesados y permite evolucionar a WebRTC/SFU sin romper control y seguridad.

## Arranque rapido (3 capas)

Server:

```bash
python server/server_core.py --host 0.0.0.0 --port 8000 --mqtt-host 127.0.0.1 --mqtt-port 1883
```

Edge (Raspi):

```bash
python edge/edge_gateway_service.py \
  --robot-id go2_01 \
  --go2-ip 192.168.123.161 \
  --mqtt-host 192.168.1.10 \
  --mqtt-port 1883 \
  --enable-camera --enable-audio --enable-lidar \
  --subscribe-profile core,lidar,audio \
  --disable-traffic-saving \
  --media-ws-url ws://192.168.1.10:8000/ws/edge-media/{robot_id} \
  --media-ws-token edge-media-dev-token
```

Frontend:

- Abrir `frontend/frontend_dashboard.html`
- Configurar API, token y robot_id

## Modo legacy: solo camara binaria por SSH

Si solo quieres video y no necesitas telemetria/lidar/control/audio, puedes usar:

- `camera_ssh_sender.py` en la maquina junto al robot
- `camera_ssh_receiver.py` en el servidor

Comando base:

```bash
/home/bensagra/Documents/go2/.venv/bin/python -u /home/bensagra/Documents/go2/camera_ssh_sender.py \
  --ip 192.168.123.161 \
  --emit-every 1 \
  --image-format jpg \
  --jpeg-quality 75 \
| ssh -T -o Compression=no user@tu-servidor \
  "python3 /ruta/en/servidor/camera_ssh_receiver.py --stats-every 30"
```

Metadata JSON (legacy):

```bash
/home/bensagra/Documents/go2/.venv/bin/python -u /home/bensagra/Documents/go2/camera_ssh_sender.py \
  --ip 192.168.123.161 --image-format jpg --jpeg-quality 75 \
| ssh -T -o Compression=no user@tu-servidor \
  "python3 /ruta/en/servidor/camera_ssh_receiver.py --stdout-json > /tmp/go2_frames_meta.jsonl"
```

Viewer remoto (legacy):

```bash
/home/bensagra/Documents/go2/.venv/bin/python -u /home/bensagra/Documents/go2/camera_ssh_sender.py \
  --ip 192.168.123.161 \
| ssh -T -Y -o Compression=no user@tu-servidor \
  "python3 /ruta/en/servidor/camera_ssh_receiver.py --show"
```

## Tuning rapido

- Menos ancho de banda: bajar `--jpeg-quality` (ej. `60-70`).
- Menos CPU local: subir `--emit-every` (ej. `2` o `3`).
- Menos latencia: mantener `--emit-every 1` y `-u` en Python.
- SSH: para JPEG suele rendir mejor `-o Compression=no`.
