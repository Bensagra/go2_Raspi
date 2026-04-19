# Go2 SSH Gateway

## Estado actual

Este archivo documenta el gateway SSH directo (NDJSON) y se mantiene como modo legacy/fallback.

La metodologia recomendada para produccion ahora es arquitectura de 3 capas con canales separados:

- Edge: `edge/edge_gateway_service.py`
- Server Core: `server/server_core.py`
- Frontend: `frontend/frontend_dashboard.html`

Ver guia completa:

- `THREE_LAYER_CONNECTION.md`

Gateway NDJSON para manejar todo por una sola sesion SSH:

- Telemetria por topics (`LOW_STATE`, `LF_SPORT_MOD_STATE`, `GAS_SENSOR`, etc.)
- Camara en vivo (`event.stream=camera`)
- LiDAR (`event.stream=topic` con `ULIDAR_*` y topics de cloud)
- Audio en vivo (`event.stream=audio`, PCM en base64)
- Control (`sport`, motion mode, publish/request generico)
- AudioHub (`list/play/pause/mode/megaphone`)

Salida: NDJSON por `stdout`.

Entrada de comandos: NDJSON por `stdin`.

## Archivos principales

- `go2_ssh_gateway.py` (lado robot/raspi)
- `go2_ssh_client.py` (lado servidor operador)

## Arranque recomendado (todo simultaneo)

En el servidor operador, lanza el cliente SSH completo:

```bash
python3 /ruta/go2_ssh_client.py \
  --remote user@raspi-o-pc-cerca-del-go2 \
  --remote-python /home/bensagra/Documents/go2/.venv/bin/python \
  --remote-gateway /home/bensagra/Documents/go2/go2_ssh_gateway.py \
  --go2-ip 192.168.123.161 \
  --enable-camera \
  --enable-lidar \
  --enable-audio \
  --subscribe-profile core \
  --subscribe-profile lidar \
  --subscribe-profile audio \
  --disable-traffic-saving \
  --play-audio
```

Controles en GUI:

- Flechas o `WASD`: mover
- `Z`/`C`: lateral
- `Space`: `StopMove`
- `Q`/`Esc`: salir

## One-shot por SSH (sin cliente interactivo)

```bash
printf '{"id":"1","op":"get_temperatures"}\n' | \
ssh -T user@pc \
  "/home/bensagra/Documents/go2/.venv/bin/python -u /home/bensagra/Documents/go2/go2_ssh_gateway.py --ip 192.168.123.161 --subscribe-profile core --exit-on-stdin-eof"
```

## Sesion persistente por SSH con NDJSON crudo

```bash
coproc GO2 {
  ssh -T user@pc \
    "/home/bensagra/Documents/go2/.venv/bin/python -u /home/bensagra/Documents/go2/go2_ssh_gateway.py --ip 192.168.123.161 --subscribe-profile core --subscribe-profile lidar --subscribe-profile audio --enable-camera --enable-lidar --enable-audio --disable-traffic-saving"
}

# Leer eventos
cat <&"${GO2[0]}" &

# Comandos
echo '{"id":"a1","op":"status"}' >&"${GO2[1]}"
echo '{"id":"a2","op":"sport","action":"Move","parameter":{"x":0.3,"y":0,"z":0}}' >&"${GO2[1]}"
echo '{"id":"a3","op":"set_audio","enabled":true}' >&"${GO2[1]}"
echo '{"id":"a4","op":"audiohub","action":"list"}' >&"${GO2[1]}"
echo '{"id":"a5","op":"sport","action":"StopMove"}' >&"${GO2[1]}"
echo '{"id":"a6","op":"exit"}' >&"${GO2[1]}"
```

## Comandos JSON soportados

- `{"op":"help"}`
- `{"op":"ping"}`
- `{"op":"status"}`
- `{"op":"list_topics"}`
- `{"op":"list_sport_cmd"}`
- `{"op":"get_latest"}`
- `{"op":"get_latest","topic":"LOW_STATE"}`
- `{"op":"get_temperatures"}`
- `{"op":"subscribe","topic":"LOW_STATE"}`
- `{"op":"unsubscribe","topic":"LOW_STATE"}`
- `{"op":"subscribe_profile","profile":"all_telemetry"}`
- `{"op":"request","topic":"MOTION_SWITCHER","api_id":1001}`
- `{"op":"publish","topic":"WIRELESS_CONTROLLER","data":{"lx":0,"ly":0,"rx":0.5,"ry":0,"keys":0}}`
- `{"op":"publish_no_ack","topic":"ULIDAR_SWITCH","data":"on"}`
- `{"op":"sport","action":"Hello"}`
- `{"op":"sport","action":"Move","parameter":{"x":0.3,"y":0,"z":0}}`
- `{"op":"set_motion_mode","name":"normal"}`
- `{"op":"get_motion_mode"}`
- `{"op":"set_video","enabled":true}`
- `{"op":"set_camera_stream","enabled":true,"emit_every":2,"format":"jpg","jpeg_quality":70}`
- `{"op":"set_lidar","enabled":true,"subscribe":true}`
- `{"op":"set_lidar_decoder","decoder":"native"}`
- `{"op":"set_audio","enabled":true,"emit_every":1,"max_bytes":24576}`
- `{"op":"audiohub","action":"list"}`
- `{"op":"audiohub","action":"play","unique_id":"..."}`
- `{"op":"audiohub","action":"pause"}`
- `{"op":"audiohub","action":"resume"}`
- `{"op":"audiohub","action":"set_mode","play_mode":"list_loop"}`
- `{"op":"audiohub","action":"raw","api_id":1001,"parameter":{}}`
- `{"op":"exit"}`

## Perfiles de suscripcion

- `core`: estado base (low/sport)
- `lidar`: `ULIDAR_ARRAY`, `ULIDAR_STATE`, `ROBOTODOM`
- `audio`: `AUDIO_HUB_PLAY_STATE`
- `navigation`: topics de SLAM/localizacion
- `all_telemetry`: amplia cobertura de topics disponibles

## Temperaturas y valores de estado

Para temperatura y sensores:

- Usa `subscribe_profile=core` para `LOW_STATE` y `LF_SPORT_MOD_STATE`
- Usa `{"op":"get_temperatures"}` para resumen rapido:
  - `temperature_ntc1`
  - temperaturas de motores
  - `bms_bq_ntc`
  - `bms_mcu_ntc`
  - `imu_temperature`
  - ultimo payload de `gas_sensor`

Para explorar todo lo disponible en tu firmware:

1. `{"op":"list_topics"}`
2. `{"op":"subscribe_profile","profile":"all_telemetry"}`

Nota: algunos topics dependen de firmware/modo y pueden no emitir en todos los robots.
