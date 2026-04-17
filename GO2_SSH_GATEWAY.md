# Go2 SSH Gateway

Este gateway permite usar una sola conexion WebRTC al Go2 y exponer por SSH:

- Telemetria por topics (lowstate, sport state, gas sensor, etc.)
- Camara en vivo (frames base64 en eventos JSON)
- LiDAR (topics y control on/off)
- Acciones y requests (sport mode, motion mode, VUI y requests genericos)

Salida: NDJSON por `stdout`.

Entrada de comandos: NDJSON por `stdin`.

## Archivo principal

- `go2_ssh_gateway.py`

## Arranque local (en la PC junto al robot)

```bash
/home/bensagra/Documents/go2/.venv/bin/python -u /home/bensagra/Documents/go2/go2_ssh_gateway.py \
  --ip 192.168.123.161 \
  --subscribe-profile core \
  --enable-lidar \
  --enable-camera \
  --disable-traffic-saving
```

## One-shot por SSH (ejecutar comando y salir)

```bash
printf '{"id":"1","op":"get_temperatures"}\n' | \
ssh -T user@pc \
  "/home/bensagra/Documents/go2/.venv/bin/python -u /home/bensagra/Documents/go2/go2_ssh_gateway.py --ip 192.168.123.161 --subscribe-profile core --exit-on-stdin-eof"
```

## Sesion persistente por SSH (simultaneo)

Ejemplo en bash (en tu servidor) para leer eventos y enviar comandos en paralelo:

```bash
coproc GO2 {
  ssh -T user@pc \
    "/home/bensagra/Documents/go2/.venv/bin/python -u /home/bensagra/Documents/go2/go2_ssh_gateway.py --ip 192.168.123.161 --subscribe-profile core --enable-lidar --enable-camera --disable-traffic-saving"
}

# Lector de eventos en segundo plano
cat <&"${GO2[0]}" &

# Comandos al gateway
echo '{"id":"a1","op":"status"}' >&"${GO2[1]}"
echo '{"id":"a2","op":"get_temperatures"}' >&"${GO2[1]}"
echo '{"id":"a3","op":"sport","action":"Hello"}' >&"${GO2[1]}"
echo '{"id":"a4","op":"sport","action":"Move","parameter":{"x":0.3,"y":0,"z":0}}' >&"${GO2[1]}"
echo '{"id":"a5","op":"sport","action":"StopMove"}' >&"${GO2[1]}"
echo '{"id":"a6","op":"set_motion_mode","name":"ai"}' >&"${GO2[1]}"
echo '{"id":"a7","op":"set_video","enabled":false}' >&"${GO2[1]}"
echo '{"id":"a8","op":"set_lidar","enabled":false}' >&"${GO2[1]}"
echo '{"id":"a9","op":"exit"}' >&"${GO2[1]}"
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
- `{"op":"exit"}`

## Temperaturas y valores de estado

Para temperatura y sensores, usa:

- `subscribe_profile=core` para recibir `LOW_STATE` y `LF_SPORT_MOD_STATE`
- comando `get_temperatures` para un resumen rapido:
  - `temperature_ntc1`
  - temperaturas de motores
  - `bms_bq_ntc`
  - `bms_mcu_ntc`
  - `imu_temperature`
  - ultimo payload de `gas_sensor`

Para pedir todos los valores disponibles por la libreria en streaming:

1. `{"op":"list_topics"}`
2. `{"op":"subscribe_profile","profile":"all_telemetry"}`

Nota: algunos topics dependen de firmware/modo y pueden no emitir en todos los robots.
