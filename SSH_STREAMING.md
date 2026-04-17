# Streaming Go2 por SSH (modo recomendado)

## Idea

No usar texto/base64 para video continuo.

Usar un stream binario:

- Emisor local: captura del Go2 y escribe paquetes binarios a stdout.
- SSH: transporta ese stdout al servidor.
- Receptor remoto: lee stdin, decodifica y procesa.

## Scripts

- `camera_ssh_sender.py`: corre en la maquina que esta junto al robot.
- `camera_ssh_receiver.py`: corre en el servidor remoto.

## Comando recomendado (local -> servidor)

Ejecutar en la maquina local (la que ve al Go2):

```bash
/home/bensagra/Documents/go2/.venv/bin/python -u /home/bensagra/Documents/go2/camera_ssh_sender.py \
  --ip 192.168.123.161 \
  --emit-every 1 \
  --image-format jpg \
  --jpeg-quality 75 \
| ssh -T -o Compression=no user@tu-servidor \
  "python3 /ruta/en/servidor/camera_ssh_receiver.py --stats-every 30"
```

## Si quieres metadata JSON en servidor

```bash
/home/bensagra/Documents/go2/.venv/bin/python -u /home/bensagra/Documents/go2/camera_ssh_sender.py \
  --ip 192.168.123.161 --image-format jpg --jpeg-quality 75 \
| ssh -T -o Compression=no user@tu-servidor \
  "python3 /ruta/en/servidor/camera_ssh_receiver.py --stdout-json > /tmp/go2_frames_meta.jsonl"
```

## Si quieres ver imagen en servidor

```bash
/home/bensagra/Documents/go2/.venv/bin/python -u /home/bensagra/Documents/go2/camera_ssh_sender.py \
  --ip 192.168.123.161 \
| ssh -T -Y -o Compression=no user@tu-servidor \
  "python3 /ruta/en/servidor/camera_ssh_receiver.py --show"
```

## Tuning rapido

- Menos ancho de banda: bajar `--jpeg-quality` (ej. 60-70).
- Menos CPU local: subir `--emit-every` (ej. 2 o 3).
- Menos latencia: mantener `--emit-every 1` y `-u` en python.
- SSH: para JPEG suele rendir mejor `-o Compression=no`.
