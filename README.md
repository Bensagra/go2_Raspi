# Go2 Raspi

Control y telemetría para un Unitree Go2 con una Raspberry Pi como gateway y
una segunda máquina como servidor. El proyecto ahora se inicia siempre desde
un único comando: `./go2`.

## Inicio rápido

En cada máquina, desde la raíz del proyecto:

```bash
./go2 init
nano config/.env
./go2 install full
```

En la máquina **servidor**:

```bash
mosquitto -p 1883          # terminal 1
./go2 server               # terminal 2
./go2 dashboard            # terminal 3
```

Luego abre `http://IP_DEL_SERVIDOR:8080/frontend_dashboard.html`.

En la **Raspberry Pi conectada al Go2**:

```bash
./go2 edge
```

Antes de arrancar, revisa en `config/.env` estas tres direcciones:

- `GO2_IP`: IP local del robot, normalmente `192.168.123.161`.
- `GO2_SERVER_PUBLIC_HOST`: IP del servidor visible desde la Raspberry.
- `GO2_EDGE_MQTT_HOST`: IP de la máquina que ejecuta Mosquitto.

El token `GO2_EDGE_MEDIA_TOKEN` debe ser idéntico en ambas máquinas.

## Qué ejecuta cada parte

```text
Go2 ──WebRTC──> edge/ (Raspberry) ──MQTT/media──> server/
                                                      │
                                                      └──> frontend/ (navegador)
```

- `./go2 edge`: conecta cámara, audio, LiDAR y control local del robot.
- `./go2 server`: recibe los datos, valida comandos y expone la API.
- `./go2 dashboard`: sirve la interfaz web en el puerto 8080.
- `./go2 doctor`: avisa si falta configuración o alguna dependencia.
- `./go2 test`: ejecuta las pruebas; requiere `./go2 install dev`.
- `./go2 help`: lista todos los comandos y acepta opciones avanzadas.

Las opciones escritas después del subcomando llegan al programa original. Por
ejemplo, `./go2 server --port 9000` cambia el puerto sin editar código.

## Estructura

```text
go2                     lanzador único
config/                 configuración local de red y tokens
edge/                   procesos de la Raspberry Pi
server/                 API, autonomía, percepción y reconstrucción 3D
frontend/               dashboard web
lidar3d/                protocolo compartido del LiDAR independiente
assets/audio/           audios que reproduce el robot
models/                 pesos descargados (no se guardan en Git)
requirements/           dependencias por tipo de instalación
scripts/                instalación como servicio del sistema
tools/camera/            herramientas de cámara para diagnóstico
tools/ssh/               conexión SSH anterior, mantenida como alternativa
docs/                    documentación técnica detallada
tests/                   pruebas automáticas
third_party/             proyectos externos descargados
```

Los archivos generados quedan fuera del código: logs en `logs/`, auditoría en
`server/audit/`, caras en `server/faces/`, mapas en `server/maps/` y resultados
LiDAR en `server/lidar_models/`.

## Configuración

`./go2 init` crea `config/.env` a partir de `config/.env.example`. El archivo
local está ignorado por Git para no publicar IP, usuarios ni tokens. Puedes usar
otra configuración sin copiar archivos:

```bash
GO2_CONFIG_FILE=/ruta/robot.env ./go2 edge
```

Los valores de `.env` cubren el uso normal. Los argumentos de cada Python siguen
disponibles para ajustes finos:

```bash
./go2 edge --camera-target-fps 20 --lidar-media-hz 1
./go2 server --perception-device cpu
```

## LiDAR 3D independiente

Este modo no necesita levantar el sistema completo y usa entornos separados
porque sus dependencias de IA pueden requerir otra versión de NumPy.

En el servidor:

```bash
./go2 install lidar-server
./go2 lidar-server
```

En la Raspberry:

```bash
./go2 install lidar-raspi
./go2 lidar-edge
```

La preparación de AdaPoinTr, pesos y exportación se explica en
[`docs/LIDAR_ADAPOINTR.md`](docs/LIDAR_ADAPOINTR.md).

## Inicio automático de la Raspberry

Cuando `./go2 edge` ya funcione manualmente:

```bash
sudo ./go2 service-install
journalctl -u go2-edge -f
```

El servicio usa la ruta real del repositorio y `config/.env`; ya no contiene un
usuario, una IP o un directorio escritos a mano.

## Documentación adicional

- [Arquitectura recomendada de tres capas](docs/THREE_LAYER_CONNECTION.md)
- [Pipeline y streaming](docs/SSH_STREAMING.md)
- [Gateway SSH legado](docs/GO2_SSH_GATEWAY.md)
- [Notas del dashboard](docs/DASHBOARD_UPDATES.md)

Para operar el robot, comienza por esta página y por `./go2 help`. Los documentos
de `docs/` quedan como referencia para configuración avanzada e implementación.
