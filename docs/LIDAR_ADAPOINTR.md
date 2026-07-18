# Pipeline aislado Go2 LiDAR → TULIP → AdaPoinTr → modelo 3D

Este flujo es independiente de cámara, audio, MQTT, autonomía y dashboard. Usa
exclusivamente `unitree-webrtc-connect` para hablar con el Go2; no importa, instala
ni utiliza el SDK oficial de Unitree:

```text
Go2 (WebRTC LocalSTA, rt/utlidar/*)
        │
        ▼
Raspberry: edge/lidar_only_sender.py
        │  WebSocket binario, último frame primero
        ▼
Servidor: server/lidar_3d_server.py
        ├─ nube medida acumulada (siempre preservada)
        ├─ TULIP opcional/experimental
        ├─ AdaPoinTr opcional
        └─ malla Poisson Open3D
```

No usa SSH para transportar LiDAR. La Raspberry abre un WebSocket saliente al
servidor, igual que el canal pesado del sistema de tres capas actual, pero el nuevo
proceso solo contiene LiDAR.

## Qué dato entrega realmente Unitree

`unitree-webrtc-connect` se conecta al Go2 con `LocalSTA`, desactiva el modo de
ahorro de tráfico, enciende `rt/utlidar/switch` y se suscribe a:

- `rt/utlidar/voxel_map_compressed`: mapa voxel LZ4 en el frame de odometría LIO.
- `rt/utlidar/robot_pose`: pose en el mismo frame del mapa.

El decoder `native` convierte la ocupación a puntos XYZ con `origin + voxel *
resolution`. Es el default recomendado en la Raspberry porque evita el runtime
WASM de `libvoxel`; `--decoder libvoxel` queda disponible para comparar firmwares.

La dependencia `unitree-webrtc-connect` agrega una clave AES por dispositivo para
Go2 con firmware `>=1.1.15`. El ejecutable acepta `--aes-key` o, de forma más
segura, `UNITREE_AES_KEY`.

## Límite importante de los modelos

TULIP y AdaPoinTr resuelven problemas distintos:

- **TULIP** aumenta la resolución vertical de una *imagen de rango organizada*.
  El checkpoint KITTI oficial espera `16×1024` y produce `64×1024`. El Go2 no
  expone ese tipo de scan por WebRTC: expone un mapa voxel ya reconstruido. Por
  eso TULIP está apagado por defecto. El adaptador incluido transforma el mapa
  local a geometría KITTI, conserva cada medición original y vuelve al frame de
  odometría; sirve para experimentar, pero necesita fine-tuning con L1 para tener
  fidelidad métrica.
- **AdaPoinTr** completa una nube parcial de un objeto; el camino oficial lleva
  cada muestra a 2048 puntos y devuelve 8192/16384 según el checkpoint. No es un
  algoritmo de SLAM ni un mallador de habitaciones. El servidor puede aplicarlo a
  todo el mapa (`whole`) o a regiones XY (`tiles`), recorta predicciones fuera del
  volumen observado y siempre guarda aparte los datos medidos.
- **Open3D Poisson** crea la malla navegable *después* del completado. Los puntos
  AdaPoinTr son predicciones y no deben usarse como distancias de seguridad.

Para datos LiDAR parciales, el checkpoint AdaPoinTr de
`Projected_ShapeNet-55` es un punto de partida más coherente que PCN, aunque
ningún checkpoint oficial fue entrenado con escenas interiores del Unitree L1.

## 1. Instalar en la Raspberry Pi

La ruta simple crea un entorno virtual separado:

```bash
./go2 install lidar-raspi
```

En Raspberry Pi OS de 64 bits puede ser necesario instalar antes los paquetes
del sistema:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip portaudio19-dev
```

Aunque este ejecutable no usa audio, las versiones actuales de
`unitree-webrtc-connect` declaran PyAudio/sounddevice como dependencias y por eso
pueden necesitar `portaudio19-dev` durante la instalación.

Verificar rutas desde la Raspberry:

```bash
ping -c 2 192.168.123.161       # Go2
ping -c 2 IP_DEL_SERVIDOR       # servidor GPU
```

La Raspberry necesita acceso simultáneo a ambos destinos. Puede ser Go2 y
Raspberry en la misma LAN (STA), o una interfaz hacia el perro y otra hacia el
servidor. No cambiar la ruta por defecto si eso rompe el acceso al servidor.

La IP del servidor se configura como `GO2_LIDAR_SERVER_HOST` en `config/.env` o
se pasa al ejecutable Python mediante `--server-ip`. MQTT y el puerto 8000 no
participan en este ejecutable aislado. Antes de iniciarlo, confirmar desde la
Raspberry que la IP elegida responde con `ping`.

### Firmware reciente y AES

Solo si la conexión informa que necesita una clave:

```bash
unitree-fetch-aes-key \
  --email TU_EMAIL \
  --password 'TU_PASSWORD' \
  --device-type Go2 \
  --sn SERIAL_DEL_GO2 \
  --quiet
```

Guardar el valor de 32 caracteres fuera del repositorio y exportarlo en la
sesión/servicio:

```bash
export UNITREE_AES_KEY='32_HEX_AQUI'
```

## 2. Instalar en el servidor

AdaPoinTr y TULIP fueron publicados con stacks CUDA antiguos. Para aislarlos del
Python del sistema, usar Python 3.10 o 3.11 si los modelos no soportan la versión
instalada:

En el servidor inspeccionado ya existen una RTX 4080 SUPER de 16 GB, el entorno
Conda `tulip` (Python 3.10, PyTorch 2.1.1+cu118) y los checkpoints en
`/home/perrobot/TULIP/trained/`. El adaptador se validó realmente con
`tulip_kitti.pth`. Ese entorno todavía necesita las dependencias del nuevo
servidor (`websockets`, `easydict`, `open3d`, etc.).

```bash
./go2 install lidar-server
```

Si hace falta una build CUDA específica, instalar el wheel PyTorch que coincida
con el driver del servidor dentro de `.venv-lidar-server`, siguiendo
<https://pytorch.org/get-started/locally/>. Después preparar los repositorios:

```bash
python -m pip install -r requirements/lidar-server.txt
mkdir -p third_party models
test -d third_party/PoinTr/.git || \
  git clone --depth 1 https://github.com/yuxumin/PoinTr.git third_party/PoinTr
```

Descargar el checkpoint **AdaPoinTr Projected_ShapeNet-55** publicado en la tabla
`Pretrained Models` de PoinTr:

```bash
test -f models/AdaPoinTr_Projected_ShapeNet55.pth || \
  curl --fail --location --retry 3 \
  'https://cloud.tsinghua.edu.cn/f/41ed3a765c4b42d98d01/?dl=1' \
  --output models/AdaPoinTr_Projected_ShapeNet55.pth
```

En el servidor actual ya están instalados el commit PoinTr
`4603257ed3db9e7dad349b712e1b2fe0da207015` y el checkpoint con SHA-256
`807ef42b3649f22dfc4cbc7ff7632d2ba593dcaa5bd93433c6137555fe3280a7`.

Comprobar la instalación con inferencia CUDA real:

```bash
python server/adapointr_self_test.py \
  --output /tmp/adapointr_self_test.npz
```

El adaptador carga únicamente `models/AdaPoinTr.py`, no los baselines que el
repositorio importa por defecto. Además incluye implementaciones PyTorch de las
cuatro operaciones PointNet2 necesarias y un Chamfer de solo inferencia. Si
`pointnet2_ops` compilado está instalado, lo usa automáticamente; no hace falta
compilar las extensiones de entrenamiento para ejecutar este pipeline.

### TULIP opcional

```bash
git clone https://github.com/ethz-asl/TULIP.git third_party/TULIP
```

Descargar el checkpoint **KITTI** del enlace `pretrained models` del repositorio
TULIP y guardarlo como `models/tulip_kitti.pth`. No habilitarlo inicialmente:
primero validar que la nube Unitree observada sea correcta.

## 3. Arranque recomendado

Usar el mismo token en las dos máquinas. No dejarlo vacío fuera de una red de
laboratorio:

```bash
export GO2_LIDAR_TOKEN='CAMBIAR_POR_UN_SECRETO_LARGO'
```

### Servidor con AdaPoinTr y malla

Desde la raíz de este repositorio:

```bash
./go2 lidar-server \
  --output-dir ./server/lidar_models \
  --device cuda:0 \
  --adapointr-repo ./third_party/PoinTr \
  --adapointr-config ./third_party/PoinTr/cfgs/Projected_ShapeNet55_models/AdaPoinTr.yaml \
  --adapointr-checkpoint ./models/AdaPoinTr_Projected_ShapeNet55.pth \
  --adapointr-mode whole \
  --mesh
```

Para un mapa grande se puede probar `--adapointr-mode tiles`; procesa por defecto
como máximo las ocho regiones de 3 m con más datos. Es más costoso y más
experimental que `whole`.

Para comprobar primero transporte y mapa sin ML ni Open3D:

```bash
./go2 lidar-server \
  --output-dir ./server/lidar_models \
  --no-mesh
```

### Raspberry

```bash
./go2 lidar-edge \
  --decoder native \
  --send-hz 1 \
  --max-points 50000
```

Si la Raspberry se conecta directamente al AP del Go2, usar
`--connection local-ap`; en ese modo la librería fija la IP del robot a
`192.168.12.1`.

### Agregar TULIP, después de validar el flujo base

Agregar al comando del servidor:

```bash
  --tulip-repo /home/perrobot/TULIP \
  --tulip-checkpoint /home/perrobot/TULIP/trained/tulip_kitti.pth
```

TULIP necesita `robot_pose`. Si ese topic no llega, el servidor lo omite, registra
el motivo en `latest.json` y continúa con la nube observada/AdaPoinTr.

## Archivos producidos

Cada `--reconstruct-interval` (30 s por defecto), y una vez más al apagar el
servidor limpiamente:

```text
server/lidar_models/go2_01/
├── latest.json             # conteos, bounds, tiempos, errores y etapas activas
├── observed/
│   ├── latest.npz          # fuente métrica: puntos Unitree acumulados
│   └── latest.ply
├── tulip/                  # solo si TULIP terminó correctamente
│   ├── latest.npz
│   └── latest.ply
├── completed/              # observado + predicción AdaPoinTr
│   ├── latest.npz
│   └── latest.ply
└── mesh/
    └── latest.ply          # malla triangular Open3D/Poisson
```

Los reemplazos de archivos son atómicos: un visor nunca debería abrir una salida
escrita a medias.

## Ajustes útiles

- `--map-voxel-size 0.05`: coincide con la resolución habitual del mapa Unitree.
- `--map-max-voxels 500000`: límite duro de RAM del acumulador.
- `--minimum-z/--maximum-z`: recorte opcional en el frame LIO; configurarlo solo
  después de observar bounds reales.
- `--completed-voxel-size 0.025`: densidad de la nube final.
- `--mesh-poisson-depth 8`: subir aumenta memoria/tiempo de Open3D.
- `--reconstruct-interval`: AdaPoinTr/TULIP no deben correr a la frecuencia del
  sensor; la Raspberry continúa enviando el mapa más reciente mientras infieren.

## Validación

Tests del protocolo, incluidos límites de descompresión:

```bash
python -m pytest tests/test_lidar3d_protocol.py -q
```

Antes de confiar en el resultado:

1. Abrir `observed/latest.ply` y confirmar ejes/escala/pose.
2. Comparar `completed/latest.ply` sin TULIP.
3. Recién entonces activar TULIP y comparar `tulip/latest.ply`.
4. Para navegación o anti-choque usar exclusivamente `observed`, nunca
   `completed` ni `mesh`.

## Fuentes upstream

- Única conexión usada con el Go2, Unitree WebRTC Connect:
  <https://github.com/legion1581/unitree_webrtc_connect>
- TULIP oficial: <https://github.com/ethz-asl/TULIP>
- AdaPoinTr en el repositorio PoinTr oficial: <https://github.com/yuxumin/PoinTr>
