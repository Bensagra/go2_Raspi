# Recuperar el canal de control en Windows

El WebSocket de video puede funcionar aunque MQTT esté desconectado. El movimiento,
los ACK y la telemetría requieren que la Raspy y Server Core alcancen el mismo broker.
`server_core.py` y `edge_gateway_service.py` son clientes MQTT, no incluyen un broker.

En la revisión del servidor `192.168.1.88`, la API 8000 respondía, MQTT 1883 no era
alcanzable desde la máquina de diagnóstico, no había telemetría ni ACK y quedaban
16 comandos pendientes. Esto identifica el canal a reparar; sin revisar Windows
no distingue un broker apagado de un listener local o un firewall.

## En Windows (máquina 192.168.1.88)

1. Instalar [Mosquitto para Windows](https://mosquitto.org/download/) si aún no está
   instalado. No alcanza con instalar `paho-mqtt` en el entorno Python.
2. Abrir **PowerShell como administrador**, en la raíz de este repositorio.
3. Comprobar si Mosquitto ya está iniciado como servicio:

   ```powershell
   Get-Service mosquitto -ErrorAction SilentlyContinue
   Get-NetTCPConnection -LocalPort 1883 -State Listen -ErrorAction SilentlyContinue
   ```

   Si el servicio está en `Running`, detener **ese servicio** para iniciar la
   instancia con la configuración preparada y evitar dos brokers en el mismo puerto:

   ```powershell
   Stop-Service mosquitto
   ```

4. Permitir el ejecutable de Mosquitto en el puerto 1883, sólo desde la subred local:

   ```powershell
   New-NetFirewallRule -DisplayName "Go2 MQTT LAN" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 1883 -LocalAddress 192.168.1.88 -RemoteAddress LocalSubnet -Program "C:\Program Files\mosquitto\mosquitto.exe" -Profile Any
   ```

   Ejecutar una sola vez. Si el ejecutable está en otro directorio, ajustar la ruta.
5. Iniciar el broker y mantener esa consola abierta:

   ```powershell
   & "C:\Program Files\mosquitto\mosquitto.exe" -c ".\mqtt\mosquitto.windows.conf" -v
   ```

   Debe indicar que abre el puerto 1883 y mostrar conexiones del servidor y la
   Raspy. Si dice que el puerto ya está ocupado, revisar el proceso que lo ocupa
   antes de arrancar otra instancia.

La configuración incluida está limitada a la IP LAN indicada y mantiene el modo
sin credenciales de los comandos actuales. Es para esta red local de desarrollo;
no publicar el puerto en Internet. Para una instalación compartida, configurar
usuarios/ACL del broker y pasar las credenciales MQTT a ambos clientes.

La forma antigua `mosquitto -p 1883` sólo acepta conexiones locales en Mosquitto 2.x;
se necesita un `listener` explícito para la Raspy.
[Documentación oficial](https://mosquitto.org/documentation/migrating-to-2-0/).

## Raspy, servidor y dashboard

Conservar `--mqtt-host 192.168.1.88 --mqtt-port 1883` en los dos comandos actuales.
El cliente reintenta la conexión MQTT automáticamente. Luego de recuperar el broker,
reconectar el dashboard para volver a aplicar la configuración de cámara/LiDAR.

Para verificar desde la Raspy sin enviar movimientos:

```bash
python -c "import socket; s=socket.create_connection(('192.168.1.88',1883),3); print('Puerto MQTT accesible'); s.close()"
```

La telemetría debe dejar de ser `{}` y los comandos deben recibir `command_ack`.
Con el código actualizado, el dashboard informa separadamente:

- Broker MQTT desconectado.
- Broker conectado pero sin telemetría reciente de la Raspy.
- Raspy conectada pero sin conexión al Go2.
- Canal de control conectado.

`GET /health` agrega `mqtt_connected`; `GET /api/robots/go2_01/state` agrega
`control_link`. Un HTTP 200 de `/health` sólo confirma que la API está viva.
Si MQTT está desconectado, los comandos ahora devuelven HTTP 503 y el control por
WebSocket devuelve un error visible. No se modificaron los límites de velocidad,
el watchdog ni la protección LiDAR.
