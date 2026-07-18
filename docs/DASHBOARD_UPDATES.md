# Dashboard Updates - Cámara, Lidar, Audio & Velocidad

## Cambios Implementados ✓

### 1. **Selector de Velocidad Simplificado (1-100)**
- **Antes**: Presets (Turtle/Normal/Sport) + escala separada (20-140%)
- **Ahora**: Slider único **1-100**
  - `1-25`: Lento (Turtle-like)
  - `25-50`: Medio-Lento → Normal
  - `50-75`: Normal → Rápido
  - `75-100`: Rápido (Sport-like)
- **Shift**: Boost x1.3 activado (igual que antes)

### 2. **Habilitación Automática de Media (Camera, Lidar, Audio)**
Ahora el dashboard:
- ✓ Habilita automáticamente **Video, Lidar y Audio** al conectar
- ✓ Los checkboxes están **checked** por defecto
- ✓ Si **cambias** un checkbox → se **aplica automáticamente**
- ✓ Botón **"✓ Aplicar media"** es ahora más visible (verde/primario)

### 3. **Mejoras en Configuración de Media**
```javascript
// Antes: timeout 1600ms
// Ahora: timeout 2000ms + mejor logging

await sendCommand(cmd.type, cmd.payload, 2000);
addEvent(`media_${cmd.type}_ok`, { enabled: cmd.payload.enabled });
```

### 4. **Mejor Timing en Conexión**
```javascript
// Ahora espera 200ms entre loadCapabilities y applyMediaSettings
await loadCapabilities();
await new Promise(resolve => setTimeout(resolve, 200));
await applyMediaSettings();
```

### 5. **Event Listeners en Checkboxes**
```javascript
// Cambiar checkbox video/lidar/audio → aplica automáticamente
els.enableVideo.addEventListener("change", async () => {
  if (state.connected) await applyMediaSettings();
});
```

## Cómo Usar

### **Velocidad**
- Mueve el slider izquierda (lento) o derecha (rápido)
- Usa **Shift + WASD** para boost temporal x1.3

### **Cámara/Lidar/Audio**
1. **Al conectar**: Se habilitan automáticamente ✓
2. **Para cambiar**: 
   - Marca/desmarca el checkbox
   - Se aplica automáticamente
3. **O usa el botón**: "✓ Aplicar media"

### **Teclado** (igual que antes)
- `WASD` o flechas: mover/girar
- `Q/E` o `Z/C`: lateral
- `Space`: stop
- `L`: toggle lidar
- `P`: toggle audio
- `Shift`: boost velocidad

## Debugging

Si camera/lidar/audio **no abre**:
1. Abre **Consola (F12)**
2. Busca eventos `media_config_error` en la lista
3. Verifica que el servidor esté corriendo
4. Verifica logs en `edge/edge_gateway_service.py` (edge)

Ejemplo error esperado:
```
media_config_error: { cmd: "set_camera_stream", error: "..." }
```

## Cambios en el Código

### Removido:
- `SPEED_PRESETS` (turtle, normal, sport)
- `state.speedPreset` y `state.speedScale`

### Agregado:
- `state.speedValue` (1-100)
- `els.speedSlider` (input range 1-100)
- Event listeners en checkboxes para auto-aplicar

### Actualizado:
- `currentPreset()` → calcula velocidad desde `speedValue`
- `applyMediaSettings()` → mejor timing + logging
- `ws.onopen()` → espera 200ms antes de applyMediaSettings
