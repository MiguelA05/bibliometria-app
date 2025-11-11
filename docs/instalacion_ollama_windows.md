# Instalación de Ollama en Windows

Esta guía explica cómo instalar Ollama y configurar los modelos necesarios para la Bibliometría App en Windows.

## Opción 1: Script PowerShell (Recomendado)

### Requisitos
- Windows 10 o superior
- PowerShell 5.1 o superior (incluido por defecto en Windows)

### Pasos

1. **Abrir PowerShell como Administrador** (opcional, pero recomendado):
   - Presiona `Win + X`
   - Selecciona "Windows PowerShell (Administrador)" o "Terminal (Administrador)"

2. **Ejecutar el script**:
   ```powershell
   cd ruta\a\bibliometria-app
   powershell -ExecutionPolicy Bypass -File scripts/install_ollama.ps1
   ```

   O si ya estás en el directorio del proyecto:
   ```powershell
   .\scripts\install_ollama.ps1
   ```

3. **Seguir las instrucciones**:
   - Si Ollama no está instalado, el script te guiará para descargarlo
   - El script descargará automáticamente el modelo `llama3.2:3b`
   - Opcionalmente puedes descargar `mistral:7b` para mejor calidad

### Solución de problemas

Si obtienes un error de "Execution Policy":
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Luego ejecuta el script nuevamente.

---

## Opción 2: Script Batch (.bat)

### Pasos

1. **Navegar al directorio del proyecto**:
   ```cmd
   cd ruta\a\bibliometria-app
   ```

2. **Ejecutar el script**:
   - Hacer doble clic en `scripts/install_ollama.bat`
   - O desde la línea de comandos: `scripts\install_ollama.bat`

3. **Seguir las instrucciones** en pantalla

---

## Opción 3: Instalación Manual

### Paso 1: Descargar e Instalar Ollama

1. Visita: https://ollama.com/download
2. Descarga `OllamaSetup.exe`
3. Ejecuta el instalador y sigue las instrucciones
4. Ollama se instalará y se iniciará automáticamente como servicio

### Paso 2: Verificar Instalación

Abre PowerShell o CMD y ejecuta:
```cmd
ollama --version
```

Deberías ver algo como: `ollama version is 0.x.x`

### Paso 3: Descargar Modelos

Descarga el modelo recomendado (ligero y rápido):
```cmd
ollama pull llama3.2:3b
```

Opcional: Descarga un modelo más potente (mejor calidad):
```cmd
ollama pull mistral:7b
```

### Paso 4: Verificar Modelos

Lista los modelos descargados:
```cmd
ollama list
```

---

## Verificar que Todo Funciona

### Verificar que el servidor está corriendo

Abre un navegador y visita: http://localhost:11434/api/tags

Deberías ver un JSON con los modelos disponibles.

### Probar desde Python

```python
import requests

response = requests.get("http://localhost:11434/api/tags")
print(response.json())
```

---

## Uso en la Aplicación

Una vez instalado Ollama y descargados los modelos, la aplicación los usará automáticamente para el algoritmo LLM-based de similitud textual.

El modelo por defecto es `llama3.2:3b`. Si deseas cambiar el modelo, edita el archivo de configuración o modifica el código en `app/utils/ollama_helper.py`.

---

## Solución de Problemas Comunes

### Error: "ollama no se reconoce como comando"

**Solución**: 
- Asegúrate de que Ollama esté instalado
- Reinicia la terminal después de instalar Ollama
- Verifica que Ollama esté en el PATH del sistema

### Error: "Connection refused" o "No se puede conectar al servidor"

**Solución**:
1. Verifica que el servicio de Ollama esté corriendo:
   ```cmd
   tasklist | findstr ollama
   ```

2. Si no está corriendo, inícialo manualmente:
   - Busca "Ollama" en el menú de inicio y ábrelo
   - O ejecuta: `ollama serve`

3. Verifica que el puerto 11434 no esté bloqueado por el firewall

### Error: "Model not found"

**Solución**:
- Asegúrate de haber descargado el modelo:
  ```cmd
  ollama pull llama3.2:3b
  ```

- Verifica que el modelo esté disponible:
  ```cmd
  ollama list
  ```

### El servidor se detiene automáticamente

**Solución**:
- En Windows, Ollama debería ejecutarse como servicio automáticamente
- Si se detiene, verifica los logs de Windows Event Viewer
- Considera ejecutar `ollama serve` en una ventana de terminal separada

---

## Notas Adicionales

- **Memoria**: Los modelos de LLM requieren RAM. `llama3.2:3b` necesita aproximadamente 2-3 GB, mientras que `mistral:7b` necesita 4-5 GB.

- **Rendimiento**: En Windows, el rendimiento puede variar según tu hardware. Se recomienda tener al menos 8 GB de RAM total.

- **Primera ejecución**: La primera vez que uses un modelo, puede tardar más en cargarse en memoria.

- **Sin Ollama**: Si Ollama no está instalado, la aplicación funcionará normalmente pero el algoritmo LLM-based no estará disponible (los otros 5 algoritmos funcionarán sin problemas).

---

## Referencias

- Documentación oficial de Ollama: https://ollama.com/docs
- Repositorio de modelos: https://ollama.com/library

