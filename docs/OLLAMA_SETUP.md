# Configuración de Ollama para Análisis LLM

Este documento explica cómo instalar y configurar Ollama para usar el algoritmo de similitud textual basado en LLM (Large Language Model).

## ¿Qué es Ollama?

Ollama es una herramienta que permite ejecutar modelos de lenguaje grandes (LLM) localmente en tu máquina, sin necesidad de conexión a servicios en la nube.

## Instalación

### Método 1: Script Automatizado (Recomendado)

Ejecuta el script de instalación incluido:

```bash
bash scripts/install_ollama.sh
```

Este script:
1. Verifica si Ollama está instalado
2. Instala Ollama si no está presente
3. Inicia el servidor Ollama
4. Descarga el modelo Llama 3.2 3B (y opcionalmente Mistral 7B)

### Método 2: Instalación Manual

1. **Instalar Ollama:**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Iniciar el servidor:**
   ```bash
   ollama serve
   ```
   (En otra terminal o en background)

3. **Descargar un modelo:**
   ```bash
   # Modelo ligero y rápido (recomendado)
   ollama pull llama3.2:3b
   
   # O modelo más potente (requiere más recursos)
   ollama pull mistral:7b
   ```

## Verificación

Para verificar que todo está funcionando correctamente:

```bash
# El test de Ollama se puede ejecutar desde el menú principal
python menu.py
```

Este script verifica:
- ✅ Instalación de Ollama
- ✅ Servidor corriendo
- ✅ Modelos disponibles
- ✅ Funcionamiento del análisis de similitud

## Uso en el Proyecto

Una vez instalado, el algoritmo LLM-based Similarity se activará automáticamente cuando ejecutes el menú:

```bash
python menu.py
```

El menú verificará automáticamente si Ollama está disponible y lo iniciará si es necesario.

## Modelos Disponibles

### Llama 3.2 3B (Recomendado)
- **Tamaño:** ~2GB
- **Velocidad:** Rápida
- **Calidad:** Buena para análisis de similitud
- **Uso:** Ideal para la mayoría de casos

### Mistral 7B
- **Tamaño:** ~4GB
- **Velocidad:** Media
- **Calidad:** Excelente
- **Uso:** Cuando necesitas mayor precisión

## Solución de Problemas

### Error: "Ollama no está instalado"
```bash
bash scripts/install_ollama.sh
```

### Error: "Servidor Ollama no está corriendo"
```bash
# Iniciar servidor manualmente
ollama serve

# O verificar si ya está corriendo
ps aux | grep ollama
```

### Error: "Modelo no encontrado"
```bash
# Listar modelos disponibles
ollama list

# Descargar modelo específico
ollama pull llama3.2:3b
```

### El análisis es muy lento
- Usa el modelo más ligero: `llama3.2:3b`
- Asegúrate de tener suficiente RAM (mínimo 4GB libres)
- Cierra otras aplicaciones que consuman recursos

### El análisis falla
- Verifica que el servidor esté corriendo: `curl http://localhost:11434/api/tags`
- Revisa los logs del servidor Ollama
- Intenta reiniciar el servidor: `pkill ollama && ollama serve`

## Configuración Avanzada

### Cambiar el modelo usado

Edita `app/services/text_similarity_service.py` y modifica el parámetro `ollama_model` en el constructor:

```python
def __init__(self, ollama_model: str = "mistral:7b"):  # Cambiar aquí
    ...
```

### Ajustar parámetros del modelo

En `app/utils/ollama_helper.py`, función `analyze_similarity_with_llm`, puedes ajustar:

- `temperature`: Controla la creatividad (0.0-1.0, default: 0.3)
- `max_tokens`: Máximo de tokens generados (default: 200)

## Notas Importantes

1. **Primera ejecución:** La primera vez que uses un modelo, Ollama lo descargará automáticamente (puede tardar varios minutos).

2. **Recursos:** Los modelos LLM requieren RAM. Asegúrate de tener al menos 4GB libres para Llama 3.2 3B y 8GB para Mistral 7B.

3. **Modo Fallback:** Si Ollama no está disponible, el algoritmo usará un modo simulado que no requiere LLM.

4. **Inicio Automático:** El menú intentará iniciar Ollama automáticamente si detecta que no está corriendo.

## Referencias

- [Documentación oficial de Ollama](https://ollama.com/docs)
- [Modelos disponibles](https://ollama.com/library)

