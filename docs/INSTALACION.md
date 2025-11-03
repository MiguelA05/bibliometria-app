# 📦 Guía de Instalación - Bibliometría App

Esta guía te ayudará a instalar todas las dependencias necesarias para ejecutar el proyecto.

## 🚀 Instalación Rápida

### Paso 1: Crear Entorno Virtual (Recomendado)

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
# En Linux/Mac:
source venv/bin/activate

# En Windows:
venv\Scripts\activate
```

### Paso 2: Instalar Dependencias

```bash
# Instalar todas las dependencias
pip install -r requirements.txt

# Descargar datos de NLTK (OBLIGATORIO)
python -m nltk.downloader punkt stopwords
```

### Paso 3: Configurar Variables de Entorno

```bash
# Copiar archivo de ejemplo
cp env.example .env

# Editar .env si es necesario (usualmente no es necesario)
```

### Paso 4: Verificar Instalación

```bash
# Ejecutar script de inicio (verifica dependencias automáticamente)
python start.py
```

## 📋 Dependencias Detalladas

### Dependencias Principales (Obligatorias)

| Paquete | Versión | Uso |
|---------|---------|-----|
| `fastapi` | >=0.104.0 | Framework web |
| `uvicorn` | >=0.24.0 | Servidor ASGI |
| `pandas` | >=2.0.0 | Manejo de datos CSV |
| `numpy` | >=1.24.0 | Operaciones numéricas |
| `requests` | >=2.31.0 | Peticiones HTTP |
| `pydantic` | >=2.5.0 | Validación de datos |
| `pydantic-settings` | >=2.1.0 | Configuración |
| `structlog` | >=23.2.0 | Logging estructurado |

### Dependencias para Similitud Textual

| Paquete | Versión | Uso | Obligatorio |
|---------|---------|-----|-------------|
| `scikit-learn` | >=1.3.0 | TF-IDF Cosine Similarity | ✅ Sí |
| `nltk` | >=3.8.0 | Preprocesamiento de texto | ✅ Sí |
| `sentence-transformers` | >=2.2.0 | Sentence-BERT (embeddings) | ⚠️ Opcional |

**Nota sobre sentence-transformers:**
- Es opcional pero altamente recomendado
- Descarga ~100MB de modelos la primera vez
- Si no está instalado, el algoritmo Sentence-BERT no funcionará

### Dependencias Opcionales

| Paquete | Versión | Uso |
|---------|---------|-----|
| `redis` | >=5.0.0 | Cache (si habilitas caching) |
| `pytest` | >=7.4.0 | Testing |
| `pytest-asyncio` | >=0.21.0 | Testing asíncrono |

## 🔧 Instalación por Categorías

### Instalación Básica (Sin IA)

Si no necesitas los algoritmos de similitud basados en IA:

```bash
pip install fastapi uvicorn pandas numpy requests \
            pydantic pydantic-settings structlog \
            scikit-learn nltk python-dotenv

python -m nltk.downloader punkt stopwords
```

### Instalación Completa (Con IA)

Para tener todas las funcionalidades incluyendo Sentence-BERT:

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords
```

### Instalación Mínima (Solo API)

Solo para usar la API sin similitud textual:

```bash
pip install fastapi uvicorn pandas requests \
            pydantic pydantic-settings structlog \
            python-dotenv
```

## ✅ Verificación de Instalación

### Verificar Dependencias Básicas

```bash
python -c "
import fastapi, uvicorn, pandas, numpy, requests, pydantic
print('✅ Dependencias básicas instaladas')
"
```

### Verificar Dependencias de Similitud

```bash
python -c "
import sklearn, nltk
try:
    import sentence_transformers
    print('✅ Todas las dependencias de similitud instaladas')
except ImportError:
    print('⚠️ sentence-transformers no instalado (opcional)')
"
```

### Verificar Datos de NLTK

```bash
python -c "
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    print('✅ Datos de NLTK descargados')
except LookupError:
    print('❌ Ejecuta: python -m nltk.downloader punkt stopwords')
"
```

## 🐛 Solución de Problemas

### Error: "No module named 'nltk'"

```bash
pip install nltk
python -m nltk.downloader punkt stopwords
```

### Error: "Resource punkt not found"

```bash
python -m nltk.downloader punkt stopwords
```

### Error: "No module named 'sklearn'"

```bash
pip install scikit-learn
```

### Error: "sentence-transformers download failed"

```bash
# Intenta actualizar pip primero
pip install --upgrade pip
pip install sentence-transformers

# Si falla, instala sin caché
pip install --no-cache-dir sentence-transformers
```

### Error: "redis connection refused"

Redis es opcional. Si no necesitas cache, simplemente desactívalo en `.env`:

```
CACHE_ENABLED=false
```

## 🎯 Próximos Pasos

Una vez instaladas las dependencias:

1. **Configurar el entorno:**
   ```bash
   cp env.example .env
   ```

2. **Iniciar el servidor:**
   ```bash
   python start.py
   ```

3. **Probar la instalación:**
   ```bash
   # Verificar salud del sistema
   curl http://localhost:8000/health
   
   # Ejecutar tests
   python -m pytest tests/ -v
   ```

## 📚 Referencias

- [Documentación de dependencias completas](docs/DEPENDENCIAS_COMPLETAS.md)
- [Guía de pruebas de similitud textual](tests/test_text_similarity_service.py)

