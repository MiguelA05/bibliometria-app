# Bibliometría App

API para extracción de metadatos de artículos académicos usando OpenAlex, la base de datos global más completa de trabajos académicos.

## Características

- **🌍 Base de datos global**: OpenAlex con 200M+ trabajos académicos
- **📊 Metadatos ricos**: Citas, Open Access, afiliaciones, financiación
- **🔬 API REST moderna**: Sin web scraping, datos estructurados
- **📈 Métricas de impacto**: Número de citas, índices de calidad
- **🔓 Información Open Access**: Estado, URLs, licencias
- **🏛️ Datos institucionales**: Afiliaciones, países, ciudades
- **💰 Información de financiación**: Agencias, proyectos
- **📚 Exportación CSV**: Datos estructurados listos para análisis
- **🧪 Pruebas completas**: Tests unitarios e integración

## Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

#### 1. Clonar el repositorio (si aún no lo has hecho)
```bash
git clone <url-del-repositorio>
cd bibliometria-app
```

#### 2. Crear y activar entorno virtual (Recomendado)
```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
# Linux/Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

#### 3. Instalar dependencias
```bash
# Instalar todas las dependencias
pip install -r requirements.txt

# Descargar datos de NLTK (OBLIGATORIO para similitud textual)
python -m nltk.downloader punkt stopwords
```

**Dependencias principales:**
- Framework web: FastAPI, Uvicorn
- Datos: pandas, numpy, requests
- Validación: pydantic, pydantic-settings
- Similitud textual: scikit-learn, nltk, sentence-transformers (opcional)

**Nota:** `sentence-transformers` es opcional pero recomendado para algoritmos de IA.

#### 4. Instalar Ollama (Opcional - para algoritmo LLM-based)

Para usar el algoritmo de similitud basado en LLM (Requerimiento 2), necesitas instalar Ollama:

```bash
# Método automático (recomendado)
bash scripts/install_ollama.sh

# O manualmente
curl -fsSL https://ollama.com/install.sh | sh
ollama serve  # En otra terminal
ollama pull llama3.2:3b  # Descargar modelo
```

**Nota:** Si Ollama no está instalado, el algoritmo LLM-based usará un modo simulado como fallback.

Ver documentación completa: [docs/OLLAMA_SETUP.md](docs/OLLAMA_SETUP.md)

#### 5. Configurar entorno
```bash
# Crear archivo de configuración desde ejemplo
cp env.example .env

# El archivo .env se crea automáticamente si no existe al ejecutar start.py
```

#### 6. Verificar instalación
```bash
# El script start.py verifica automáticamente las dependencias
python start.py
```

### 📖 Guía de Instalación Detallada

Para una guía completa con solución de problemas, ver: [docs/INSTALACION.md](docs/INSTALACION.md)

### Ejecutar la aplicación

```bash
# Opción 1: Script de inicio (recomendado)
python start.py

# Opción 2: Comando directo
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Opción 3: Con configuración personalizada
python start.py --host 127.0.0.1 --port 8080 --reload
```

## Uso

### 🎯 Menú Interactivo (Recomendado)

Para cumplir con los requerimientos del proyecto, se ha creado un menú interactivo que permite:

1. **Probar Web Scraping y Generar Resultados** (Requerimiento 1)
   - Ejecutar proceso completo de automatización
   - Descargar datos de múltiples bases de datos (OpenAlex, PubMed, ArXiv)
   - Unificar información en un solo archivo
   - Eliminar duplicados automáticamente
   - Generar archivos: unificado, duplicados y reportes

2. **Evaluar Algoritmos de Similitud Textual** (Requerimiento 2)
   - Seleccionar archivo CSV unificado
   - Elegir 2 o más artículos para comparar
   - Ejecutar 6 algoritmos de similitud:
     - 4 algoritmos clásicos: Levenshtein, Damerau-Levenshtein, Jaccard, TF-IDF
     - 2 algoritmos de IA: Sentence-BERT, LLM-based
   - Ver explicación detallada paso a paso de cada algoritmo

**Ejecutar el menú:**
```bash
python menu.py
```

El menú guiará paso a paso a través de todas las funcionalidades con explicaciones detalladas.

### API Endpoints

#### Endpoint principal

```bash
POST /api/v1/fetch-metadata
```

### Ejemplo de uso

```bash
curl -X POST http://127.0.0.1:8000/api/v1/fetch-metadata \
     -H "Content-Type: application/json" \
     -d '{
       "query": "machine learning",
       "max_articles": 10,
       "email": "tu@email.com",
       "filters": {
         "publication_year": "2024",
         "type": "journal-article"
       }
     }'
```

### Scripts de prueba

```bash
# Probar API
python test_api.py

# Pruebas unitarias
python -m pytest tests/ -v

# Solo ejecutar pruebas
python start.py --test

# Verificar salud del sistema
curl http://localhost:8000/health

# Ver métricas de rendimiento
curl http://localhost:8000/metrics
```

## Documentación

- **[Guía de Instalación](docs/INSTALACION.md)** - Instrucciones detalladas de instalación
- **[Dependencias Completas](docs/DEPENDENCIAS_COMPLETAS.md)** - Lista completa de dependencias
- [README de Documentación](docs/README.md) - Índice de toda la documentación técnica

## Estructura del proyecto

```
app/
├── api/
│   └── endpoints.py              # Endpoints de la API
├── models/
│   └── article.py                # Modelos de datos para OpenAlex
├── services/
│   └── openalex_service.py       # Servicio OpenAlex
├── utils/
│   ├── logger.py                 # Sistema de logging estructurado
│   ├── validators.py             # Validación robusta de entrada
│   ├── exceptions.py             # Manejo de errores
│   ├── metrics.py                # Métricas de rendimiento
│   └── cache.py                  # Sistema de caché
├── config.py                     # Configuración de la aplicación
└── main.py                       # Aplicación principal

tests/
└── test_openalex_service.py      # Pruebas del servicio OpenAlex

results/                          # Archivos CSV generados
requirements.txt                  # Dependencias del proyecto
env.example                       # Ejemplo de configuración
start.py                          # Script de inicio mejorado
```

## Ventajas de OpenAlex

| Característica | OpenAlex |
|----------------|----------|
| **🌍 Cobertura** | ✅ Global (200M+ trabajos) |
| **⚡ Velocidad** | ✅ Rápido (1-2 segundos) |
| **📊 Metadatos** | ✅ Muy ricos y estructurados |
| **📈 Métricas** | ✅ Citas, impacto, calidad |
| **🔓 Open Access** | ✅ Información completa |
| **🏛️ Instituciones** | ✅ Afiliaciones detalladas |
| **💰 Financiación** | ✅ Datos de financiación |
| **🔧 Mantenimiento** | ✅ Bajo (API estable) |
| **🌐 Dependencia** | ✅ API REST confiable |
| **📚 Filtros** | ✅ Filtros avanzados |

## Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT.
