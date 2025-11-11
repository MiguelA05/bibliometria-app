# 📚 Bibliometría App

Sistema completo de análisis bibliométrico para la extracción, unificación, análisis y visualización de datos de producción científica desde múltiples fuentes académicas.

## 🎯 Descripción del Proyecto

Bibliometría App es una aplicación integral diseñada para automatizar el proceso de recopilación, análisis y visualización de datos bibliométricos. El sistema integra múltiples bases de datos académicas (OpenAlex, PubMed, ArXiv), implementa algoritmos avanzados de similitud textual, análisis de frecuencia de palabras, clustering jerárquico y generación de visualizaciones interactivas.

### Características Principales

- **🌍 Múltiples Fuentes de Datos**: Integración con OpenAlex, PubMed y ArXiv
- **🔄 Unificación Automática**: Proceso automatizado de unificación y eliminación de duplicados
- **🔍 Análisis de Similitud Textual**: 6 algoritmos (4 clásicos + 2 basados en IA)
- **📊 Análisis de Frecuencia**: Cálculo de frecuencia de palabras y términos asociados
- **🌳 Clustering Jerárquico**: Agrupamiento de abstracts con dendrogramas
- **📈 Visualizaciones Interactivas**: Mapas de calor, nubes de palabras, líneas temporales
- **🚀 API REST**: Servidor FastAPI con endpoints para todos los servicios
- **💻 Interfaz Interactiva**: Menú CLI para facilitar el uso

---

## 📋 Requerimientos Implementados

El proyecto cumple con 5 requerimientos principales:

### Requerimiento 1: Automatización de Descarga y Unificación de Datos

**Funcionalidad**: Proceso automatizado de descarga de información desde múltiples bases de datos académicas, unificación en un solo archivo y eliminación de duplicados.

**Características**:
- ✅ Descarga automática desde **OpenAlex**, **PubMed** y **ArXiv**
- ✅ Unificación de datos en formato estructurado
- ✅ Detección y eliminación de duplicados mediante algoritmos de similitud
- ✅ Generación de archivos:
  - `unified/`: Archivo CSV unificado con todos los artículos únicos
  - `duplicates/`: Registro de artículos duplicados eliminados
  - `raw_data/`: Datos crudos por cada fuente
  - `reports/`: Reportes de procesamiento

**Campos incluidos**: Título, autores, abstract, keywords/topics, año de publicación, DOI, URL, afiliaciones, países, ciudades, journal, citas, y más.

### Requerimiento 2: Análisis de Similitud Textual

**Funcionalidad**: Implementación de 6 algoritmos de similitud textual para comparar abstracts de artículos científicos.

**Algoritmos Implementados**:

1. **Levenshtein (Distancia de Edición)**: Mide la distancia mínima de edición entre textos
2. **Damerau-Levenshtein**: Extiende Levenshtein incluyendo transposiciones
3. **Jaccard (n-grams)**: Similitud basada en intersección de shingles
4. **TF-IDF Cosine Similarity**: Vectorización estadística con importancia de términos
5. **Sentence-BERT**: Embeddings semánticos usando transformers
6. **LLM-based (Ollama)**: Análisis semántico profundo con modelos LLM locales

**Características**:
- ✅ Explicación detallada paso a paso de cada algoritmo
- ✅ Análisis matemático y algorítmico completo
- ✅ Comparación de 2 o más artículos simultáneamente
- ✅ Extracción automática de abstracts desde CSV unificado
- ✅ Resultados con scores, tiempos de procesamiento y detalles técnicos

### Requerimiento 3: Análisis de Frecuencia de Palabras

**Funcionalidad**: Cálculo de frecuencia de aparición de palabras de una categoría específica y generación de palabras asociadas.

**Características**:
- ✅ Categoría predefinida: "Concepts of Generative AI in Education"
- ✅ Cálculo de frecuencia de aparición en abstracts
- ✅ Generación automática de palabras asociadas (máximo 15)
- ✅ Análisis de precisión de palabras asociadas
- ✅ Identificación de palabras por proximidad contextual

### Requerimiento 4: Agrupamiento Jerárquico de Abstracts

**Funcionalidad**: Implementación de clustering jerárquico para agrupar abstracts científicos relacionados.

**Características**:
- ✅ 3 métodos de linkage: Single, Complete, Average
- ✅ Preprocesamiento: Vectorización TF-IDF con normalización
- ✅ Generación de dendrogramas en formato PNG
- ✅ Evaluación de calidad de clusters (correlación cophenética)
- ✅ Determinación del mejor algoritmo según métricas
- ✅ Guardado en `results/reports/clustering/`

### Requerimiento 5: Análisis Visual

**Funcionalidad**: Generación de visualizaciones interactivas y estáticas de la producción científica.

**Visualizaciones Incluidas**:

1. **Mapa de Calor Geográfico**: Distribución geográfica por países de instituciones (choropleth interactivo)
2. **Nubes de Palabras**: 
   - Abstracts
   - Keywords
   - Combinada
3. **Línea Temporal**: Publicaciones por año y por revista/fuente
4. **Exportación PDF**: Reporte combinado con todas las visualizaciones

**Características**:
- ✅ Visualizaciones interactivas (Plotly) y estáticas (Matplotlib)
- ✅ Nubes de palabras dinámicas que se actualizan con más datos
- ✅ Exportación automática a PDF con formato profesional
- ✅ Guardado en `results/reports/visualizations/`

---

## 🚀 Instalación

### Requisitos Previos

- **Python**: 3.8 o superior
- **pip**: Gestor de paquetes de Python
- **Git**: Para clonar el repositorio (opcional)

### Pasos de Instalación

#### 1. Clonar el Repositorio

```bash
git clone <url-del-repositorio>
cd bibliometria-app
```

#### 2. Crear Entorno Virtual (Recomendado)

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
# Linux/Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

#### 3. Instalar Dependencias

```bash
# Instalar todas las dependencias
pip install -r requirements.txt

# Si pip no funciona, usa el módulo de Python:
python -m pip install -r requirements.txt
# O con Python 3:
python3 -m pip install -r requirements.txt

# Descargar datos de NLTK (OBLIGATORIO para similitud textual)
python -m nltk.downloader punkt stopwords
```

**⚠️ Problema con pip?** Si `pip` no está instalado o no se reconoce, consulta la [guía de solución de problemas](docs/solucion_problemas_pip.md).

**Dependencias Principales**:
- **Framework Web**: FastAPI, Uvicorn
- **Manejo de Datos**: pandas, numpy, requests
- **Validación**: pydantic, pydantic-settings
- **Similitud Textual**: scikit-learn, nltk, sentence-transformers
- **Clustering**: scipy
- **Visualización**: matplotlib, wordcloud, plotly, pillow, kaleido

#### 4. Instalar Ollama (Opcional - para algoritmo LLM-based)

Para usar el algoritmo de similitud basado en LLM (Requerimiento 2), necesitas instalar Ollama:

**Linux/Mac:**
```bash
# Método automático (recomendado)
bash scripts/install_ollama.sh

# O manualmente
curl -fsSL https://ollama.com/install.sh | sh
ollama serve  # En otra terminal
ollama pull llama3.2:3b  # Descargar modelo
```

**Windows:**
```powershell
# Método automático con PowerShell (recomendado)
powershell -ExecutionPolicy Bypass -File scripts/install_ollama.ps1

# O ejecutar el script .bat
scripts\install_ollama.bat

# O manualmente:
# 1. Descargar desde https://ollama.com/download
# 2. Ejecutar OllamaSetup.exe
# 3. Descargar modelo: ollama pull llama3.2:3b
```

**Nota**: Si Ollama no está instalado, el algoritmo LLM-based no estará disponible, pero los otros 5 algoritmos funcionarán normalmente.

**Ver guía detallada para Windows**: Ver [docs/instalacion_ollama_windows.md](docs/instalacion_ollama_windows.md)

#### 5. Configurar Entorno

```bash
# El archivo .env se crea automáticamente si no existe
# Puedes personalizar la configuración editando .env
```

#### 6. Verificar Instalación

```bash
# Ejecutar el menú principal (verifica dependencias automáticamente)
python menu.py
```

---

## 💻 Uso

### Menú Interactivo (Recomendado)

El menú interactivo es la forma más sencilla de usar todas las funcionalidades del proyecto:

```bash
python menu.py
```

El menú principal incluye:

1. **Probar Web Scraping y Generar Resultados** (Requerimiento 1)
   - Configurar consulta de búsqueda
   - Establecer límite de artículos por fuente
   - Configurar umbral de similitud para duplicados
   - Ejecutar proceso completo de automatización
   - Ver archivos generados

2. **Evaluar Algoritmos de Similitud Textual** (Requerimiento 2)
   - Seleccionar archivo CSV unificado
   - Elegir 2 o más artículos para comparar
   - Seleccionar algoritmos a ejecutar (todos, clásicos, IA, o individual)
   - Ver resultados detallados con explicaciones paso a paso

3. **Análisis de Frecuencia de Palabras** (Requerimiento 3)
   - Seleccionar archivo CSV unificado
   - Configurar categoría y palabras asociadas
   - Ver frecuencias y palabras asociadas generadas

4. **Agrupamiento Jerárquico de Abstracts** (Requerimiento 4)
   - Seleccionar archivo CSV unificado
   - Configurar parámetros de clustering
   - Generar dendrogramas con diferentes métodos
   - Ver evaluación de calidad de clusters

5. **Análisis Visual** (Requerimiento 5)
   - Seleccionar archivo CSV unificado
   - Generar todas las visualizaciones
   - Exportar a PDF

**Nota**: El servidor FastAPI se inicia automáticamente al ejecutar el menú.

### API REST

El proyecto incluye una API REST completa con FastAPI. El servidor se inicia automáticamente con el menú, o puedes iniciarlo manualmente:

```bash
# Opción 1: Script de inicio
python start.py

# Opción 2: Comando directo
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Endpoints Principales

- `POST /api/v1/automation/unified-data`: Ejecutar proceso de automatización
- `POST /api/v1/text-similarity/analyze`: Analizar similitud textual
- `GET /docs`: Documentación interactiva de la API (Swagger UI)
- `GET /health`: Estado de salud del sistema

#### Ejemplo de Uso de la API

```bash
# Ejecutar proceso de automatización
curl -X POST http://127.0.0.1:8000/api/v1/automation/unified-data \
     -H "Content-Type: application/json" \
     -d '{
       "base_query": "generative artificial intelligence",
       "max_articles_per_source": 100,
       "similarity_threshold": 0.75
     }'
```

### Scripts de Prueba

```bash
# Pruebas unitarias
python -m pytest tests/ -v

# Verificar salud del sistema
curl http://localhost:8000/health
```

---

## 📁 Estructura del Proyecto

```
bibliometria-app/
├── app/                              # Código principal de la aplicación
│   ├── api/                          # Endpoints de la API REST
│   │   └── endpoints.py              # Definición de endpoints
│   ├── models/                       # Modelos de datos
│   │   ├── article.py                # Modelo ArticleMetadata
│   │   └── schemas.py                # Esquemas Pydantic
│   ├── services/                     # Servicios principales
│   │   ├── openalex_service.py       # Servicio OpenAlex
│   │   ├── pubmed_service.py         # Servicio PubMed
│   │   ├── arxiv_service.py          # Servicio ArXiv
│   │   ├── data_unification_service.py  # Unificación y detección de duplicados
│   │   ├── text_similarity_service.py    # 6 algoritmos de similitud
│   │   ├── word_frequency_service.py     # Análisis de frecuencia
│   │   ├── hierarchical_clustering_service.py  # Clustering jerárquico
│   │   ├── visualization_service.py      # Visualizaciones
│   │   └── geographic_service.py        # Extracción de datos geográficos
│   ├── utils/                        # Utilidades
│   │   ├── logger.py                 # Sistema de logging
│   │   ├── csv_reader.py             # Lectura de CSVs
│   │   ├── text_extractor.py         # Extracción de textos
│   │   ├── ollama_helper.py           # Integración con Ollama
│   │   ├── server_helper.py           # Gestión del servidor FastAPI
│   │   ├── cache.py                   # Sistema de caché
│   │   ├── metrics.py                # Métricas de rendimiento
│   │   └── exceptions.py              # Manejo de excepciones
│   ├── config.py                      # Configuración de la aplicación
│   └── main.py                        # Aplicación FastAPI
│
├── tests/                            # Pruebas
│   ├── test_openalex_service.py      # Tests del servicio OpenAlex
│   ├── test_system.py                # Tests de integración
│   ├── test_text_similarity_service.py  # Tests de similitud textual
│   └── conftest.py                   # Configuración de pytest
│
├── results/                          # Archivos generados
│   ├── raw_data/                     # Datos crudos por fuente
│   ├── unified/                      # Archivos CSV unificados
│   ├── duplicates/                   # Registro de duplicados
│   ├── reports/                       # Reportes de procesamiento
│   │   ├── clustering/               # Dendrogramas
│   │   └── visualizations/           # Visualizaciones y PDFs
│
├── docs/                             # Documentación
│   └── README.md                     # Índice de documentación
│
├── scripts/                          # Scripts auxiliares
│   └── install_ollama.sh             # Instalación de Ollama
│
├── menu.py                           # Menú interactivo principal
├── start.py                          # Script de inicio del servidor
├── requirements.txt                  # Dependencias del proyecto
├── pytest.ini                        # Configuración de pytest
├── env.example                       # Ejemplo de configuración
└── README.md                         # Este archivo
```

---

## 🔧 Configuración

### Variables de Entorno

El proyecto usa un archivo `.env` para configuración (se crea automáticamente). Variables principales:

```env
# API
API_HOST=0.0.0.0
API_PORT=8000

# Bases de Datos
OPENALEX_BASE_URL=https://api.openalex.org
PUBMED_BASE_URL=https://eutils.ncbi.nlm.nih.gov/entrez/eutils
ARXIV_BASE_URL=https://export.arxiv.org/api/query

# Límites
MAX_ARTICLES_DEFAULT=10
MAX_ARTICLES_LIMIT=1000

# Archivos
RESULTS_DIR=results
CSV_ENCODING=utf-8-sig

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

---

## 🧪 Testing

```bash
# Ejecutar todas las pruebas
python -m pytest tests/ -v

# Pruebas con cobertura
python -m pytest tests/ --cov=app --cov-report=html

# Pruebas específicas
python -m pytest tests/test_openalex_service.py -v
```

---

## 📊 Fuentes de Datos

### OpenAlex
- **Cobertura**: 200M+ trabajos académicos globales
- **Metadatos**: Ricos y estructurados
- **API**: REST gratuita y sin límites estrictos
- **Datos**: Citas, Open Access, afiliaciones, financiación

### PubMed
- **Cobertura**: Base de datos biomédica del NLM
- **Metadatos**: MeSH terms, keywords, abstracts
- **API**: Entrez/eutils REST API
- **Datos**: Información médica y biomédica especializada

### ArXiv
- **Cobertura**: Preprints de física, matemáticas, ciencias de la computación
- **Metadatos**: Categorías, abstracts, autores
- **API**: REST API pública
- **Datos**: Preprints antes de publicación

---

## 🛠️ Tecnologías Utilizadas

### Backend
- **FastAPI**: Framework web moderno y rápido
- **Uvicorn**: Servidor ASGI de alto rendimiento
- **Pydantic**: Validación de datos y configuración

### Procesamiento de Datos
- **pandas**: Manipulación y análisis de datos
- **numpy**: Cálculos numéricos
- **scikit-learn**: Machine learning (TF-IDF, clustering)
- **scipy**: Algoritmos científicos (clustering jerárquico)

### Análisis de Texto
- **NLTK**: Procesamiento de lenguaje natural
- **sentence-transformers**: Embeddings semánticos
- **Ollama**: Modelos LLM locales

### Visualización
- **matplotlib**: Visualizaciones estáticas
- **plotly**: Visualizaciones interactivas
- **wordcloud**: Nubes de palabras
- **Pillow**: Procesamiento de imágenes

### Utilidades
- **requests**: Cliente HTTP
- **structlog**: Logging estructurado
- **python-dotenv**: Gestión de variables de entorno

---

## 📈 Características Avanzadas

### Detección de Duplicados
- Algoritmo híbrido que combina similitud de título, autores, DOI y año
- Pesos configurables para diferentes criterios
- Manejo especial de preprints vs. versiones publicadas

### Extracción Geográfica
- Extracción automática de países y ciudades desde afiliaciones
- Normalización de nombres geográficos
- Soporte para múltiples formatos de afiliación

### Preprocesamiento de Texto
- Normalización de caracteres
- Eliminación de stopwords
- Stemming (opcional)
- Limpieza de puntuación y espacios

### Logging Estructurado
- Logs en formato JSON para fácil análisis
- Niveles configurables
- Métricas de rendimiento integradas

---

## 🐛 Solución de Problemas

### Error: "No module named 'nltk'"
```bash
pip install nltk
python -m nltk.downloader punkt stopwords
```

### Error: "Ollama no disponible"
```bash
# Verificar que Ollama esté instalado
ollama --version

# Iniciar servidor Ollama
ollama serve

# Descargar modelo
ollama pull llama3.2:3b
```

### Error: "TF-IDF falló"
- Verificar que scikit-learn esté instalado: `pip install scikit-learn`
- Asegurar que los textos tengan al menos 20 caracteres

### PubMed no encuentra artículos
- Verificar que la consulta sea apropiada para PubMed
- Intentar con términos más específicos
- Revisar logs para ver la consulta transformada

---

## 📝 Notas Importantes

1. **Primera Ejecución**: La primera vez que ejecutes el proyecto, puede tardar más debido a la descarga de modelos de IA (Sentence-BERT).

2. **Ollama**: El algoritmo LLM-based requiere Ollama instalado y un modelo descargado. Sin esto, solo estarán disponibles 5 algoritmos.

3. **Memoria**: El procesamiento de grandes volúmenes de datos puede requerir memoria adicional. Se recomienda al menos 4GB de RAM.

4. **Internet**: Se requiere conexión a internet para acceder a las APIs de OpenAlex, PubMed y ArXiv.

---

## 🤝 Contribución

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

---

## 👥 Autor

Desarrollado como parte de un proyecto académico de análisis bibliométrico.

---

## 🔗 Enlaces Útiles

- [OpenAlex Documentation](https://docs.openalex.org/)
- [PubMed API](https://www.ncbi.nlm.nih.gov/books/NBK25497/)
- [ArXiv API](https://arxiv.org/help/api)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ollama Documentation](https://ollama.ai/docs)

---

## 📞 Soporte

Para problemas, preguntas o sugerencias, por favor abre un issue en el repositorio del proyecto.

---

**Última actualización**: Noviembre 2025
