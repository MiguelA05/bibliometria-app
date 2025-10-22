# OpenAlex Integration - Documentación Completa

## Descripción

Esta implementación integra [OpenAlex](https://openalex.org/) como la nueva fuente principal de datos académicos, reemplazando completamente el web scraping con una API REST moderna y robusta. OpenAlex proporciona acceso a una base de datos global de trabajos académicos con metadatos muy ricos.

## Características Principales

### 🌍 **Base de Datos Global**
- **Más de 200 millones de trabajos** académicos
- **Múltiples fuentes**: arXiv, PubMed, Crossref, Microsoft Academic, etc.
- **Actualización continua** de datos
- **Cobertura temporal**: Desde 1800 hasta la actualidad

### 📊 **Metadatos Ricos**
- **Información básica**: Título, autores, abstract, fechas
- **Métricas de impacto**: Número de citas, índices de calidad
- **Información institucional**: Afiliaciones, países, ciudades
- **Datos de Open Access**: Estado, URLs, licencias
- **Clasificación temática**: Conceptos, temas, categorías
- **Información de financiación**: Agencias, proyectos
- **Metadatos bibliográficos**: Volumen, número, páginas

### 🔬 **API REST Moderna**
- **Sin autenticación requerida** (gratuita)
- **Límite generoso**: 100,000 requests/día
- **Respuestas estructuradas** en JSON
- **Filtros avanzados** por año, tipo, fuente, etc.
- **Ordenamiento flexible** por citas, fecha, relevancia

## Estructura de Archivos

```
app/services/
├── scraper_service.py              # Scraper original (arXiv)
├── playwright_scraper_service.py   # Scraper con Playwright (arXiv)
└── openalex_service.py             # Servicio OpenAlex (NUEVO)

app/models/
└── article.py                      # Modelo actualizado con campos OpenAlex

app/api/
└── endpoints.py                    # Endpoints actualizados

tests/
├── test_scraper.py                 # Pruebas scraper original
├── test_playwright_scraper.py      # Pruebas Playwright
└── test_openalex_service.py        # Pruebas OpenAlex (NUEVO)

test_openalex_api.py                # Pruebas de integración OpenAlex
migracion_openalex.py               # Script de migración y comparación
```

## Endpoints Disponibles

### 1. OpenAlex (RECOMENDADO)
```bash
POST /api/v1/fetch-metadata-openalex
```

**Parámetros:**
```json
{
    "query": "machine learning",
    "max_articles": 10,
    "email": "tu@email.com",  // Opcional, para polite pool
    "filters": {               // Opcional
        "publication_year": "2024",
        "type": "journal-article",
        "is_oa": true
    }
}
```

**Respuesta:**
```json
{
    "articles": [...],
    "total_articles": 10,
    "csv_file_path": "results/resultados_openalex_machine_learning_20241201_143022.csv",
    "scraper_type": "openalex",
    "data_source": "OpenAlex API",
    "message": "Se encontraron 10 artículos usando OpenAlex..."
}
```

### 2. arXiv (Requests) - Respaldo
```bash
POST /api/v1/fetch-metadata
```

### 3. arXiv (Playwright) - Respaldo
```bash
POST /api/v1/fetch-metadata-playwright
```

## Campos de Datos Disponibles

### Campos Básicos
- `title`: Título del trabajo
- `authors`: Lista de autores
- `affiliations`: Afiliaciones institucionales
- `abstract`: Resumen
- `publication_date`: Fecha de publicación
- `article_url`: URL del artículo

### Campos Específicos de OpenAlex
- `openalex_id`: ID único de OpenAlex
- `doi`: DOI del trabajo
- `doi_url`: URL del DOI
- `publication_year/month/day`: Fecha desglosada
- `type`: Tipo de trabajo (journal-article, conference-paper, etc.)
- `language`: Idioma del trabajo
- `is_oa`: ¿Es Open Access?
- `oa_status`: Estado OA (gold, green, hybrid, closed)
- `oa_url`: URL de acceso abierto

### Información de la Fuente
- `source_title`: Título de la revista/conferencia
- `source_type`: Tipo de fuente (journal, conference, repository)
- `source_url`: URL de la fuente
- `source_issn`: ISSN de la fuente
- `source_is_oa`: ¿La fuente es OA?

### Información del Editor
- `publisher`: Editorial
- `publisher_url`: URL del editor

### Métricas de Impacto
- `cited_by_count`: Número de citas
- `cited_by_api_url`: URL de la API de citas
- `citing_works_count`: Número de trabajos que citan

### Clasificación Temática
- `concepts`: Conceptos con scores de relevancia
- `topics`: Lista de temas principales

### Información de Financiación
- `funding`: Lista de agencias financiadoras

### Metadatos Adicionales
- `biblio`: Información bibliográfica (volumen, número, páginas)
- `mesh`: Medical Subject Headings
- `license`: Licencia del trabajo
- `quality_score`: Score de calidad
- `sustainable_development_goals`: Objetivos de desarrollo sostenible

## Uso y Ejemplos

### Instalación
```bash
# No requiere instalación adicional
pip install -r requirements.txt
```

### Uso Básico
```python
from app.services.openalex_service import OpenAlexService

# Crear servicio
service = OpenAlexService(email="tu@email.com")

# Buscar trabajos
articles, csv_path = service.search_works(
    query="artificial intelligence",
    max_articles=10
)

# Con filtros
articles, csv_path = service.search_works(
    query="machine learning",
    max_articles=10,
    filters={
        "publication_year": "2024",
        "type": "journal-article",
        "is_oa": True
    }
)
```

### Uso con API
```bash
# Búsqueda básica
curl -X POST http://127.0.0.1:8000/api/v1/fetch-metadata-openalex \
     -H "Content-Type: application/json" \
     -d '{"query": "machine learning", "max_articles": 5}'

# Con filtros
curl -X POST http://127.0.0.1:8000/api/v1/fetch-metadata-openalex \
     -H "Content-Type: application/json" \
     -d '{
       "query": "artificial intelligence",
       "max_articles": 10,
       "email": "tu@email.com",
       "filters": {
         "publication_year": "2024",
         "type": "journal-article"
       }
     }'
```

## Filtros Disponibles

### Filtros Temporales
- `publication_year`: Año de publicación
- `from_publication_date`: Fecha desde
- `to_publication_date`: Fecha hasta

### Filtros de Tipo
- `type`: Tipo de trabajo
  - `journal-article`: Artículo de revista
  - `conference-paper`: Artículo de conferencia
  - `book-chapter`: Capítulo de libro
  - `dataset`: Conjunto de datos
  - `software`: Software

### Filtros de Acceso
- `is_oa`: Solo Open Access
- `oa_status`: Estado específico de OA
  - `gold`: Gold OA
  - `green`: Green OA
  - `hybrid`: Hybrid OA
  - `closed`: No OA

### Filtros de Fuente
- `source_type`: Tipo de fuente
- `source_id`: ID específico de fuente
- `publisher_id`: ID del editor

### Filtros de Autor
- `author_id`: ID específico de autor
- `institutions.id`: ID de institución

### Filtros de Concepto
- `concepts.id`: ID de concepto específico
- `concepts.display_name`: Nombre del concepto

## Ventajas de OpenAlex vs Web Scraping

| Aspecto | Web Scraping | OpenAlex |
|---------|--------------|----------|
| **Datos** | Limitados a arXiv | Global (200M+ trabajos) |
| **Calidad** | Dependiente del HTML | Estructurados y validados |
| **Mantenimiento** | Alto (cambios en sitios) | Bajo (API estable) |
| **Velocidad** | Variable | Consistente y rápida |
| **Robustez** | Frágil | Muy robusta |
| **Metadatos** | Básicos | Muy ricos |
| **Métricas** | No disponibles | Citas, impacto, etc. |
| **Filtros** | Limitados | Muy avanzados |
| **Escalabilidad** | Limitada | Excelente |

## Casos de Uso Recomendados

### ✅ **Usar OpenAlex cuando:**
- Necesitas datos globales (no solo arXiv)
- Quieres métricas de citas e impacto
- Necesitas información de Open Access
- Quieres análisis bibliométricos serios
- Necesitas datos institucionales
- Quieres filtros avanzados
- Necesitas datos de financiación
- Quieres análisis temporal amplio

### ⚠️ **Usar arXiv cuando:**
- Necesitas datos específicos de arXiv
- Quieres máxima velocidad para arXiv
- Tienes limitaciones de conectividad
- Necesitas datos en tiempo real de arXiv

## Rendimiento

### Tiempos Típicos
- **OpenAlex**: 0.8-1.2 segundos para 3 artículos
- **arXiv (Requests)**: 3-4 segundos para 3 artículos
- **arXiv (Playwright)**: 9-11 segundos para 3 artículos

### Límites de API
- **OpenAlex**: 100,000 requests/día (gratuito)
- **arXiv**: Sin límites oficiales

## Monitoreo y Logs

El servicio incluye logging detallado:
- 🔍 Búsquedas iniciadas
- 📄 Resultados encontrados
- ✅ Artículos procesados
- ⚠️ Errores y advertencias
- 📊 Estadísticas de exportación

## Troubleshooting

### Error: "No se encontraron artículos"
- Verificar que la consulta sea válida
- Probar con términos más generales
- Verificar filtros aplicados

### Error: "400 Bad Request"
- Verificar parámetros de la consulta
- Revisar formato de filtros
- Verificar límites de API

### Error: "Timeout"
- Verificar conectividad a internet
- Reducir número de artículos solicitados
- Verificar estado de la API de OpenAlex

## Migración desde arXiv

### Script de Migración
```bash
python migracion_openalex.py
```

### Comparación de Datos
```bash
python test_openalex_api.py
```

### Verificación Completa
```bash
python verificacion_final.py
```

## Contribución

Para contribuir a la integración de OpenAlex:
1. Mantén compatibilidad con la API existente
2. Añade pruebas para nuevas funcionalidades
3. Documenta cambios en la API
4. Verifica que los tests pasen

## Recursos Adicionales

- [Documentación oficial de OpenAlex](https://docs.openalex.org/)
- [API Reference](https://docs.openalex.org/api-entities)
- [Ejemplos de consultas](https://docs.openalex.org/api-entities/works)
- [Política de uso](https://docs.openalex.org/api-entities/works)

## Licencia

Esta implementación mantiene la misma licencia que el proyecto principal. OpenAlex es una base de datos abierta bajo licencia CC0.

