# 🔬 OPENALEX SERVICE - Lógica de Implementación

## 📋 Objetivo

El `OpenAlexService` es el servicio principal que interactúa con la API de OpenAlex para descargar, procesar y exportar datos de artículos académicos, incluyendo información geográfica integrada.

---

## 🏗️ Arquitectura del Servicio

### Clase Principal

```python
class OpenAlexService:
    """
    Servicio para interactuar con la API de OpenAlex.
    Reemplaza completamente el web scraping con llamadas a la API REST.
    """
```

**Responsabilidades:**
- Hacer peticiones a la API de OpenAlex
- Procesar metadatos de artículos académicos
- Extraer información completa (título, autores, instituciones, etc.)
- Integrar datos geográficos
- Exportar resultados a CSV organizados

---

## 🔄 Flujo Principal: search_works()

**Ubicación:** Líneas 38-122

### Proceso Completo

```
1. BUSCAR EN OPENALEX
   POST /api/works?search=query&per_page=30
   ↓
2. PROCESAR CADA WORK
   for work in works:
       article = _process_work(work)
       ↓
3. EXPORTAR A CSV
   CSV guardado en results/raw_data/
   ↓
4. RETORNAR RESULTADOS
   (articles, csv_file_path)
```

**Código Clave:**
```python
def search_works(self, query, max_articles, filters):
    # Línea 60-63: Construir parámetros
    params = {
        'search': query,
        'per_page': max_articles
    }
    
    # Línea 76: Hacer petición a API
    response = self.session.get(
        f"{self.base_url}/works",
        params=params,
        timeout=settings.openalex_timeout
    )
    
    # Línea 80: Obtener results
    works = response.json().get('results', [])
    
    # Línea 90-98: Procesar cada work
    for work in works:
        article = self._process_work(work)
        articles.append(article)
    
    # Línea 116: Exportar a CSV
    csv_file_path = self._export_to_csv(articles, query)
    
    return articles, csv_file_path
```

---

## 🧩 Componentes Clave

### 1. Procesamiento de un Work (_process_work)

**Ubicación:** Líneas 124-233

**Lógica Secuencial:**

```python
def _process_work(work):
    # Paso 1: Extraer título
    title = work.get('title')  # Línea 136
    
    # Paso 2: Extraer autores y afiliaciones
    authors, affiliations = _extract_authors_and_affiliations(work)  # Línea 145
    
    # Paso 3: Extraer fechas
    publication_date = _extract_publication_date(work)  # Línea 148
    publication_year = work.get('publication_year')  # Línea 149
    
    # Paso 4: Extraer URLs
    article_url = _extract_article_url(work)  # Línea 154
    doi = work.get('doi')  # Línea 155
    
    # Paso 5: Extraer información de la fuente
    source_info = _extract_source_info(work)  # Línea 169
    
    # Paso 6: Extraer información Open Access
    oa_info = _extract_open_access_info(work)  # Línea 172
    
    # Paso 7: Extraer conceptos/temas
    concepts, topics = _extract_concepts_and_topics(work)  # Línea 175
    
    # Paso 8: Extraer datos geográficos (🆕)
    geographic_data = self.geographic_service.extract_geographic_data(work)  # Línea 184
    
    # Paso 9: Crear objeto ArticleMetadata
    article = ArticleMetadata(
        title=title,
        authors=authors,
        affiliations=affiliations,
        # ... todos los campos
        # + datos geográficos
        institution_countries=geographic_data.get('institution_countries'),
        geographic_coordinates=geographic_data.get('geographic_coordinates')
    )
    
    return article
```

---

### 2. Extracción de Autores y Afiliaciones

**Función:** `_extract_authors_and_affiliations()` (líneas 265-325)

**Lógica:**
```python
def _extract_authors_and_affiliations(work):
    authors = []
    affiliations = []
    
    # Iterar sobre authorships
    for authorship in work.get('authorships', []):
        # Extraer nombre del autor
        author = authorship.get('author', {})
        authors.append(author.get('display_name'))
        
        # Extraer información de instituciones con datos geográficos
        for institution in authorship.get('institutions', []):
            parts = [institution.get('display_name')]
            
            # Agregar ciudad, región, país
            if institution.get('city'):
                parts.append(institution.get('city'))
            if institution.get('country_code'):
                parts.append(institution.get('country_code'))
            
            affiliations.append(", ".join(parts))
    
    return authors, affiliations
```

**Resultado:**
```python
authors = ['John Smith', 'Jane Doe']
affiliations = [
    'MIT, Cambridge, US',
    'Harvard University, Boston, US'
]
```

---

### 3. Extracción de Fecha de Publicación

**Función:** `_extract_publication_date()` (líneas 327-353)

**Estrategia Cascada:**
```python
def _extract_publication_date(work):
    # Prioridad 1: Fecha completa
    if work.get('publication_date'):
        return work['publication_date']  # "2023-07-13"
    
    # Prioridad 2: Construir desde componentes
    year = work.get('publication_year')   # 2023
    month = work.get('publication_month') # 7
    day = work.get('publication_day')    # 13
    
    if year:
        date_parts = [str(year)]
        if month:
            date_parts.append(f"{int(month):02d}")  # "07"
            if day:
                date_parts.append(f"{int(day):02d}")  # "13"
        return '-'.join(date_parts)  # "2023-07-13"
    
    # Prioridad 3: Usar fecha de creación del registro
    if work.get('created_date'):
        return work['created_date'].split('T')[0]
    
    # Fallback
    return "Date not available"
```

---

### 4. Extracción de URL del Artículo

**Función:** `_extract_article_url()` (líneas 355-374)

**Prioridades:**
```python
def _extract_article_url(work):
    # 1. URL de Open Access (preferida)
    oa_url = work.get('open_access', {}).get('oa_url')
    if oa_url:
        return oa_url
    
    # 2. URL primaria del artículo
    landing_page = work.get('primary_location', {}).get('landing_page_url')
    if landing_page:
        return landing_page
    
    # 3. URL de OpenAlex
    if work.get('id'):
        return work.get('id')
    
    # 4. Fallback
    return "URL not available"
```

---

### 5. Exportación a CSV

**Función:** `_export_to_csv()` (líneas 423-493)

**Ubicación de Archivo:**
```python
base_dir = settings.results_dir           # "results"
raw_data_dir = os.path.join(base_dir, "raw_data")  # "results/raw_data"

filename = f"resultados_openalex_{query}_{timestamp}.csv"
file_path = os.path.join(raw_data_dir, filename)
```

**Estructura de Datos Exportada:**
```python
for article in articles:
    article_dict = {
        # Campos básicos
        'title': article.title,
        'authors': '; '.join(article.authors),
        'abstract': article.abstract,
        
        # Campos de OpenAlex
        'doi': article.doi,
        'cited_by_count': article.cited_by_count,
        
        # Campos geográficos (🆕)
        'institution_countries': '; '.join(article.institution_countries),
        'institution_cities': '; '.join(article.institution_cities),
        'geographic_coordinates': json.dumps(article.geographic_coordinates),
        
        # ... todos los demás campos
    }
```

**Encoding:**
```python
# Configurado en settings
encoding = settings.csv_encoding  # 'utf-8'
```

---

## 🌍 Integración con Geographic Service

### Línea 14: Import
```python
from app.services.geographic_service import GeographicDataService
```

### Línea 36: Inicialización
```python
def __init__(self):
    # ...
    self.geographic_service = GeographicDataService()
```

### Línea 184: Uso
```python
geographic_data = self.geographic_service.extract_geographic_data(work)
```

### Líneas 222-226: Integración en ArticleMetadata
```python
article = ArticleMetadata(
    # ... otros campos
    author_countries=geographic_data.get('author_countries'),
    author_cities=geographic_data.get('author_cities'),
    institution_countries=geographic_data.get('institution_countries'),
    institution_cities=geographic_data.get('institution_cities'),
    geographic_coordinates=geographic_data.get('geographic_coordinates')
)
```

### Líneas 476-480: Exportación Geográfica
```python
article_dict = {
    # ... otros campos
    'author_countries': '; '.join(article.author_countries) if article.author_countries else '',
    'author_cities': '; '.join(article.author_cities) if article.author_cities else '',
    'institution_countries': '; '.join(article.institution_countries) if article.institution_countries else '',
    'institution_cities': '; '.join(article.institution_cities) if article.institution_cities else '',
    'geographic_coordinates': json.dumps(article.geographic_coordinates) if article.geographic_coordinates else ''
}
```

---

## 🔧 Funciones Auxiliares

### 1. Extracción de Abstract

**Función:** `_extract_abstract()` (líneas 235-263)

**Estrategia Multi-fuente:**
```python
def _extract_abstract(work):
    # Prioridad 1: Abstract directo
    if work.get('abstract'):
        return work['abstract']
    
    # Prioridad 2: Reconstruir desde índice invertido
    if work.get('abstract_inverted_index'):
        # Algoritmo de reconstrucción
        words = []
        for word, positions in work['abstract_inverted_index'].items():
            for pos in positions:
                words.append((pos, word))
        words.sort()
        return ' '.join([word for _, word in words])
    
    # Prioridad 3: Campos alternativos
    for field in ['summary', 'description', 'content']:
        if work.get(field):
            return work[field]
    
    # Fallback
    return "Abstract not available"
```

---

### 2. Extracción de Conceptos y Topics

**Función:** `_extract_concepts_and_topics()` (líneas 403-412)

```python
def _extract_concepts_and_topics(work):
    concepts = work.get('concepts', [])
    topics = []
    
    for concept in concepts:
        if concept.get('display_name'):
            topics.append(concept.get('display_name'))
    
    return concepts, topics
```

**Ejemplo:**
```python
# Input OpenAlex
concepts = [
    {'display_name': 'Machine Learning', 'score': 0.9},
    {'display_name': 'Artificial Intelligence', 'score': 0.8}
]

# Output
topics = ['Machine Learning', 'Artificial Intelligence']
```

---

### 3. Extracción de Open Access Info

**Función:** `_extract_open_access_info()` (líneas 393-401)

```python
def _extract_open_access_info(work):
    open_access = work.get('open_access', {})
    
    return {
        'is_oa': open_access.get('is_oa'),          # True/False
        'oa_url': open_access.get('oa_url'),         # URL del PDF
        'oa_status': open_access.get('oa_status')    # 'gold', 'green', etc.
    }
```

---

## 📊 Estructura de Datos OpenAlex

### Campos Mapeados

```
OpenAlex Work → ArticleMetadata
├── title → title
├── abstract → abstract
├── authorships → authors + affiliations
│   ├── author.display_name → authors[]
│   └── institutions[].display_name → affiliations[]
├── publication_year → publication_year
├── publication_date → publication_date
├── doi → doi
├── openalex_id → openalex_id
├── open_access → is_oa, oa_url, oa_status
├── primary_location → article_url
├── concepts → topics[]
├── cited_by_count → cited_by_count
└── authorships → geographic_data (🆕)
    └── institutions[] → institution_countries, geographic_coordinates
```

---

## 🎯 Casos de Uso

### Caso 1: Artículo Completo

**Input OpenAlex:**
```json
{
    "title": "Machine Learning Applications",
    "doi": "10.1234/ml",
    "authorships": [{
        "author": {"display_name": "John Smith"},
        "institutions": [{
            "display_name": "MIT",
            "country_code": "US",
            "city": "Cambridge"
        }]
    }],
    "cited_by_count": 100
}
```

**Output ArticleMetadata:**
```python
ArticleMetadata(
    title="Machine Learning Applications",
    doi="10.1234/ml",
    authors=["John Smith"],
    affiliations=["MIT, Cambridge, US"],
    cited_by_count=100,
    institution_countries=["United States"],
    institution_cities=["Cambridge"],
    geographic_coordinates=[...]
)
```

---

### Caso 2: Manejo de Errores

```python
def _process_work(work):
    try:
        # Procesar...
        return article
    except Exception as e:
        print(f"⚠️ Error procesando trabajo: {e}")
        return None  # Retorna None si hay error (se omite)
```

**Estrategia:** Fail gracefully - omite artículos con error

---

## 🔐 Configuración y Seguridad

### Headers HTTP

**Línea 29-32:**
```python
self.headers = {
    'User-Agent': f'{settings.openalex_user_agent} (mailto:{email})'
}
self.session.headers.update(self.headers)
```

**Polite Pool:**
- Incluir email en User-Agent da acceso a "polite pool"
- Límites más generosos
- Prioridad en peticiones

---

## 📁 Organización de Archivos Exportados

**Ruta:**
```python
results/
└── raw_data/
    └── resultados_openalex_{query}_{timestamp}.csv
```

**Formato de Nombre:**
```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
safe_query = re.sub(r'[^\w\s-]', '', search_query).strip()
filename = f"resultados_openalex_{safe_query}_{timestamp}.csv"
```

**Ejemplo:**
```
resultados_openalex_generative_artificial_intelligence_20251027_123045.csv
```

---

## ✅ Ventajas de Esta Implementación

1. **API REST:** No depende de web scraping
2. **Manejo de Errores:** Fail gracefully
3. **Logging:** Trazabilidad completa
4. **Performance:** Timeout configurable
5. **Flexibilidad:** Filtros y parámetros configurables
6. **Geografía Integrada:** Datos geográficos automáticos
7. **Organización:** CSV en carpetas específicas

---

## 🎯 Resumen

**File:** `app/services/openalex_service.py`  
**Líneas:** 513 líneas  
**Clase principal:** `OpenAlexService`  
**Función principal:** `search_works()`  
**Dependencia:** `GeographicDataService` (línea 14)  
**Exportación:** CSV en `results/raw_data/`  
**Encoding:** UTF-8 configurado en settings  
**Geografía:** Integrada automáticamente (líneas 184, 222-226)
