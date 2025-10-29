# 🌍 GEOGRAPHIC SERVICE - Lógica de Implementación

## 📋 Objetivo

El `GeographicDataService` extrae y procesa información geográfica de artículos académicos para permitir análisis geoespaciales y creación de mapas de calor bibliométricos.

---

## 🏗️ Arquitectura del Servicio

### Clase Principal

```python
class GeographicDataService:
    """Servicio para extraer y procesar datos geográficos de OpenAlex."""
```

**Responsabilidades:**
- Extraer datos geográficos de metadatos de OpenAlex
- Normalizar y limpiar información geográfica
- Convertir códigos de país a nombres completos
- Generar resúmenes estadísticos geográficos
- Exportar datos para herramientas de visualización

---

## 🔄 Flujo Principal

### 1. Extracción de Datos Geográficos

```python
def extract_geographic_data(work: Dict[str, Any]) -> Dict[str, Any]
```

**Ubicación:** Líneas 26-91

**Proceso:**
```
OpenAlex Work Data
    ↓
Iterar sobre authorships
    ↓
Para cada authorship:
    ├─ Extraer info del autor
    │   └─ → author_countries, author_cities
    └─ Extraer info de instituciones
        └─ → institution_countries, institution_cities, coordinates
    ↓
Limpiar y deduplicar datos
    ↓
Retornar estructura geográfica
```

**Estructura de Datos Retornada:**
```python
{
    'author_countries': ['United States', 'Canada'],
    'author_cities': ['Cambridge', 'Boston'],
    'institution_countries': ['United States'],
    'institution_cities': ['Cambridge'],
    'geographic_coordinates': [
        {
            'institution': 'MIT',
            'country': 'United States',
            'city': 'Cambridge',
            'latitude': 42.3601,
            'longitude': -71.0942
        }
    ]
}
```

---

## 🧩 Componentes Clave

### 1. Extracción de Información Geográfica de Autores

**Función:** `_get_author_geographic_info()` (líneas 93-111)

**Lógica:**
```python
def _get_author_geographic_info(author):
    # Método 1: Datos directos del autor
    if author tiene 'last_known_institution':
        return info de la institución
    
    # Método 2: Hacer petición API adicional
    if author tiene 'id':
        return datos completos del autor desde API
    
    # Fallback
    return None
```

**Estrategia:**
- Si el autor tiene `last_known_institution` → usar esos datos
- Si no, hacer petición a `/authors/{id}` para obtener datos completos
- Utiliza cache para evitar peticiones repetidas

---

### 2. Extracción de Información Geográfica de Instituciones

**Función:** `_get_institution_geographic_info()` (líneas 113-146)

**Lógica:**
```python
def _get_institution_geographic_info(institution):
    geo_data = {
        'countries': [],
        'cities': [],
        'coordinates': None
    }
    
    # Extraer país del código de país
    country_code = institution.get('country_code')  # Ej: 'US'
    geo_data['countries'].append(get_country_name(country_code))  # → 'United States'
    
    # Extraer ciudad
    city = institution.get('city')  # Ej: 'Cambridge'
    geo_data['cities'].append(city)
    
    # Extraer coordenadas
    geo = institution.get('geo')
    if geo:
        geo_data['coordinates'] = [geo['lat'], geo['lng']]  # [42.3601, -71.0942]
    
    return geo_data
```

**Datos Extraídos:**
- **País:** De `country_code` → nombre completo
- **Ciudad:** De `city`
- **Coordenadas:** De `geo.lat` y `geo.lng`

---

### 3. Conversión de Códigos de País

**Función:** `_get_country_name()` (líneas 170-357)

**Lógica:**
```python
# Mapeo completo de códigos ISO a nombres completos
country_mapping = {
    'US': 'United States',
    'GB': 'United Kingdom',
    'CO': 'Colombia',
    # ... todos los países del mundo
}

def _get_country_name(code):
    return country_mapping.get(code.upper())
```

**Coverage:** 190+ países mapeados

---

### 4. Limpieza y Deduplicación

**Función:** `_clean_geographic_data()` (líneas 359-389)

**Problema a resolver:**
- Datos duplicados en listas
- Coordenadas repetidas
- Valores vacíos o None

**Algoritmo de Limpieza:**
```python
for lista in [countries, cities, ...]:
    seen = set()
    unique_list = []
    for item in lista:
        if item and item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list
```

**Para Coordenadas:**
```python
# Usar (lat, lng) como clave única
seen_coords = set()
for coord in coordinates:
    key = (coord['latitude'], coord['longitude'])
    if key not in seen_coords:
        unique_coords.append(coord)
        seen_coords.add(key)
```

---

### 5. Resumen Geográfico

**Función:** `get_geographic_summary()` (líneas 391-469)

**Propósito:** Generar estadísticas de la colección de artículos

**Proceso:**
```python
summary = {
    'total_articles': len(articles),
    'countries_count': 0,
    'cities_count': 0,
    'coordinates_count': 0,
    'top_countries': [],
    'top_cities': [],
    'geographic_coverage': {
        'articles_with_countries': X,
        'articles_with_cities': Y,
        'articles_with_coordinates': Z
    }
}

# Calcular usando Counter
country_counts = Counter(all_countries)
summary['top_countries'] = country_counts.most_common(10)
```

**Salida:**
```python
{
    'total_articles': 30,
    'countries_count': 15,
    'cities_count': 8,
    'coordinates_count': 0,
    'top_countries': [('United States', 10), ('Canada', 5), ...],
    'top_cities': [('Cambridge', 5), ...],
    'geographic_coverage': {
        'articles_with_countries': 28,  # 93.3%
        'articles_with_cities': 0,      # 0%
        'articles_with_coordinates': 0  # 0%
    }
}
```

---

### 6. Exportación para Mapas de Calor

**Función:** `export_geographic_data()` (líneas 471-546)

**Propósito:** Preparar CSV para herramientas de visualización (Folium, Plotly)

**Estrategia de Exportación:**
```python
# Si hay coordenadas:
for coord in coordinates:
    export_row({
        'title': article.title,
        'country': coord['country'],
        'city': coord['city'],
        'institution': coord['institution'],
        'latitude': coord['latitude'],
        'longitude': coord['longitude'],
        'cited_by_count': article.cited_by_count
    })

# Si NO hay coordenadas:
for country in countries:
    export_row({
        'title': article.title,
        'country': country,
        'latitude': '',  # Vacío
        'longitude': '',
        'cited_by_count': article.cited_by_count
    })
```

**Resultado:** CSV con una fila por combinación país/ciudad

---

## 🎯 Casos de Uso

### Caso 1: Artículo con Datos Completos

**Input (OpenAlex):**
```json
{
    "authorships": [{
        "author": {"id": "A123"},
        "institutions": [{
            "display_name": "MIT",
            "country_code": "US",
            "city": "Cambridge",
            "geo": {"lat": 42.3601, "lng": -71.0942}
        }]
    }]
}
```

**Output:**
```python
{
    'institution_countries': ['United States'],
    'institution_cities': ['Cambridge'],
    'geographic_coordinates': [{
        'institution': 'MIT',
        'country': 'United States',
        'city': 'Cambridge',
        'latitude': 42.3601,
        'longitude': -71.0942
    }]
}
```

### Caso 2: Artículo sin Coordenadas

**Input:**
```json
{
    "authorships": [{
        "institutions": [{
            "country_code": "CO",
            "city": "Bogotá"
        }]
    }]
}
```

**Output:**
```python
{
    'institution_countries': ['Colombia'],
    'institution_cities': ['Bogotá'],
    'geographic_coordinates': []  # Sin coordenadas
}
```

---

## 🔍 Detalles Técnicos

### Manejo de Errores

**Estrategia:** Fail gracefully

```python
try:
    # Procesar datos
    return geographic_data
except Exception as e:
    self.logger.error(f"Error: {e}")
    return {
        'author_countries': [],
        'author_cities': [],
        'institution_countries': [],
        'institution_cities': [],
        'geographic_coordinates': []
    }
```

**Resultado:** Si hay error, retorna estructura vacía en lugar de crash

---

### Optimización: Cache

```python
def __init__(self):
    self.coordinates_cache = {}  # Cache de coordenadas geográficas
```

**Uso:** Evita peticiones repetidas a la API de OpenAlex

---

## 📊 Integración con el Sistema

### Flujo Completo

```
openalex_service.py
    ↓ (llama a)
GeographicDataService.extract_geographic_data(work)
    ↓ (retorna)
geographic_data = {
    'institution_countries': [...],
    'geographic_coordinates': [...]
}
    ↓ (se agrega a)
ArticleMetadata(
    institution_countries=...,
    geographic_coordinates=...
)
    ↓ (se exporta a)
CSV con columnas geográficas
```

**Línea clave en openalex_service.py:**
```python
# Línea 184
geographic_data = self.geographic_service.extract_geographic_data(work)
```

---

## ✅ Ventajas de Esta Implementación

1. **Extracción Automática:** Geografía incluida en todos los CSV
2. **Normalización:** Códigos de país → nombres completos
3. **Limpieza:** Eliminación de duplicados
4. **Flexibilidad:** Funciona con o sin coordenadas
5. **Exportación Compatible:** CSV listo para Folium, Plotly, etc.

---

## 🎯 Resumen

**File:** `app/services/geographic_service.py`  
**Líneas:** 552 líneas  
**Funciones principales:** 6 funciones públicas + 6 privadas  
**Propósito:** Extraer, normalizar y limpiar datos geográficos  
**Integración:** Usado por `OpenAlexService` en línea 184
