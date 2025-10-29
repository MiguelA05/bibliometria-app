# 🔗 ALGORITMO DE UNIFICACIÓN DE DATOS

## 📊 Descripción General

El algoritmo de unificación combina datos de múltiples fuentes (OpenAlex_General, OpenAlex_Articles, OpenAlex_Conferences) y elimina duplicados usando un sistema de puntuación de similitud ponderado.

---

## 🔄 FLUJO DEL ALGORITMO

### 1️⃣ DESCARGAR DE MÚLTIPLES FUENTES

```python
def download_from_sources(sources: List[DataSource]) -> List[ArticleMetadata]:
```

**Proceso:**
- Descarga datos de cada fuente configurada (3 por defecto)
- Marca cada artículo con su `source` (OpenAlex_General, etc.)
- Combina todos los artículos en una lista única

**Resultado:** Lista de TODOS los artículos (con duplicados potenciales)

---

### 2️⃣ DETECTAR Y ELIMINAR DUPLICADOS

```python
def detect_and_remove_duplicates(articles: List[ArticleMetadata], 
                                 similarity_threshold: float = 0.8)
```

**Algoritmo:**
```python
unique_articles = []
duplicates_log = []

for article in articles:
    is_duplicate = False
    
    # Comparar con artículos ya procesados
    for unique_article in unique_articles:
        similarity = calculate_similarity_score(article, unique_article)
        
        if similarity >= threshold:  # Por defecto: 0.8
            duplicates_log.append(article)
            is_duplicate = True
            break
    
    if not is_duplicate:
        unique_articles.append(article)
```

**Características:**
- **Algoritmo:** Comparación secuencial (O(n²))
- **Estrategia:** Mantener el primer artículo, marcar los duplicados
- **Orden:** Se mantiene el orden de llegada de los artículos

---

## 🎯 CÁLCULO DE SIMILITUD (Core del Algoritmo)

### Fórmula de Puntuación

```python
def calculate_similarity_score(article1, article2) -> float:
    score = 0.0
    
    # 1. TÍTULO (40% peso)
    title_similarity = _calculate_text_similarity(
        article1.title.lower(), 
        article2.title.lower()
    )
    score += title_similarity * 0.4
    
    # 2. DOI (30% peso)
    if article1.doi == article2.doi:
        score += 0.3
    else:
        doi_sim = _normalize_and_compare(article1.doi, article2.doi)
        score += doi_sim * 0.3
    
    # 3. AUTORES (20% peso)
    author_similarity = _calculate_author_similarity(
        article1.authors, 
        article2.authors
    )
    score += author_similarity * 0.2
    
    # 4. AÑO DE PUBLICACIÓN (10% peso)
    if article1.year == article2.year:
        score += 0.1
    
    return min(score, 1.0)
```

---

## 📐 TÉCNICAS DE SIMILITUD UTILIZADAS

### 1. Similitud de Título (40% peso)

**Método:** Jaccard Similarity sobre palabras

```python
def _calculate_text_similarity(text1, text2) -> float:
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union
```

**Ejemplo:**
- Texto 1: "Machine Learning in Healthcare"
- Texto 2: "Machine Learning Applications in Healthcare"
- **Similitud:** 3/6 = 0.5

---

### 2. Similitud de DOI (30% peso)

**Normalización:**
- Remover prefijos: `https://doi.org/`, `doi:`
- Convertir a minúsculas
- Comparar strings normalizados

**Ejemplo:**
- DOI 1: "https://doi.org/10.1234/example"
- DOI 2: "doi:10.1234/example"
- **Normalizados:** "10.1234/example"
- **Resultado:** Score completo (0.3)

---

### 3. Similitud de Autores (20% peso)

**Método:** Jaccard Similarity sobre conjunto de autores

```python
def _calculate_author_similarity(authors1, authors2) -> float:
    # Normalizar nombres
    norm1 = [normalize(author) for author in authors1]
    norm2 = [normalize(author) for author in authors2]
    
    # Calcular intersección/unión
    set1, set2 = set(norm1), set(norm2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union
```

**Normalización de nombre:**
- Minúsculas
- Remover caracteres especiales
- Remover espacios extra

---

### 4. Año de Publicación (10% peso)

**Lógica:**
- Si años coinciden: +0.1
- Si no coinciden: +0.0

---

## ⚙️ PARÁMETROS CONFIGURABLES

### Umbral de Similitud (`similarity_threshold`)

**Por defecto:** `0.8`

**Efecto:**
- Umbral más alto (0.9): Solo duplicados muy obvios
- Umbral bajo (0.6): Detección agresiva de duplicados

**Ejemplo:**
```python
# Score = 0.85
similarity_threshold = 0.8
# Resultado: ✅ DUPLICADO (0.85 >= 0.8)

# Score = 0.75
similarity_threshold = 0.8
# Resultado: ❌ NO DUPLICADO (0.75 < 0.8)
```

---

## 📊 EJEMPLO PRÁCTICO

### Caso 1: Duplicado Exacto
```
Artículo 1: "Machine Learning in AI"
            DOI: 10.1234/ml
            Autores: ["Smith", "Jones"]
            Año: 2023

Artículo 2: "Machine Learning in AI"
            DOI: 10.1234/ml
            Autores: ["Smith", "Jones"]
            Año: 2023

Cálculo:
- Título: 1.0 × 0.4 = 0.4
- DOI: 1.0 × 0.3 = 0.3
- Autores: 1.0 × 0.2 = 0.2
- Año: 1.0 × 0.1 = 0.1
- TOTAL: 1.0

Resultado: ✅ DUPLICADO (score: 1.0 >= 0.8)
```

### Caso 2: Artículo Similar pero Diferente
```
Artículo 1: "Machine Learning in AI" (2023)
Artículo 2: "Deep Learning in AI" (2024)

Cálculo:
- Título: 0.25 × 0.4 = 0.1
- DOI: 0.0 × 0.3 = 0.0
- Autores: 0.0 × 0.2 = 0.0
- Año: 0.0 × 0.1 = 0.0
- TOTAL: 0.1

Resultado: ❌ NO DUPLICADO (score: 0.1 < 0.8)
```

---

## 🎯 VENTAJAS DEL ALGORITMO

✅ **Ponderación inteligente:** Más peso a identificadores únicos (DOI)  
✅ **Flexible:** Títulos con variaciones se detectan  
✅ **Trazable:** Registro completo de duplicados  
✅ **Configurable:** Umbral ajustable según necesidades  

---

## ⚠️ LIMITACIONES

❌ **Complejidad O(n²):** Cada artículo comparado con todos los anteriores  
❌ **Orden dependiente:** El primer artículo siempre se mantiene  
❌ **Puede perder variaciones:** Verificaciones exactas solo para DOI  

---

## 💡 OPCIONES DE MEJORA

### 1. Optimización de Rendimiento
```python
# Usar hashing para detectar duplicados exactos rápidamente
doi_hash = {article.doi for article in processed}
if article.doi in doi_hash:
    # Es probablemente un duplicado
```

### 2. Algoritmo de Agrupación
```python
# Usar DBSCAN o similar para clustering
from sklearn.cluster import DBSCAN
clusters = DBSCAN(eps=0.8, min_samples=1).fit(article_vectors)
```

### 3. Embeddings Semánticos
```python
# Usar embeddings de título para similitud semántica
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
similarity = cosine_similarity(emb1, emb2)
```

---

## 📋 RESUMEN

**Algoritmo Actual:**
- **Tipo:** Comparación secuencial con similitud ponderada
- **Pesos:** Título (40%), DOI (30%), Autores (20%), Año (10%)
- **Métrica:** Jaccard Similarity + Comparación exacta
- **Umbral:** 0.8 por defecto
- **Complejidad:** O(n²) en tiempo

**Justificación de la División:**
- ✅ **raw_data/**: Fuente original (necesario para auditabilidad)
- ✅ **unified/**: Resultado del algoritmo (esencial para análisis)
- ⚠️ **duplicates/**: Registro del proceso (útil pero opcional)
- ✅ **reports/**: Estadísticas (útil para monitoreo)
