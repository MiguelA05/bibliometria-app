# 🔬 Sistema de Análisis de Similitud Textual

## 📋 Objetivo

Implementación de **6 algoritmos de similitud textual** (4 clásicos + 2 basados en IA) para analizar la similitud entre abstracts de artículos académicos extraídos de CSV unificados.

---

## 🎯 Algoritmos Implementados

### **Clásicos (4 algoritmos):**

#### 1. **Levenshtein (Distancia de Edición)**
- **Tipo:** Caracteres
- **Complejidad:** O(n²)
- **Output:**
  - Distancia (número de operaciones)
  - Matriz DP (opcional)
  - Backtrace (transformaciones)

#### 2. **Damerau-Levenshtein**
- **Tipo:** Caracteres con transposición
- **Diferencia:** Permite intercambiar caracteres adyacentes como 1 operación
- **Output:**
  - Distancia
  - Transposiciones detectadas
  - Ejemplos de transposiciones

#### 3. **Jaccard sobre Shingles (n-grams)**
- **Tipo:** Tokens/conjuntos
- **Configurable:** longitud de n-grams (n=1,2,3...)
- **Output:**
  - Score de Jaccard
  - Shingles comunes
  - Lista de n-grams compartidos

#### 4. **TF-IDF Cosine Similarity**
- **Tipo:** Vectorización estadística
- **Características:**
  - TF-IDF para pesos
  - Cosine similarity para comparación
  - n-grams (1-3)
- **Output:**
  - Score de similitud
  - Top términos que contribuyen
  - TF-IDF de cada término

### **IA (2 algoritmos):**

#### 5. **Sentence-BERT (Embeddings Semánticos)**
- **Modelo:** paraphrase-MiniLM-L6-v2
- **Dimensiones:** 384
- **Tipo:** Similitud semántica (no léxica)
- **Output:**
  - Coseno de similitud semántica
  - Interpretación del score
  - Análisis conceptual

#### 6. **LLM-based Similarity (Simulado)**
- **Tipo:** Simulación de razonamiento LLM
- **Características:**
  - Análisis de temas comunes
  - Overlap semántico
  - Justificación textual
- **Nota:** Listo para integrar con API real (OpenAI, GPT, etc.)

---

## 🔧 Preprocesamiento de Texto

### **Pipelines según Algoritmo:**

#### **Char-level** (Levenshtein, Damerau):
```python
1. Normalización Unicode (NFKC)
2. Minúsculas
3. Limpiar espacios
```

#### **Token-level** (Jaccard):
```python
1. Normalización Unicode
2. Minúsculas
3. Tokenización
4. Remover puntuación
```

#### **Standard** (TF-IDF, Sentence-BERT):
```python
1. Normalización Unicode
2. Minúsculas
3. Tokenización
4. Remover stopwords
5. Stemming (Porter)
```

### **Pasos de Normalización:**

1. **Unicode NFKC:** Normaliza caracteres especiales
2. **Lowercase:** Estandarización
3. **Tokenización:** Por algoritmo
4. **Stopwords Removal:** Elimina palabras comunes
5. **Stemming:** Reduce palabras a raíz
6. **n-grams:** Genera shingles

---

## 📊 Estructura de Salida

### **Formato SimilarityResult:**
```python
{
    "algorithm_name": "Levenshtein (Edit Distance)",
    "similarity_score": 0.856,
    "explanation": "...detallada...",
    "details": {
        "distance": 45,
        "max_length": 320,
        "matrix": [...],
        "backtrace": [...]
    },
    "processing_time": 0.023
}
```

### **Ejemplo de Explicación:**
```
Levenshtein Distance: 45 operaciones
- Insertions: 10 caracteres a agregar
- Deletions: 20 caracteres a eliminar
- Substitutions: 15 caracteres a reemplazar
- Distance/Max_length ratio: 45/320 = 0.141
Similarity = 1 - ratio = 0.859
```

---

## 🚀 API Endpoints

### 1. **Analizar Similitud**
```
POST /api/v1/text-similarity/analyze

Request:
{
    "csv_file_path": "results/unified/unified_xxx.csv",
    "article_indices": [0, 1, 2]
}

Response:
{
    "articles": [...],
    "results": [
        {
            "algorithm": "...",
            "score": 0.85,
            "explanation": "...",
            "details": {...}
        }
    ]
}
```

### 2. **Listar CSVs Disponibles**
```
GET /api/v1/text-similarity/csv-list

Response:
{
    "csvs": [
        {
            "filename": "unified_xxx.csv",
            "filepath": "...",
            "size_kb": 42.3
        }
    ],
    "total": 1
}
```

---

## 💻 Uso del Sistema

### **1. Preparar Datos:**
```bash
# Generar CSV unificado
python test_system.py
```

### **2. Analizar Similitud:**
```bash
# Probar con artículos específicos
python test_text_similarity.py
```

### **3. Instalar Dependencias:**
```bash
pip install scikit-learn nltk sentence-transformers
python -m nltk.downloader punkt stopwords
```

---

## 📁 Archivos del Sistema

```
app/
├── services/
│   └── text_similarity_service.py    # 6 algoritmos implementados
├── utils/
│   └── text_extractor.py             # Lectura de CSVs
├── api/
│   └── text_similarity_endpoints.py  # Endpoints API
└── main.py                           # Integración

tests/
└── test_text_similarity.py           # Script de prueba
```

---

## 🎯 Características Clave

✅ **6 algoritmos:** 4 clásicos + 2 IA  
✅ **Preprocesamiento adaptativo:** según algoritmo  
✅ **Output detallado:** explicaciones paso a paso  
✅ **Matemática visible:** matrices, operaciones, transformaciones  
✅ **Integrado:** lee CSV unificados automáticamente  
✅ **Extensible:** fácil agregar más algoritmos  

---

## 📊 Ejemplo de Resultado Completo

```json
{
  "articles_analyzed": [
    {"index": 0, "title": "Machine Learning in AI"},
    {"index": 1, "title": "Deep Learning Applications"}
  ],
  "similarity_results": [
    {
      "algorithm": "Levenshtein",
      "score": 0.856,
      "explanation": "...",
      "details": {
        "distance": 45,
        "operations": ["insert", "substitute"]
      }
    },
    {
      "algorithm": "Sentence-BERT",
      "score": 0.912,
      "explanation": "...",
      "interpretation": "Very similar (likely same topic)"
    }
  ],
  "summary": {
    "avg_similarity": 0.884,
    "algorithms_used": 6
  }
}
```

---

## 🔬 Matemática Detallada

Ver documentación específica en código:
- Líneas 121-166: `calculate_similarity_score()` - Levenshtein
- Líneas 168-182: `_calculate_text_similarity()` - Jaccard
- Líneas 423-493: `tfidf_cosine_similarity()` - TF-IDF
- Sentence-BERT: utiliza modelo pre-entrenado

---

¡Sistema completo y listo para usar!
