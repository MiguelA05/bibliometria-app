# 🔬 Sistema de Análisis de Similitud Textual - Guía de Uso

## ✅ Integración Completada

Los endpoints de similitud textual están integrados en `app/api/endpoints.py` y funcionan junto con los demás endpoints del sistema.

---

## 🎯 Endpoints Disponibles

### 1. **Analizar Similitud Textual**
```
POST /api/v1/text-similarity/analyze
```

**Request Body:**
```json
{
    "csv_file_path": "results/unified/unified_xxx.csv",
    "article_indices": [0, 1, 2]
}
```

**Response:**
```json
{
    "articles": [
        {"index": 0, "title": "Article 1"},
        {"index": 1, "title": "Article 2"}
    ],
    "results": [
        {
            "algorithm": "Levenshtein (Edit Distance)",
            "score": 0.856,
            "explanation": "...",
            "details": {...},
            "time": 0.023
        },
        ...
    ],
    "summary": {
        "algorithms_used": 6,
        "avg_similarity": 0.782
    }
}
```

### 2. **Listar CSVs Disponibles**
```
GET /api/v1/text-similarity/csv-list
```

**Response:**
```json
{
    "csvs": [
        {
            "filename": "unified_xxx.csv",
            "filepath": "results/unified/unified_xxx.csv",
            "size_kb": 42.3,
            "modified": 1234567890
        }
    ],
    "total": 1
}
```

---

## 🔬 Algoritmos Implementados

### **Clásicos:**

1. **Levenshtein** - Distancia de edición
2. **Damerau-Levenshtein** - Con transposición
3. **Jaccard** - Sobre shingles (n-grams)
4. **TF-IDF Cosine** - Vectorización estadística

### **IA:**

5. **Sentence-BERT** - Embeddings semánticos
6. **LLM-based** - Similarity simulado

---

## 📝 Uso Rápido

### **Ejemplo con Python:**
```python
import requests

# Analizar similitud
response = requests.post(
    "http://127.0.0.1:8000/api/v1/text-similarity/analyze",
    json={
        "csv_file_path": "results/unified/unified_xxx.csv",
        "article_indices": [0, 1, 2]
    }
)

results = response.json()
print(results['summary'])
```

### **Ejemplo con Script:**
```bash
python test_text_similarity.py
```

---

## 🔧 Instalación de Dependencias

```bash
# Instalar librerías necesarias
pip install scikit-learn nltk sentence-transformers

# Descargar datos de NLTK
python -m nltk.downloader punkt stopwords
```

---

## 📊 Todos los Endpoints del Sistema

```
GET  /                              - Raíz de la API
GET  /health                        - Estado del sistema
GET  /metrics                       - Métricas de rendimiento

POST /api/v1/fetch-metadata         - Extraer metadatos de OpenAlex
POST /api/v1/uniquindio/generative-ai  - Endpoint universitario
POST /api/v1/automation/unified-data    - Automatización completa

POST /api/v1/text-similarity/analyze   - Analizar similitud textual ✨
GET  /api/v1/text-similarity/csv-list   - Listar CSVs disponibles ✨
```

---

## ✅ Integración Completa

**Archivos clave:**
- `app/api/endpoints.py` - Todos los endpoints en un solo archivo
- `app/services/text_similarity_service.py` - Lógica de los algoritmos
- `app/utils/text_extractor.py` - Lectura de CSVs
- `app/main.py` - Aplicación principal

**¡Todo listo para usar!** 🎉
