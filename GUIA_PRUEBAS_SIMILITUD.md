# 🧪 GUÍA DE PRUEBAS - Sistema de Similitud Textual

## 📋 Cómo Probar la Implementación

### **Paso 1: Iniciar el Servidor**
```bash
python start.py
```

**Salida esperada:**
```
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

### **Paso 2: Ejecutar Prueba Completa**
```bash
python test_similarity_complete.py
```

**O pruebas individuales:**
```bash
python test_system.py         # Poblar datos primero
python test_similarity_complete.py  # Probar similitud
```

---

## 📊 Salidas que Deberías Esperar

### **1. Verificación de Dependencias**
```
✅ scikit-learn instalado
✅ nltk instalado
✅ sentence-transformers instalado (o ⚠️ si no está)
```

---

### **2. Listado de CSVs**
```
✅ CSVs disponibles: 1

   📄 unified_generative_ai_20251027_092624_unified.csv
      Tamaño: 65.2 KB
      Ruta: results/unified/unified_xxx.csv
```

---

### **3. Análisis de Similitud**
```
🎯 RESULTADOS DE LOS 6 ALGORITMOS:

   1. Levenshtein (Edit Distance)
      Score: 0.856
      Tiempo: 0.023s
      ⚡ Distancia: 45
      ⚡ Max length: 320

   2. Damerau-Levenshtein (with Transposition)
      Score: 0.892
      Tiempo: 0.019s
      🔄 Transposiciones: 2

   3. Jaccard over 3-grams
      Score: 0.634
      Tiempo: 0.015s
      📊 Shingles: 15/45 comunes

   4. TF-IDF Cosine Similarity
      Score: 0.721
      Tiempo: 0.187s
      🔑 Top términos: machine, learning, artificial

   5. Sentence-BERT Semantic Similarity
      Score: 0.912
      Tiempo: 1.234s  (descarga de modelo la primera vez)
      💡 Very similar (likely same topic and argument)

   6. LLM-based Similarity (Simulated)
      Score: 0.789
      Tiempo: 0.012s

📊 RESUMEN GENERAL:
   Algoritmos ejecutados: 6
   Similitud promedio: 0.817
```

---

## 🔍 Análisis de Resultados

### **Interpretación de Scores:**
- **0.9 - 1.0:** Textos muy similares (mismo tema/argumento)
- **0.7 - 0.9:** Textos similares (temas relacionados)
- **0.4 - 0.7:** Algo similar (conceptos relacionados)
- **0.0 - 0.4:** Textos diferentes (temas distintos)

### **Diferencias por Algoritmo:**

| Algoritmo | Mide | Mejor Para |
|-----------|------|------------|
| Levenshtein | Caracteres editados | Errores tipográficos |
| Damerau-Levenshtein | + Transposiciones | Errores de teclado |
| Jaccard | Overlap de términos | Contenido temático |
| TF-IDF | Relevancia estadística | Análisis semántico básico |
| Sentence-BERT | Significado profundo | Similitud semántica |
| LLM | Comprensión contextual | Análisis conceptual |

---

## 🎯 Resultados Esperados por Algoritmo

### **1. Levenshtein (Score 0.6-0.9 típico):**
```json
{
  "algorithm": "Levenshtein",
  "score": 0.856,
  "details": {
    "distance": 45,
    "max_length": 320,
    "operations_needed": 45
  }
}
```

### **2. Damerau-Levenshtein (Score similar o ligeramente mayor):**
```json
{
  "algorithm": "Damerau-Levenshtein",
  "score": 0.892,
  "details": {
    "distance": 38,
    "transpositions_count": 2,
    "transpositions": [("34", "35", "ab", "ba")]
  }
}
```

### **3. Jaccard (Score 0.4-0.8 típico):**
```json
{
  "algorithm": "Jaccard over 3-grams",
  "score": 0.634,
  "details": {
    "shingles1_count": 120,
    "shingles2_count": 135,
    "intersection_size": 48,
    "union_size": 207,
    "common_shingles": ["machine learning applications", ...]
  }
}
```

### **4. TF-IDF (Score 0.5-0.9):**
```json
{
  "algorithm": "TF-IDF Cosine Similarity",
  "score": 0.721,
  "details": {
    "top_contributing_terms": [
      {"term": "machine", "contribution": 0.234},
      {"term": "learning", "contribution": 0.189}
    ]
  }
}
```

### **5. Sentence-BERT (Score 0.7-0.95):**
```json
{
  "algorithm": "Sentence-BERT",
  "score": 0.912,
  "details": {
    "embedding_dim": 384,
    "model": "paraphrase-MiniLM-L6-v2",
    "interpretation": "Very similar (likely same topic and argument)"
  }
}
```

---

## ⚠️ Problemas Comunes y Soluciones

### **Error: "sentence-transformers not available"**
```bash
pip install sentence-transformers
```
- Los algoritmos 5 y 6 no funcionarán
- Los algoritmos 1-4 sí funcionan

### **Error: "CSV file not found"**
```bash
# Primero poblar datos:
python test_system.py

# Luego probar similitud:
python test_similarity_complete.py
```

### **Error: "Need at least 2 articles"**
- Verifica que el CSV tenga al menos 2 artículos
- Ajusta los índices en la petición

### **Timeouts largos:**
- Normal en primera ejecución (descarga de modelo Sentence-BERT)
- Siguientes ejecuciones son más rápidas (~1-2s)

---

## 📈 Métricas de Rendimiento Esperadas

| Algoritmo | Tiempo Típico | Notas |
|-----------|---------------|-------|
| Levenshtein | 10-50ms | Rápido, O(n²) |
| Damerau-Levenshtein | 10-50ms | Similar a Levenshtein |
| Jaccard | 5-20ms | Muy rápido |
| TF-IDF | 100-200ms | Análisis más profundo |
| Sentence-BERT | 1000-2000ms | Primera vez: ~10s (descarga modelo) |
| LLM-based | 5-15ms | Simulado |

---

## ✅ Checklist de Prueba

- [ ] Servidor corriendo en puerto 8000
- [ ] Al menos un CSV unificado disponible
- [ ] Dependencias instaladas
- [ ] Los 6 algoritmos se ejecutan
- [ ] Scores están entre 0.0 y 1.0
- [ ] Tiempos de procesamiento mostrados
- [ ] Detalles específicos por algoritmo

---

## 🎉 Resultado Esperado Final

```
✅ TODAS LAS PRUEBAS EXITOSAS

🎉 El sistema de similitud textual está funcionando correctamente

📋 LO QUE DEBERÍAS VER:
   • 2 endpoints funcionando
   • 6 algoritmos ejecutándose
   • Score de similitud entre 0.0 y 1.0
   • Tiempos de procesamiento para cada algoritmo
   • Detalles específicos por algoritmo
```

---

¡Listo para probar! 🚀
