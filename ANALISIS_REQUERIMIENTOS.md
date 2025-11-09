# Análisis de Cumplimiento de Requerimientos

## Estado Actual del Proyecto

### ✅ Requerimiento 1: Automatización de proceso de descarga de datos
**Estado: COMPLETO**

**Implementación:**
- ✅ Descarga automática de datos de múltiples bases de datos (OpenAlex, PubMed, ArXiv)
- ✅ Unificación de información en un solo archivo
- ✅ Detección y eliminación de duplicados por similitud
- ✅ Archivo separado con registros de duplicados eliminados
- ✅ Proceso totalmente automático desde búsqueda hasta generación de archivo
- ✅ Accesible desde menú: Opción 1 → "Ejecutar proceso completo de automatización"

**Archivos relacionados:**
- `app/services/data_unification_service.py`
- `app/services/openalex_service.py`
- `app/services/pubmed_service.py`
- `app/services/arxiv_service.py`
- `menu.py` (líneas 188-251)

---

### ✅ Requerimiento 2: Algoritmos de similitud textual
**Estado: COMPLETO**

**Implementación:**
- ✅ 4 algoritmos clásicos:
  1. Levenshtein (Distancia de Edición)
  2. Damerau-Levenshtein (con transposición)
  3. Jaccard (n-grams)
  4. TF-IDF Cosine Similarity (Vectorización estadística)
- ✅ 2 algoritmos con modelos de IA:
  5. Sentence-BERT (Embeddings semánticos)
  6. LLM-based Similarity (Ollama - Llama 3.2 3B/Mistral 7B)
- ✅ Explicación detallada paso a paso del funcionamiento matemático y algorítmico
- ✅ Permite seleccionar dos o más artículos
- ✅ Extrae abstract y realiza análisis
- ✅ Accesible desde menú: Opción 2 → "Seleccionar artículos y analizar"

**Archivos relacionados:**
- `app/services/text_similarity_service.py`
- `menu.py` (líneas 344-625)

---

### ❌ Requerimiento 3: Frecuencia de aparición de palabras
**Estado: NO IMPLEMENTADO**

**Requisitos:**
- ❌ Calcular frecuencia de aparición de palabras asociadas a categoría "Concepts of Generative AI in Education"
- ❌ Generar listado de palabras asociadas (máximo 15) desde abstracts
- ❌ Determinar precisión de las nuevas palabras
- ❌ No accesible desde menú

**Acción requerida:** Implementar servicio de análisis de frecuencia de palabras

---

### ❌ Requerimiento 4: Agrupamiento jerárquico
**Estado: NO IMPLEMENTADO**

**Requisitos:**
- ❌ Implementar 3 algoritmos de agrupamiento jerárquico
- ❌ Construir dendrograma que represente similitud entre abstracts
- ❌ Preprocesamiento del texto (transformar abstract)
- ❌ Cálculo de similitud
- ❌ Aplicación de clustering
- ❌ Representación mediante dendrograma
- ❌ Determinar cuál algoritmo produce agrupamientos más coherentes
- ❌ No accesible desde menú

**Acción requerida:** Implementar servicio de clustering jerárquico con dendrogramas

---

### ❌ Requerimiento 5: Análisis visual
**Estado: NO IMPLEMENTADO**

**Requisitos:**
- ❌ Mapa de calor con distribución geográfica (primer autor)
- ❌ Nube de palabras dinámica (términos más frecuentes en abstracts y keywords)
- ❌ Línea temporal de publicaciones por año y por revista
- ❌ Exportar los tres anteriores a formato PDF
- ❌ No accesible desde menú

**Acción requerida:** Implementar servicio de visualización y exportación a PDF

---

## Resumen

- ✅ **Requerimiento 1**: COMPLETO y accesible desde menú
- ✅ **Requerimiento 2**: COMPLETO y accesible desde menú
- ❌ **Requerimiento 3**: NO IMPLEMENTADO
- ❌ **Requerimiento 4**: NO IMPLEMENTADO
- ❌ **Requerimiento 5**: NO IMPLEMENTADO

**Progreso: 2/5 requerimientos completos (40%)**

