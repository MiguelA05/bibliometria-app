# Estado de Cumplimiento de Requerimientos

## ✅ Requerimiento 1: Automatización de proceso de descarga de datos
**ESTADO: COMPLETO Y VERIFICADO**

### Verificación:
- ✅ Descarga automática de múltiples bases de datos (OpenAlex, PubMed, ArXiv)
- ✅ Unificación en un solo archivo CSV
- ✅ Detección y eliminación de duplicados por similitud
- ✅ Archivo separado con duplicados eliminados
- ✅ Proceso totalmente automático
- ✅ **Accesible desde menú**: Opción 1 → "Ejecutar proceso completo de automatización"

### Archivos:
- `app/services/data_unification_service.py`
- `menu.py` (líneas 188-251)

---

## ✅ Requerimiento 2: Algoritmos de similitud textual
**ESTADO: COMPLETO Y VERIFICADO**

### Verificación:
- ✅ 4 algoritmos clásicos implementados:
  1. Levenshtein (Distancia de Edición)
  2. Damerau-Levenshtein
  3. Jaccard (n-grams)
  4. TF-IDF Cosine Similarity
- ✅ 2 algoritmos con IA:
  5. Sentence-BERT
  6. LLM-based (Ollama)
- ✅ Explicación detallada paso a paso con detalles matemáticos
- ✅ Selección de 2 o más artículos
- ✅ Extracción de abstract y análisis
- ✅ **Accesible desde menú**: Opción 2 → "Seleccionar artículos y analizar"

### Archivos:
- `app/services/text_similarity_service.py`
- `menu.py` (líneas 344-625)

---

## ❌ Requerimiento 3: Frecuencia de aparición de palabras
**ESTADO: NO IMPLEMENTADO - REQUIERE IMPLEMENTACIÓN**

### Requisitos faltantes:
- ❌ Calcular frecuencia de palabras asociadas a "Concepts of Generative AI in Education"
- ❌ Generar listado de palabras asociadas (máximo 15) desde abstracts
- ❌ Determinar precisión de nuevas palabras
- ❌ **No accesible desde menú**

### Acción requerida:
Crear servicio `app/services/word_frequency_service.py` e integrar en menú

---

## ❌ Requerimiento 4: Agrupamiento jerárquico
**ESTADO: NO IMPLEMENTADO - REQUIERE IMPLEMENTACIÓN**

### Requisitos faltantes:
- ❌ 3 algoritmos de agrupamiento jerárquico
- ❌ Construir dendrograma
- ❌ Preprocesamiento de texto
- ❌ Cálculo de similitud
- ❌ Aplicación de clustering
- ❌ Representación mediante dendrograma
- ❌ Determinar algoritmo más coherente
- ❌ **No accesible desde menú**

### Acción requerida:
Crear servicio `app/services/hierarchical_clustering_service.py` e integrar en menú

### Dependencias necesarias:
- `scipy>=1.11.0` (para clustering jerárquico)
- `matplotlib>=3.7.0` (para dendrogramas)

---

## ❌ Requerimiento 5: Análisis visual
**ESTADO: NO IMPLEMENTADO - REQUIERE IMPLEMENTACIÓN**

### Requisitos faltantes:
- ❌ Mapa de calor geográfico (primer autor)
- ❌ Nube de palabras dinámica (abstracts y keywords)
- ❌ Línea temporal (publicaciones por año y revista)
- ❌ Exportar a PDF
- ❌ **No accesible desde menú**

### Acción requerida:
Crear servicio `app/services/visualization_service.py` e integrar en menú

### Dependencias necesarias:
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0` (opcional, para mapas de calor)
- `wordcloud>=1.9.0` (para nube de palabras)
- `reportlab>=4.0.0` o `fpdf>=2.5.0` (para exportar PDF)
- `plotly>=5.17.0` (opcional, para visualizaciones interactivas)

---

## Resumen Ejecutivo

| Requerimiento | Estado | Accesible desde Menú | Progreso |
|---------------|--------|---------------------|----------|
| 1. Automatización | ✅ COMPLETO | ✅ SÍ | 100% |
| 2. Similitud Textual | ✅ COMPLETO | ✅ SÍ | 100% |
| 3. Frecuencia Palabras | ❌ FALTANTE | ❌ NO | 0% |
| 4. Agrupamiento | ❌ FALTANTE | ❌ NO | 0% |
| 5. Análisis Visual | ❌ FALTANTE | ❌ NO | 0% |

**Progreso General: 2/5 (40%)**

---

## Plan de Implementación

### Fase 1: Requerimiento 3 (Frecuencia de Palabras)
1. Crear `app/services/word_frequency_service.py`
2. Implementar cálculo de frecuencias
3. Implementar generación de palabras asociadas
4. Implementar cálculo de precisión
5. Integrar en menú

### Fase 2: Requerimiento 4 (Agrupamiento Jerárquico)
1. Agregar dependencias (scipy, matplotlib)
2. Crear `app/services/hierarchical_clustering_service.py`
3. Implementar 3 algoritmos de linkage
4. Implementar generación de dendrogramas
5. Implementar evaluación de coherencia
6. Integrar en menú

### Fase 3: Requerimiento 5 (Análisis Visual)
1. Agregar dependencias (matplotlib, wordcloud, reportlab)
2. Crear `app/services/visualization_service.py`
3. Implementar mapa de calor geográfico
4. Implementar nube de palabras
5. Implementar línea temporal
6. Implementar exportación a PDF
7. Integrar en menú

### Fase 4: Integración Final
1. Actualizar menú principal con todas las opciones
2. Verificar accesibilidad de todos los requerimientos
3. Documentación completa

