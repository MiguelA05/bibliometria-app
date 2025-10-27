# 📁 Análisis de Estructura de Carpetas - Sistema Bibliométrico

## 🔍 Explicación de Datos en Cada Carpeta

### 1. 📂 `raw_data/` - Datos sin procesar

**¿Qué contiene?**
- Datos descargados directamente de OpenAlex sin procesar
- Cada archivo corresponde a una fuente específica (OpenAlex_General, OpenAlex_Articles, etc.)
- Formato: CSV con todos los campos completos + datos geográficos

**Ejemplo de archivo:**
```
resultados_openalex_generative_artificial_intelligence_20251027_092620.csv
```

**Columnas incluidas:**
- Básicas: title, authors, affiliations, abstract, doi, etc.
- Geográficas: institution_countries, author_countries, etc.
- **NO incluye:** campo `data_source` (porque TODOS son de esa fuente)

**¿Por qué existe?**
- ✅ **NECESARIO** - Trazabilidad de las fuentes originales
- Permite auditabilidad del proceso
- Facilita debugging cuando hay problemas con una fuente específica
- Permite re-procesar solo una fuente sin re-descargar todo

**Justificación:** ✅ **MANTENER**

---

### 2. 📂 `unified/` - Datos unificados sin duplicados

**¿Qué contiene?**
- Datos de TODAS las fuentes combinadas
- Duplicados eliminados mediante algoritmo de similitud
- Datos listos para análisis final
- Formato: CSV único por consulta

**Ejemplo de archivo:**
```
unified_generative_ai_20251027_092624_unified.csv
```

**Columnas incluidas:**
- Todas las de raw_data +
- **`data_source`**: origen (OpenAlex_General, etc.)
- Todas las columnas geográficas integradas

**¿Por qué existe?**
- ✅ **NECESARIO** - Es el archivo PRINCIPAL de trabajo
- Elimina duplicados de múltiples fuentes
- Combina información de diferentes búsquedas
- Es el input directo para análisis bibliométricos

**Justificación:** ✅ **MANTENER** (Es el objetivo principal del sistema)

---

### 3. 📂 `duplicates/` - Registro de duplicados eliminados

**¿Qué contiene?**
- Registro de artículos identificados como duplicados
- Información sobre qué artículo se conservó (original)
- Score de similitud que causó la eliminación
- Fuente del duplicado eliminado

**Ejemplo de archivo:**
```
unified_generative_ai_20251027_092624_duplicates.csv
```

**Columnas:**
- `duplicate_title`, `original_title`
- `similarity_score` (0.8, 0.9, etc.)
- `duplicate_source`, `elimination_reason`
- `duplicate_doi`, `original_doi`
- `duplicate_authors`, `original_authors`

**¿Por qué existe?**
- ⚠️ **OPCIONAL** - Puede ser útil o redundante según el caso
- Ventajas:
  - Transparencia en el proceso de deduplicación
  - Auditoría de qué se eliminó y por qué
  - Debugging de falsos positivos en deduplicación
- Desventajas:
  - Archivos pequeños (casi vacíos si no hay duplicados)
  - Información ya está en el procesamiento

**Recomendación:** ⚠️ **CONDICIONAL** - Solo útil si hay muchos duplicados. Considerar eliminarlo si siempre está vacío.

---

### 4. 📂 `reports/` - Reportes de procesamiento

**¿Qué contiene?**
- Estadísticas del proceso de unificación
- Métricas de rendimiento
- Distribución de datos por tipo/año/fuente
- Resumen ejecutivo del procesamiento

**Ejemplo de archivo:**
```
unified_generative_ai_20251027_092624_processing_report.csv
```

**Datos incluidos:**
```csv
metric,value,description
Total Articles Processed,30,Total articles downloaded from all sources
Unique Articles,30,Articles after duplicate removal
Duplicates Removed,0,Articles identified as duplicates
Duplication Rate,0.0%,Percentage of articles that were duplicates
Processing Date,2025-10-27 09:26:24,Date and time of processing
Articles from OpenAlex_General,30,Unique articles from OpenAlex_General source
Type: article,29,Articles of type article
Year: 2023,28,Articles published in 2023
```

**¿Por qué existe?**
- ✅ **ÚTIL** - Proporciona métricas rápidas
- No requiere abrir archivos grandes para ver resúmenes
- Útil para dashboards o reportes ejecutivos
- Trazabilidad de parámetros de búsqueda

**Justificación:** ✅ **MANTENER** - Es informativo y ligero

---

## 🎯 EVALUACIÓN FINAL

### Estructura ACTUAL:
```
results/
├── raw_data/      ✅ NECESARIO - Trazabilidad de fuentes
├── unified/       ✅ NECESARIO - Datos finales para análisis
├── duplicates/    ⚠️  OPCIONAL - Solo útil si hay muchos duplicados
└── reports/       ✅ ÚTIL - Estadísticas rápidas
```

### Recomendación de CAMBIOS:

#### Opción A: Mantener Actual (RECOMENDADO)
**Ventajas:**
- Máxima trazabilidad
- Separación clara de responsabilidades
- Fácil debugging

**Desventajas:**
- Muchos archivos para gestión
- `duplicates/` puede ser redundante

#### Opción B: Simplificar (SI duplicate está siempre vacío)
```
results/
├── raw_data/     ✅ Mantener
├── unified/      ✅ Mantener  
└── reports/      ✅ Mantener
```

**Eliminar:**
- `duplicates/` (si siempre está vacío)

---

## 📊 Conclusiones

### Carpetas CLARAMENTE necesarias:
1. **`raw_data/`** - Fuente de datos individuales por fuente
2. **`unified/`** - Dataset final listo para análisis

### Carpetas útiles pero opcionales:
3. **`reports/`** - Estadísticas y métricas rápidas
4. **`duplicates/`** - Solo útil si hay duplicados significativos

### Sugerencia:
- **MANTENER** la estructura actual es correcto
- Considerar eliminar `duplicates/` si siempre está vacío
- Los archivos son pequeños y no generan overhead significativo

---

## 💡 Comparación con tu Imagen

Tu imagen muestra exactamente esta estructura:
- ✅ `results/` (carpeta principal)
- ✅ `raw_data/` con datos sin procesar
- ✅ `duplicates/` con un archivo casi vacío
- ✅ `reports/` con reporte de procesamiento
- ✅ `unified/` con el archivo unificado final

**Todo está funcionando correctamente según el diseño del sistema.**
