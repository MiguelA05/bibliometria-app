# Bibliometría App

API para extracción de metadatos de artículos académicos usando OpenAlex, la base de datos global más completa de trabajos académicos.

## Características

- **🌍 Base de datos global**: OpenAlex con 200M+ trabajos académicos
- **📊 Metadatos ricos**: Citas, Open Access, afiliaciones, financiación
- **🔬 API REST moderna**: Sin web scraping, datos estructurados
- **📈 Métricas de impacto**: Número de citas, índices de calidad
- **🔓 Información Open Access**: Estado, URLs, licencias
- **🏛️ Datos institucionales**: Afiliaciones, países, ciudades
- **💰 Información de financiación**: Agencias, proyectos
- **📚 Exportación CSV**: Datos estructurados listos para análisis
- **🧪 Pruebas completas**: Tests unitarios e integración

## Instalación

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Configurar entorno
```bash
# Crear archivo de configuración
cp env.example .env

# O usar el script de inicio automático
python start.py --setup
```

### 3. Ejecutar la aplicación
```bash
# Opción 1: Script de inicio (recomendado)
python start.py

# Opción 2: Comando directo
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Opción 3: Con configuración personalizada
python start.py --host 127.0.0.1 --port 8080 --reload
```

## Uso

### Endpoint disponible

```bash
POST /api/v1/fetch-metadata
```

### Ejemplo de uso

```bash
curl -X POST http://127.0.0.1:8000/api/v1/fetch-metadata \
     -H "Content-Type: application/json" \
     -d '{
       "query": "machine learning",
       "max_articles": 10,
       "email": "tu@email.com",
       "filters": {
         "publication_year": "2024",
         "type": "journal-article"
       }
     }'
```

### Scripts de prueba

```bash
# Probar API
python test_api.py

# Pruebas unitarias
python -m pytest tests/ -v

# Solo ejecutar pruebas
python start.py --test

# Verificar salud del sistema
curl http://localhost:8000/health

# Ver métricas de rendimiento
curl http://localhost:8000/metrics
```

## Documentación

- [OPENALEX_README.md](OPENALEX_README.md) - **Documentación completa de OpenAlex**

## Estructura del proyecto

```
app/
├── api/
│   └── endpoints.py              # Endpoints de la API
├── models/
│   └── article.py                # Modelos de datos para OpenAlex
├── services/
│   └── openalex_service.py       # Servicio OpenAlex
├── utils/
│   ├── logger.py                 # Sistema de logging estructurado
│   ├── validators.py             # Validación robusta de entrada
│   ├── exceptions.py             # Manejo de errores
│   ├── metrics.py                # Métricas de rendimiento
│   └── cache.py                  # Sistema de caché
├── config.py                     # Configuración de la aplicación
└── main.py                       # Aplicación principal

tests/
└── test_openalex_service.py      # Pruebas del servicio OpenAlex

results/                          # Archivos CSV generados
requirements.txt                  # Dependencias del proyecto
env.example                       # Ejemplo de configuración
start.py                          # Script de inicio mejorado
```

## Ventajas de OpenAlex

| Característica | OpenAlex |
|----------------|----------|
| **🌍 Cobertura** | ✅ Global (200M+ trabajos) |
| **⚡ Velocidad** | ✅ Rápido (1-2 segundos) |
| **📊 Metadatos** | ✅ Muy ricos y estructurados |
| **📈 Métricas** | ✅ Citas, impacto, calidad |
| **🔓 Open Access** | ✅ Información completa |
| **🏛️ Instituciones** | ✅ Afiliaciones detalladas |
| **💰 Financiación** | ✅ Datos de financiación |
| **🔧 Mantenimiento** | ✅ Bajo (API estable) |
| **🌐 Dependencia** | ✅ API REST confiable |
| **📚 Filtros** | ✅ Filtros avanzados |

## Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT.
