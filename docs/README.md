# Documentación del Proyecto

Esta carpeta contiene toda la documentación técnica del sistema bibliométrico.

## Archivos de Documentación

- **ALGORITMO_UNIFICACION.md**: Explicación del algoritmo de unificación y detección de duplicados
- **DEPENDENCIAS_COMPLETAS.md**: Lista completa de dependencias del proyecto
- **GEOGRAPHIC_SERVICE_LOGIC.md**: Lógica implementada en el servicio geográfico
- **IMPLEMENTACION_CROSSREF.md**: Documentación sobre la implementación de CrossRef (descontinuado)
- **MEJORAS_PUBMED.md**: Mejoras implementadas en la extracción de keywords de PubMed
- **OPENALEX_SERVICE_LOGIC.md**: Lógica implementada en el servicio de OpenAlex

## Estructura del Proyecto

```
bibliometria-app/
├── docs/           # Documentación
├── tests/          # Scripts de pruebas
├── app/            # Código de la aplicación
│   ├── api/       # Endpoints de la API
│   ├── services/  # Servicios de bases de datos
│   ├── models/    # Modelos de datos
│   └── utils/     # Utilidades
├── results/        # Archivos CSV generados
│   ├── raw_data/  # Datos crudos por fuente
│   ├── unified/   # Datos unificados
│   ├── duplicates/ # Duplicados detectados
│   └── reports/   # Reportes de procesamiento
└── README.md       # Documentación principal del proyecto
```


