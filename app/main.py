from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.endpoints import router as api_router
from app.config import settings
from app.utils.logger import get_logger
from app.utils.metrics import get_health_status, performance_monitor
from app.utils.exceptions import error_handler, BibliometriaAppError
import time

# Configurar logging
logger = get_logger("main")

app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware para logging de peticiones
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Procesar petición
    response = await call_next(request)
    
    # Calcular tiempo de procesamiento
    process_time = time.time() - start_time
    
    # Log de la petición
    logger.info(
        "Request processed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time,
        client_ip=request.client.host if request.client else None
    )
    
    # Agregar headers de métricas
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Incluir las rutas de la API
app.include_router(api_router)

@app.get("/")
async def root():
    """Endpoint raíz de la API."""
    return {
        "message": settings.api_description,
        "version": settings.api_version,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Endpoint de salud del sistema."""
    health_data = get_health_status()
    return JSONResponse(content=health_data)

@app.get("/metrics")
async def get_metrics():
    """Endpoint de métricas de rendimiento."""
    stats = performance_monitor.get_stats()
    return JSONResponse(content=stats)

# Manejador global de excepciones
@app.exception_handler(BibliometriaAppError)
async def bibliometria_exception_handler(request: Request, exc: BibliometriaAppError):
    """Manejador de excepciones personalizadas."""
    logger.error(
        "Application error occurred",
        error=str(exc),
        error_code=exc.error_code,
        url=str(request.url),
        method=request.method
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details
        }
    )
