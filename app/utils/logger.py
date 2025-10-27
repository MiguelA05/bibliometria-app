"""
Sistema de logging estructurado para Bibliometría App.
Proporciona logging consistente y estructurado en toda la aplicación.
"""

import logging
import sys
from typing import Any, Dict, Optional
from datetime import datetime
import structlog
from app.config import settings


def configure_logging() -> None:
    """Configurar el sistema de logging estructurado."""
    
    # Configurar structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.log_format == "json" 
            else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configurar logging estándar
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Obtener un logger estructurado."""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin para agregar logging a cualquier clase."""
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Obtener logger para la clase."""
        return get_logger(self.__class__.__name__)


def log_api_request(
    endpoint: str,
    method: str,
    query: str,
    max_articles: int,
    email: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None
) -> None:
    """Registrar una petición a la API."""
    logger = get_logger("api")
    logger.info(
        "API request received",
        endpoint=endpoint,
        method=method,
        query=query,
        max_articles=max_articles,
        email=email,
        filters=filters,
        timestamp=datetime.utcnow().isoformat()
    )


def log_openalex_request(
    query: str,
    max_articles: int,
    filters: Optional[Dict[str, Any]] = None,
    response_time: Optional[float] = None,
    articles_found: Optional[int] = None,
    error: Optional[str] = None
) -> None:
    """Registrar una petición a OpenAlex."""
    logger = get_logger("openalex")
    
    log_data = {
        "query": query,
        "max_articles": max_articles,
        "filters": filters,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if response_time is not None:
        log_data["response_time"] = response_time
    
    if articles_found is not None:
        log_data["articles_found"] = articles_found
    
    if error:
        log_data["error"] = error
        logger.error("OpenAlex request failed", **log_data)
    else:
        logger.info("OpenAlex request successful", **log_data)


def log_csv_export(
    file_path: str,
    articles_count: int,
    query: str,
    error: Optional[str] = None
) -> None:
    """Registrar exportación de CSV."""
    logger = get_logger("csv_export")
    
    log_data = {
        "file_path": file_path,
        "articles_count": articles_count,
        "query": query,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if error:
        log_data["error"] = error
        logger.error("CSV export failed", **log_data)
    else:
        logger.info("CSV export successful", **log_data)


def log_performance_metrics(
    operation: str,
    duration: float,
    success: bool,
    additional_data: Optional[Dict[str, Any]] = None
) -> None:
    """Registrar métricas de rendimiento."""
    logger = get_logger("performance")
    
    log_data = {
        "operation": operation,
        "duration": duration,
        "success": success,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if additional_data:
        log_data.update(additional_data)
    
    logger.info("Performance metric", **log_data)


# Configurar logging al importar el módulo
configure_logging()






