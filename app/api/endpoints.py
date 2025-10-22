from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from app.services.openalex_service import fetch_articles_metadata_openalex
from app.models.article import ArticleMetadata
from app.utils.validators import SearchRequest, validate_search_parameters
from app.utils.logger import get_logger, log_api_request
from app.utils.metrics import PerformanceTimer, check_rate_limit, get_remaining_requests
from app.utils.exceptions import ValidationError, RateLimitError, convert_to_http_exception
from app.utils.cache import get_cached_openalex_result, cache_openalex_result
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Dict, Any

router = APIRouter()
logger = get_logger("endpoints")

def get_client_ip(request: Request) -> str:
    """Obtener IP del cliente."""
    return request.client.host if request.client else "unknown"

class UniversitySearchRequest(BaseModel):
    """Modelo específico para el endpoint universitario (sin campo query requerido)."""
    max_articles: int = 10
    email: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None

@router.post("/api/v1/fetch-metadata")
async def fetch_metadata(request: SearchRequest, http_request: Request):
    """
    Endpoint principal para extraer metadatos de artículos académicos usando OpenAlex.
    
    Args:
        request: Objeto con query (término de búsqueda), max_articles, email (opcional) y filters (opcional)
        http_request: Objeto de petición HTTP para obtener IP del cliente
        
    Returns:
        Lista de objetos ArticleMetadata con metadatos completos de OpenAlex
    """
    client_ip = get_client_ip(http_request)
    
    # Log de la petición
    log_api_request(
        endpoint="/api/v1/fetch-metadata",
        method="POST",
        query=request.query,
        max_articles=request.max_articles,
        email=request.email,
        filters=request.filters
    )
    
    # Verificar límite de velocidad
    if not check_rate_limit(client_ip):
        remaining = get_remaining_requests(client_ip)
        raise HTTPException(
            status_code=429,
            detail={
                "error": True,
                "error_code": "RATE_LIMIT_ERROR",
                "message": "Límite de velocidad excedido",
                "retry_after": 60,
                "remaining_requests": remaining
            }
        )
    
    # Medir rendimiento
    with PerformanceTimer("fetch_metadata") as timer:
        try:
            # Validar parámetros
            validated_params = validate_search_parameters(
                query=request.query,
                max_articles=request.max_articles,
                email=request.email,
                filters=request.filters
            )
            
            timer.add_data("query", request.query)
            timer.add_data("max_articles", request.max_articles)
            timer.add_data("client_ip", client_ip)
            
            # Verificar caché
            cached_result = get_cached_openalex_result(
                query=request.query,
                max_articles=request.max_articles,
                filters=request.filters
            )
            
            if cached_result:
                logger.info(
                    "Cache hit",
                    query=request.query,
                    max_articles=request.max_articles,
                    client_ip=client_ip
                )
                timer.add_data("cache_hit", True)
                return cached_result
            
            # Realizar búsqueda
            articles, csv_file_path = fetch_articles_metadata_openalex(
                search_query=validated_params["query"],
                max_articles=validated_params["max_articles"],
                email=validated_params["email"],
                filters=validated_params["filters"]
            )
            
            # Preparar respuesta
            response_data = {
                "articles": [article.dict() for article in articles],
                "total_articles": len(articles),
                "csv_file_path": csv_file_path,
                "data_source": "OpenAlex API",
                "message": f"Se encontraron {len(articles)} artículos usando OpenAlex" + 
                          (f" y se exportaron a {csv_file_path}" if csv_file_path else ""),
                "timestamp": datetime.utcnow().isoformat(),
                "cache_hit": False
            }
            
            # Cachear resultado
            cache_openalex_result(
                query=request.query,
                max_articles=request.max_articles,
                filters=request.filters,
                result=response_data
            )
            
            timer.add_data("articles_found", len(articles))
            timer.add_data("cache_hit", False)
            
            logger.info(
                "Search completed successfully",
                query=request.query,
                articles_found=len(articles),
                client_ip=client_ip
            )
            
            return response_data
            
        except ValidationError as e:
            logger.warning(
                "Validation error",
                error=str(e),
                query=request.query,
                client_ip=client_ip
            )
            raise convert_to_http_exception(e)
            
        except Exception as e:
            logger.error(
                "Unexpected error in fetch_metadata",
                error=str(e),
                query=request.query,
                client_ip=client_ip
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error": True,
                    "error_code": "INTERNAL_ERROR",
                    "message": f"Error durante la extracción de metadatos: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

@router.post("/api/v1/uniquindio/generative-ai")
async def fetch_generative_ai_research(request: UniversitySearchRequest, http_request: Request):
    """
    Endpoint específico para el proyecto de la Universidad del Quindío.
    Busca artículos sobre "generative artificial intelligence" usando OpenAlex.
    """
    client_ip = get_client_ip(http_request)
    
    # Forzar búsqueda específica del dominio
    search_query = "generative artificial intelligence"
    
    # Log de la petición específica
    log_api_request(
        endpoint="/api/v1/uniquindio/generative-ai",
        method="POST",
        query=search_query,
        max_articles=request.max_articles,
        email=request.email,
        filters=request.filters
    )
    
    # Verificar límite de velocidad
    if not check_rate_limit(client_ip):
        remaining = get_remaining_requests(client_ip)
        raise HTTPException(
            status_code=429,
            detail={
                "error": True,
                "error_code": "RATE_LIMIT_ERROR",
                "message": "Límite de velocidad excedido",
                "retry_after": 60,
                "remaining_requests": remaining
            }
        )
    
    # Medir rendimiento
    with PerformanceTimer("uniquindio_generative_ai") as timer:
        try:
            # Validar parámetros con dominio específico
            validated_params = validate_search_parameters(
                query=search_query,
                max_articles=request.max_articles,
                email=request.email,
                filters=request.filters
            )
            
            timer.add_data("query", search_query)
            timer.add_data("max_articles", request.max_articles)
            timer.add_data("client_ip", client_ip)
            timer.add_data("university", "Universidad del Quindío")
            
            # Realizar búsqueda específica
            articles, csv_file_path = fetch_articles_metadata_openalex(
                search_query=validated_params["query"],
                max_articles=validated_params["max_articles"],
                email=validated_params["email"],
                filters=validated_params["filters"]
            )
            
            # Preparar respuesta específica para la universidad
            response_data = {
                "university_project": {
                    "institution": "Universidad del Quindío",
                    "course": "Análisis de Algoritmos",
                    "domain": "Generative Artificial Intelligence",
                    "search_query": search_query,
                    "database_source": "OpenAlex",
                    "export_format": "CSV"
                },
                "research_results": {
                    "total_articles": len(articles),
                    "articles": [article.dict() for article in articles],
                    "csv_file_path": csv_file_path,
                    "data_source": "OpenAlex API",
                    "message": f"Proyecto Universidad del Quindío: {len(articles)} artículos sobre IA Generativa exportados a CSV"
                },
                "content_types": {
                    "available_types": ["journal-article", "conference-paper", "book-chapter", "book", "thesis", "report"],
                    "found_types": list(set([article.type for article in articles if article.type]))
                },
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "cache_hit": False,
                    "university_access": True
                }
            }
            
            timer.add_data("articles_found", len(articles))
            
            logger.info(
                "University project search completed",
                university="Universidad del Quindío",
                articles_found=len(articles),
                csv_file=csv_file_path,
                client_ip=client_ip
            )
            
            return response_data
            
        except ValidationError as e:
            logger.warning(
                "University project validation error",
                error=str(e),
                university="Universidad del Quindío",
                client_ip=client_ip
            )
            raise convert_to_http_exception(e)
            
        except Exception as e:
            logger.error(
                "University project unexpected error",
                error=str(e),
                university="Universidad del Quindío",
                client_ip=client_ip
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error": True,
                    "error_code": "UNIVERSITY_PROJECT_ERROR",
                    "message": f"Error en proyecto Universidad del Quindío: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
