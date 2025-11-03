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
from app.services.data_unification_service import DataUnificationService
from app.config import settings
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os

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

class AutomationRequest(BaseModel):
    """Modelo para solicitudes de automatización."""
    base_query: str = "generative artificial intelligence"
    similarity_threshold: float = 0.75
    max_articles_per_source: int = 350  # Configurado para obtener ~300 artículos únicos (considerando duplicados)

class TextSimilarityRequest(BaseModel):
    """Modelo para análisis de similitud textual."""
    csv_file_path: str
    article_indices: List[int]
    algorithms: Optional[List[str]] = None

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

@router.post("/api/v1/automation/unified-data")
async def automated_data_unification(request: AutomationRequest, http_request: Request):
    """
    Endpoint para automatización completa de descarga y unificación de datos.
    Descarga de múltiples fuentes, elimina duplicados y genera archivos unificados.
    """
    client_ip = get_client_ip(http_request)
    
    # Log de la petición
    log_api_request(
        endpoint="/api/v1/automation/unified-data",
        method="POST",
        query=request.base_query,
        max_articles=request.max_articles_per_source,
        email=None,
        filters={"similarity_threshold": request.similarity_threshold}
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
    with PerformanceTimer("automated_unification") as timer:
        try:
            timer.add_data("base_query", request.base_query)
            timer.add_data("similarity_threshold", request.similarity_threshold)
            timer.add_data("max_articles_per_source", request.max_articles_per_source)
            timer.add_data("client_ip", client_ip)
            
            # Inicializar servicio de unificación
            unification_service = DataUnificationService()
            
            # Ejecutar proceso automatizado
            result = unification_service.run_automated_process(
                base_query=request.base_query,
                similarity_threshold=request.similarity_threshold,
                max_articles_per_source=request.max_articles_per_source
            )
            
            if not result['success']:
                raise Exception(result.get('error', 'Unknown error in automation process'))
            
            # Preparar respuesta
            response_data = {
                "automation_result": {
                    "success": True,
                    "process_type": "Multi-source data download and unification",
                    "base_query": request.base_query,
                    "similarity_threshold": request.similarity_threshold
                },
                "data_statistics": {
                    "total_articles_downloaded": result['total_articles_downloaded'],
                    "unique_articles": result['unique_articles'],
                    "duplicates_removed": result['duplicates_removed'],
                    "sources_processed": result['sources_processed'],
                    "duplication_rate": f"{(result['duplicates_removed'] / result['total_articles_downloaded'] * 100):.1f}%" if result['total_articles_downloaded'] > 0 else "0%"
                },
                "generated_files": {
                    "unified_file": result['unified_file'],
                    "duplicates_file": result['duplicates_file'],
                    "unified_file_size": f"{os.path.getsize(result['unified_file']) / 1024:.1f} KB" if os.path.exists(result['unified_file']) else "N/A",
                    "duplicates_file_size": f"{os.path.getsize(result['duplicates_file']) / 1024:.1f} KB" if os.path.exists(result['duplicates_file']) else "N/A"
                },
                "performance": {
                    "processing_time_seconds": result['processing_time_seconds'],
                    "articles_per_second": f"{result['total_articles_downloaded'] / result['processing_time_seconds']:.2f}" if result['processing_time_seconds'] > 0 else "0"
                },
                "metadata": {
                    "timestamp": result['timestamp'],
                    "cache_hit": False,
                    "automation_completed": True
                }
            }
            
            timer.add_data("total_articles", result['total_articles_downloaded'])
            timer.add_data("unique_articles", result['unique_articles'])
            timer.add_data("duplicates_removed", result['duplicates_removed'])
            timer.add_data("processing_time", result['processing_time_seconds'])
            
            logger.info(
                "Automated unification completed",
                total_articles=result['total_articles_downloaded'],
                unique_articles=result['unique_articles'],
                duplicates_removed=result['duplicates_removed'],
                processing_time=result['processing_time_seconds'],
                client_ip=client_ip
            )
            
            return response_data
            
        except Exception as e:
            logger.error(
                "Automated unification error",
                error=str(e),
                base_query=request.base_query,
                client_ip=client_ip
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error": True,
                    "error_code": "AUTOMATION_ERROR",
                    "message": f"Error en proceso de automatización: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

@router.post("/api/v1/text-similarity/analyze")
async def analyze_text_similarity(request: TextSimilarityRequest, http_request: Request):
    """Analizar similitud textual de abstracts usando 6 algoritmos."""
    from app.services.text_similarity_service import TextSimilarityService
    from app.utils.text_extractor import TextExtractor
    
    client_ip = get_client_ip(http_request)
    
    try:
        logger.info("Text similarity requested", csv=request.csv_file_path, articles=request.article_indices)
        
        if not os.path.exists(request.csv_file_path):
            raise HTTPException(status_code=404, detail={"error": "CSV not found"})
        
        extractor = TextExtractor()
        df = extractor.read_unified_csv(request.csv_file_path)
        articles_data = extractor.extract_abstracts(df, request.article_indices)
        
        if len(articles_data) < 2:
            raise HTTPException(status_code=400, detail={"error": "Need at least 2 articles"})
        
        similarity_service = TextSimilarityService()
        texts = [article['abstract'] for article in articles_data]
        results = similarity_service.analyze_texts_similarity(texts)
        
        # Convertir detalles a tipos serializables
        def make_serializable(obj):
            """Convertir objetos numpy y otros tipos no serializables a tipos nativos."""
            import numpy as np
            
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            return obj
        
        # Preparar datos serializables con manejo especial para detalles complejos
        response_data = {
            "articles": [{"index": int(a['index']), "title": str(a['title'])} for a in articles_data],
            "results": []
        }
        
        for r in results:
            # Simplificar detalles para evitar tipos numpy problemáticos
            simple_details = {}
            for key, value in r.details.items():
                try:
                    # Casos especiales
                    if key == 'matrix' and value is not None:
                        # Matrices muy grandes: solo info de tamaño
                        if isinstance(value, (list, tuple)) and len(value) > 10:
                            simple_details[key] = f"Matrix {len(value)}x{len(value[0]) if isinstance(value[0], (list, tuple)) else 1} (too large to serialize)"
                        else:
                            simple_details[key] = make_serializable(value)
                    elif key in ['backtrace', 'top_contributing_terms']:
                        # Limitar listas largas a primeros elementos
                        if isinstance(value, list):
                            simple_details[key] = value[:10] if len(value) > 10 else make_serializable(value)
                        else:
                            simple_details[key] = make_serializable(value)
                    elif isinstance(value, (str, int, float, bool)):
                        simple_details[key] = value
                    else:
                        simple_details[key] = make_serializable(value)
                except Exception as e:
                    simple_details[key] = f"Error serializing {type(value).__name__}: {str(e)}"
            
            response_data["results"].append({
                "algorithm": str(r.algorithm_name),
                "score": float(r.similarity_score),
                "explanation": str(r.explanation)[:1000],  # Limitar longitud
                "details": simple_details,
                "time": float(r.processing_time)
            })
        
        # Summary
        response_data["summary"] = {
            "algorithms_used": int(len(set(r.algorithm_name for r in results))),
            "avg_similarity": float(sum(r.similarity_score for r in results) / len(results)) if results else 0.0
        }
        
        logger.info("Text similarity completed", results=len(results))
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in text similarity: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get("/api/v1/text-similarity/csv-list")
async def list_unified_csvs():
    """Listar CSVs unificados disponibles."""
    from app.utils.text_extractor import get_unified_csv_list
    
    try:
        csv_files = get_unified_csv_list()
        return {"csvs": csv_files, "total": len(csv_files)}
    except Exception as e:
        logger.error(f"Error listing CSVs: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})
