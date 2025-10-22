from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from app.services.openalex_service import fetch_articles_metadata_openalex
from app.models.article import ArticleMetadata

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    max_articles: int = 10
    email: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None

@router.post("/api/v1/fetch-metadata")
def fetch_metadata(request: SearchRequest):
    """
    Endpoint principal para extraer metadatos de artículos académicos usando OpenAlex.
    
    Args:
        request: Objeto con query (término de búsqueda), max_articles, email (opcional) y filters (opcional)
        
    Returns:
        Lista de objetos ArticleMetadata con metadatos completos de OpenAlex
    """
    try:
        articles, csv_file_path = fetch_articles_metadata_openalex(
            search_query=request.query,
            max_articles=request.max_articles,
            email=request.email,
            filters=request.filters
        )
        
        # Agregar información sobre el archivo CSV a la respuesta
        response_data = {
            "articles": articles,
            "total_articles": len(articles),
            "csv_file_path": csv_file_path,
            "data_source": "OpenAlex API",
            "message": f"Se encontraron {len(articles)} artículos usando OpenAlex y se exportaron a {csv_file_path}" if csv_file_path else f"Se encontraron {len(articles)} artículos usando OpenAlex"
        }
        
        return response_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la extracción de metadatos: {str(e)}"
        )
