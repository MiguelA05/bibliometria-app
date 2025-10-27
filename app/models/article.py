from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ArticleMetadata(BaseModel):
    # Campos básicos (requeridos)
    title: str
    authors: List[str]
    affiliations: List[str]
    abstract: str
    publication_date: str
    article_url: str
    
    # Campos principales de OpenAlex
    openalex_id: Optional[str] = None
    doi: Optional[str] = None
    doi_url: Optional[str] = None
    publication_year: Optional[int] = None
    type: Optional[str] = None  # journal-article, conference-paper, etc.
    language: Optional[str] = None
    is_oa: Optional[bool] = None  # Open Access
    oa_url: Optional[str] = None
    oa_status: Optional[str] = None  # gold, green, hybrid, closed
    
    # Información de la fuente
    source_title: Optional[str] = None
    source_type: Optional[str] = None  # journal, conference, repository
    publisher: Optional[str] = None
    
    # Métricas de impacto
    cited_by_count: Optional[int] = None
    
    # Clasificación temática
    topics: Optional[List[str]] = None
    
    # Información de licencia
    license: Optional[str] = None
    source: Optional[str] = None  # Fuente de datos (OpenAlex_General, etc.)
    
    # Información geográfica
    author_countries: Optional[List[str]] = None  # Países de los autores
    author_cities: Optional[List[str]] = None     # Ciudades de los autores
    institution_countries: Optional[List[str]] = None  # Países de las instituciones
    institution_cities: Optional[List[str]] = None     # Ciudades de las instituciones
    geographic_coordinates: Optional[List[Dict[str, Any]]] = None  # Coordenadas lat/lng
