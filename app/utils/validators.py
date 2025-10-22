"""
Validación robusta de entrada para la API de Bibliometría App.
Incluye validación de parámetros, límites y filtros.
"""

import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator, model_validator
from app.config import settings


class SearchRequest(BaseModel):
    """Modelo de validación para peticiones de búsqueda."""
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Término de búsqueda (1-500 caracteres)"
    )
    max_articles: int = Field(
        default=settings.max_articles_default,
        ge=1,
        le=settings.max_articles_limit,
        description=f"Número máximo de artículos (1-{settings.max_articles_limit})"
    )
    email: Optional[str] = Field(
        default=None,
        description="Email para polite pool de OpenAlex"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filtros adicionales para la búsqueda"
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Validar el término de búsqueda."""
        if not v or not v.strip():
            raise ValueError("El término de búsqueda no puede estar vacío")
        
        # Limpiar caracteres especiales peligrosos
        cleaned = re.sub(r'[<>"\']', '', v.strip())
        if not cleaned:
            raise ValueError("El término de búsqueda debe contener caracteres válidos")
        
        return cleaned
    
    @validator('email')
    def validate_email(cls, v):
        """Validar formato de email."""
        if v is None:
            return v
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError("Formato de email inválido")
        
        return v.lower().strip()
    
    @validator('filters')
    def validate_filters(cls, v):
        """Validar filtros de búsqueda."""
        if v is None:
            return v
        
        # Filtros válidos de OpenAlex
        valid_filters = {
            'publication_year', 'from_publication_date', 'to_publication_date',
            'type', 'is_oa', 'oa_status', 'source_type', 'source_id',
            'author_id', 'institutions.id', 'concepts.id', 'concepts.display_name',
            'language', 'publisher', 'venue', 'has_doi', 'has_pmid', 'has_pmcid'
        }
        
        for key in v.keys():
            if key not in valid_filters:
                raise ValueError(f"Filtro '{key}' no es válido. Filtros válidos: {', '.join(sorted(valid_filters))}")
        
        # Validar filtros específicos
        if 'publication_year' in v:
            year = v['publication_year']
            if isinstance(year, str):
                try:
                    year_int = int(year)
                    if not (1800 <= year_int <= 2030):
                        raise ValueError("Año de publicación debe estar entre 1800 y 2030")
                    v['publication_year'] = str(year_int)
                except ValueError:
                    raise ValueError("Año de publicación debe ser un número válido")
        
        if 'type' in v:
            valid_types = {
                'journal-article', 'conference-paper', 'book-chapter', 'book',
                'dataset', 'dissertation', 'editorial', 'erratum', 'letter',
                'other', 'peer-review', 'posted-content', 'proceedings-article',
                'reference-entry', 'report', 'review', 'standard', 'thesis'
            }
            if v['type'] not in valid_types:
                raise ValueError(f"Tipo '{v['type']}' no es válido. Tipos válidos: {', '.join(sorted(valid_types))}")
        
        if 'oa_status' in v:
            valid_statuses = {'gold', 'green', 'hybrid', 'bronze', 'closed'}
            if v['oa_status'] not in valid_statuses:
                raise ValueError(f"Estado OA '{v['oa_status']}' no es válido. Estados válidos: {', '.join(sorted(valid_statuses))}")
        
        return v
    
    @model_validator(mode='after')
    def validate_request(self):
        """Validación adicional de la petición completa."""
        query = self.query
        max_articles = self.max_articles
        
        # Validar que la consulta no sea demasiado genérica
        if len(query.split()) < 2 and len(query) < 10:
            raise ValueError("La consulta debe tener al menos 2 palabras o 10 caracteres para obtener resultados relevantes")
        
        return self


class FilterValidator:
    """Validador especializado para filtros de OpenAlex."""
    
    @staticmethod
    def validate_date_filter(date_str: str, field_name: str) -> str:
        """Validar formato de fecha."""
        date_pattern = r'^\d{4}-\d{2}-\d{2}$'
        if not re.match(date_pattern, date_str):
            raise ValueError(f"{field_name} debe tener formato YYYY-MM-DD")
        
        # Validar que la fecha sea razonable
        from datetime import datetime
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            if date_obj.year < 1800 or date_obj.year > 2030:
                raise ValueError(f"{field_name} debe estar entre 1800 y 2030")
        except ValueError as e:
            raise ValueError(f"{field_name} inválida: {str(e)}")
        
        return date_str
    
    @staticmethod
    def validate_id_filter(id_str: str, field_name: str) -> str:
        """Validar formato de ID."""
        if not id_str or not id_str.strip():
            raise ValueError(f"{field_name} no puede estar vacío")
        
        # Validar formato básico de ID
        if not re.match(r'^[a-zA-Z0-9._-]+$', id_str):
            raise ValueError(f"{field_name} contiene caracteres inválidos")
        
        return id_str.strip()


def validate_search_parameters(
    query: str,
    max_articles: int,
    email: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validar parámetros de búsqueda y devolver datos limpios.
    
    Args:
        query: Término de búsqueda
        max_articles: Número máximo de artículos
        email: Email opcional
        filters: Filtros opcionales
        
    Returns:
        Diccionario con parámetros validados
        
    Raises:
        ValueError: Si los parámetros no son válidos
    """
    try:
        # Crear modelo de validación
        request = SearchRequest(
            query=query,
            max_articles=max_articles,
            email=email,
            filters=filters
        )
        
        return request.dict()
        
    except Exception as e:
        raise ValueError(f"Error de validación: {str(e)}")


def sanitize_query(query: str) -> str:
    """
    Sanitizar término de búsqueda para OpenAlex.
    
    Args:
        query: Término de búsqueda original
        
    Returns:
        Término sanitizado
    """
    # Remover caracteres especiales peligrosos
    sanitized = re.sub(r'[<>"\']', '', query)
    
    # Normalizar espacios
    sanitized = re.sub(r'\s+', ' ', sanitized.strip())
    
    # Limitar longitud
    if len(sanitized) > 500:
        sanitized = sanitized[:500].rsplit(' ', 1)[0]
    
    return sanitized


def validate_rate_limit(user_ip: str, requests_count: int) -> bool:
    """
    Validar límite de velocidad de peticiones.
    
    Args:
        user_ip: IP del usuario
        requests_count: Número de peticiones en el último minuto
        
    Returns:
        True si está dentro del límite, False si excede
    """
    return requests_count <= settings.rate_limit_per_minute
