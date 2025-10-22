"""
Configuración de la aplicación Bibliometría App.
Maneja variables de entorno y configuración por defecto.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Configuración de la aplicación."""
    
    # Configuración de la API
    api_title: str = Field(default="Bibliometría App", env="API_TITLE")
    api_description: str = Field(default="API para extracción de metadatos de artículos académicos", env="API_DESCRIPTION")
    api_version: str = Field(default="1.0.0", env="API_VERSION")
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # Configuración de OpenAlex
    openalex_base_url: str = Field(default="https://api.openalex.org", env="OPENALEX_BASE_URL")
    openalex_user_agent: str = Field(default="BibliometriaApp/1.0", env="OPENALEX_USER_AGENT")
    openalex_timeout: int = Field(default=30, env="OPENALEX_TIMEOUT")
    openalex_max_per_page: int = Field(default=200, env="OPENALEX_MAX_PER_PAGE")
    
    # Configuración de límites
    max_articles_default: int = Field(default=10, env="MAX_ARTICLES_DEFAULT")
    max_articles_limit: int = Field(default=1000, env="MAX_ARTICLES_LIMIT")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Configuración de archivos
    results_dir: str = Field(default="results", env="RESULTS_DIR")
    csv_encoding: str = Field(default="utf-8-sig", env="CSV_ENCODING")
    
    # Configuración de logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Configuración de caché
    cache_enabled: bool = Field(default=False, env="CACHE_ENABLED")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Configuración de desarrollo
    debug: bool = Field(default=False, env="DEBUG")
    reload: bool = Field(default=False, env="RELOAD")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Instancia global de configuración
settings = Settings()


def get_settings() -> Settings:
    """Obtener la configuración de la aplicación."""
    return settings
