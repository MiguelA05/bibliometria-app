import requests
import re
import pandas as pd
import os
import time
import json
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from app.models.article import ArticleMetadata
from app.config import settings
from app.utils.logger import get_logger, log_openalex_request, log_csv_export
from app.utils.metrics import PerformanceTimer
from app.utils.exceptions import OpenAlexError, CSVExportError, error_handler
from app.services.geographic_service import GeographicDataService

class OpenAlexService:
    """
    Servicio para interactuar con la API de OpenAlex.
    Reemplaza completamente el web scraping con llamadas a la API REST.
    """
    
    def __init__(self, email: Optional[str] = None):
        """
        Inicializar el servicio de OpenAlex.
        
        Args:
            email: Email para acceder al "polite pool" (recomendado)
        """
        self.base_url = settings.openalex_base_url
        self.headers = {
            'User-Agent': f'{settings.openalex_user_agent} (mailto:{email})' if email else settings.openalex_user_agent
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.logger = get_logger("openalex_service")
        self.geographic_service = GeographicDataService()
    
    def search_works(self, query: str, max_articles: int = 10, 
                    filters: Optional[Dict[str, Any]] = None) -> Tuple[List[ArticleMetadata], str]:
        """
        Buscar trabajos académicos en OpenAlex.
        
        Args:
            query: Término de búsqueda
            max_articles: Número máximo de artículos a devolver
            filters: Filtros adicionales (año, tipo, etc.)
            
        Returns:
            Tupla con (lista_de_artículos, ruta_del_archivo_csv)
        """
        articles = []
        csv_file_path = ""
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Searching OpenAlex: {query}")
            
            # Construir parámetros de búsqueda
            params = {
                'search': query,
                'per_page': min(max_articles, settings.openalex_max_per_page),
            }
            
            # Agregar email solo si está disponible y es válido
            # email = self.headers.get('User-Agent', '').split('mailto:')[-1].split(')')[0] if 'mailto:' in self.headers.get('User-Agent', '') else None
            # if email and '@' in email:
            #     params['mailto'] = email
            
            # Agregar filtros si se proporcionan
            if filters:
                for key, value in filters.items():
                    params[key] = value
            
            # Realizar búsqueda
            response = self.session.get(f"{self.base_url}/works", params=params, timeout=settings.openalex_timeout)
            response.raise_for_status()
            
            data = response.json()
            works = data.get('results', [])
            
            if not works:
                self.logger.warning("No results found in OpenAlex", query=query)
                log_openalex_request(query, max_articles, filters, time.time() - start_time, 0)
                return articles, csv_file_path
            
            self.logger.info(f"Found {len(works)} results in OpenAlex", query=query)
            
            # Procesar cada trabajo
            for i, work in enumerate(works[:max_articles]):
                try:
                    article = self._process_work(work)
                    if article:
                        articles.append(article)
                        self.logger.debug(f"Processed article {len(articles)}: {article.title[:50]}...")
                except Exception as e:
                    self.logger.error(f"Error processing article {i+1}: {e}")
                    continue
            
            # Log de éxito
            response_time = time.time() - start_time
            log_openalex_request(query, max_articles, filters, response_time, len(articles))
            
        except requests.exceptions.RequestException as e:
            response_time = time.time() - start_time
            log_openalex_request(query, max_articles, filters, response_time, 0, str(e))
            raise error_handler.handle_openalex_error(e, query)
        except Exception as e:
            response_time = time.time() - start_time
            log_openalex_request(query, max_articles, filters, response_time, 0, str(e))
            raise error_handler.handle_unexpected_error(e, "search_works")
        
        # Exportar a CSV si hay artículos
        if articles:
            try:
                csv_file_path = self._export_to_csv(articles, query)
                log_csv_export(csv_file_path, len(articles), query)
            except Exception as e:
                log_csv_export("", len(articles), query, str(e))
                raise error_handler.handle_csv_export_error(e)
        
        return articles, csv_file_path
    
    def _process_work(self, work: Dict[str, Any]) -> Optional[ArticleMetadata]:
        """
        Procesar un trabajo individual de OpenAlex.
        
        Args:
            work: Datos del trabajo en formato JSON de OpenAlex
            
        Returns:
            Objeto ArticleMetadata o None si hay error
        """
        try:
            # Extraer información básica
            title = work.get('title')
            if title and isinstance(title, str):
                title = title.strip() or 'Title not available'
            else:
                title = 'Title not available'
            
            abstract = self._extract_abstract(work)
            
            # Extraer autores y afiliaciones
            authors, affiliations = self._extract_authors_and_affiliations(work)
            
            # Extraer información de publicación
            publication_date = self._extract_publication_date(work)
            publication_year = work.get('publication_year')
            publication_month = work.get('publication_month')
            publication_day = work.get('publication_day')
            
            # Extraer URLs
            article_url = self._extract_article_url(work)
            doi = work.get('doi')
            doi_url = f"https://doi.org/{doi}" if doi else None
            
            # Asegurar que los campos requeridos no sean None
            if not authors:
                authors = []
            if not affiliations:
                affiliations = []
            if not abstract:
                abstract = "Abstract not available"
            if not article_url:
                article_url = "URL not available"
            
            # Extraer información de la fuente
            source_info = self._extract_source_info(work)
            
            # Extraer información de Open Access
            oa_info = self._extract_open_access_info(work)
            
            # Extraer conceptos y temas
            concepts, topics = self._extract_concepts_and_topics(work)
            
            # Extraer información de financiación
            funding = self._extract_funding_info(work)
            
            # Extraer metadatos bibliográficos
            biblio = self._extract_biblio_info(work)
            
            # Extraer datos geográficos
            geographic_data = self.geographic_service.extract_geographic_data(work)
            
            # Crear objeto ArticleMetadata
            article = ArticleMetadata(
                # Campos básicos
                title=title,
                authors=authors,
                affiliations=affiliations,
                abstract=abstract,
                publication_date=publication_date,
                article_url=article_url,
                
                # Campos principales de OpenAlex
                openalex_id=work.get('id'),
                doi=doi,
                doi_url=doi_url,
                publication_year=publication_year,
                type=work.get('type'),
                language=work.get('language'),
                is_oa=oa_info.get('is_oa'),
                oa_url=oa_info.get('oa_url'),
                oa_status=oa_info.get('oa_status'),
                
                # Información de la fuente
                source_title=source_info.get('title'),
                source_type=source_info.get('type'),
                publisher=work.get('primary_location', {}).get('publisher'),
                
                # Journal mapeado desde source_title (siempre que esté disponible)
                journal=source_info.get('title') or None,
                
                # Métricas de impacto
                cited_by_count=work.get('cited_by_count'),
                
                # Clasificación temática
                topics=topics,
                
                # Información de licencia
                license=work.get('license'),
                
                # Información geográfica
                author_countries=geographic_data.get('author_countries'),
                author_cities=geographic_data.get('author_cities'),
                institution_countries=geographic_data.get('institution_countries'),
                institution_cities=geographic_data.get('institution_cities'),
                geographic_coordinates=geographic_data.get('geographic_coordinates')
            )
            
            return article
            
        except Exception as e:
            print(f"[WARNING] Error procesando trabajo: {e}")
            return None
    
    def _extract_abstract(self, work: Dict[str, Any]) -> str:
        """Extraer abstract del trabajo."""
        # Priorizar abstract directo
        abstract = work.get('abstract')
        if abstract and abstract.strip() and abstract != 'None':
            return abstract.strip()
        
        # Intentar reconstruir desde índice invertido
        abstract_inverted = work.get('abstract_inverted_index')
        if abstract_inverted and isinstance(abstract_inverted, dict):
            try:
                words = []
                for word, positions in abstract_inverted.items():
                    if isinstance(positions, list):
                        for pos in positions:
                            words.append((pos, word))
                words.sort()
                reconstructed = ' '.join([word for _, word in words])
                if reconstructed.strip():
                    return reconstructed.strip()
            except Exception as e:
                print(f"[WARNING] Error reconstruyendo abstract: {e}")
        
        # Buscar en otros campos posibles
        for field in ['summary', 'description', 'content']:
            if work.get(field) and work[field].strip():
                return work[field].strip()
        
        return "Abstract not available"
    
    def _extract_authors_and_affiliations(self, work: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Extraer autores y afiliaciones."""
        authors = []
        affiliations = []
        
        authorships = work.get('authorships', [])
        
        if not authorships:
            # Si no hay authorships, intentar extraer información alternativa
            corresponding_author_ids = work.get('corresponding_author_ids', [])
            if corresponding_author_ids:
                authors.append("Author information not fully available")
                affiliations.append("Institution information not available")
            return authors, affiliations
        
        for authorship in authorships:
            # Extraer nombre del autor
            author = authorship.get('author', {})
            if author:
                display_name = author.get('display_name', '')
                if display_name:
                    authors.append(display_name)
            
            # Extraer afiliaciones con información geográfica
            institutions = authorship.get('institutions', [])
            if institutions:
                for institution in institutions:
                    if institution:
                        display_name = institution.get('display_name', '')
                        country = institution.get('country_code', '')
                        city = institution.get('city', '')
                        region = institution.get('region', '')
                        
                        if display_name:
                            # Construir afiliación con información geográfica
                            affiliation_parts = [display_name]
                            
                            # Agregar ciudad si está disponible
                            if city:
                                affiliation_parts.append(city)
                            
                            # Agregar región si está disponible y es diferente de la ciudad
                            if region and region != city:
                                affiliation_parts.append(region)
                            
                            # Agregar país si está disponible
                            if country:
                                affiliation_parts.append(country)
                            
                            affiliation = ", ".join(affiliation_parts)
                            affiliations.append(affiliation)
            else:
                # Si no hay instituciones específicas, usar información del autor
                author_country = author.get('country_code', '')
                if author_country:
                    affiliations.append(f"Country: {author_country}")
                else:
                    # Si no hay información geográfica del autor, agregar mensaje informativo
                    affiliations.append("Institution information not available")
        
        return authors, affiliations
    
    def _extract_publication_date(self, work: Dict[str, Any]) -> str:
        """Extraer fecha de publicación."""
        # Priorizar fecha completa
        publication_date = work.get('publication_date')
        if publication_date and publication_date != 'None':
            return publication_date
        
        # Construir fecha desde componentes individuales
        year = work.get('publication_year')
        month = work.get('publication_month')
        day = work.get('publication_day')
        
        if year:
            date_parts = [str(year)]
            if month and month != 'None':
                date_parts.append(f"{int(month):02d}")
                if day and day != 'None':
                    date_parts.append(f"{int(day):02d}")
            return '-'.join(date_parts)
        
        # Si no hay fecha específica, usar fecha de creación del registro
        created_date = work.get('created_date')
        if created_date:
            # Extraer solo la parte de la fecha (YYYY-MM-DD)
            return created_date.split('T')[0]
        
        return "Date not available"
    
    def _extract_article_url(self, work: Dict[str, Any]) -> str:
        """Extraer URL del artículo."""
        # Priorizar URL de Open Access
        open_access = work.get('open_access', {})
        oa_url = open_access.get('oa_url')
        if oa_url:
            return oa_url
        
        # Usar URL primaria
        primary_location = work.get('primary_location', {})
        landing_page_url = primary_location.get('landing_page_url')
        if landing_page_url:
            return landing_page_url
        
        # Usar ID de OpenAlex como fallback
        openalex_id = work.get('id')
        if openalex_id:
            return openalex_id
        
        return "URL not available"
    
    def _extract_source_info(self, work: Dict[str, Any]) -> Dict[str, Any]:
        """Extraer información de la fuente."""
        primary_location = work.get('primary_location', {})
        source = primary_location.get('source', {}) if primary_location else {}
        
        # Asegurar que source es un diccionario
        if not isinstance(source, dict):
            source = {}
        
        return {
            'title': source.get('display_name'),
            'type': source.get('type'),
            'url': source.get('homepage_url'),
            'issn': source.get('issn'),
            'is_oa': source.get('is_oa')
        }
    
    def _extract_open_access_info(self, work: Dict[str, Any]) -> Dict[str, Any]:
        """Extraer información de Open Access."""
        open_access = work.get('open_access', {})
        
        return {
            'is_oa': open_access.get('is_oa'),
            'oa_url': open_access.get('oa_url'),
            'oa_status': open_access.get('oa_status')
        }
    
    def _extract_concepts_and_topics(self, work: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Extraer conceptos y temas."""
        concepts = work.get('concepts', [])
        topics = []
        
        for concept in concepts:
            if concept.get('display_name'):
                topics.append(concept.get('display_name'))
        
        return concepts, topics
    
    def _extract_funding_info(self, work: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extraer información de financiación."""
        return work.get('funding', [])
    
    def _extract_biblio_info(self, work: Dict[str, Any]) -> Dict[str, Any]:
        """Extraer metadatos bibliográficos."""
        biblio = work.get('biblio', {})
        return biblio
    
    def _export_to_csv(self, articles: List[ArticleMetadata], search_query: str) -> str:
        """
        Exportar artículos a CSV.
        
        Args:
            articles: Lista de artículos
            search_query: Término de búsqueda
            
        Returns:
            Ruta del archivo CSV
        """
        try:
            # Crear estructura de directorios organizada
            base_dir = settings.results_dir
            raw_data_dir = os.path.join(base_dir, "raw_data")
            
            # Crear directorio si no existe
            os.makedirs(raw_data_dir, exist_ok=True)
            
            # Generar nombre de archivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_query = re.sub(r'[^\w\s-]', '', search_query).strip()
            safe_query = re.sub(r'[-\s]+', '_', safe_query)
            filename = f"resultados_openalex_{safe_query}_{timestamp}.csv"
            file_path = os.path.join(raw_data_dir, filename)
            
            # Convertir a DataFrame
            articles_data = []
            for article in articles:
                article_dict = {
                    'title': article.title,
                    'authors': '; '.join(article.authors) if article.authors else '',
                    'affiliations': '; '.join(article.affiliations) if article.affiliations else '',
                    'abstract': article.abstract or '',
                    'publication_date': article.publication_date,
                    'article_url': article.article_url,
                    'doi': article.doi,
                    'publication_year': article.publication_year,
                    'type': article.type,
                    'language': article.language or 'en',
                    'keywords': '; '.join(article.topics) if article.topics else '',
                    'license': article.license,
                    'journal': article.journal or '',
                    
                    # Datos geográficos
                    'author_countries': '; '.join(article.author_countries) if article.author_countries else '',
                    'author_cities': '; '.join(article.author_cities) if article.author_cities else '',
                    'institution_countries': '; '.join(article.institution_countries) if article.institution_countries else '',
                    'institution_cities': '; '.join(article.institution_cities) if article.institution_cities else '',
                    'geographic_coordinates': json.dumps(article.geographic_coordinates) if article.geographic_coordinates else '',
                    'data_source': getattr(article, 'source', 'OpenAlex')
                }
                articles_data.append(article_dict)
            
            # Exportar CSV
            df = pd.DataFrame(articles_data)
            df.to_csv(file_path, index=False, encoding=settings.csv_encoding)
            
            self.logger.info(f"Exported {len(articles)} articles to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error exporting CSV: {e}")
            raise CSVExportError(f"Error al exportar CSV: {e}", file_path)

# Función de conveniencia para mantener compatibilidad
def fetch_articles_metadata_openalex(search_query: str, max_articles: int = 10, 
                                   email: Optional[str] = None,
                                   filters: Optional[Dict[str, Any]] = None) -> Tuple[List[ArticleMetadata], str]:
    """
    Función de conveniencia para buscar artículos en OpenAlex.
    
    Args:
        search_query: Término de búsqueda
        max_articles: Número máximo de artículos
        email: Email para polite pool (opcional)
        filters: Filtros adicionales (opcional)
        
    Returns:
        Tupla con (lista_de_artículos, ruta_del_archivo_csv)
    """
    service = OpenAlexService(email=email)
    return service.search_works(search_query, max_articles, filters)
