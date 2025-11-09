"""
Servicio para interactuar con la API de ArXiv.
ArXiv es un repositorio de preprints en ciencias.
"""

import requests
import re
import os
import time
import json
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from app.models.article import ArticleMetadata
from app.config import settings
from app.utils.logger import get_logger
from app.services.geographic_service import GeographicDataService


class ArXivService:
    """
    Servicio para interactuar con la API de ArXiv.
    ArXiv ofrece una API REST gratuita para acceder a preprints.
    """
    
    def __init__(self, email: Optional[str] = None):
        """
        Inicializar el servicio de ArXiv.
        
        Args:
            email: Email para identificar la aplicación (recomendado)
        """
        self.base_url = "https://export.arxiv.org/api/query"
        self.headers = {
            'User-Agent': 'BibliometriaApp/1.0'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.logger = get_logger("arxiv_service")
        # Aumentar timeout para ArXiv
        self.timeout = 60
    
    def search_works(self, query: str, max_articles: int = 10, 
                    filters: Optional[Dict[str, Any]] = None) -> Tuple[List[ArticleMetadata], str]:
        """
        Buscar preprints en ArXiv.
        
        Args:
            query: Término de búsqueda
            max_articles: Número máximo de artículos a devolver
            filters: Filtros adicionales (año, etc.)
            
        Returns:
            Tupla con (lista_de_artículos, ruta_del_archivo_csv)
        """
        articles = []
        csv_file_path = ""
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Searching ArXiv: {query}")
            
            # Construir búsqueda
            search_query = query
            if filters and 'year' in filters:
                # ArXiv permite filtrar por fecha
                search_query = f"{query} AND submittedDate:[{filters['year']}01010000 TO {filters['year']}12312359]"
            
            params = {
                'search_query': search_query,
                'start': 0,
                'max_results': min(max_articles, 100)
            }
            
            response = self.session.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            # Parsear respuesta XML/Atom
            xml_content = response.text
            articles_data = self._parse_arxiv_response(xml_content)
            
            # Buscar más artículos si es necesario para compensar los sin abstract
            if len(articles_data) < max_articles:
                # Intentar obtener más resultados
                additional_needed = max_articles - len(articles_data)
                params['start'] = len(articles_data)
                params['max_results'] = min(additional_needed * 2, 100)  # Buscar 2x para compensar
                
                try:
                    response = self.session.get(self.base_url, params=params, timeout=self.timeout)
                    response.raise_for_status()
                    additional_data = self._parse_arxiv_response(response.text)
                    articles_data.extend(additional_data)
                except Exception as e:
                    self.logger.warning(f"Could not fetch additional ArXiv articles: {e}")
            
            # Procesar cada artículo hasta obtener max_articles con abstracts válidos
            for article_data in articles_data:
                if len(articles) >= max_articles:
                    break
                    
                article = self._process_article(article_data)
                if article:
                    # Verificar que el abstract sea válido
                    abstract = article.abstract or ""
                    abstract_lower = abstract.lower().strip()
                    
                    if (abstract and 
                        abstract_lower != '' and 
                        abstract_lower != 'none' and 
                        abstract_lower != 'nan' and
                        'abstract not available' not in abstract_lower and
                        len(abstract) > 20):  # Mínimo 20 caracteres
                        articles.append(article)
                    else:
                        self.logger.debug(f"Skipping ArXiv article without valid abstract: {article.title[:50]}...")
            
            if len(articles) < max_articles:
                self.logger.warning(
                    f"Only found {len(articles)} ArXiv articles with valid abstracts out of {max_articles} requested"
                )
            
            # Exportar a CSV
            csv_file_path = self._export_to_csv(articles, query)
            
            elapsed_time = time.time() - start_time
            
            self.logger.info(
                f"ArXiv search completed: {len(articles)} articles in {elapsed_time:.2f}s"
            )
            
            return articles, csv_file_path
            
        except Exception as e:
            self.logger.error(f"Error searching ArXiv: {e}")
            raise
    
    def _parse_arxiv_response(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parsear respuesta XML de ArXiv."""
        articles = []
        
        try:
            # Extraer entradas de Atom de forma más eficiente
            entry_pattern = re.compile(r'<entry>(.*?)</entry>', re.DOTALL)
            entries = entry_pattern.findall(xml_content)
            
            for entry_xml in entries:
                try:
                    article = {}
                    
                    # Extraer título (limpiar saltos de línea)
                    title_match = re.search(r'<title>(.*?)</title>', entry_xml, re.DOTALL)
                    if title_match:
                        title = title_match.group(1).strip()
                        title = re.sub(r'\s+', ' ', title)  # Normalizar espacios
                        article['title'] = title
                    else:
                        continue  # Skip si no hay título
                    
                    # Extraer resumen
                    abstract_match = re.search(r'<summary>(.*?)</summary>', entry_xml, re.DOTALL)
                    if abstract_match:
                        abstract = abstract_match.group(1).strip()
                        abstract = re.sub(r'\s+', ' ', abstract)
                        article['abstract'] = abstract
                    else:
                        article['abstract'] = ''
                    
                    # Extraer autores - ArXiv los tiene dentro de <author>
                    authors = []
                    # Buscar todos los bloques <author>...</author>
                    author_blocks = re.findall(r'<author(?:.*?)>(.*?)</author>', entry_xml, re.DOTALL)
                    for author_block in author_blocks:
                        # Dentro de cada bloque, buscar <name>
                        name_match = re.search(r'<name>(.*?)</name>', author_block)
                        if name_match:
                            author_name = name_match.group(1).strip()
                            author_name = re.sub(r'\s+', ' ', author_name)
                            if author_name:
                                authors.append(author_name)
                    article['authors'] = authors
                    
                    # Extraer fecha de publicación
                    published_match = re.search(r'<published>(.*?)</published>', entry_xml)
                    if published_match:
                        date_str = published_match.group(1).strip()
                        year_match = re.search(r'(\d{4})-\d{2}-\d{2}', date_str)
                        if year_match:
                            article['year'] = int(year_match.group(1))
                    
                    # Extraer ID (arXiv)
                    id_match = re.search(r'<id>(.*?)</id>', entry_xml)
                    if id_match:
                        arxiv_id = id_match.group(1).strip()
                        article['arxiv_id'] = arxiv_id
                        article['url'] = arxiv_id
                    
                    # Extraer categorías
                    categories = []
                    category_pattern = re.compile(r'<category.*?term="(.*?)"')
                    for match in category_pattern.finditer(entry_xml):
                        categories.append(match.group(1).strip())
                    article['categories'] = categories
                    
                    articles.append(article)
                    
                except Exception as e:
                    self.logger.warning(f"Error parsing single ArXiv entry: {e}")
                    continue
            
        except Exception as e:
            self.logger.error(f"Error parsing ArXiv response: {e}")
        
        return articles
    
    def _process_article(self, article_data: Dict[str, Any]) -> Optional[ArticleMetadata]:
        """Procesar artículo de ArXiv a ArticleMetadata."""
        try:
            title = article_data.get('title', 'No title')
            authors = article_data.get('authors', [])
            abstract_raw = article_data.get('abstract')
            year = article_data.get('year')
            url = article_data.get('url')
            categories = article_data.get('categories', [])
            affiliations_list = article_data.get('affiliations', [])
            
            # Asegurar que abstract sea siempre un string (no None)
            abstract = abstract_raw if abstract_raw else ""
            
            # ArXiv no proporciona afiliaciones ni datos geográficos
            # Pero asignamos campos vacíos para consistencia
            geo_data = {
                'author_countries': [],
                'author_cities': [],
                'institution_countries': [],
                'institution_cities': [],
                'geographic_coordinates': []
            }
            
            # Crear objeto ArticleMetadata (ArXiv es preprint, sin algunos campos)
            article = ArticleMetadata(
                title=title,
                authors=authors if authors else [],
                abstract=abstract,
                affiliations=affiliations_list if affiliations_list else [],  # ArXiv no proporciona afiliaciones
                publication_year=year,
                publication_date=f"{year}-01-01" if year else None,
                type='preprint',
                article_url=url,
                language='en',  # Por defecto inglés para ArXiv
                topics=None,  # ArXiv no tiene keywords reales, solo categorías
                journal='ArXiv Preprint',  # ArXiv es un repositorio de preprints
                # ArXiv NO tiene datos geográficos
                author_countries=geo_data.get('author_countries', []),
                author_cities=geo_data.get('author_cities', []),
                institution_countries=geo_data.get('institution_countries', []),
                institution_cities=geo_data.get('institution_cities', []),
                geographic_coordinates=geo_data.get('geographic_coordinates', [])
            )
            
            return article
            
        except Exception as e:
            self.logger.error(f"Error processing ArXiv article: {e}")
            return None
    
    def _export_to_csv(self, articles: List[ArticleMetadata], search_query: str) -> str:
        """Exportar artículos a CSV."""
        try:
            base_dir = settings.results_dir
            raw_data_dir = os.path.join(base_dir, "raw_data")
            os.makedirs(raw_data_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_query = re.sub(r'[^\w\s-]', '', search_query).strip()
            safe_query = re.sub(r'[-\s]+', '_', safe_query)
            filename = f"resultados_arxiv_{safe_query}_{timestamp}.csv"
            file_path = os.path.join(raw_data_dir, filename)
            
            # Convertir a DataFrame con formato correcto
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
                    'journal': article.journal or 'ArXiv Preprint',
                    'data_source': 'ArXiv',
                    # ArXiv NO tiene datos geográficos estructurados
                    'author_countries': '',
                    'author_cities': '',
                    'institution_countries': '',
                    'institution_cities': '',
                    'geographic_coordinates': ''
                }
                articles_data.append(article_dict)
            
            df = pd.DataFrame(articles_data)
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"Exported {len(articles)} articles to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error exporting CSV: {e}")
            return ""

