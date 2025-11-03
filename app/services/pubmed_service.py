"""
Servicio para interactuar con la API de PubMed/NLM.
Usa la API REST gratuita de Entrez/eutils de NCBI.
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
from app.utils.logger import get_logger, log_openalex_request
from app.utils.metrics import PerformanceTimer
from app.utils.exceptions import OpenAlexError, CSVExportError, error_handler
from app.services.geographic_service import GeographicDataService


class PubMedService:
    """
    Servicio para interactuar con la API de PubMed.
    PubMed es una base de datos del National Library of Medicine (NLM).
    """
    
    def __init__(self, email: Optional[str] = None):
        """
        Inicializar el servicio de PubMed.
        
        Args:
            email: Email para identificar la aplicación (recomendado)
        """
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.headers = {
            'User-Agent': 'BibliometriaApp/1.0',
            'Accept': 'application/json'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.logger = get_logger("pubmed_service")
        self.geographic_service = GeographicDataService()
    
    def search_works(self, query: str, max_articles: int = 10, 
                    filters: Optional[Dict[str, Any]] = None) -> Tuple[List[ArticleMetadata], str]:
        """
        Buscar artículos en PubMed.
        
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
            self.logger.info(f"Searching PubMed: {query}")
            
            # 1. Buscar IDs de artículos
            pmids = self._search_pmids(query, max_articles, filters)
            
            if not pmids:
                self.logger.warning("No articles found in PubMed")
                return [], ""
            
            # 2. Obtener detalles de cada artículo
            article_details = self._fetch_article_details(pmids)
            
            # 3. Procesar cada artículo
            for article_data in article_details:
                if article_data:
                    article = self._process_article(article_data)
                    if article:
                        articles.append(article)
            
            # 4. Exportar a CSV
            csv_file_path = self._export_to_csv(articles, query)
            
            elapsed_time = time.time() - start_time
            
            self.logger.info(
                f"PubMed search completed: {len(articles)} articles in {elapsed_time:.2f}s"
            )
            
            return articles, csv_file_path
            
        except Exception as e:
            self.logger.error(f"Error searching PubMed: {e}")
            raise
    
    def _search_pmids(self, query: str, max_articles: int, 
                     filters: Optional[Dict[str, Any]]) -> List[str]:
        """Buscar IDs de artículos en PubMed."""
        try:
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': min(max_articles, 100),
                'retmode': 'json'
            }
            
            if filters and 'year' in filters:
                params['term'] = f"{query} AND {filters['year']}[PDAT]"
            
            response = self.session.get(
                f"{self.base_url}/esearch.fcgi",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            pmids = data.get('esearchresult', {}).get('idlist', [])
            
            return pmids[:max_articles]
            
        except Exception as e:
            self.logger.error(f"Error searching PMIDs: {e}")
            return []
    
    def _fetch_article_details(self, pmids: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Obtener detalles de artículos por sus IDs."""
        if not pmids:
            return []
        
        # Obtener todos los artículos en una sola llamada
        try:
            params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml',
                'rettype': 'abstract'
            }
            
            response = self.session.post(
                f"{self.base_url}/efetch.fcgi",
                data=params,
                timeout=60
            )
            response.raise_for_status()
            
            # Parsear XML completo
            articles = self._parse_pubmed_xml(response.text)
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching article details: {e}")
            return []
    
    def _parse_pubmed_xml(self, xml_content: str) -> List[Optional[Dict[str, Any]]]:
        """Parsear XML de PubMed con múltiples artículos."""
        articles = []
        
        try:
            # Separar cada artículo PubMed
            pubmed_article_pattern = re.compile(r'<PubmedArticle>(.*?)</PubmedArticle>', re.DOTALL)
            articles_xml = pubmed_article_pattern.findall(xml_content)
            
            for article_xml in articles_xml:
                try:
                    # Extraer PMID
                    pmid_match = re.search(r'<PMID Version=".*?">(.*?)</PMID>', article_xml)
                    pmid = pmid_match.group(1).strip() if pmid_match else ''
                    
                    # Extraer título
                    title_match = re.search(r'<ArticleTitle>(.*?)</ArticleTitle>', article_xml, re.DOTALL)
                    title = title_match.group(1).strip() if title_match else 'No title'
                    
                    # Extraer abstract
                    abstract = ''
                    abstract_match = re.search(r'<AbstractText.*?>(.*?)</AbstractText>', article_xml, re.DOTALL)
                    if abstract_match:
                        abstract = abstract_match.group(1).strip()
                    
                    # Extraer autores con afiliaciones
                    authors = []
                    affiliations_list = []
                    
                    # Buscar cada bloque <Author>...</Author>
                    author_blocks = re.findall(r'<Author.*?>(.*?)</Author>', article_xml, re.DOTALL)
                    for author_block in author_blocks:
                        last_match = re.search(r'<LastName>(.*?)</LastName>', author_block)
                        fore_match = re.search(r'<ForeName>(.*?)</ForeName>', author_block)
                        
                        if last_match and fore_match:
                            authors.append(f"{fore_match.group(1).strip()} {last_match.group(1).strip()}")
                        elif last_match:
                            authors.append(last_match.group(1).strip())
                        
                        # Extraer afiliaciones
                        aff_match = re.search(r'<Affiliation>(.*?)</Affiliation>', author_block)
                        if aff_match:
                            affiliations_list.append(aff_match.group(1).strip())
                    
                    # Extraer año
                    year = None
                    year_match = re.search(r'<PubDate>.*?<Year>(\d{4})</Year>', article_xml, re.DOTALL)
                    if year_match:
                        year = int(year_match.group(1))
                    
                    # Extraer DOI (buscar en ELocationID o ArticleIdList)
                    doi = None
                    doi_match = re.search(r'<ELocationID EIdType="doi".*?>(.*?)</ELocationID>', article_xml)
                    if doi_match:
                        doi = doi_match.group(1).strip()
                    else:
                        # Buscar en ArticleIdList
                        doi_match = re.search(r'<ArticleId IdType="doi">(.*?)</ArticleId>', article_xml)
                        if doi_match:
                            doi = doi_match.group(1).strip()
                    
                    # Extraer journal/revista (buscar en múltiples campos del XML)
                    journal = None
                    # Prioridad 1: MedlineTA (Medline Title Abbreviation)
                    journal_match = re.search(r'<MedlineTA>(.*?)</MedlineTA>', article_xml, re.DOTALL)
                    if journal_match:
                        journal = journal_match.group(1).strip()
                    else:
                        # Prioridad 2: Title completo
                        title_match = re.search(r'<Title>(.*?)</Title>', article_xml, re.DOTALL)
                        if title_match:
                            journal = title_match.group(1).strip()
                        else:
                            # Prioridad 3: ISOAbbreviation
                            iso_match = re.search(r'<ISOAbbreviation>(.*?)</ISOAbbreviation>', article_xml, re.DOTALL)
                            if iso_match:
                                journal = iso_match.group(1).strip()
                    
                    # Extraer keywords/topics (MEJORADO con múltiples fuentes)
                    keywords = []
                    
                    # Fuente 1: Keyword tags tradicionales
                    keyword_pattern = re.compile(r'<Keyword.*?>(.*?)</Keyword>', re.DOTALL)
                    for match in keyword_pattern.finditer(article_xml):
                        keyword_text = match.group(1).strip()
                        # Limpiar el formato raro de PubMed si viene con tags
                        if '<' in keyword_text or keyword_text.startswith('>') or 'MajorTopicYN' in keyword_text:
                            clean_keyword = re.sub(r'<[^>]+>', '', keyword_text).strip()
                            if clean_keyword:
                                keywords.append(clean_keyword)
                        else:
                            keywords.append(keyword_text)
                    
                    # Fuente 2: MeSH Descriptor Names (más común y relevante)
                    descriptor_pattern = re.compile(r'<DescriptorName.*?>(.*?)</DescriptorName>', re.DOTALL)
                    for match in descriptor_pattern.finditer(article_xml):
                        descriptor = match.group(1).strip()
                        if descriptor and descriptor not in keywords:
                            keywords.append(descriptor)
                    
                    # Fuente 3: NameOfSubstance (MeSH terms de sustancias)
                    substance_pattern = re.compile(r'<NameOfSubstance.*?>(.*?)</NameOfSubstance>', re.DOTALL)
                    for match in substance_pattern.finditer(article_xml):
                        substance = match.group(1).strip()
                        if substance and substance not in keywords:
                            keywords.append(substance)
                    
                    # Fuente 4: Concept Terms
                    concept_pattern = re.compile(r'<Concept Term="(.*?)"', re.DOTALL)
                    for match in concept_pattern.finditer(article_xml):
                        concept = match.group(1).strip()
                        if concept and concept not in keywords:
                            keywords.append(concept)
                    
                    articles.append({
                        'title': title,
                        'abstract': abstract,
                        'authors': authors if authors else [],
                        'affiliations': list(set(affiliations_list)) if affiliations_list else [],  # Eliminar duplicados
                        'year': year,
                        'doi': doi,
                        'pmid': pmid,
                        'topics': keywords,
                        'journal': journal
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error parsing single PubMed article: {e}")
                    continue
            
        except Exception as e:
            self.logger.error(f"Error parsing PubMed XML: {e}")
        
        return articles
    
    def _process_article(self, article_data: Dict[str, Any]) -> Optional[ArticleMetadata]:
        """Procesar artículo de PubMed a ArticleMetadata."""
        try:
            # Extraer información básica
            title = article_data.get('title', 'No title')
            abstract = article_data.get('abstract', '')
            authors_list = article_data.get('authors', [])
            affiliations_list = article_data.get('affiliations', [])
            year = article_data.get('year')
            doi = article_data.get('doi')
            pmid = article_data.get('pmid')
            keywords = article_data.get('topics', [])  # En PubMed, topics son keywords
            journal = article_data.get('journal')  # Journal extraído del XML
            
            # Construir URL de PubMed
            article_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}" if pmid else None
            
            # Extraer datos geográficos de afiliaciones usando GeographicDataService
            geo_data = self.geographic_service.extract_geographic_data_from_affiliation_text(affiliations_list)
            
            # Crear objeto ArticleMetadata
            article = ArticleMetadata(
                title=title,
                authors=authors_list if authors_list else [],
                abstract=abstract if abstract else None,
                affiliations=affiliations_list if affiliations_list else [],
                publication_year=year,
                publication_date=f"{year}-01-01" if year else None,  # Inferir fecha
                type='journal-article',
                doi=doi,
                article_url=article_url,
                language='en',
                topics=keywords if keywords else None,  # En ArticleMetadata siguen siendo topics pero con keywords limpias
                journal=journal,  # Journal extraído
                # Asignar datos geográficos extraídos
                author_countries=geo_data.get('author_countries', []),
                author_cities=geo_data.get('author_cities', []),
                institution_countries=geo_data.get('institution_countries', []),
                institution_cities=geo_data.get('institution_cities', []),
                geographic_coordinates=geo_data.get('geographic_coordinates', [])
            )
            
            return article
            
        except Exception as e:
            self.logger.error(f"Error processing PubMed article: {e}")
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
            filename = f"resultados_pubmed_{safe_query}_{timestamp}.csv"
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
                    'journal': article.journal or '',
                    'data_source': 'PubMed',
                    # Datos geográficos extraídos de afiliaciones
                    'author_countries': '; '.join(article.author_countries) if article.author_countries else '',
                    'author_cities': '; '.join(article.author_cities) if article.author_cities else '',
                    'institution_countries': '; '.join(article.institution_countries) if article.institution_countries else '',
                    'institution_cities': '; '.join(article.institution_cities) if article.institution_cities else '',
                    'geographic_coordinates': json.dumps(article.geographic_coordinates) if article.geographic_coordinates else ''
                }
                articles_data.append(article_dict)
            
            df = pd.DataFrame(articles_data)
            df.to_csv(file_path, index=False, encoding=settings.csv_encoding)
            
            self.logger.info(f"Exported {len(articles)} articles to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error exporting CSV: {e}")
            return ""

