"""
Sistema de automatización para descarga y unificación de datos académicos.
Maneja múltiples fuentes de datos, detección de duplicados y generación de archivos unificados.
"""

import os
import hashlib
import json
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
from app.models.article import ArticleMetadata
from app.services.openalex_service import OpenAlexService
from app.services.pubmed_service import PubMedService
from app.services.arxiv_service import ArXivService
from app.utils.logger import get_logger
from app.config import settings


@dataclass
class DataSource:
    """Configuración de una fuente de datos."""
    name: str
    service: Any
    query: str
    max_articles: int
    filters: Optional[Dict[str, Any]] = None
    email: Optional[str] = None


@dataclass
class DuplicateRecord:
    """Registro de un artículo duplicado."""
    original_article: ArticleMetadata
    duplicate_article: ArticleMetadata
    similarity_score: float
    duplicate_source: str
    elimination_reason: str


class DataUnificationService:
    """Servicio para unificar datos de múltiples fuentes eliminando duplicados."""
    
    def __init__(self):
        self.logger = get_logger("data_unification")
        self.duplicates_log: List[DuplicateRecord] = []
        
    def create_data_sources(self, base_query: str = "generative artificial intelligence", 
                          max_articles_per_source: int = 50) -> List[DataSource]:
        """
        Crear configuraciones para múltiples fuentes de datos REALES.
        Por ahora solo OpenAlex está activo hasta completar las pruebas de PubMed y ArXiv.
        
        Args:
            base_query: Consulta base para todas las fuentes
            max_articles_per_source: Máximo de artículos por fuente
            
        Returns:
            Lista de fuentes de datos configuradas
        """
        sources = [
            DataSource(
                name="OpenAlex",
                service=OpenAlexService(),
                query=base_query,
                max_articles=max_articles_per_source // 3,  # Dividir entre 3 fuentes
                filters=None
            ),
            DataSource(
                name="PubMed",
                service=PubMedService(),
                query=base_query,
                max_articles=max_articles_per_source // 3,
                filters=None
            ),
            DataSource(
                name="ArXiv",
                service=ArXivService(),
                query=base_query,
                max_articles=max_articles_per_source // 3,
                filters=None
            )
        ]
        
        self.logger.info(f"Created {len(sources)} data sources for query: {base_query}")
        self.logger.info(f"Sources: OpenAlex, PubMed, ArXiv")
        return sources
    
    def download_from_sources(self, sources: List[DataSource]) -> List[ArticleMetadata]:
        """
        Descargar datos de todas las fuentes configuradas.
        
        Args:
            sources: Lista de fuentes de datos
            
        Returns:
            Lista combinada de todos los artículos descargados
        """
        all_articles = []
        
        for source in sources:
            try:
                self.logger.info(f"Downloading from {source.name}")
                articles, csv_path = source.service.search_works(
                    query=source.query,
                    max_articles=source.max_articles,
                    filters=source.filters
                )
                
                # Marcar cada artículo con su fuente
                for article in articles:
                    article.source = source.name
                
                all_articles.extend(articles)
                self.logger.info(f"Downloaded {len(articles)} articles from {source.name}")
                
            except Exception as e:
                self.logger.error(f"Error downloading from {source.name}: {e}")
                continue
        
        self.logger.info(f"Total articles downloaded: {len(all_articles)}")
        return all_articles
    
    def calculate_similarity_score(self, article1: ArticleMetadata, article2: ArticleMetadata) -> float:
        """
        Calcular puntuación de similitud entre dos artículos.
        
        Args:
            article1: Primer artículo
            article2: Segundo artículo
            
        Returns:
            Puntuación de similitud (0.0 a 1.0)
        """
        # Caso especial: verificación rápida de DOI idéntico
        if article1.doi and article2.doi:
            doi1_norm = self._normalize_doi(article1.doi)
            doi2_norm = self._normalize_doi(article2.doi)
            if doi1_norm == doi2_norm and doi1_norm:
                return 1.0  # Mismo DOI = mismo artículo
        
        score = 0.0
        
        # Comparar títulos (peso: 45% - aumentado porque es más confiable)
        title_similarity = 0.0
        if article1.title and article2.title:
            title_similarity = self._calculate_text_similarity(
                article1.title.lower(), 
                article2.title.lower()
            )
            score += title_similarity * 0.45
        
        # Comparar autores (peso: 30% - aumentado)
        author_similarity = 0.0
        if article1.authors and article2.authors:
            author_similarity = self._calculate_author_similarity(
                article1.authors, 
                article2.authors
            )
            score += author_similarity * 0.30
        
        # Comparar DOI (peso: 15% - reducido, solo si ambos tienen DOI)
        if article1.doi and article2.doi:
            doi1_norm = self._normalize_doi(article1.doi)
            doi2_norm = self._normalize_doi(article2.doi)
            if doi1_norm == doi2_norm:
                score += 0.15
            # Si los DOIs son diferentes pero título y autores son muy similares,
            # podría ser el mismo artículo con diferentes IDs (preprint vs published)
        
        # Comparar año de publicación (peso: 10%)
        if (article1.publication_year and article2.publication_year and 
            article1.publication_year == article2.publication_year):
            score += 0.10
        
        # CASO ESPECIAL: Si título y autores son casi idénticos (>95%), aumentar score
        # Esto cubre casos donde el mismo artículo tiene diferentes DOIs (ej: preprint vs published)
        if title_similarity >= 0.95 and author_similarity >= 0.9:
            # Bonus por alta similitud en criterios principales
            bonus = (title_similarity * 0.5 + author_similarity * 0.5) * 0.15
            score = min(score + bonus, 1.0)
        
        return min(score, 1.0)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calcular similitud entre dos textos usando múltiples métodos."""
        if not text1 or not text2:
            return 0.0
        if text1 == text2:
            return 1.0
        
        # Normalizar textos (eliminar puntuación extra, espacios)
        text1_norm = ' '.join(text1.lower().split())
        text2_norm = ' '.join(text2.lower().split())
        
        if text1_norm == text2_norm:
            return 1.0
        
        # Método 1: Jaccard sobre palabras (más robusto)
        words1 = set(text1_norm.split())
        words2 = set(text2_norm.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        jaccard = len(words1.intersection(words2)) / len(words1.union(words2)) if words1.union(words2) else 0.0
        
        # Método 2: Ratio de caracteres comunes (para detectar truncamientos)
        # Calcular similitud usando longitud de subsecuencia común
        min_len = min(len(text1_norm), len(text2_norm))
        max_len = max(len(text1_norm), len(text2_norm))
        
        if max_len == 0:
            return 1.0
        
        # Ratio de longitud (si uno es truncamiento del otro)
        length_ratio = min_len / max_len if max_len > 0 else 0.0
        
        # Si uno es casi un prefijo del otro y la ratio es alta, probablemente es el mismo
        if length_ratio > 0.85:
            shorter = text1_norm if len(text1_norm) < len(text2_norm) else text2_norm
            longer = text2_norm if len(text1_norm) < len(text2_norm) else text1_norm
            if longer.startswith(shorter):
                return 0.98  # Muy probable que sea truncamiento
        
        # Método 3: Ratio de palabras compartidas
        shared_words_ratio = len(words1.intersection(words2)) / min(len(words1), len(words2)) if min(len(words1), len(words2)) > 0 else 0.0
        
        # Combinar métodos: priorizar el más alto
        return max(jaccard, shared_words_ratio * 0.95, length_ratio * 0.7)
    
    def _normalize_doi(self, doi: str) -> str:
        """Normalizar DOI para comparación."""
        if not doi:
            return ""
        
        # Remover prefijos comunes
        doi = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
        doi = doi.replace("doi:", "").strip()
        
        return doi.lower()
    
    def _calculate_author_similarity(self, authors1: List[str], authors2: List[str]) -> float:
        """Calcular similitud entre listas de autores."""
        if not authors1 or not authors2:
            return 0.0
        
        # Normalizar nombres de autores
        norm_authors1 = [self._normalize_author_name(author) for author in authors1]
        norm_authors2 = [self._normalize_author_name(author) for author in authors2]
        
        # Calcular intersección
        set1 = set(norm_authors1)
        set2 = set(norm_authors2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _normalize_author_name(self, author: str) -> str:
        """Normalizar nombre de autor para comparación."""
        if not author:
            return ""
        
        # Convertir a minúsculas y remover caracteres especiales
        normalized = author.lower().strip()
        normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
        
        # Remover espacios extra
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def detect_and_remove_duplicates(self, articles: List[ArticleMetadata], 
                                   similarity_threshold: float = 0.75) -> Tuple[List[ArticleMetadata], List[DuplicateRecord]]:
        """
        Detectar y eliminar duplicados de la lista de artículos.
        
        Args:
            articles: Lista de artículos a procesar
            similarity_threshold: Umbral de similitud para considerar duplicados
            
        Returns:
            Tupla con (artículos_únicos, registros_de_duplicados)
        """
        unique_articles = []
        duplicates_log = []
        
        self.logger.info(f"Starting duplicate detection for {len(articles)} articles")
        
        for i, article in enumerate(articles):
            is_duplicate = False
            
            # Comparar con artículos ya procesados
            for j, unique_article in enumerate(unique_articles):
                similarity = self.calculate_similarity_score(article, unique_article)
                
                if similarity >= similarity_threshold:
                    # Es un duplicado
                    duplicate_record = DuplicateRecord(
                        original_article=unique_article,
                        duplicate_article=article,
                        similarity_score=similarity,
                        duplicate_source=article.source,
                        elimination_reason=f"Similarity score: {similarity:.3f}"
                    )
                    
                    duplicates_log.append(duplicate_record)
                    is_duplicate = True
                    
                    self.logger.debug(
                        f"Duplicate found: '{article.title[:50]}...' "
                        f"similar to '{unique_article.title[:50]}...' "
                        f"(score: {similarity:.3f})"
                    )
                    break
            
            if not is_duplicate:
                unique_articles.append(article)
        
        self.logger.info(
            f"Duplicate detection completed: "
            f"{len(unique_articles)} unique articles, "
            f"{len(duplicates_log)} duplicates removed"
        )
        
        return unique_articles, duplicates_log
    
    def export_unified_data(self, unique_articles: List[ArticleMetadata], 
                          duplicates_log: List[DuplicateRecord],
                          base_filename: str = None) -> Tuple[str, str]:
        """
        Exportar datos unificados y registro de duplicados a archivos CSV organizados por carpetas.
        
        Args:
            unique_articles: Lista de artículos únicos
            duplicates_log: Lista de registros de duplicados
            base_filename: Nombre base para los archivos
            
        Returns:
            Tupla con (ruta_archivo_unificado, ruta_archivo_duplicados)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not base_filename:
            base_filename = f"unified_generative_ai_{timestamp}"
        
        # Crear estructura de directorios organizada
        base_dir = settings.results_dir
        unified_dir = os.path.join(base_dir, "unified")
        duplicates_dir = os.path.join(base_dir, "duplicates")
        reports_dir = os.path.join(base_dir, "reports")
        
        # Crear directorios si no existen
        os.makedirs(unified_dir, exist_ok=True)
        os.makedirs(duplicates_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        
        # Exportar artículos únicos en carpeta unified
        unified_file = os.path.join(unified_dir, f"{base_filename}_unified.csv")
        self._export_articles_to_csv(unique_articles, unified_file)
        
        # Exportar registro de duplicados en carpeta duplicates
        duplicates_file = os.path.join(duplicates_dir, f"{base_filename}_duplicates.csv")
        self._export_duplicates_to_csv(duplicates_log, duplicates_file)
        
        # Generar reporte de procesamiento en carpeta reports
        report_file = os.path.join(reports_dir, f"{base_filename}_processing_report.csv")
        self._export_processing_report(unique_articles, duplicates_log, report_file)
        
        self.logger.info(
            f"Exported unified data: {len(unique_articles)} articles to {unified_file}"
        )
        self.logger.info(
            f"Exported duplicates log: {len(duplicates_log)} records to {duplicates_file}"
        )
        self.logger.info(
            f"Exported processing report to {report_file}"
        )
        
        return unified_file, duplicates_file
    
    def _export_articles_to_csv(self, articles: List[ArticleMetadata], file_path: str):
        """Exportar lista de artículos a CSV."""
        if not articles:
            return
        
        # Convertir a DataFrame
        data = []
        for article in articles:
            data.append({
                'title': article.title,
                'authors': '; '.join(article.authors) if article.authors else '',
                'affiliations': '; '.join(article.affiliations) if article.affiliations else '',
                'abstract': article.abstract,
                'publication_date': article.publication_date,
                'article_url': article.article_url,
                'doi': article.doi,
                'publication_year': article.publication_year,
                'type': article.type,
                'language': article.language,
                'keywords': '; '.join(article.topics) if article.topics else '',
                'license': article.license,
                'journal': article.journal or '',
                'data_source': getattr(article, 'source', 'Unknown'),
                
                # Datos geográficos
                'author_countries': '; '.join(article.author_countries) if article.author_countries else '',
                'author_cities': '; '.join(article.author_cities) if article.author_cities else '',
                'institution_countries': '; '.join(article.institution_countries) if article.institution_countries else '',
                'institution_cities': '; '.join(article.institution_cities) if article.institution_cities else '',
                'geographic_coordinates': json.dumps(article.geographic_coordinates) if article.geographic_coordinates else ''
            })
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False, encoding=settings.csv_encoding)
    
    def _export_duplicates_to_csv(self, duplicates_log: List[DuplicateRecord], file_path: str):
        """Exportar registro de duplicados a CSV."""
        if not duplicates_log:
            # Crear archivo vacío con encabezados
            empty_df = pd.DataFrame(columns=[
                'duplicate_title', 'original_title', 'similarity_score',
                'duplicate_source', 'elimination_reason', 'duplicate_doi',
                'original_doi', 'duplicate_authors', 'original_authors'
            ])
            empty_df.to_csv(file_path, index=False, encoding=settings.csv_encoding)
            return
        
        # Convertir a DataFrame
        data = []
        for duplicate in duplicates_log:
            data.append({
                'duplicate_title': duplicate.duplicate_article.title,
                'original_title': duplicate.original_article.title,
                'similarity_score': duplicate.similarity_score,
                'duplicate_source': duplicate.duplicate_source,
                'elimination_reason': duplicate.elimination_reason,
                'duplicate_doi': duplicate.duplicate_article.doi,
                'original_doi': duplicate.original_article.doi,
                'duplicate_authors': '; '.join(duplicate.duplicate_article.authors) if duplicate.duplicate_article.authors else '',
                'original_authors': '; '.join(duplicate.original_article.authors) if duplicate.original_article.authors else ''
            })
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False, encoding=settings.csv_encoding)
    
    def _export_processing_report(self, unique_articles: List[ArticleMetadata], 
                                duplicates_log: List[DuplicateRecord], file_path: str):
        """Exportar reporte de procesamiento con estadísticas detalladas."""
        timestamp = datetime.now()
        
        # Calcular estadísticas
        total_unique = len(unique_articles)
        total_duplicates = len(duplicates_log)
        total_processed = total_unique + total_duplicates
        
        # Estadísticas por fuente
        source_stats = {}
        for article in unique_articles:
            source = getattr(article, 'source', 'Unknown')
            source_stats[source] = source_stats.get(source, 0) + 1
        
        # Estadísticas por tipo
        type_stats = {}
        for article in unique_articles:
            article_type = article.type or 'Unknown'
            type_stats[article_type] = type_stats.get(article_type, 0) + 1
        
        # Estadísticas por año
        year_stats = {}
        for article in unique_articles:
            year = article.publication_year or 'Unknown'
            year_stats[year] = year_stats.get(year, 0) + 1
        
        # Crear datos del reporte
        report_data = [
            {
                'metric': 'Total Articles Processed',
                'value': total_processed,
                'description': 'Total articles downloaded from all sources'
            },
            {
                'metric': 'Unique Articles',
                'value': total_unique,
                'description': 'Articles after duplicate removal'
            },
            {
                'metric': 'Duplicates Removed',
                'value': total_duplicates,
                'description': 'Articles identified as duplicates'
            },
            {
                'metric': 'Duplication Rate',
                'value': f"{(total_duplicates / total_processed * 100):.1f}%" if total_processed > 0 else "0%",
                'description': 'Percentage of articles that were duplicates'
            },
            {
                'metric': 'Processing Date',
                'value': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'description': 'Date and time of processing'
            }
        ]
        
        # Agregar estadísticas por fuente
        for source, count in source_stats.items():
            report_data.append({
                'metric': f'Articles from {source}',
                'value': count,
                'description': f'Unique articles from {source} source'
            })
        
        # Agregar estadísticas por tipo
        for article_type, count in type_stats.items():
            report_data.append({
                'metric': f'Type: {article_type}',
                'value': count,
                'description': f'Articles of type {article_type}'
            })
        
        # Agregar estadísticas por año
        for year, count in year_stats.items():
            report_data.append({
                'metric': f'Year: {year}',
                'value': count,
                'description': f'Articles published in {year}'
            })
        
        # Crear DataFrame y exportar
        df = pd.DataFrame(report_data)
        df.to_csv(file_path, index=False, encoding=settings.csv_encoding)
    
    def run_automated_process(self, base_query: str = "generative artificial intelligence",
                            similarity_threshold: float = 0.75,
                            max_articles_per_source: int = 350) -> Dict[str, Any]:
        """
        Ejecutar proceso completo de automatización.
        
        Args:
            base_query: Consulta base para todas las fuentes
            similarity_threshold: Umbral de similitud para duplicados
            
        Returns:
            Diccionario con resultados del proceso
        """
        start_time = datetime.now()
        
        self.logger.info("Starting automated data download and unification process")
        
        try:
            # 1. Crear fuentes de datos
            sources = self.create_data_sources(base_query, max_articles_per_source)
            
            # 2. Descargar datos de todas las fuentes
            all_articles = self.download_from_sources(sources)
            
            if not all_articles:
                raise ValueError("No articles downloaded from any source")
            
            # 3. Detectar y eliminar duplicados
            unique_articles, duplicates_log = self.detect_and_remove_duplicates(
                all_articles, similarity_threshold
            )
            
            # 4. Exportar resultados
            unified_file, duplicates_file = self.export_unified_data(
                unique_articles, duplicates_log
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            result = {
                'success': True,
                'total_articles_downloaded': len(all_articles),
                'unique_articles': len(unique_articles),
                'duplicates_removed': len(duplicates_log),
                'unified_file': unified_file,
                'duplicates_file': duplicates_file,
                'processing_time_seconds': processing_time,
                'sources_processed': len(sources),
                'similarity_threshold': similarity_threshold,
                'timestamp': end_time.isoformat()
            }
            
            self.logger.info(
                f"Automated process completed successfully: "
                f"{len(unique_articles)} unique articles, "
                f"{len(duplicates_log)} duplicates removed, "
                f"processed in {processing_time:.2f} seconds"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in automated process: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
