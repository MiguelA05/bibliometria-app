"""
Utilidad para leer abstracts de CSV unificados y preparar para análisis de similitud.
"""

import pandas as pd
import os
from typing import List, Dict, Any
from pathlib import Path
from app.utils.logger import get_logger
from app.config import settings


class TextExtractor:
    """Extraer y normalizar abstracts de archivos CSV."""
    
    def __init__(self):
        self.logger = get_logger("text_extractor")
    
    def read_unified_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Leer CSV unificado y validar estructura.
        
        Args:
            csv_path: Ruta al archivo CSV unificado
            
        Returns:
            DataFrame con los artículos
        """
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            
            # Validar columnas requeridas
            required_cols = ['title', 'abstract']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            self.logger.info(f"Loaded {len(df)} articles from {csv_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading CSV: {e}")
            raise
    
    def extract_abstracts(self, df: pd.DataFrame, 
                         article_indices: List[int]) -> List[Dict[str, Any]]:
        """
        Extraer abstracts de artículos específicos.
        
        Args:
            df: DataFrame con artículos
            article_indices: Índices de los artículos a extraer
            
        Returns:
            Lista de diccionarios con información del artículo y abstract
        """
        articles_data = []
        
        for idx in article_indices:
            if idx >= len(df):
                self.logger.warning(f"Index {idx} out of range")
                continue
            
            row = df.iloc[idx]
            
            article_info = {
                'index': idx,
                'title': row.get('title', 'No title'),
                'abstract': row.get('abstract', ''),
                'authors': row.get('authors', ''),
                'publication_year': row.get('publication_year', ''),
                'doi': row.get('doi', ''),
                'cited_by_count': row.get('cited_by_count', 0)
            }
            
            articles_data.append(article_info)
        
        return articles_data
    
    def find_latest_unified_csv(self, base_dir: str = None) -> str:
        """
        Encontrar el CSV unificado más reciente.
        
        Args:
            base_dir: Directorio base (por defecto settings.results_dir)
            
        Returns:
            Ruta al CSV más reciente
        """
        if base_dir is None:
            base_dir = settings.results_dir
        
        unified_dir = os.path.join(base_dir, "unified")
        
        if not os.path.exists(unified_dir):
            raise FileNotFoundError(f"Unified directory not found: {unified_dir}")
        
        csv_files = list(Path(unified_dir).glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {unified_dir}")
        
        # Devolver el más reciente
        latest = max(csv_files, key=lambda p: p.stat().st_mtime)
        
        self.logger.info(f"Found latest unified CSV: {latest}")
        return str(latest)


def get_unified_csv_list() -> List[Dict[str, Any]]:
    """Listar todos los CSVs unificados disponibles."""
    try:
        base_dir = settings.results_dir
        unified_dir = os.path.join(base_dir, "unified")
        
        if not os.path.exists(unified_dir):
            return []
        
        csv_files = []
        for path in Path(unified_dir).glob("*.csv"):
            stat = path.stat()
            csv_files.append({
                'filename': path.name,
                'filepath': str(path),
                'size_kb': stat.st_size / 1024,
                'modified': stat.st_mtime
            })
        
        # Ordenar por fecha de modificación (más reciente primero)
        csv_files.sort(key=lambda x: x['modified'], reverse=True)
        
        return csv_files
        
    except Exception as e:
        logger = get_logger("text_extractor")
        logger.error(f"Error listing CSVs: {e}")
        return []
