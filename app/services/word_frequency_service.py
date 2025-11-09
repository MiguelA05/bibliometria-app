"""
Servicio para análisis de frecuencia de palabras en abstracts.
Requerimiento 3: Calcular frecuencia de aparición y generar palabras asociadas.
"""

import re
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import Counter
from dataclasses import dataclass
import pandas as pd

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from app.utils.logger import get_logger
from app.utils.csv_reader import read_unified_csv, resolve_column_index, normalize_header
from app.config import settings


@dataclass
class WordFrequencyResult:
    """Resultado del análisis de frecuencia de palabras."""
    category: str
    category_words: List[str]
    category_frequencies: Dict[str, int]
    associated_words: List[Tuple[str, int, float]]  # (word, frequency, precision)
    total_articles: int
    total_words_analyzed: int


# Stopwords por defecto
DEFAULT_STOPWORDS = {
    'the', 'and', 'of', 'in', 'to', 'a', 'is', 'for', 'on', 'that', 'with', 'as',
    'by', 'an', 'are', 'this', 'from', 'be', 'at', 'or', 'we', 'it', 'which',
    'can', 'has', 'have', 'these', 'their', 'our', 'was', 'were', 'will', 'such',
    'but', 'not', 'they', 'its', 'may', 'also', 'more', 'other', 'than'
}


class WordFrequencyService:
    """Servicio para análisis de frecuencia de palabras en abstracts."""
    
    # Palabras asociadas a "Concepts of Generative AI in Education"
    GENERATIVE_AI_EDUCATION_WORDS = {
        'generative', 'artificial', 'intelligence', 'ai',
        'education', 'learning', 'teaching', 'pedagogy', 'pedagogical',
        'student', 'students', 'teacher', 'teachers', 'classroom',
        'curriculum', 'instruction', 'assessment', 'evaluation',
        'chatgpt', 'gpt', 'llm', 'large language model', 'language model',
        'machine learning', 'deep learning', 'neural network',
        'personalized', 'adaptive', 'intelligent tutoring',
        'educational technology', 'edtech', 'digital learning'
    }
    
    def __init__(self):
        self.logger = get_logger("word_frequency")
        
        # Inicializar NLTK si está disponible
        self.stop_words = DEFAULT_STOPWORDS.copy()
        if NLTK_AVAILABLE:
            try:
                nltk_stopwords = set(stopwords.words('english'))
                self.stop_words.update(nltk_stopwords)
            except:
                pass
    
    def analyze_word_frequency(
        self,
        csv_path: Optional[str] = None,
        category: str = "Concepts of Generative AI in Education",
        max_associated_words: int = 15,
        text_field: str = "abstract",
        keywords_field: str = "keywords"
    ) -> WordFrequencyResult:
        """
        Analizar frecuencia de palabras en abstracts.
        
        Args:
            csv_path: Ruta al CSV unificado (opcional, usa el más reciente si no se proporciona)
            category: Categoría de análisis
            max_associated_words: Máximo de palabras asociadas a generar
            text_field: Campo de texto a analizar (default: "abstract")
            keywords_field: Campo de keywords (default: "keywords")
        
        Returns:
            WordFrequencyResult con resultados del análisis
        """
        # Leer CSV
        rows = read_unified_csv(csv_path)
        if not rows or len(rows) < 2:
            raise ValueError("El CSV unificado está vacío o no contiene datos suficientes.")
        
        header = normalize_header(rows[0])
        
        # Resolver índices de columnas
        text_idx, _ = resolve_column_index(header, text_field)
        keywords_idx, _ = resolve_column_index(header, keywords_field)
        
        # Obtener palabras de la categoría
        category_words = self._get_category_words(category)
        
        # Extraer todos los abstracts y keywords
        abstracts = []
        keywords_list = []
        for row in rows[1:]:
            if text_idx < len(row):
                abstracts.append(str(row[text_idx]).strip())
            if keywords_idx < len(row):
                keywords_list.append(str(row[keywords_idx]).strip())
        
        # Calcular frecuencias de palabras de la categoría
        category_frequencies = self._calculate_category_frequencies(
            abstracts, category_words
        )
        
        # Generar palabras asociadas
        associated_words = self._generate_associated_words(
            abstracts, category_words, max_associated_words
        )
        
        # Calcular precisión de palabras asociadas
        associated_words_with_precision = self._calculate_precision(
            associated_words, abstracts, category_words
        )
        
        total_words = sum(len(self._tokenize(abstract)) for abstract in abstracts)
        
        return WordFrequencyResult(
            category=category,
            category_words=list(category_words),
            category_frequencies=category_frequencies,
            associated_words=associated_words_with_precision,
            total_articles=len(abstracts),
            total_words_analyzed=total_words
        )
    
    def get_top_words_from_fields(
        self,
        field: str = 'abstract',
        top_n: int = 15,
        csv_path: Optional[str] = None,
        stopwords: Optional[Set[str]] = None,
        min_word_length: int = 2
    ) -> List[Tuple[str, int]]:
        """
        Obtener las top-N palabras más frecuentes en un campo.
        
        Args:
            field: Nombre de la columna
            top_n: Número de palabras a devolver
            csv_path: Ruta al CSV (opcional)
            stopwords: Conjunto de stopwords (opcional)
            min_word_length: Longitud mínima de palabra
        
        Returns:
            Lista de tuplas (palabra, frecuencia) ordenada por frecuencia descendente
        """
        if stopwords is None:
            stopwords = self.stop_words
        
        try:
            rows = read_unified_csv(csv_path)
        except FileNotFoundError as exc:
            self.logger.error(str(exc))
            return []
        
        if not rows or len(rows) < 2:
            return []
        
        header = normalize_header(rows[0])
        idx, _ = resolve_column_index(header, field)
        
        counter = Counter()
        word_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
        
        for row in rows[1:]:
            text = row[idx] if idx < len(row) else ''
            if not text:
                continue
            for w in word_re.findall(text):
                w_lower = w.lower()
                if len(w_lower) < min_word_length:
                    continue
                if w_lower in stopwords:
                    continue
                counter[w_lower] += 1
        
        return counter.most_common(top_n)
    
    def _get_category_words(self, category: str) -> Set[str]:
        """Obtener palabras asociadas a la categoría."""
        if "Generative AI in Education" in category or "Generative AI" in category:
            return self.GENERATIVE_AI_EDUCATION_WORDS
        # Puedes agregar más categorías aquí
        return set()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenizar texto."""
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text.lower())
                tokens = [t for t in tokens if t.isalnum() and t not in self.stop_words and len(t) > 2]
                return tokens
            except:
                pass
        
        # Fallback simple
        text_lower = text.lower()
        text_clean = re.sub(r'[^\w\s]', '', text_lower)
        tokens = text_clean.split()
        return [t for t in tokens if len(t) > 2 and t not in self.stop_words]
    
    def _calculate_category_frequencies(
        self,
        abstracts: List[str],
        category_words: Set[str]
    ) -> Dict[str, int]:
        """Calcular frecuencia de aparición de palabras de la categoría."""
        frequencies = Counter()
        
        for abstract in abstracts:
            tokens = self._tokenize(abstract)
            for token in tokens:
                token_lower = token.lower()
                for cat_word in category_words:
                    if cat_word.lower() in token_lower or token_lower in cat_word.lower():
                        frequencies[cat_word] += 1
                        break
        
        return dict(frequencies)
    
    def _generate_associated_words(
        self,
        abstracts: List[str],
        category_words: Set[str],
        max_words: int = 15
    ) -> List[Tuple[str, int]]:
        """Generar listado de palabras asociadas desde abstracts."""
        word_scores = Counter()
        
        for abstract in abstracts:
            tokens = self._tokenize(abstract)
            
            # Encontrar posiciones de palabras de categoría
            category_positions = []
            for i, token in enumerate(tokens):
                token_lower = token.lower()
                for cat_word in category_words:
                    if cat_word.lower() in token_lower or token_lower in cat_word.lower():
                        category_positions.append(i)
                        break
            
            # Contar palabras cercanas a palabras de categoría (ventana de ±3 palabras)
            window = 3
            for pos in category_positions:
                start = max(0, pos - window)
                end = min(len(tokens), pos + window + 1)
                for i in range(start, end):
                    if i != pos:
                        word = tokens[i].lower()
                        if word not in category_words and len(word) > 2:
                            word_scores[word] += 1
        
        return word_scores.most_common(max_words)
    
    def _calculate_precision(
        self,
        associated_words: List[Tuple[str, int]],
        abstracts: List[str],
        category_words: Set[str]
    ) -> List[Tuple[str, int, float]]:
        """
        Calcular precisión de palabras asociadas.
        
        La precisión se calcula como:
        - Frecuencia de aparición junto con palabras de categoría / Frecuencia total
        """
        results = []
        
        for word, freq_with_category in associated_words:
            total_freq = 0
            freq_with_category_actual = 0
            
            for abstract in abstracts:
                tokens = self._tokenize(abstract)
                tokens_lower = [t.lower() for t in tokens]
                
                if word.lower() in tokens_lower:
                    total_freq += tokens_lower.count(word.lower())
                    
                    # Verificar si aparece cerca de palabras de categoría
                    word_positions = [i for i, t in enumerate(tokens_lower) if t == word.lower()]
                    category_positions = []
                    for i, token in enumerate(tokens_lower):
                        for cat_word in category_words:
                            if cat_word.lower() in token or token in cat_word.lower():
                                category_positions.append(i)
                                break
                    
                    # Verificar proximidad
                    window = 5
                    for wp in word_positions:
                        for cp in category_positions:
                            if abs(wp - cp) <= window:
                                freq_with_category_actual += 1
                                break
            
            precision = freq_with_category_actual / total_freq if total_freq > 0 else 0.0
            results.append((word, freq_with_category, precision))
        
        return results
