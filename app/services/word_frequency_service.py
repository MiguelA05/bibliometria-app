"""
Servicio para análisis de frecuencia de palabras en abstracts.

Requerimiento 3: Calcular frecuencia de aparición de palabras y generar
palabras asociadas basadas en proximidad contextual.
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

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
    chart_paths: Optional[Dict[str, str]] = None  # Rutas de gráficos generados


DEFAULT_STOPWORDS = {
    'the', 'and', 'of', 'in', 'to', 'a', 'is', 'for', 'on', 'that', 'with', 'as',
    'by', 'an', 'are', 'this', 'from', 'be', 'at', 'or', 'we', 'it', 'which',
    'can', 'has', 'have', 'these', 'their', 'our', 'was', 'were', 'will', 'such',
    'but', 'not', 'they', 'its', 'may', 'also', 'more', 'other', 'than'
}


class WordFrequencyService:
    """
    Servicio para análisis de frecuencia de palabras en abstracts.
    
    Calcula frecuencias de palabras, identifica palabras asociadas por proximidad
    contextual y genera estadísticas de uso de términos clave.
    """
    
    # Palabras asociadas según el Requerimiento 3
    GENERATIVE_AI_EDUCATION_WORDS = {
        # Palabras exactas del requerimiento
        'generative models', 'generative model',
        'prompting', 'prompt',
        'machine learning', 'machine-learning',
        'multimodality', 'multimodal',
        'fine-tuning', 'fine tuning', 'finetuning',
        'training data', 'training dataset',
        'algorithmic bias', 'algorithm bias', 'bias',
        'explainability', 'explainable',
        'transparency', 'transparent',
        'ethics', 'ethical',
        'privacy', 'privacy-preserving',
        'personalization', 'personalized', 'personalize',
        'human-ai interaction', 'human ai interaction', 'human-ai', 'human ai',
        'ai literacy', 'ai-literacy', 'literacy',
        'co-creation', 'co creation', 'cocreation'
    }
    
    def __init__(self):
        """Inicializar el servicio de frecuencia de palabras."""
        self.logger = get_logger("word_frequency")
        self.stop_words = DEFAULT_STOPWORDS.copy()
        
        if NLTK_AVAILABLE:
            try:
                nltk_stopwords = set(stopwords.words('english'))
                self.stop_words.update(nltk_stopwords)
            except Exception:
                pass
    
    def analyze_word_frequency(
        self,
        csv_path: Optional[str] = None,
        category: str = "Concepts of Generative AI in Education",
        max_associated_words: int = 15,
        text_field: str = "abstract",
        generate_charts: bool = True,
        output_dir: Optional[Path] = None
    ) -> WordFrequencyResult:
        """
        Analizar frecuencia de palabras en abstracts.
        
        Calcula la frecuencia de aparición de palabras de una categoría específica
        y genera palabras asociadas basadas en proximidad contextual.
        
        Args:
            csv_path: Ruta al CSV unificado (opcional, usa el más reciente si no se proporciona)
            category: Categoría de análisis (default: "Concepts of Generative AI in Education")
            max_associated_words: Máximo de palabras asociadas a generar (default: 15)
            text_field: Campo de texto a analizar (default: "abstract")
        
        Returns:
            WordFrequencyResult con frecuencias, palabras asociadas y estadísticas
        
        Raises:
            ValueError: Si el CSV está vacío o no contiene datos suficientes
        """
        rows = read_unified_csv(csv_path)
        if not rows or len(rows) < 2:
            raise ValueError("El CSV unificado está vacío o no contiene datos suficientes.")
        
        header = normalize_header(rows[0])
        text_idx, _ = resolve_column_index(header, text_field)
        
        category_words = self._get_category_words(category)
        
        abstracts = []
        for row in rows[1:]:
            if text_idx < len(row):
                abstracts.append(str(row[text_idx]).strip())
        
        category_frequencies = self._calculate_category_frequencies(
            abstracts, category_words
        )
        
        associated_words = self._generate_associated_words(
            abstracts, category_words, max_associated_words
        )
        
        associated_words_with_precision = self._calculate_precision(
            associated_words, abstracts, category_words
        )
        
        total_words = sum(len(self._tokenize(abstract)) for abstract in abstracts)
        
        # Generar gráficos si se solicita
        chart_paths = None
        if generate_charts:
            try:
                chart_paths = self._generate_frequency_charts(
                    category_frequencies,
                    associated_words_with_precision,
                    category,
                    output_dir
                )
            except Exception as e:
                self.logger.warning(f"No se pudieron generar gráficos: {e}")
        
        return WordFrequencyResult(
            category=category,
            category_words=list(category_words),
            category_frequencies=category_frequencies,
            associated_words=associated_words_with_precision,
            total_articles=len(abstracts),
            total_words_analyzed=total_words,
            chart_paths=chart_paths
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
        """Obtener conjunto de palabras asociadas a la categoría especificada."""
        if "Generative AI in Education" in category or "Generative AI" in category:
            return self.GENERATIVE_AI_EDUCATION_WORDS
        return set()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenizar texto removiendo stopwords y caracteres no alfanuméricos."""
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text.lower())
                tokens = [t for t in tokens if t.isalnum() and t not in self.stop_words and len(t) > 2]
                return tokens
            except Exception:
                pass
        
        text_lower = text.lower()
        text_clean = re.sub(r'[^\w\s]', '', text_lower)
        tokens = text_clean.split()
        return [t for t in tokens if len(t) > 2 and t not in self.stop_words]
    
    def _calculate_category_frequencies(
        self,
        abstracts: List[str],
        category_words: Set[str]
    ) -> Dict[str, int]:
        """
        Calcular frecuencia de aparición de palabras de la categoría.
        
        Busca tanto palabras individuales como frases completas (bigramas/trigramas).
        """
        frequencies = Counter()
        
        for abstract in abstracts:
            abstract_lower = abstract.lower()
            
            # Buscar frases completas primero (bigramas/trigramas)
            for cat_phrase in category_words:
                if ' ' in cat_phrase or '-' in cat_phrase:  # Es una frase o compuesta
                    # Buscar la frase completa en el abstract
                    if cat_phrase.lower() in abstract_lower:
                        frequencies[cat_phrase] += abstract_lower.count(cat_phrase.lower())
                else:  # Es una palabra individual
                    # Buscar la palabra completa (no como substring)
                    # Usar regex para buscar palabra completa
                    pattern = r'\b' + re.escape(cat_phrase.lower()) + r'\b'
                    matches = re.findall(pattern, abstract_lower)
                    if matches:
                        frequencies[cat_phrase] += len(matches)
        
        return dict(frequencies)
    
    def _generate_associated_words(
        self,
        abstracts: List[str],
        category_words: Set[str],
        max_words: int = 15
    ) -> List[Tuple[str, int]]:
        """
        Generar palabras asociadas basadas en proximidad contextual.
        
        Identifica palabras que aparecen cerca de palabras de categoría
        (ventana de ±3 palabras) y las ordena por frecuencia.
        """
        word_scores = Counter()
        
        for abstract in abstracts:
            tokens = self._tokenize(abstract)
            
            category_positions = []
            for i, token in enumerate(tokens):
                token_lower = token.lower()
                for cat_word in category_words:
                    if cat_word.lower() in token_lower or token_lower in cat_word.lower():
                        category_positions.append(i)
                        break
            
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
        
        La precisión es la proporción de apariciones de una palabra que ocurren
        cerca de palabras de categoría (ventana de ±5 palabras) sobre su frecuencia total.
        
        Interpretación:
        - 100%: La palabra SIEMPRE aparece cerca de palabras de categoría
        - <100%: La palabra aparece en otros contextos también (palabras más generales)
        """
        results = []
        
        for word, freq_with_category in associated_words:
            total_freq = 0
            freq_with_category_actual = 0
            
            for abstract in abstracts:
                tokens = self._tokenize(abstract)
                tokens_lower = [t.lower() for t in tokens]
                
                if word.lower() in tokens_lower:
                    # Contar todas las apariciones de la palabra en este abstract
                    word_count = tokens_lower.count(word.lower())
                    total_freq += word_count
                    
                    # Obtener posiciones de la palabra y de palabras de categoría
                    word_positions = [i for i, t in enumerate(tokens_lower) if t == word.lower()]
                    category_positions = []
                    for i, token in enumerate(tokens_lower):
                        for cat_word in category_words:
                            # Verificar si el token contiene o es parte de una palabra de categoría
                            if cat_word.lower() in token or token in cat_word.lower():
                                category_positions.append(i)
                                break
                    
                    # Verificar para cada aparición de la palabra si está cerca de categoría
                    window = 5
                    for wp in word_positions:
                        found_near_category = False
                        for cp in category_positions:
                            if abs(wp - cp) <= window:
                                freq_with_category_actual += 1
                                found_near_category = True
                                break  # Solo contar una vez por aparición de palabra
            
            precision = freq_with_category_actual / total_freq if total_freq > 0 else 0.0
            results.append((word, freq_with_category, precision))
        
        return results
    
    def _generate_frequency_charts(
        self,
        category_frequencies: Dict[str, int],
        associated_words: List[Tuple[str, int, float]],
        category: str,
        output_dir: Optional[Path] = None
    ) -> Dict[str, str]:
        """
        Generar gráficos de barras para frecuencias de palabras.
        
        Genera dos gráficos:
        1. Top palabras de la categoría (por frecuencia)
        2. Top palabras asociadas (por frecuencia)
        
        Args:
            category_frequencies: Frecuencias de palabras de la categoría
            associated_words: Lista de palabras asociadas con frecuencia y precisión
            category: Nombre de la categoría
            output_dir: Directorio de salida (opcional)
        
        Returns:
            Diccionario con rutas de los gráficos generados
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("matplotlib no está disponible, no se generarán gráficos")
            return {}
        
        if output_dir is None:
            output_dir = Path(settings.results_dir) / "reports" / "word_frequency"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        chart_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Gráfico 1: Top palabras de la categoría
        if category_frequencies:
            # Ordenar por frecuencia descendente y tomar top 15
            sorted_category = sorted(
                category_frequencies.items(),
                key=lambda x: x[1],
                reverse=True
            )[:15]
            
            if sorted_category:
                words, freqs = zip(*sorted_category)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                bars = ax.barh(range(len(words)), freqs, color='steelblue', alpha=0.7)
                ax.set_yticks(range(len(words)))
                ax.set_yticklabels(words, fontsize=10)
                ax.set_xlabel('Frecuencia de Aparición', fontsize=12, fontweight='bold')
                ax.set_title(
                    f'Top Palabras de la Categoría: {category}',
                    fontsize=14,
                    fontweight='bold',
                    pad=20
                )
                ax.grid(axis='x', alpha=0.3, linestyle='--')
                
                # Agregar valores en las barras
                for i, (bar, freq) in enumerate(zip(bars, freqs)):
                    ax.text(
                        freq + max(freqs) * 0.01,
                        i,
                        str(freq),
                        va='center',
                        fontsize=9,
                        fontweight='bold'
                    )
                
                plt.tight_layout()
                chart_path_1 = output_dir / f"category_words_frequency_{timestamp}.png"
                fig.savefig(chart_path_1, dpi=200, bbox_inches='tight')
                plt.close(fig)
                chart_paths['category_words'] = str(chart_path_1)
                self.logger.info(f"Gráfico de palabras de categoría guardado: {chart_path_1}")
        
        # Gráfico 2: Top palabras asociadas
        if associated_words:
            # Ordenar por frecuencia descendente y tomar top 15
            sorted_associated = sorted(
                associated_words,
                key=lambda x: x[1],  # Ordenar por frecuencia
                reverse=True
            )[:15]
            
            if sorted_associated:
                words, freqs, precisions = zip(*sorted_associated)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                bars = ax.barh(range(len(words)), freqs, color='coral', alpha=0.7)
                ax.set_yticks(range(len(words)))
                ax.set_yticklabels(words, fontsize=10)
                ax.set_xlabel('Frecuencia de Aparición', fontsize=12, fontweight='bold')
                ax.set_title(
                    f'Top Palabras Asociadas (Generadas) - {category}',
                    fontsize=14,
                    fontweight='bold',
                    pad=20
                )
                ax.grid(axis='x', alpha=0.3, linestyle='--')
                
                # Agregar valores en las barras (frecuencia y precisión)
                # La precisión indica qué % de apariciones están cerca de palabras de categoría
                for i, (bar, freq, precision) in enumerate(zip(bars, freqs, precisions)):
                    label = f"{freq} (precisión: {precision:.1%})"
                    ax.text(
                        freq + max(freqs) * 0.01,
                        i,
                        label,
                        va='center',
                        fontsize=9,
                        fontweight='bold'
                    )
                
                plt.tight_layout()
                chart_path_2 = output_dir / f"associated_words_frequency_{timestamp}.png"
                fig.savefig(chart_path_2, dpi=200, bbox_inches='tight')
                plt.close(fig)
                chart_paths['associated_words'] = str(chart_path_2)
                self.logger.info(f"Gráfico de palabras asociadas guardado: {chart_path_2}")
        
        return chart_paths
