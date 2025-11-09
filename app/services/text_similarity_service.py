"""
Servicio de similitud textual con 6 algoritmos:
- 4 algoritmos clásicos: Levenshtein, Damerau-Levenshtein, Jaccard, TF-IDF Cosine
- 2 algoritmos basados en IA: Sentence-BERT, LLM-based (Ollama)

Incluye preprocesamiento normalizado y análisis detallado de resultados.
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import unicodedata

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("[WARNING] scikit-learn no está instalado. Algunos algoritmos no estarán disponibles.")

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
except ImportError:
    print("[WARNING] nltk no está instalado. Instala con: pip install nltk && python -m nltk.downloader punkt stopwords")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_BERT_AVAILABLE = True
except ImportError:
    SENTENCE_BERT_AVAILABLE = False
    print("[WARNING] sentence-transformers no está instalado. Algoritmos de embeddings no estarán disponibles.")

try:
    from app.utils.ollama_helper import (
        ensure_ollama_ready,
        analyze_similarity_with_llm,
        OLLAMA_AVAILABLE
    )
except ImportError:
    OLLAMA_AVAILABLE = False
    ensure_ollama_ready = None
    analyze_similarity_with_llm = None
    print("[WARNING] Ollama helper no disponible. Algoritmo LLM-based usará modo simulado.")

from app.utils.logger import get_logger


@dataclass
class SimilarityResult:
    """Resultado de un algoritmo de similitud textual."""
    algorithm_name: str
    similarity_score: float
    explanation: str
    details: Dict[str, Any]
    processing_time: float


@dataclass
class TextPreprocessingResult:
    """Resultado del preprocesamiento de texto."""
    original_text: str
    processed_text: str
    language: str
    tokens: List[str]
    stats: Dict[str, Any]


class TextSimilarityService:
    """
    Servicio para análisis de similitud textual con múltiples algoritmos.
    
    Implementa 6 algoritmos de similitud:
    - Levenshtein: Distancia de edición estándar
    - Damerau-Levenshtein: Distancia de edición con transposiciones
    - Jaccard: Similitud sobre n-grams (shingles)
    - TF-IDF Cosine: Vectorización estadística con similitud coseno
    - Sentence-BERT: Embeddings semánticos con transformers
    - LLM-based: Análisis semántico profundo con modelos LLM locales (Ollama)
    """
    
    def __init__(self, ollama_model: str = "llama3.2:3b"):
        """
        Inicializar el servicio de similitud textual.
        
        Args:
            ollama_model: Nombre del modelo de Ollama a usar (default: "llama3.2:3b")
        """
        self.logger = get_logger("text_similarity")
        self.ollama_model = ollama_model
        
        try:
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stemmer = None
            self.stop_words = set()
        
        self.sbert_model = None
        if SENTENCE_BERT_AVAILABLE:
            try:
                self.sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                self.logger.info("Sentence-BERT model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Could not load Sentence-BERT model: {e}")
                self.sbert_model = None
        
        self.ollama_available = False
        if OLLAMA_AVAILABLE:
            try:
                if ensure_ollama_ready():
                    self.ollama_available = True
                    self.logger.info(f"Ollama disponible. Modelo configurado: {ollama_model}")
                else:
                    self.logger.warning("Ollama no está disponible. LLM-based similarity no funcionará.")
            except Exception as e:
                self.logger.warning(f"Error verificando Ollama: {e}")
    
    def preprocess_text(self, text: str, method: str = 'standard') -> TextPreprocessingResult:
        """
        Preprocesar texto según el método especificado.
        
        Args:
            text: Texto original a procesar
            method: Método de preprocesamiento:
                - 'standard': Tokenización completa con stemming y stopwords
                - 'char-level': Solo normalización básica (para algoritmos de caracteres)
                - 'token-level': Tokenización sin stemming (para TF-IDF)
                - 'minimal': Mínimo procesamiento (solo minúsculas y split)
            
        Returns:
            TextPreprocessingResult con texto procesado, tokens y estadísticas
        """
        if not text or not isinstance(text, str):
            return TextPreprocessingResult(
                original_text=str(text),
                processed_text="",
                language="unknown",
                tokens=[],
                stats={}
            )
        
        original = text
        normalized = unicodedata.normalize('NFKC', text)
        text_lower = normalized.lower()
        detected_lang = self._detect_language(text_lower)
        
        if method == 'char-level':
            processed = self._clean_for_char_similarity(text_lower)
            tokens = list(processed)
        elif method == 'minimal':
            processed = text_lower
            tokens = processed.split()
        elif method == 'token-level':
            tokens = self._tokenize(text_lower)
            processed = ' '.join(tokens)
        else:  # standard
            tokens = self._tokenize_advanced(text_lower)
            processed = ' '.join(tokens)
        
        stats = {
            'original_length': len(original),
            'processed_length': len(processed),
            'num_tokens': len(tokens),
            'vocab_size': len(set(tokens)),
            'compression_ratio': len(processed) / len(original) if len(original) > 0 else 0
        }
        
        return TextPreprocessingResult(
            original_text=original,
            processed_text=processed,
            language=detected_lang,
            tokens=tokens,
            stats=stats
        )
    
    def _clean_for_char_similarity(self, text: str) -> str:
        """Normalizar espacios para similitud a nivel de caracteres."""
        return re.sub(r'\s+', ' ', text.strip())
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenización básica: remover puntuación y dividir en palabras."""
        text_no_punct = re.sub(r'[^\w\s]', '', text)
        return text_no_punct.split()
    
    def _tokenize_advanced(self, text: str) -> List[str]:
        """Tokenización avanzada: remover stopwords y aplicar stemming."""
        if not self.stemmer:
            return self._tokenize(text)
        
        tokens = self._tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens
    
    def _detect_language(self, text: str) -> str:
        """Detección simple de idioma basada en palabras comunes."""
        english_words = {'the', 'and', 'is', 'are', 'in', 'on', 'to', 'of', 'a', 'an'}
        words = text.split()
        english_count = sum(1 for word in words if word.lower() in english_words)
        
        if words and english_count / len(words) > 0.1:
            return 'english'
        return 'unknown'
    
    def levenshtein_similarity(self, text1: str, text2: str, 
                               include_matrix: bool = False) -> SimilarityResult:
        """
        Algoritmo 1: Levenshtein (Distancia de Edición).
        
        Calcula el número mínimo de operaciones (inserción, eliminación, sustitución)
        necesarias para convertir un texto en otro. Usa programación dinámica.
        
        Args:
            text1: Primer texto a comparar
            text2: Segundo texto a comparar
            include_matrix: Si True, incluye la matriz DP y backtrace en los detalles
        
        Returns:
            SimilarityResult con score de similitud (0-1), distancia y explicación detallada
        """
        start_time = datetime.now()
        
        prep1 = self.preprocess_text(text1, 'char-level')
        prep2 = self.preprocess_text(text2, 'char-level')
        
        s1, s2 = prep1.processed_text, prep2.processed_text
        n, m = len(s1), len(s2)
        
        dp = np.zeros((n + 1, m + 1), dtype=int)
        
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + cost
                )
        
        distance = dp[n][m]
        max_len = max(n, m)
        similarity = 1 - (distance / max_len) if max_len > 0 else 0
        
        backtrace = []
        if include_matrix:
            i, j = n, m
            while i > 0 or j > 0:
                if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
                    backtrace.append(('match', s1[i-1], i-1, j-1))
                    i -= 1
                    j -= 1
                elif i > 0 and (j == 0 or dp[i][j-1] > dp[i-1][j]):
                    backtrace.append(('delete', s1[i-1], i-1, j))
                    i -= 1
                else:
                    backtrace.append(('insert', s2[j-1], i, j-1))
                    j -= 1
            backtrace.reverse()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        explanation = f"""
        Levenshtein Distance: {distance} operaciones
        - Insertions: número de caracteres a agregar
        - Deletions: número de caracteres a eliminar
        - Substitutions: número de caracteres a reemplazar
        - Distance/Max_length ratio: {distance}/{max_len} = {distance/max_len:.3f}
        Similarity = 1 - ratio = {similarity:.3f}
        """
        
        details = {
            'distance': distance,
            'max_length': max_len,
            'operations_needed': distance,
            'matrix': dp.tolist() if include_matrix else None,
            'backtrace': backtrace if include_matrix else None,
            'text1_length': n,
            'text2_length': m
        }
        
        return SimilarityResult(
            algorithm_name="Levenshtein (Edit Distance)",
            similarity_score=similarity,
            explanation=explanation,
            details=details,
            processing_time=elapsed
        )
    
    def damerau_levenshtein_similarity(self, text1: str, text2: str) -> SimilarityResult:
        """
        Algoritmo 2: Damerau-Levenshtein (con transposición).
        
        Similar a Levenshtein pero incluye transposición de caracteres adyacentes
        como operación adicional. Ejemplo: 'ab' → 'ba' cuenta como 1 operación vs 2.
        
        Args:
            text1: Primer texto a comparar
            text2: Segundo texto a comparar
        
        Returns:
            SimilarityResult con score de similitud (0-1), distancia y transposiciones detectadas
        """
        start_time = datetime.now()
        
        prep1 = self.preprocess_text(text1, 'char-level')
        prep2 = self.preprocess_text(text2, 'char-level')
        
        s1, s2 = prep1.processed_text, prep2.processed_text
        n, m = len(s1), len(s2)
        
        dp = np.zeros((n + 1, m + 1), dtype=int)
        
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        
        transpositions = []
        transpositions_used = set()
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                
                # Calcular operaciones estándar
                standard_min = min(
                    dp[i-1][j] + 1,      # inserción
                    dp[i][j-1] + 1,      # eliminación
                    dp[i-1][j-1] + cost   # sustitución
                )
                
                # Verificar transposición (Damerau)
                transposition_applied = False
                if (i > 1 and j > 1 and 
                    s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1] and
                    s1[i-1] != s1[i-2]):  # Solo si los caracteres son diferentes
                    transposition_cost = dp[i-2][j-2] + 1
                    if transposition_cost < standard_min:
                        dp[i][j] = transposition_cost
                        transposition_applied = True
                        # Registrar transposición solo si realmente se aplicó
                        trans_pos = (i-2, j-2)
                        if trans_pos not in transpositions_used:
                            transpositions_used.add(trans_pos)
                            transpositions.append((i-2, j-2, s1[i-2:i], s2[j-2:j]))
                
                if not transposition_applied:
                    dp[i][j] = standard_min
        
        distance = dp[n][m]
        max_len = max(n, m)
        similarity = 1 - (distance / max_len) if max_len > 0 else 0
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        explanation = f"""
        Damerau-Levenshtein Distance: {distance} operaciones
        - Incluye transposición de caracteres adyacentes
        - Transposiciones detectadas: {len(transpositions)}
        - Ejemplo: 'ab' → 'ba' cuenta como 1 operación vs 2 en Levenshtein estándar
        - Similarity = {similarity:.3f}
        """
        
        details = {
            'distance': distance,
            'max_length': max_len,
            'transpositions_count': len(transpositions),
            'transpositions': transpositions[:5],  # Primeras 5
            'text1_length': n,
            'text2_length': m
        }
        
        return SimilarityResult(
            algorithm_name="Damerau-Levenshtein (with Transposition)",
            similarity_score=similarity,
            explanation=explanation,
            details=details,
            processing_time=elapsed
        )
    
    def jaccard_similarity(self, text1: str, text2: str, n: int = 3) -> SimilarityResult:
        """
        Algoritmo 3: Jaccard sobre Shingles (n-grams).
        
        Mide la similitud entre dos textos usando la intersección sobre la unión
        de n-grams (shingles). Jaccard = |A ∩ B| / |A ∪ B|
        
        Args:
            text1: Primer texto a comparar
            text2: Segundo texto a comparar
            n: Longitud de los n-grams (shingles). Default: 3 (trigramas)
            
        Returns:
            SimilarityResult con score de similitud (0-1) y shingles comunes
        """
        start_time = datetime.now()
        
        prep1 = self.preprocess_text(text1, 'token-level')
        prep2 = self.preprocess_text(text2, 'token-level')
        
        def generate_ngrams(tokens, n):
            """Generar n-grams de tokens."""
            ngrams = []
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i+n])
                ngrams.append(ngram)
            return set(ngrams)
        
        shingles1 = generate_ngrams(prep1.tokens, n)
        shingles2 = generate_ngrams(prep2.tokens, n)
        
        intersection = shingles1 & shingles2
        union = shingles1 | shingles2
        similarity = len(intersection) / len(union) if union else 0
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        explanation = f"""
        Jaccard Similarity sobre Shingles de longitud {n}:
        - Shingles en texto 1: {len(shingles1)}
        - Shingles en texto 2: {len(shingles2)}
        - Shingles comunes (intersección): {len(intersection)}
        - Shingles totales (unión): {len(union)}
        - Jaccard = |intersection| / |union| = {len(intersection)}/{len(union)} = {similarity:.3f}
        """
        
        common_examples = list(intersection)[:10]
        
        details = {
            'shingles1_count': len(shingles1),
            'shingles2_count': len(shingles2),
            'intersection_size': len(intersection),
            'union_size': len(union),
            'n_gram_length': n,
            'common_shingles': common_examples,
            'all_shingles1': list(shingles1)[:20],
            'all_shingles2': list(shingles2)[:20]
        }
        
        return SimilarityResult(
            algorithm_name=f"Jaccard over {n}-grams",
            similarity_score=similarity,
            explanation=explanation,
            details=details,
            processing_time=elapsed
        )
    
    def tfidf_cosine_similarity(self, text1: str, text2: str) -> SimilarityResult:
        """
        Algoritmo 4: TF-IDF Cosine Similarity.
        
        Usa Term Frequency-Inverse Document Frequency para vectorizar textos y calcula
        similitud del coseno entre los vectores. Captura importancia de términos.
        
        Args:
            text1: Primer texto a comparar
            text2: Segundo texto a comparar
        
        Returns:
            SimilarityResult con score de similitud (0-1) y términos con mayor contribución
        """
        start_time = datetime.now()
        
        # Limpieza más robusta del texto
        def clean_text_for_tfidf(text: str) -> str:
            """Limpiar texto para TF-IDF de manera robusta."""
            if not text or not isinstance(text, str):
                return ""
            
            # Convertir a string y normalizar
            text = str(text).strip()
            
            # Remover caracteres no ASCII problemáticos pero mantener letras acentuadas
            text = unicodedata.normalize('NFKD', text)
            
            # Remover caracteres especiales excepto espacios y letras/números
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Normalizar espacios
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Asegurar que solo contenga palabras separadas por espacios
            words = text.split()
            # Filtrar palabras que sean solo números o muy cortas
            words = [w for w in words if len(w) > 1 and not w.isdigit()]
            
            return ' '.join(words)
        
        text1_clean = clean_text_for_tfidf(text1)
        text2_clean = clean_text_for_tfidf(text2)
        
        if not text1_clean or not text2_clean or len(text1_clean) < 3 or len(text2_clean) < 3:
            similarity = 0
            top_terms = []
            feature_names = []
            term_contributions = []
        else:
            corpus = [text1_clean, text2_clean]
            
            try:
                # Configuración más robusta del vectorizador
                vectorizer = TfidfVectorizer(
                    max_features=500,
                    ngram_range=(1, 1),
                    min_df=1,
                    max_df=1.0,
                    token_pattern=r'(?u)\b\w{2,}\b',  # Mínimo 2 caracteres por token
                    lowercase=True,
                    analyzer='word',
                    strip_accents='unicode',
                    sublinear_tf=False  # Evitar problemas numéricos
                )
                
                tfidf_matrix = vectorizer.fit_transform(corpus)
                
                if tfidf_matrix.shape[0] < 2 or tfidf_matrix.shape[1] == 0:
                    raise ValueError("Matriz TF-IDF insuficiente")
                
                # Convertir a arrays densos de forma segura
                matrix_array = tfidf_matrix.toarray()
                
                if matrix_array.shape[0] < 2:
                    raise ValueError("Matriz TF-IDF insuficiente")
                
                similarity = cosine_similarity([matrix_array[0]], [matrix_array[1]])[0][0]
                
                # Asegurar que los valores sean escalares
                feature_names = vectorizer.get_feature_names_out()
                tfidf1 = matrix_array[0]
                tfidf2 = matrix_array[1]
                
                term_contributions = []
                for i, term in enumerate(feature_names):
                    val1 = float(tfidf1[i])
                    val2 = float(tfidf2[i])
                    contrib = val1 * val2
                    if contrib > 0.01:
                        term_contributions.append({
                            'term': str(term),
                            'contribution': contrib,
                            'tfidf1': val1,
                            'tfidf2': val2
                        })
                
                term_contributions.sort(key=lambda x: x['contribution'], reverse=True)
                top_terms = term_contributions[:20]
            
            except Exception as e:
                try:
                    words1 = set(text1_clean.split())
                    words2 = set(text2_clean.split())
                    common_words = words1 & words2
                    all_words = words1 | words2
                    
                    similarity = len(common_words) / len(all_words) if all_words else 0
                    top_terms = [{'term': w, 'contribution': 1.0, 'tfidf1': 1.0, 'tfidf2': 1.0} 
                                for w in list(common_words)[:20]]
                    feature_names = list(all_words)
                    term_contributions = top_terms
                    self.logger.warning(f"TF-IDF falló, usando método de respaldo (Jaccard): {e}")
                except Exception as e2:
                    similarity = 0
                    top_terms = []
                    feature_names = []
                    term_contributions = []
                    self.logger.error(f"Error crítico en TF-IDF: {e2}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        explanation = f"""
        TF-IDF Cosine Similarity:
        - TF-IDF: Term Frequency × Inverse Document Frequency
        - Coseno: mide similitud direccional entre vectores
        - Similarity = {similarity:.3f}
        - Los términos con mayor contribución son los que aparecen en ambos textos
        """
        
        details = {
            'similarity': similarity,
            'top_contributing_terms': top_terms,
            'total_terms_analyzed': len(feature_names) if 'feature_names' in locals() else 0,
            'terms_with_contribution': len(term_contributions) if 'term_contributions' in locals() else 0
        }
        
        return SimilarityResult(
            algorithm_name="TF-IDF Cosine Similarity",
            similarity_score=similarity,
            explanation=explanation,
            details=details,
            processing_time=elapsed
        )
    
    def sentence_bert_similarity(self, text1: str, text2: str) -> SimilarityResult:
        """
        Algoritmo 5: Sentence-BERT (Embeddings semánticos).
        
        Usa modelos transformer pre-entrenados para generar embeddings semánticos
        y calcular similitud del coseno. Captura significado, no solo palabras.
        
        Args:
            text1: Primer texto a comparar
            text2: Segundo texto a comparar
        
        Returns:
            SimilarityResult con score de similitud semántica (0-1) e interpretación
        """
        start_time = datetime.now()
        
        if not self.sbert_model:
            return SimilarityResult(
                algorithm_name="Sentence-BERT",
                similarity_score=0.0,
                explanation="Sentence-BERT model not available. Install with: pip install sentence-transformers",
                details={},
                processing_time=0.0
            )
        
        embeddings = self.sbert_model.encode([text1, text2], convert_to_numpy=True)
        similarity = float(np.dot(embeddings[0], embeddings[1]) / 
                          (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        explanation = f"""
        Sentence-BERT Semantic Similarity:
        - Usa el modelo 'paraphrase-MiniLM-L6-v2'
        - Genera embeddings de 384 dimensiones
        - Captura similitud semántica, no solo léxica
        - Cosine similarity = {similarity:.3f}
        - Interpretación: {self._interpret_semantic_score(similarity)}
        """
        
        details = {
            'similarity': similarity,
            'embedding_dim': len(embeddings[0]),
            'model': 'paraphrase-MiniLM-L6-v2',
            'interpretation': self._interpret_semantic_score(similarity)
        }
        
        return SimilarityResult(
            algorithm_name="Sentence-BERT Semantic Similarity",
            similarity_score=similarity,
            explanation=explanation,
            details=details,
            processing_time=elapsed
        )
    
    def llm_based_similarity(self, text1: str, text2: str) -> SimilarityResult:
        """
        Algoritmo 6: LLM-based Similarity usando Ollama (modelo local).
        
        Usa un modelo LLM local (Llama 3.2 3B o Mistral 7B) para análisis semántico profundo.
        El modelo analiza los textos y proporciona un score de similitud con justificación.
        
        REQUIERE Ollama instalado y el servidor corriendo.
        
        Args:
            text1: Primer texto a comparar
            text2: Segundo texto a comparar
        
        Returns:
            SimilarityResult con score de similitud (0-1) y justificación del modelo
        
        Raises:
            RuntimeError: Si Ollama no está disponible o no se puede conectar
        """
        start_time = datetime.now()
        
        if not OLLAMA_AVAILABLE or not self.ollama_available:
            error_msg = (
                "Ollama no está disponible. "
                "Para usar el algoritmo LLM-based Similarity, necesitas:\n"
                "1. Instalar Ollama: bash scripts/install_ollama.sh\n"
                "2. Iniciar el servidor: ollama serve\n"
                "3. Descargar un modelo: ollama pull llama3.2:3b"
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        prep1 = self.preprocess_text(text1, 'standard')
        prep2 = self.preprocess_text(text2, 'standard')
        
        try:
            self.logger.info(f"Usando Ollama con modelo {self.ollama_model} para análisis LLM")
            
            if analyze_similarity_with_llm is None:
                error_msg = (
                    f"No se puede usar Ollama. "
                    f"Verifica que el servidor esté corriendo y que el modelo {self.ollama_model} esté disponible."
                )
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            llm_result = analyze_similarity_with_llm(
                text1=prep1.processed_text,
                text2=prep2.processed_text,
                model=self.ollama_model
            )
            
            if llm_result is None or llm_result.get("score") is None:
                error_msg = (
                    f"Error al obtener respuesta de Ollama. "
                    f"Verifica que el servidor esté corriendo y que el modelo {self.ollama_model} esté disponible."
                )
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            score = llm_result["score"]
            justification = llm_result["justification"]
            raw_response = llm_result.get("raw_response", "")
            
            common_topics = self._extract_common_topics(prep1.tokens, prep2.tokens)
            semantic_overlap = len(common_topics) / max(len(set(prep1.tokens)), len(set(prep2.tokens))) if max(len(set(prep1.tokens)), len(set(prep2.tokens))) > 0 else 0
            
            explanation = f"""
            LLM-based Similarity (Ollama - {self.ollama_model}):
            - Score de similitud: {score:.3f}
            - Justificación del modelo: {justification}
            - Temas comunes identificados: {len(common_topics)}
            - Overlap semántico: {semantic_overlap:.2%}
            - Modelo usado: {self.ollama_model}
            - Método: Análisis LLM real (no simulado)
            """
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            details = {
                'similarity': score,
                'model': self.ollama_model,
                'justification': justification,
                'raw_llm_response': raw_response,
                'common_topics': list(common_topics)[:10],
                'semantic_overlap': semantic_overlap,
                'method': 'ollama_llm',
                'ollama_available': True
            }
            
            return SimilarityResult(
                algorithm_name=f"LLM-based Similarity (Ollama - {self.ollama_model})",
                similarity_score=score,
                explanation=explanation,
                details=details,
                processing_time=elapsed
            )
            
        except RuntimeError:
            # Re-lanzar errores de RuntimeError (Ollama no disponible)
            raise
        except Exception as e:
            error_msg = (
                f"Error al usar Ollama para análisis LLM: {e}\n"
                f"Verifica que:\n"
                f"1. El servidor Ollama esté corriendo: ollama serve\n"
                f"2. El modelo {self.ollama_model} esté disponible: ollama list\n"
                f"3. Si el modelo no está disponible, descárgalo: ollama pull {self.ollama_model}"
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _interpret_semantic_score(self, score: float) -> str:
        """Interpretar score semántico en categorías."""
        if score >= 0.8:
            return "Very similar (likely same topic and argument)"
        elif score >= 0.6:
            return "Similar (related topics or arguments)"
        elif score >= 0.4:
            return "Somewhat similar (touches on related concepts)"
        else:
            return "Different (distinct topics or arguments)"
    
    def _extract_common_topics(self, tokens1: List[str], tokens2: List[str]) -> set:
        """Extraer tokens comunes entre dos listas."""
        return set(tokens1) & set(tokens2)
    
    def analyze_texts_similarity(self, texts: List[str]) -> List[SimilarityResult]:
        """
        Analizar similitud entre múltiples textos usando todos los algoritmos.
        
        Compara cada par de textos con los 6 algoritmos disponibles y retorna
        todos los resultados.
        
        Args:
            texts: Lista de textos a analizar (mínimo 2)
            
        Returns:
            Lista de SimilarityResult, uno por algoritmo por cada par de textos
        """
        results = []
        
        if len(texts) < 2:
            self.logger.warning("Need at least 2 texts to compare")
            return results
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                text1, text2 = texts[i], texts[j]
                
                algorithms = [
                    self.levenshtein_similarity(text1, text2),
                    self.damerau_levenshtein_similarity(text1, text2),
                    self.jaccard_similarity(text1, text2, n=3),
                    self.tfidf_cosine_similarity(text1, text2),
                    self.sentence_bert_similarity(text1, text2),
                    self.llm_based_similarity(text1, text2)
                ]
                
                results.extend(algorithms)
        
        return results
