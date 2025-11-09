"""
Servicio de similitud textual con 4 algoritmos clásicos y 2 algoritmos basados en IA.
Incluye preprocesamiento normalizado y análisis detallado paso a paso.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import unicodedata
import json

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
        check_model_available,
        OLLAMA_AVAILABLE
    )
except ImportError:
    OLLAMA_AVAILABLE = False
    ensure_ollama_ready = None
    analyze_similarity_with_llm = None
    check_model_available = None
    print("[WARNING] Ollama helper no disponible. Algoritmo LLM-based usará modo simulado.")

from app.utils.logger import get_logger
from app.config import settings


@dataclass
class SimilarityResult:
    """Resultado de un algoritmo de similitud."""
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
    """Servicio para análisis de similitud textual con múltiples algoritmos."""
    
    def __init__(self, ollama_model: str = "llama3.2:3b"):
        self.logger = get_logger("text_similarity")
        self.ollama_model = ollama_model
        
        # Inicializar stemmer
        try:
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stemmer = None
            self.stop_words = set()
        
        # Cargar modelo de embeddings si está disponible
        self.sbert_model = None
        if SENTENCE_BERT_AVAILABLE:
            try:
                self.sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                self.logger.info("Sentence-BERT model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Could not load Sentence-BERT model: {e}")
                self.sbert_model = None
        
        # Verificar Ollama si está disponible
        self.ollama_available = False
        if OLLAMA_AVAILABLE:
            try:
                if ensure_ollama_ready():
                    self.ollama_available = True
                    self.logger.info(f"Ollama disponible. Modelo configurado: {ollama_model}")
                else:
                    self.logger.warning("Ollama no está disponible. LLM-based similarity usará modo simulado.")
            except Exception as e:
                self.logger.warning(f"Error verificando Ollama: {e}")
    
    def preprocess_text(self, text: str, method: str = 'standard') -> TextPreprocessingResult:
        """
        Preprocesar texto según el método especificado.
        
        Args:
            text: Texto original
            method: 'standard', 'char-level', 'token-level', 'minimal'
            
        Returns:
            TextPreprocessingResult con texto procesado y estadísticas
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
        
        # Paso 1: Normalización Unicode
        normalized = unicodedata.normalize('NFKC', text)
        
        # Paso 2: Minúsculas
        text_lower = normalized.lower()
        
        # Paso 3: Detectar idioma (simple)
        detected_lang = self._detect_language(text_lower)
        
        # Paso 4: Procesamiento según método
        if method == 'char-level':
            # Para Levenshtein: solo normalización básica
            processed = self._clean_for_char_similarity(text_lower)
            tokens = list(processed)
            
        elif method == 'minimal':
            # Mínimo procesamiento
            processed = text_lower
            tokens = processed.split()
            
        elif method == 'token-level':
            # Tokenización completa
            tokens = self._tokenize(text_lower)
            processed = ' '.join(tokens)
            
        else:  # standard
            # Procesamiento estándar completo
            tokens = self._tokenize_advanced(text_lower)
            processed = ' '.join(tokens)
        
        # Estadísticas
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
        """Limpiar texto para similitud a nivel de caracteres."""
        # Remover espacios extra pero mantener estructura
        return re.sub(r'\s+', ' ', text.strip())
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenización básica."""
        # Remover puntuación y tokenizar
        text_no_punct = re.sub(r'[^\w\s]', '', text)
        return text_no_punct.split()
    
    def _tokenize_advanced(self, text: str) -> List[str]:
        """Tokenización avanzada con stemmer y stopwords."""
        if not self.stemmer:
            return self._tokenize(text)
        
        tokens = self._tokenize(text)
        
        # Remover stopwords
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        # Stemming
        tokens = [self.stemmer.stem(t) for t in tokens]
        
        return tokens
    
    def _detect_language(self, text: str) -> str:
        """Detección simple de idioma."""
        # Contar palabras comunes en inglés
        english_words = {'the', 'and', 'is', 'are', 'in', 'on', 'to', 'of', 'a', 'an'}
        
        words = text.split()
        english_count = sum(1 for word in words if word.lower() in english_words)
        
        if english_count / len(words) > 0.1 if words else False:
            return 'english'
        return 'unknown'
    
    # ==================== ALGORITMOS CLÁSICOS ====================
    
    def levenshtein_similarity(self, text1: str, text2: str, 
                               include_matrix: bool = False) -> SimilarityResult:
        """
        Algoritmo 1: Levenshtein (Distancia de Edición).
        
        Returns:
            SimilarityResult con distancia, matriz DP opcional y backtrace
        """
        start_time = datetime.now()
        
        # Preprocesar para char-level
        prep1 = self.preprocess_text(text1, 'char-level')
        prep2 = self.preprocess_text(text2, 'char-level')
        
        s1, s2 = prep1.processed_text, prep2.processed_text
        n, m = len(s1), len(s2)
        
        # Matriz DP: dp[i][j] = distancia para s1[0:i] y s2[0:j]
        dp = np.zeros((n + 1, m + 1), dtype=int)
        
        # Inicializar primera fila y columna
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        
        # Llenar matriz DP
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if s1[i-1] == s2[j-1]:
                    cost = 0
                else:
                    cost = 1
                
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # inserción
                    dp[i][j-1] + 1,      # eliminación
                    dp[i-1][j-1] + cost   # sustitución
                )
        
        # Distancia final
        distance = dp[n][m]
        max_len = max(n, m)
        similarity = 1 - (distance / max_len) if max_len > 0 else 0
        
        # Backtrace (opcional)
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
        
        Returns:
            SimilarityResult con distancia, transposiciones detectadas
        """
        start_time = datetime.now()
        
        prep1 = self.preprocess_text(text1, 'char-level')
        prep2 = self.preprocess_text(text2, 'char-level')
        
        s1, s2 = prep1.processed_text, prep2.processed_text
        n, m = len(s1), len(s2)
        
        dp = np.zeros((n + 1, m + 1), dtype=int)
        
        # Inicializar
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        
        # Llenar con Damerau
        transpositions = []
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if s1[i-1] == s2[j-1]:
                    cost = 0
                else:
                    cost = 1
                
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # inserción
                    dp[i][j-1] + 1,      # eliminación
                    dp[i-1][j-1] + cost   # sustitución
                )
                
                # Transposición (Damerau)
                if (i > 1 and j > 1 and 
                    s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]):
                    dp[i][j] = min(dp[i][j], dp[i-2][j-2] + 1)
                    
                    if not transpositions or transpositions[-1] != (i-2, j-2):
                        transpositions.append((i-2, j-2, s1[i-2:i], s2[j-2:j]))
        
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
        
        Args:
            text1, text2: Textos a comparar
            n: Longitud de los n-grams (shingles)
            
        Returns:
            SimilarityResult con shingles comunes y proporción
        """
        start_time = datetime.now()
        
        prep1 = self.preprocess_text(text1, 'token-level')
        prep2 = self.preprocess_text(text2, 'token-level')
        
        # Generar n-grams
        def generate_ngrams(tokens, n):
            """Generar n-grams de tokens."""
            ngrams = []
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i+n])
                ngrams.append(ngram)
            return set(ngrams)
        
        shingles1 = generate_ngrams(prep1.tokens, n)
        shingles2 = generate_ngrams(prep2.tokens, n)
        
        # Calcular Jaccard
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
        
        # Ejemplos de shingles comunes
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
        
        Returns:
            SimilarityResult con top términos y contribuciones
        """
        start_time = datetime.now()
        
        prep1 = self.preprocess_text(text1, 'standard')
        prep2 = self.preprocess_text(text2, 'standard')
        
        # Validar que hay texto suficiente para vectorizar
        corpus = [prep1.processed_text, prep2.processed_text]
        if not corpus[0].strip() or not corpus[1].strip():
            similarity = 0
            top_terms = []
            feature_names = []
            term_contributions = []
        else:
            # Vectorizar con TF-IDF
            vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3), min_df=1)
            
            try:
                tfidf_matrix = vectorizer.fit_transform(corpus)
            
                # Calcular similitud coseno
                similarity = cosine_similarity([tfidf_matrix[0]], [tfidf_matrix[1]])[0][0]
                
                # Extraer términos importantes
                feature_names = vectorizer.get_feature_names_out()
                tfidf1 = tfidf_matrix[0].toarray()[0]
                tfidf2 = tfidf_matrix[1].toarray()[0]
                
                # Producto punto por término
                term_contributions = []
                for i, term in enumerate(feature_names):
                    contrib = tfidf1[i] * tfidf2[i]
                    if contrib > 0.01:  # Umbral
                        term_contributions.append({
                            'term': term,
                            'contribution': float(contrib),
                            'tfidf1': float(tfidf1[i]),
                            'tfidf2': float(tfidf2[i])
                        })
                
                term_contributions.sort(key=lambda x: x['contribution'], reverse=True)
                top_terms = term_contributions[:20]
                
            except Exception as e:
                similarity = 0
                top_terms = []
                feature_names = []
                term_contributions = []
                self.logger.warning(f"Error in TF-IDF (texts may be too short): {e}")
        
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
    
    # ==================== ALGORITMOS IA ====================
    
    def sentence_bert_similarity(self, text1: str, text2: str) -> SimilarityResult:
        """
        Algoritmo 5: Sentence-BERT (Embeddings semánticos) + Cosine.
        
        Returns:
            SimilarityResult con embeddings y explicación semántica
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
        
        # Generar embeddings
        embeddings = self.sbert_model.encode([text1, text2], convert_to_numpy=True)
        
        # Calcular similitud coseno
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
        
        REQUIERE Ollama instalado y el servidor corriendo.
        Usa un modelo LLM local (Llama 3.2 3B o Mistral 7B) para análisis semántico profundo.
        
        Raises:
            RuntimeError: Si Ollama no está disponible o no se puede conectar.
        """
        start_time = datetime.now()
        
        # Verificar que Ollama esté disponible
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
        
        # Preprocesar textos
        prep1 = self.preprocess_text(text1, 'standard')
        prep2 = self.preprocess_text(text2, 'standard')
        
        # Usar Ollama para análisis de similitud
        try:
            self.logger.info(f"Usando Ollama con modelo {self.ollama_model} para análisis LLM")
            
            # Verificar que el modelo esté disponible
            if analyze_similarity_with_llm is None:
                error_msg = (
                    f"No se puede usar Ollama. "
                    f"Verifica que el servidor esté corriendo y que el modelo {self.ollama_model} esté disponible."
                )
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Usar Ollama para análisis de similitud
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
            
            # Análisis adicional con tokens comunes
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
    
    # ==================== MÉTODOS AUXILIARES ====================
    
    def _interpret_semantic_score(self, score: float) -> str:
        """Interpretar score semántico."""
        if score >= 0.8:
            return "Very similar (likely same topic and argument)"
        elif score >= 0.6:
            return "Similar (related topics or arguments)"
        elif score >= 0.4:
            return "Somewhat similar (touches on related concepts)"
        else:
            return "Different (distinct topics or arguments)"
    
    def _extract_common_topics(self, tokens1: List[str], tokens2: List[str]) -> set:
        """Extraer temas comunes (simulado)."""
        set1, set2 = set(tokens1), set(tokens2)
        return set1 & set2
    
    def analyze_texts_similarity(self, texts: List[str]) -> List[SimilarityResult]:
        """
        Analizar similitud entre múltiples textos usando todos los algoritmos.
        
        Args:
            texts: Lista de textos a analizar
            
        Returns:
            Lista de resultados para cada par
        """
        results = []
        
        if len(texts) < 2:
            self.logger.warning("Need at least 2 texts to compare")
            return results
        
        # Comparar cada par
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                text1, text2 = texts[i], texts[j]
                
                # Ejecutar todos los algoritmos
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
