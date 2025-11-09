"""
Servicio para agrupamiento jerárquico de abstracts.
Requerimiento 4: Implementar 3 algoritmos de agrupamiento jerárquico con dendrogramas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Optional, Dict, Any
from datetime import datetime

try:
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")  # Backend sin display
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from app.utils.logger import get_logger
from app.utils.csv_reader import read_unified_csv, resolve_column_index, normalize_header
from app.services.word_frequency_service import DEFAULT_STOPWORDS
from app.config import settings


@dataclass
class DocumentRecord:
    """Registro de documento con índice, etiqueta y texto."""
    row_index: int
    label: str
    text: str


@dataclass
class ClusteringResult:
    """Resultado del clustering jerárquico."""
    method: str
    metric: str
    dendrogram_path: str
    cluster_count: int
    cophenetic_correlation: float
    clusters: List[Tuple[int, List[str]]]  # (cluster_id, members)
    cluster_assignments: List[int]
    best_method: Optional[str] = None


class HierarchicalClusteringService:
    """Servicio para agrupamiento jerárquico de abstracts."""
    
    def __init__(self):
        self.logger = get_logger("hierarchical_clustering")
        
        if not SCIPY_AVAILABLE:
            self.logger.error("SciPy no está disponible. Instala con: pip install scipy")
        if not SKLEARN_AVAILABLE:
            self.logger.error("scikit-learn no está disponible. Instala con: pip install scikit-learn")
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("matplotlib no está disponible. Instala con: pip install matplotlib")
    
    def perform_hierarchical_clustering(
        self,
        csv_path: Optional[str] = None,
        text_field: str = "abstract",
        label_field: str = "title",
        limit: Optional[int] = None,
        min_chars: int = 40,
        max_features: int = 1500,
        methods: List[str] = None,
        distance_threshold: float = 1.0,
        output_dir: Optional[Path] = None,
        drop_duplicates: bool = True
    ) -> Dict[str, ClusteringResult]:
        """
        Realizar agrupamiento jerárquico con múltiples métodos.
        
        Args:
            csv_path: Ruta al CSV unificado
            text_field: Campo de texto a analizar
            label_field: Campo para etiquetas
            limit: Límite de documentos a procesar
            min_chars: Longitud mínima de texto
            max_features: Máximo de características TF-IDF
            methods: Lista de métodos de linkage (default: ["single", "complete", "average"])
            distance_threshold: Umbral de distancia para clusters
            output_dir: Directorio de salida
            drop_duplicates: Eliminar duplicados
        
        Returns:
            Diccionario con resultados por método
        """
        if not SCIPY_AVAILABLE or not SKLEARN_AVAILABLE:
            raise ImportError("SciPy y scikit-learn son requeridos para clustering jerárquico")
        
        if methods is None:
            methods = ["single", "complete", "average"]
        
        valid_methods = {"single", "complete", "average", "ward"}
        for method in methods:
            if method.lower() not in valid_methods:
                raise ValueError(f"Método de linkage no soportado: {method}")
        
        methods = [m.lower() for m in methods]
        
        # Recopilar documentos
        documents = self._collect_documents(
            csv_path=csv_path,
            text_field=text_field,
            label_field=label_field,
            limit=limit,
            min_chars=min_chars,
            drop_duplicates=drop_duplicates
        )
        
        self.logger.info(f"Documentos cargados: {len(documents)}")
        
        # Vectorizar documentos
        dense_matrix, vectorizer = self._vectorize_documents(
            documents, max_features=max_features
        )
        
        labels = [f"{doc.row_index}. {doc.label}" for doc in documents]
        
        # Número máximo de hojas para dendrograma (visualización)
        max_for_dendrogram = 30
        if len(documents) > max_for_dendrogram:
            self.logger.info(f"Generando dendrogramas usando muestra de {max_for_dendrogram} documentos")
        
        if output_dir is None:
            output_dir = Path(settings.results_dir) / "reports"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        best_cophenetic = float("-inf")
        best_method = None
        
        for method in methods:
            metric = "euclidean" if method == "ward" else "cosine"
            self.logger.info(f"Ejecutando método: {method} (métrica: {metric})")
            
            # Calcular linkage
            linkage_matrix, distance_vector = self._compute_linkage(
                dense_matrix, method=method, metric=metric
            )
            
            # Generar dendrograma
            sample_n = min(len(documents), max_for_dendrogram)
            if sample_n >= 2 and sample_n < len(documents):
                linkage_sample, _ = self._compute_linkage(
                    dense_matrix[:sample_n], method=method, metric=metric
                )
                sample_labels = labels[:sample_n]
                dendrogram_path = self._plot_dendrogram(
                    linkage_sample, sample_labels, method, metric, output_dir
                )
            else:
                dendrogram_path = self._plot_dendrogram(
                    linkage_matrix, labels, method, metric, output_dir
                )
            
            # Asignar clusters
            cluster_assignments = hierarchy.fcluster(
                linkage_matrix, t=distance_threshold, criterion="distance"
            )
            
            # Evaluar calidad
            clusters = self._summarize_clusters(cluster_assignments, labels)
            metrics = self._evaluate_cluster_quality(
                dense_matrix, cluster_assignments, metric,
                linkage_matrix, distance_vector
            )
            
            result = ClusteringResult(
                method=method,
                metric=metric,
                dendrogram_path=str(dendrogram_path),
                cluster_count=metrics["cluster_count"],
                cophenetic_correlation=metrics["cophenetic_correlation"],
                clusters=clusters,
                cluster_assignments=list(cluster_assignments)
            )
            
            results[method] = result
            
            if metrics["cophenetic_correlation"] > best_cophenetic:
                best_cophenetic = metrics["cophenetic_correlation"]
                best_method = method
        
        # Marcar el mejor método
        if best_method:
            results[best_method].best_method = best_method
            self.logger.info(
                f"Método con mayor correlación cophenética ({best_cophenetic:.3f}): {best_method}"
            )
        
        return results
    
    def _collect_documents(
        self,
        csv_path: Optional[str],
        text_field: str,
        label_field: str,
        limit: Optional[int],
        min_chars: int,
        drop_duplicates: bool
    ) -> List[DocumentRecord]:
        """Recopilar y filtrar documentos."""
        rows = read_unified_csv(csv_path)
        if not rows or len(rows) < 2:
            raise ValueError("El CSV unificado no contiene datos suficientes.")
        
        header = normalize_header(rows[0])
        text_idx, text_name = resolve_column_index(header, text_field)
        label_idx = None
        if label_field:
            label_idx, _ = resolve_column_index(header, label_field)
        
        documents = []
        seen_texts = set()
        
        for row_number, row in enumerate(rows[1:], start=1):
            text = row[text_idx].strip() if text_idx < len(row) else ""
            if len(text) < min_chars:
                continue
            
            if drop_duplicates:
                key = text.lower()
                if key in seen_texts:
                    continue
                seen_texts.add(key)
            
            if label_idx is not None and label_idx < len(row):
                label = row[label_idx].strip() or f"Fila {row_number}"
            else:
                label = f"Fila {row_number}"
            
            documents.append(DocumentRecord(
                row_index=row_number,
                label=label,
                text=text
            ))
            
            if limit is not None and len(documents) >= limit:
                break
        
        if not documents:
            raise ValueError(f"No se encontraron textos válidos en la columna '{text_name}'.")
        
        return documents
    
    def _build_vectorizer(self, stopwords: Optional[Sequence[str]] = None, max_features: int = 1500) -> TfidfVectorizer:
        """Crear TfidfVectorizer."""
        token_pattern = r"(?u)[\wÀ-ÖØ-öø-ÿ]{2,}"
        stop_words = sorted(set(stopwords)) if stopwords else None
        
        return TfidfVectorizer(
            lowercase=True,
            stop_words=stop_words,
            token_pattern=token_pattern,
            max_features=max_features,
        )
    
    def _vectorize_documents(
        self,
        documents: Sequence[DocumentRecord],
        max_features: int = 1500
    ) -> Tuple[Any, TfidfVectorizer]:
        """Vectorizar documentos con TF-IDF."""
        vectorizer = self._build_vectorizer(
            stopwords=DEFAULT_STOPWORDS,
            max_features=max_features
        )
        matrix = vectorizer.fit_transform([doc.text for doc in documents])
        return matrix.toarray(), vectorizer
    
    def _compute_linkage(
        self,
        dense_matrix: Any,
        method: str,
        metric: str = "cosine"
    ) -> Tuple[Any, Any]:
        """Calcular linkage jerárquico."""
        if dense_matrix.shape[0] < 2:
            raise ValueError("Se necesitan al menos dos documentos para agrupar.")
        
        distance_vector = pdist(dense_matrix, metric=metric)
        linkage_matrix = hierarchy.linkage(distance_vector, method=method)
        return linkage_matrix, distance_vector
    
    def _plot_dendrogram(
        self,
        linkage_matrix: Any,
        labels: Sequence[str],
        method: str,
        metric: str,
        output_dir: Path,
        font_size: int = 9
    ) -> Path:
        """Generar y guardar dendrograma."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib es requerido para generar dendrogramas")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_width = max(8.0, min(16.0, len(labels) * 0.35))
        fig, ax = plt.subplots(figsize=(fig_width, 0.38 * len(labels) + 2))
        
        # Truncar etiquetas
        truncated_labels = [self._truncate_label(label) for label in labels]
        
        hierarchy.dendrogram(
            linkage_matrix,
            labels=truncated_labels,
            orientation="right",
            leaf_font_size=font_size,
            color_threshold=None,
            above_threshold_color="dimgray",
            ax=ax,
        )
        
        ax.set_title(f"Dendrograma - linkage {method}")
        ax.set_xlabel(f"Distancia ({metric})")
        ax.set_ylabel("Documentos")
        fig.tight_layout()
        
        output_path = output_dir / f"dendrogram_{method}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        
        return output_path
    
    def _truncate_label(self, label: str, max_chars: int = 50) -> str:
        """Truncar etiquetas largas."""
        if len(label) <= max_chars:
            return label
        return label[:max_chars - 1].rstrip() + "…"
    
    def _summarize_clusters(
        self,
        cluster_assignments: Sequence[int],
        labels: Sequence[str]
    ) -> List[Tuple[int, List[str]]]:
        """Resumir clusters con sus miembros."""
        groups: Dict[int, List[str]] = {}
        
        for cluster_id, label in zip(cluster_assignments, labels):
            groups.setdefault(cluster_id, []).append(label)
        
        sorted_groups = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
        return [(cluster_id, members) for cluster_id, members in sorted_groups]
    
    def _evaluate_cluster_quality(
        self,
        dense_matrix: Any,
        cluster_assignments: Sequence[int],
        metric: str,
        linkage_matrix: Any,
        distance_vector: Any
    ) -> Dict[str, Any]:
        """Evaluar calidad de clusters."""
        unique_clusters = set(cluster_assignments)
        cluster_count = len(unique_clusters)
        cophenetic_corr, _ = hierarchy.cophenet(linkage_matrix, distance_vector)
        
        return {
            "cluster_count": cluster_count,
            "cophenetic_correlation": float(cophenetic_corr),
        }

