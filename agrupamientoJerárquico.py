"""Herramientas para agrupar abstracts con clustering jerárquico.

Este módulo carga los textos desde el CSV unificado, vectoriza con TF-IDF y
genera dendrogramas para distintos métodos de *linkage*.  Se puede ejecutar
desde la línea de comandos o importar las funciones principales.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from contadorPalabras import DEFAULT_STOPWORDS
from resultsUtil import read_unified


try:  # SciPy y scikit-learn son necesarios para el pipeline
	from scipy.cluster import hierarchy
	from scipy.spatial.distance import pdist
except ImportError as exc:  # pragma: no cover - validamos en tiempo de ejecución
	raise SystemExit(
		"SciPy es requerida para agrupar abstracts. Ejecuta 'pip install scipy'."
	) from exc

try:
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.metrics import silhouette_score
except ImportError as exc:  # pragma: no cover - validamos en tiempo de ejecución
	raise SystemExit(
		"scikit-learn es requerida para vectorizar textos. Ejecuta 'pip install scikit-learn'."
	) from exc


# \ufeff aparece a veces en la primera cabecera; lo normalizamos al leer.
def _normalize_header(raw_header: Sequence[str]) -> List[str]:
	"""Remove BOM markers that sometimes appear in the first CSV header cell."""

	return [value.lstrip("\ufeff") if isinstance(value, str) else value for value in raw_header]


def _resolve_column_index(header: Sequence[str], field: str | int) -> Tuple[int, str]:
	"""Translate a column specifier (name or index) into a zero-based index.

	The CSV reader delivers every header exactly as stored, so this helper adds a
	case-insensitive fallback when a string is provided.  It raises a ``ValueError``
	with a friendly message if the column does not exist.
	"""

	lowered = [h.lower() for h in header]
	if isinstance(field, int):
		if field < 0 or field >= len(header):
			raise ValueError(f"Índice de columna fuera de rango: {field}")
		return field, header[field]

	if field in header:
		return header.index(field), field
	if field.lower() in lowered:
		index = lowered.index(field.lower())
		return index, header[index]

	raise ValueError(
		f"Columna '{field}' no encontrada. Columnas disponibles: {', '.join(header)}"
	)


@dataclass
class DocumentRecord:
	"""Small data container joining the row index, label and raw text."""

	row_index: int
	label: str
	text: str


def collect_documents(
	*,
	text_field: str | int = "abstract",
	label_field: str | int | None = "title",
	limit: int | None = 40,
	min_chars: int = 40,
	drop_duplicates: bool = True,
) -> List[DocumentRecord]:
	"""Load and pre-filter the documents that will feed the clustering step.

	Los puntos principales:
	- Filtramos filas sin contenido o con textos muy cortos (``min_chars``) porque
	  aportan ruido al vectorizador.
	- ``limit`` corta la lista para mantener los dendrogramas legibles.
	- ``drop_duplicates`` evita procesar abstracts repetidos, algo habitual en
	  datasets unificados.
	El resultado es una lista ordenada de ``DocumentRecord`` lista para
	vectorizarse.
	"""

	rows = read_unified()
	if not rows or len(rows) < 2:
		raise ValueError("El CSV `unified` no contiene datos suficientes.")

	header = _normalize_header(rows[0])
	text_idx, text_name = _resolve_column_index(header, text_field)
	label_idx = None
	if label_field is not None:
		label_idx, _ = _resolve_column_index(header, label_field)

	documents: List[DocumentRecord] = []
	seen_texts: set[str] = set()

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

		documents.append(DocumentRecord(row_index=row_number, label=label, text=text))

		if limit is not None and len(documents) >= limit:
			break

	if not documents:
		raise ValueError(
			f"No se encontraron textos válidos en la columna '{text_name}'."
		)

	return documents


def build_vectorizer(
	*, stopwords: Iterable[str] | None = None, max_features: int = 1500
) -> TfidfVectorizer:
	"""Crear un ``TfidfVectorizer`` con tokenización y stopwords amigables.

	El patrón de tokens acepta letras con acentos, requiere al menos dos
	caracteres por palabra y, si se proporcionan, transforma las stopwords en una
	lista ordenada compatible con scikit-learn.
	"""

	token_pattern = r"(?u)[\wÀ-ÖØ-öø-ÿ]{2,}"  # admite caracteres Unicode y filtra tokens de 1 letra
	stop_words = sorted(set(stopwords)) if stopwords is not None else None

	return TfidfVectorizer(
		lowercase=True,
		stop_words=stop_words,
		token_pattern=token_pattern,
		max_features=max_features,
	)


def vectorize_documents(
	documents: Sequence[DocumentRecord],
	*,
	stopwords: Iterable[str] | None = None,
	max_features: int = 1500,
):
	"""Transformar la lista de ``DocumentRecord`` a una matriz TF-IDF densa."""

	vectorizer = build_vectorizer(stopwords=stopwords, max_features=max_features)
	matrix = vectorizer.fit_transform([doc.text for doc in documents])
	return matrix.toarray(), vectorizer


def compute_linkage(
	dense_matrix,
	*,
	method: str,
	metric: str = "cosine",
):
	"""Calcular el linkage jerárquico y devolver también el vector de distancias.

	El vector de distancias condensado se reutiliza más adelante para calcular la
	correlación cophenética sin tener que reconstruirlo.
	"""

	if dense_matrix.shape[0] < 2:
		raise ValueError("Se necesitan al menos dos documentos para agrupar.")

	distance_vector = pdist(dense_matrix, metric=metric)
	linkage_matrix = hierarchy.linkage(distance_vector, method=method)
	return linkage_matrix, distance_vector


def configure_matplotlib(show_plots: bool):
	"""Seleccionar el backend apropiado y exponer ``matplotlib``/``pyplot``."""

	import matplotlib

	if not show_plots:
		matplotlib.use("Agg")

	import matplotlib.pyplot as plt

	return matplotlib, plt


def truncate_label(label: str, max_chars: int = 50) -> str:
	"""Recortar etiquetas largas para que el dendrograma siga siendo legible."""

	if len(label) <= max_chars:
		return label
	return label[: max_chars - 1].rstrip() + "…"


def plot_dendrogram(
	linkage_matrix,
	labels: Sequence[str],
	*,
	method: str,
	metric: str,
	output_dir: Path,
	show_plot: bool,
	font_size: int = 9,
) -> Path:
	"""Generar y guardar un dendrograma en disco usando etiquetas truncadas."""

	_, plt = configure_matplotlib(show_plot)

	output_dir.mkdir(parents=True, exist_ok=True)
	fig_width = max(8.0, min(16.0, len(labels) * 0.35))
	fig, ax = plt.subplots(figsize=(fig_width, 0.38 * len(labels) + 2))

	hierarchy.dendrogram(
		linkage_matrix,
		labels=[truncate_label(label) for label in labels],
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

	if show_plot:
		plt.show()

	plt.close(fig)
	return output_path


def summarize_clusters(cluster_assignments: Sequence[int], labels: Sequence[str]) -> List[Tuple[int, List[str]]]:
	"""Build an ordered summary that lists the members of cada cluster."""

	groups: dict[int, List[str]] = {}

	for cluster_id, label in zip(cluster_assignments, labels):
		groups.setdefault(cluster_id, []).append(label)

	sorted_groups = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
	return [(cluster_id, members) for cluster_id, members in sorted_groups]


def evaluate_cluster_quality(
	dense_matrix,
	cluster_assignments: Sequence[int],
	*,
	metric: str,
	linkage_matrix,
	distance_vector,
):
	"""Calcular cuántos clusters se formaron y dos indicadores de coherencia."""

	unique_clusters = set(cluster_assignments)
	cluster_count = len(unique_clusters)

	cophenetic_corr, _ = hierarchy.cophenet(linkage_matrix, distance_vector)

	silhouette_val = None
	if 1 < cluster_count < len(cluster_assignments):
		try:
			silhouette_val = float(silhouette_score(dense_matrix, cluster_assignments, metric=metric))
		except ValueError:
			silhouette_val = None

	return {
		"cluster_count": cluster_count,
		"cophenetic_correlation": float(cophenetic_corr),
		"silhouette": silhouette_val,
	}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	"""Definir los argumentos de línea de comandos para reproducir el análisis."""
	parser = argparse.ArgumentParser(description="Clustering jerárquico de abstracts del CSV unificado")
	parser.add_argument("--field", default="abstract", help="Columna de texto a analizar (nombre o índice)")
	parser.add_argument(
		"--label-field",
		default="title",
		help="Columna usada como etiqueta en el dendrograma (nombre o índice)",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=40,
		help="Número máximo de documentos a procesar (más de 50 dificulta la lectura del dendrograma)",
	)
	parser.add_argument(
		"--min-chars",
		type=int,
		default=40,
		help="Longitud mínima de texto para considerar un documento",
	)
	parser.add_argument(
		"--max-features",
		type=int,
		default=1500,
		help="Características máximas en la matriz TF-IDF",
	)
	parser.add_argument(
		"--methods",
		nargs="+",
		default=["single", "complete", "average"],
		help="Métodos de linkage a ejecutar (por ejemplo: single complete average)",
	)
	parser.add_argument(
		"--distance-threshold",
		type=float,
		default=1.0,
		help="Umbral de distancia para el resumen textual de clusters",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("results") / "reports",
		help="Directorio donde se guardarán los dendrogramas",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Mostrar las figuras además de guardarlas en disco",
	)
	parser.add_argument(
		"--clusters-json",
		type=Path,
		default=None,
		help="Ruta opcional para exportar el resumen de clusters en JSON",
	)
	parser.add_argument(
		"--keep-duplicates",
		action="store_true",
		help="No eliminar abstracts duplicados (por defecto se deduplican)",
	)
	return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
	"""Ejecutar el pipeline completo desde la terminal."""

	args = parse_args(argv)

	methods = [method.lower() for method in args.methods]
	valid_methods = {"single", "complete", "average", "ward"}

	for method in methods:
		if method not in valid_methods:
			raise SystemExit(f"Método de linkage no soportado: {method}")

	documents = collect_documents(
		text_field=args.field,
		label_field=args.label_field,
		limit=args.limit,
		min_chars=args.min_chars,
		drop_duplicates=not args.keep_duplicates,
	)

	print(f"Documentos cargados: {len(documents)}")
	print("Primeras etiquetas:")
	for doc in documents[:5]:
		preview = truncate_label(doc.text, max_chars=80)
		print(f"  - {doc.label} :: {preview}")

	dense_matrix, _ = vectorize_documents(
		documents,
		stopwords=DEFAULT_STOPWORDS,
		max_features=args.max_features,
	)

	labels = [f"{doc.row_index}. {doc.label}" for doc in documents]

	results_summary = []
	best_silhouette = float("-inf")
	best_cophenetic = float("-inf")
	best_by_silhouette = None
	best_by_cophenetic = None
	for method in methods:
		metric = "euclidean" if method == "ward" else "cosine"
		linkage_matrix, distance_vector = compute_linkage(dense_matrix, method=method, metric=metric)
		output_path = plot_dendrogram(
			linkage_matrix,
			labels,
			method=method,
			metric=metric,
			output_dir=args.output_dir,
			show_plot=args.show,
		)
		print(f"Dendrograma guardado en: {output_path}")

		cluster_assignments = hierarchy.fcluster(
			linkage_matrix,
			t=args.distance_threshold,
			criterion="distance",
		)
		clusters = summarize_clusters(cluster_assignments, labels)
		metrics = evaluate_cluster_quality(
			dense_matrix,
			cluster_assignments,
			metric=metric,
			linkage_matrix=linkage_matrix,
			distance_vector=distance_vector,
		)
		results_summary.append({
			"method": method,
			"dendrogram_path": str(output_path),
			"clusters": clusters,
			"cluster_assignments": list(cluster_assignments),
			"metrics": metrics,
		})

		if metrics["silhouette"] is not None and metrics["silhouette"] > best_silhouette:
			best_silhouette = metrics["silhouette"]
			best_by_silhouette = method

		if metrics["cophenetic_correlation"] > best_cophenetic:
			best_cophenetic = metrics["cophenetic_correlation"]
			best_by_cophenetic = method

		print(f"Resumen de clusters (umbral {args.distance_threshold}):")
		for cluster_id, members in clusters[:5]:
			joined = "; ".join(truncate_label(label, 60) for label in members)
			print(f"  - Cluster {cluster_id} ({len(members)} docs): {joined}")

		print("Métricas de coherencia:")
		print(f"  - Correlación cophenética: {metrics['cophenetic_correlation']:.3f}")
		if metrics["silhouette"] is not None:
			print(f"  - Coeficiente silhouette: {metrics['silhouette']:.3f}")
		else:
			print("  - Coeficiente silhouette: no disponible (solo un cluster)")

	if args.clusters_json:
		serializable = [
			{
				"method": entry["method"],
				"dendrogram_path": entry["dendrogram_path"],
				"metrics": entry["metrics"],
				"cluster_assignments": entry["cluster_assignments"],
				"clusters": [
					{"cluster_id": cluster_id, "members": list(members)}
					for cluster_id, members in entry["clusters"]
				],
			}
			for entry in results_summary
		]

		args.clusters_json.parent.mkdir(parents=True, exist_ok=True)
		args.clusters_json.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
		print(f"Resumen completo exportado a {args.clusters_json}")

	if best_by_silhouette is not None:
		print(
			f"Método con mayor silhouette ({best_silhouette:.3f}): {best_by_silhouette}."
		)
	elif best_by_cophenetic is not None:
		print(
			f"Método con correlación cophenética más alta ({best_cophenetic:.3f}): {best_by_cophenetic}."
		)


if __name__ == "__main__":  # pragma: no cover - ejecución manual
	main()
