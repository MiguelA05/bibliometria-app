"""Linea temporal simple: publicaciones por año desglosadas por data_source.

Este script es una versión autónoma que lee el CSV unificado (usando
`resultsUtil.read_unified()`), agrupa por año y por `data_source`, y
genera un HTML interactivo (Plotly) y/o un PNG (Kaleido o Matplotlib)
en `results/reports/`.

Uso:
    from lineaTiempo1 import plot_publications_by_year_source
    artifact, pivot = plot_publications_by_year_source(limit=500, top_n_sources=10)

O ejecútalo directamente:
    .venv/Scripts/python.exe lineaTiempo1.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from resultsUtil import read_unified


def plot_publications_by_year_source(
    *,
    date_field: str = "publication_date",
    source_field: str = "data_source",
    limit: Optional[int] = None,
    top_n_sources: int = 8,
    output_dir: Path | str = Path("results") / "reports",
    show_plot: bool = False,
) -> Tuple[Path, pd.DataFrame]:
    """Cuenta artículos por año y por fuente y genera una gráfica.

    Retorna (ruta_al_artifact, pivot_df) donde pivot_df es un DataFrame con
    años como índice y columnas por fuente (top_n_sources + 'Other').
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_unified()
    if not rows or len(rows) < 2:
        raise FileNotFoundError("El CSV unificado está vacío o no existe.")

    header = rows[0]
    lowered = [h.lower() for h in header]

    # resolver índices
    if date_field in header:
        date_idx = header.index(date_field)
    elif date_field.lower() in lowered:
        date_idx = lowered.index(date_field.lower())
    else:
        raise ValueError(f"Columna de fecha '{date_field}' no encontrada. Columnas: {header}")

    if source_field in header:
        src_idx = header.index(source_field)
    elif source_field.lower() in lowered:
        src_idx = lowered.index(source_field.lower())
    else:
        raise ValueError(f"Columna de fuente '{source_field}' no encontrada. Columnas: {header}")

    records = []
    for i, row in enumerate(rows[1:]):
        if limit is not None and i >= limit:
            break
        date_raw = row[date_idx] if date_idx < len(row) else ""
        src_raw = row[src_idx] if src_idx < len(row) else ""
        records.append({"publication_date": date_raw, "data_source": src_raw})

    df = pd.DataFrame(records)
    df["year"] = pd.to_datetime(df["publication_date"], errors="coerce").dt.year
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df["data_source"] = df["data_source"].fillna("Unknown").astype(str)

    top_sources = df["data_source"].value_counts().nlargest(top_n_sources).index.tolist()
    df["source_group"] = df["data_source"].where(df["data_source"].isin(top_sources), other="Other")

    grouped = df.groupby(["year", "source_group"]).size().reset_index(name="count")
    pivot = grouped.pivot(index="year", columns="source_group", values="count").fillna(0).sort_index()

    # Intentar plotly
    try:
        import plotly.express as px

        fig = px.line(pivot.reset_index(), x="year", y=pivot.columns.tolist(), markers=True)
        fig.update_layout(title="Publicaciones por año y fuente", xaxis_title="Año", yaxis_title="Publicaciones")
        html_path = output_dir / "publications_timeline_by_source.html"
        fig.write_html(str(html_path))

        try:
            png_path = output_dir / "publications_timeline_by_source.png"
            fig.write_image(str(png_path), engine="kaleido")
            main_artifact = png_path
        except Exception:
            main_artifact = html_path

        if show_plot:
            fig.show()

        # Además guardar el CSV de conteos
        counts_csv = output_dir / "publications_by_year_source_counts.csv"
        pivot.to_csv(counts_csv)

        return main_artifact, pivot

    except Exception:
        # Fallback matplotlib
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        for col in pivot.columns:
            ax.plot(pivot.index, pivot[col], marker="o", label=str(col))
        ax.set_xlabel("Año")
        ax.set_ylabel("Publicaciones")
        ax.set_title("Publicaciones por año y fuente")
        ax.legend(loc="best")
        plt.tight_layout()
        png_path = output_dir / "publications_timeline_by_source.png"
        fig.savefig(str(png_path), dpi=200)
        if show_plot:
            plt.show()
        plt.close(fig)

        counts_csv = output_dir / "publications_by_year_source_counts.csv"
        pivot.to_csv(counts_csv)

        return png_path, pivot


if __name__ == "__main__":
    # Ejecución de ejemplo
    try:
        artifact, pivot = plot_publications_by_year_source(limit=500, top_n_sources=10, show_plot=False)
        print("Artefacto generado:", artifact)
        print(pivot.head())
    except Exception as e:
        print("Error ejecutando lineaTiempo1:", e)
