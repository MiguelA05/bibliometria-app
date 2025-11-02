"""Linea temporal de eventos: etiquetas tipo caja por publicación.

Este script genera una representación similar a la vista en `analisisVisual.py`
pero como un script independiente: cada evento aparece en el eje X por año y
con una etiqueta en caja que contiene el título (o el campo que indiques).

Uso:
    from lineaTiempo2 import plot_publications_timeline_events
    artifact, df = plot_publications_timeline_events(limit=500, max_events=800)

O ejecútalo directamente:
    .venv/Scripts/python.exe lineaTiempo2.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from resultsUtil import read_unified


def plot_publications_timeline_events(
    *,
    date_field: str = "publication_date",
    label_field: str = "title",
    source_field: str = "data_source",
    limit: Optional[int] = None,
    max_events: Optional[int] = 1000,
    output_dir: Path | str = Path("results") / "reports",
    show_plot: bool = False,
) -> Tuple[Path, pd.DataFrame]:
    """Genera una línea temporal con etiquetas tipo caja por evento.

    Devuelve la ruta al artefacto (HTML o PNG) y el DataFrame con columnas
    ['year','label','source','y_plot'] usado para dibujar.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_unified()
    if not rows or len(rows) < 2:
        raise FileNotFoundError("El CSV unificado está vacío o no existe.")

    header = rows[0]
    lowered = [h.lower() for h in header]

    def _resolve(colname: str) -> int:
        if colname in header:
            return header.index(colname)
        if colname.lower() in lowered:
            return lowered.index(colname.lower())
        return -1

    date_idx = _resolve(date_field)
    src_idx = _resolve(source_field)
    lbl_idx = _resolve(label_field)
    if date_idx == -1:
        raise ValueError(f"Columna de fecha '{date_field}' no encontrada. Columnas: {header}")
    if src_idx == -1:
        raise ValueError(f"Columna de fuente '{source_field}' no encontrada. Columnas: {header}")

    records = []
    for i, row in enumerate(rows[1:]):
        if limit is not None and i >= limit:
            break
        date_raw = row[date_idx] if date_idx < len(row) else ""
        try:
            year = pd.to_datetime(date_raw, errors="coerce").year
        except Exception:
            year = None
        if pd.isna(year):
            continue
        src_raw = row[src_idx] if src_idx < len(row) else ""
        if lbl_idx != -1 and lbl_idx < len(row):
            lbl_raw = row[lbl_idx]
        else:
            lbl_raw = src_raw
        records.append({"year": int(year), "label": str(lbl_raw).strip(), "source": str(src_raw).strip()})
        if max_events is not None and len(records) >= max_events:
            break

    if not records:
        raise ValueError("No se encontraron publicaciones con fecha válida para generar la línea temporal.")

    df = pd.DataFrame(records).sort_values("year")

    # Agrupar por año y asignar posiciones verticales
    year_groups = df.groupby("year")
    placements = []
    for year, group in year_groups:
        texts = group["label"].tolist()
        sources = group["source"].tolist()
        for i, (t, s) in enumerate(zip(texts, sources)):
            placements.append({"year": year, "label": t, "source": s, "y_pos": i})

    placed_df = pd.DataFrame(placements)
    max_stack = placed_df["y_pos"].max() if not placed_df.empty else 0
    if max_stack > 0:
        placed_df["y_plot"] = placed_df["y_pos"] / (max_stack + 1) * 0.8 + 0.1
    else:
        placed_df["y_plot"] = 0.1

    # Intentar Plotly
    try:
        import plotly.graph_objects as go

        years = placed_df["year"].astype(int)
        yvals = placed_df["y_plot"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[years.min(), years.max()], y=[0, 0], mode="lines", line=dict(color="#888"), hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=years, y=yvals, mode="markers", marker=dict(size=6, color="#ff7f0e"), hoverinfo="text", text=placed_df["label"]))

        annotations = []
        for r in placed_df.to_dict(orient="records"):
            annotations.append(
                dict(
                    x=r["year"],
                    y=r["y_plot"],
                    xref="x",
                    yref="y",
                    text=str(r["label"]),
                    showarrow=False,
                    bgcolor="#d62728",
                    bordercolor="#800000",
                    font=dict(color="white", size=12),
                    align="center",
                    opacity=0.9,
                    ax=0,
                    ay=-10,
                )
            )

        fig.update_layout(annotations=annotations)
        fig.update_yaxes(visible=False)
        fig.update_xaxes(title_text="Año", dtick=1)
        fig.update_layout(title_text="Línea temporal de publicaciones (eventos)", height=400 + int(max_stack) * 6)

        html_path = Path(output_dir) / "publications_timeline_events.html"
        fig.write_html(str(html_path))

        try:
            png_path = Path(output_dir) / "publications_timeline_events.png"
            fig.write_image(str(png_path), engine="kaleido")
            main_artifact = png_path
        except Exception:
            main_artifact = html_path

        if show_plot:
            fig.show()

        return main_artifact, placed_df

    except Exception:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, max(4, 0.4 * (max_stack + 1))))
        ax.hlines(0, placed_df["year"].min() - 1, placed_df["year"].max() + 1, colors="#888", linewidth=1)
        for _, r in placed_df.iterrows():
            x = r["year"]
            y = r["y_plot"]
            bbox = dict(boxstyle="round,pad=0.3", facecolor="#d62728", edgecolor="#800000", alpha=0.9)
            ax.text(x, y, str(r["label"]), ha="center", va="center", fontsize=9, color="white", bbox=bbox)

        ax.set_ylim(-0.1, 1.05)
        ax.set_yticks([])
        ax.set_xlabel("Año")
        ax.set_title("Línea temporal de publicaciones (eventos)")
        plt.tight_layout()
        png_path = Path(output_dir) / "publications_timeline_events.png"
        fig.savefig(str(png_path), dpi=200)
        if show_plot:
            plt.show()
        plt.close(fig)
        return png_path, placed_df


if __name__ == "__main__":
    try:
        artifact, df = plot_publications_timeline_events(limit=500, max_events=800, show_plot=False)
        print("Artefacto generado:", artifact)
        print(df.head())
    except Exception as e:
        print("Error generando linea temporal (events):", e)
