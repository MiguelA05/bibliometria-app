"""Linea temporal alternante (arriba/abajo) para muchos eventos.

Genera una timeline donde los eventos se colocan alternando por encima y
por debajo de la línea base a medida que avanzan los años. Está diseñada
para manejar etiquetas largas (se dibujan en horizontal) y un número
considerable de eventos; la imagen se hará ancha en función del número
de años/eventos para mantener legibilidad.

Salida: `results/reports/publications_timeline_alternating.html` y/o .png

Uso:
    from lineaTiempo3 import plot_publications_timeline_alternating
    artifact, df = plot_publications_timeline_alternating(limit=1000, max_events=1000)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from resultsUtil import read_unified


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def plot_publications_timeline_alternating(
    *,
    date_field: str = "publication_date",
    label_field: str = "title",
    source_field: str = "data_source",
    limit: Optional[int] = None,
    max_events: Optional[int] = 2000,
    output_dir: Path | str = Path("results") / "reports",
    show_plot: bool = False,
    label_max_chars: int = 180,
) -> Tuple[Path, pd.DataFrame]:
    """Genera una timeline alternante (arriba/abajo) apropiada para muchos eventos.

    - label_max_chars: longitud máxima que se dibuja; el hover (Plotly) mostrará
      el texto completo.
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

    df = pd.DataFrame(records).sort_values(["year"]).reset_index(drop=True)

    # Asignar posiciones alternadas: para cada evento i => sign = (-1)**i,
    # altura = 0.15 * (1 + floor(i/2) % 6) para escalonar.
    placements = []
    for i, row in df.iterrows():
        sign = 1 if (i % 2 == 0) else -1
        layer = (i // 2) % 8  # limitar el escalado de capas
        height = (0.12 + 0.06 * layer) * sign
        placements.append({"year": int(row["year"]), "label": row["label"], "source": row["source"], "y_plot": height, "i": i})

    placed_df = pd.DataFrame(placements)

    # Calcular ancho dinámico: basado en número de años únicos y eventos
    n_years = placed_df["year"].nunique()
    n_events = len(placed_df)
    # width per year ~ 80 px, cap to reasonable limits
    width = int(min(16000, max(1200, n_years * 80, n_events * 30)))
    height = int(min(4000, 400 + (placed_df["y_plot"].abs().max() * 120)))

    # Truncar etiquetas dibujadas para que no estorben; dejar completo en hover
    placed_df["label_short"] = placed_df["label"].apply(lambda t: _truncate(t, label_max_chars))

    # Intentar Plotly
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        # línea base
        fig.add_trace(go.Scatter(x=[placed_df["year"].min() - 1, placed_df["year"].max() + 1], y=[0, 0], mode="lines", line=dict(color="#444"), hoverinfo='skip'))

        # puntos invisibles para hover
        fig.add_trace(
            go.Scatter(
                x=placed_df["year"],
                y=placed_df["y_plot"],
                mode="markers",
                marker=dict(size=6, color="#1f77b4"),
                hoverinfo="text",
                text=placed_df["label"],
            )
        )

        annotations = []
        for r in placed_df.to_dict(orient="records"):
            annotations.append(
                dict(
                    x=r["year"],
                    y=r["y_plot"],
                    xref="x",
                    yref="y",
                    text=str(r["label_short"]),
                    showarrow=False,
                    bgcolor="#2ca02c" if r.get("i", 0) % 2 == 0 else "#d62728",
                    bordercolor="#555",
                    font=dict(color="white", size=10),
                    align="center",
                    opacity=0.95,
                    ax=0,
                    ay=-6 if r["y_plot"] > 0 else 6,
                )
            )

        fig.update_layout(annotations=annotations)
        fig.update_yaxes(visible=False)
        fig.update_xaxes(title_text="Año", dtick=1)
        fig.update_layout(title_text="Línea temporal alternante de publicaciones", width=width, height=height)

        html_path = output_dir / "publications_timeline_alternating.html"
        fig.write_html(str(html_path))

        try:
            png_path = output_dir / "publications_timeline_alternating.png"
            fig.write_image(str(png_path), engine="kaleido")
            main_artifact = png_path
        except Exception:
            main_artifact = html_path

        if show_plot:
            fig.show()

        return main_artifact, placed_df

    except Exception:
        # Fallback matplotlib
        import matplotlib.pyplot as plt

        fig_width_inches = max(12, width / 100)
        fig_height_inches = max(6, height / 100)
        fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))
        ax.hlines(0, placed_df["year"].min() - 1, placed_df["year"].max() + 1, colors="#444", linewidth=1)

        # dibujar markers y cajas
        for _, r in placed_df.iterrows():
            x = r["year"]
            y = r["y_plot"]
            ax.plot(x, 0, marker="o", color="#333")
            bbox = dict(boxstyle="round,pad=0.3", facecolor=("#2ca02c" if r.get("i", 0) % 2 == 0 else "#d62728"), edgecolor="#555", alpha=0.95)
            ax.text(x, y, _truncate(r["label"], label_max_chars), ha="center", va="center", fontsize=8, color="white", bbox=bbox)

        ax.set_ylim(placed_df["y_plot"].min() - 0.2, placed_df["y_plot"].max() + 0.2)
        ax.set_yticks([])
        ax.set_xlabel("Año")
        ax.set_title("Línea temporal alternante de publicaciones")
        plt.tight_layout()
        png_path = output_dir / "publications_timeline_alternating.png"
        fig.savefig(str(png_path), dpi=200)
        if show_plot:
            plt.show()
        plt.close(fig)
        return png_path, placed_df


if __name__ == "__main__":
    try:
        artifact, df = plot_publications_timeline_alternating(limit=1000, max_events=1000, show_plot=False)
        print("Artefacto generado:", artifact)
        print(df.head())
    except Exception as e:
        print("Error generando linea temporal alternante:", e)
