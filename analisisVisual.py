"""Visualizaciones rápidas sobre el CSV unificado.

Contiene utilidades para generar un mapa de calor geográfico basado en la
columna `institution_countries`. El código intenta usar plotly para un
choropleth interactivo y cae en una visualización estática con matplotlib si
plotly no está disponible.

Uso recomendado (desde la raíz del repo):
    python -c "from analisisVisual import plot_institution_countries_heatmap; plot_institution_countries_heatmap(limit=200)"

El resultado se guarda en `results/reports/` (HTML y/o PNG) y se exporta un
CSV con los conteos por país.
"""

from __future__ import annotations

from collections import Counter
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from resultsUtil import read_unified
from contadorPalabras import get_top_words_from_fields
import numpy as np
from PIL import Image


#Punto 1 Mostrar un mapa de calor con la distribución geográfica
def _parse_country_cell(value: str) -> List[str]:
    """Intenta parsear diversos formatos comunes en `institution_countries`.

    Soporta:
    - JSON arrays: ["United States", "Spain"]
    - Listas separadas por coma, punto y coma o barra vertical
    - Texto único
    """
    if not value:
        return []
    value = value.strip()

    # Intentar JSON primero
    if value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if x]
        except Exception:
            pass

    # Separadores comunes
    parts = re.split(r"[;,|]\s*|\s+\|\s+", value)
    parts = [p.strip() for p in parts if p and p.strip()]
    if parts:
        return parts

    return [value]


def plot_institution_countries_heatmap(
    *,
    field: str = "institution_countries",
    limit: Optional[int] = None,
    top_n: int = 100,
    output_dir: Path | str = Path("results") / "reports",
    show_plot: bool = False,
) -> Tuple[Path, pd.DataFrame]:
    """Genera un mapa de calor por país (o una gráfica alternativa).

    Parámetros:
    - field: columna a usar (por defecto 'institution_countries').
    - limit: número máximo de filas a procesar (None = todas).
    - top_n: número máximo de países a mostrar en la alternativa estática.
    - output_dir: directorio donde guardar resultados.
    - show_plot: si True muestra la figura (requiere entorno gráfico).

    Retorna la ruta (Path) del archivo principal generado (HTML o PNG) y el
    DataFrame con conteos por país.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_unified()
    if not rows or len(rows) < 2:
        raise FileNotFoundError("El CSV unificado está vacío o no existe.")

    header = rows[0]
    lowered = [h.lower() for h in header]
    if field in header:
        idx = header.index(field)
    elif field.lower() in lowered:
        idx = lowered.index(field.lower())
    else:
        raise ValueError(f"Columna '{field}' no encontrada. Columnas: {header}")

    counter: Counter = Counter()
    rows_iter = rows[1:]
    if limit is not None:
        rows_iter = rows_iter[:limit]

    for row in rows_iter:
        cell = row[idx] if idx < len(row) else ""
        for country in _parse_country_cell(cell):
            name = country.strip().rstrip(".")
            if name:
                counter[name] += 1

    df = pd.DataFrame(counter.items(), columns=["country", "count"]).sort_values(
        "count", ascending=False
    )

    # Guardar conteos
    counts_csv = output_dir / "institution_countries_counts.csv"
    df.to_csv(counts_csv, index=False, encoding="utf-8")

    # Intentar mapa coroplético con plotly si está disponible
    try:
        import plotly.express as px

        fig = px.choropleth(
            df,
            locations="country",
            locationmode="country names",
            color="count",
            hover_name="country",
            color_continuous_scale="YlOrRd",
            title="Distribución de instituciones por país",
        )

        html_path = output_dir / "institution_countries_choropleth.html"
        fig.write_html(str(html_path))

        # Intentar guardar PNG si kaleido está instalado
        try:
            png_path = output_dir / "institution_countries_choropleth.png"
            fig.write_image(str(png_path), engine="kaleido")
            main_artifact = png_path
        except Exception:
            main_artifact = html_path

        if show_plot:
            fig.show()

        return main_artifact, df

    except Exception:
        # Fallback: gráfico de barras horizontales con matplotlib
        import matplotlib.pyplot as plt

        top_df = df.head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, max(4, 0.2 * len(top_df))))
        ax.barh(top_df["country"], top_df["count"], color="#3478b6")
        ax.set_xlabel("Número de instituciones")
        ax.set_title("Top países por instituciones (fallback, no choropleth)")
        plt.tight_layout()

        png_path = output_dir / "institution_countries_bar.png"
        fig.savefig(str(png_path), dpi=200)
        if show_plot:
            plt.show()
        plt.close(fig)
        return png_path, df



#Punto 2 Mostrar una nube de palabras
def _make_wordcloud_from_frequencies(freqs: dict[str, int], output_path: Path, *, mask_path: Path | None = None, show: bool = False) -> Path:
    """Genera y guarda una nube de palabras a partir de un dict {word: count}.

    Si se proporciona ``mask_path`` (ruta a imagen), la nube tomará la forma
    de la máscara (las zonas no en blanco serán la forma).

    Devuelve la ruta al PNG generado.
    """
    try:
        from wordcloud import WordCloud
    except Exception as exc:
        raise SystemExit("La librería 'wordcloud' no está instalada. Ejecuta 'pip install wordcloud'.") from exc

    import matplotlib.pyplot as plt

    mask_array = None
    if mask_path is not None:
        try:
            mask_img = Image.open(mask_path).convert("L")
            mask_arr = np.array(mask_img)
            # Normalizar a máscara binaria: 255 donde está la forma
            mask_array = (mask_arr > 0).astype(np.uint8) * 255
        except Exception:
            mask_array = None

    wc = WordCloud(width=1200, height=600, background_color="white", mask=mask_array)
    wc = wc.generate_from_frequencies(freqs)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def generate_wordclouds_for_fields(
    *,
    fields: Iterable[str] = ("abstract", "keywords"),
    top_n: int = 15,
    output_dir: Path | str = Path("results") / "reports",
    show_plot: bool = False,
    limit: Optional[int] = None,
    masks: dict | None = None,
    combined_mask: Path | None = None,
    mask: Path | None = None,
):
    """Genera nubes de palabras para cada campo en `fields` y una nube combinada.

    - Usa `get_top_words_from_fields` para obtener las top-N palabras por campo.
    - Genera PNGs: `wordcloud_<field>.png` y `wordcloud_combined.png`.
    - Devuelve un dict con rutas y los dataframes/tuplas usadas.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ruta de máscara por defecto (bundled)
    default_mask_path = Path("app") / "utils" / "resources" / "imagen.png"
    if masks is None:
        masks = {}

    # If the caller passed a single `mask`, apply it to all fields and combined cloud
    if mask is not None:
        mask_path = Path(mask)
        for f in fields:
            masks.setdefault(f, str(mask_path))
        if combined_mask is None:
            combined_mask = mask_path
    else:
        # If no explicit mask was provided, try the bundled default and apply it to all
        if default_mask_path.exists():
            for f in fields:
                masks.setdefault(f, str(default_mask_path))
            if combined_mask is None:
                combined_mask = default_mask_path

    results = {}

    combined_freq: dict[str, int] = {}

    for field in fields:
        top = get_top_words_from_fields(field=field, top_n=top_n)
        # top is list[tuple[word, count]]
        freqs = {w: c for w, c in top}

        # if limit is provided, the underlying reader isn't limited; but we reuse the function as-is
        out_path = output_dir / f"wordcloud_{field}.png"
        mask_path = None
        if masks and field in masks:
            mask_path = Path(masks[field])
        _make_wordcloud_from_frequencies(freqs, out_path, mask_path=mask_path, show=show_plot)
        results[field] = {"top": top, "path": out_path}

        for w, c in freqs.items():
            combined_freq[w] = combined_freq.get(w, 0) + c

    # nube combinada (suma de frecuencias)
    combined_path = output_dir / "wordcloud_combined.png"
    _make_wordcloud_from_frequencies(combined_freq, combined_path, mask_path=combined_mask, show=show_plot)
    results["combined"] = {"freqs": combined_freq, "path": combined_path}

    return results


# Punto 3: Línea temporal de publicaciones por año y por revista
def plot_publications_timeline(
    *,
    date_field: str = "publication_date",
    source_field: str = "data_source",
    limit: Optional[int] = None,
    top_n_sources: int = 8,
    output_dir: Path | str = Path("results") / "reports",
    show_plot: bool = False,
) -> Tuple[Path, pd.DataFrame]:
    """Genera una línea temporal (publicaciones por año) desglosada por fuente.

    Parámetros:
    - date_field: nombre de la columna con la fecha de publicación.
    - source_field: nombre de la columna con la fuente/revista (data_source por defecto).
    - limit: número máximo de filas a procesar (None = todas).
    - top_n_sources: cuántas fuentes mostrar individualmente (el resto se agrupa en 'Other').
    - output_dir: directorio donde guardar los artefactos.
    - show_plot: si True abre la figura (requiere entorno gráfico).

    Devuelve (ruta_al_artifact, dataframe_con_los_conteos_por_año_y_fuente).
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

    # Construir DataFrame mínimo
    records = []
    for i, row in enumerate(rows[1:]):
        if limit is not None and i >= limit:
            break
        date_raw = row[date_idx] if date_idx < len(row) else ""
        src_raw = row[src_idx] if src_idx < len(row) else ""
        records.append({"publication_date": date_raw, "data_source": src_raw})

    df = pd.DataFrame(records)
    # Normalizar fechas y extraer año
    df["year"] = pd.to_datetime(df["publication_date"], errors="coerce").dt.year
    # Filtrar filas sin año
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    # Normalizar fuente simple
    df["data_source"] = df["data_source"].fillna("Unknown").astype(str)

    # Seleccionar top N fuentes por volumen total
    top_sources = (
        df["data_source"].value_counts().nlargest(top_n_sources).index.tolist()
    )
    df["source_group"] = df["data_source"].where(df["data_source"].isin(top_sources), other="Other")

    # Agrupar por año y fuente
    grouped = df.groupby(["year", "source_group"]).size().reset_index(name="count")

    # Pivot para series (años x fuentes)
    pivot = grouped.pivot(index="year", columns="source_group", values="count").fillna(0).sort_index()

    # Intentar plotly para interactividad
    try:
        import plotly.express as px

        fig = px.line(pivot.reset_index(), x="year", y=pivot.columns.tolist(), markers=True)
        fig.update_layout(title="Publicaciones por año y fuente", xaxis_title="Año", yaxis_title="Publicaciones")
        html_path = output_dir / "publications_timeline.html"
        fig.write_html(str(html_path))

        # intentar PNG con kaleido
        try:
            png_path = output_dir / "publications_timeline.png"
            fig.write_image(str(png_path), engine="kaleido")
            main_artifact = png_path
        except Exception:
            main_artifact = html_path

        if show_plot:
            fig.show()

        return main_artifact, pivot

    except Exception:
        # Fallback matplotlib: múltiples líneas
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        for col in pivot.columns:
            ax.plot(pivot.index, pivot[col], marker="o", label=str(col))
        ax.set_xlabel("Año")
        ax.set_ylabel("Publicaciones")
        ax.set_title("Publicaciones por año y fuente")
        ax.legend(loc="best")
        plt.tight_layout()
        png_path = output_dir / "publications_timeline.png"
        fig.savefig(str(png_path), dpi=200)
        if show_plot:
            plt.show()
        plt.close(fig)
        return png_path, pivot


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
    """Genera una línea temporal estilo "event labels" similar a la imagen adjunta.

    Cada publicación se marca en el eje X por su año y se coloca una etiqueta
    tipo caja con el texto tomado de `label_field` (por defecto 'title'). Si
    `label_field` no existe en el CSV, se usa `source_field`.

    Devuelve la ruta al artefacto (HTML o PNG) y el DataFrame de eventos
    con columnas ['year','label','source'] ordenado por year.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_unified()
    if not rows or len(rows) < 2:
        raise FileNotFoundError("El CSV unificado está vacío o no existe.")

    header = rows[0]
    lowered = [h.lower() for h in header]

    # Resolver índices con tolerancia a mayúsculas
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

    # Agrupar por año para preparar posiciones verticales (stack)
    year_groups = df.groupby("year")
    placements = []
    for year, group in year_groups:
        texts = group["label"].tolist()
        sources = group["source"].tolist()
        for i, (t, s) in enumerate(zip(texts, sources)):
            placements.append({"year": year, "label": t, "source": s, "y_pos": i})

    placed_df = pd.DataFrame(placements)

    # Normalizar posiciones verticales para que no crezcan indefinidamente
    max_stack = placed_df["y_pos"].max() if not placed_df.empty else 0
    # Scale to a small visual range
    if max_stack > 0:
        placed_df["y_plot"] = placed_df["y_pos"] / (max_stack + 1) * 0.8 + 0.1
    else:
        placed_df["y_plot"] = 0.1

    # Crear figura con Plotly si está disponible
    try:
        import plotly.graph_objects as go

        years = placed_df["year"].astype(int)
        yvals = placed_df["y_plot"]

        fig = go.Figure()
        # Línea base
        fig.add_trace(go.Scatter(x=[years.min(), years.max()], y=[0, 0], mode="lines", line=dict(color="#888"), hoverinfo='skip'))

        # Marcadores invisibles para hover
        fig.add_trace(go.Scatter(x=years, y=yvals, mode="markers", marker=dict(size=6, color="#ff7f0e"), hoverinfo="text", text=placed_df["label"]))

        # Anotaciones con cajas
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

        html_path = output_dir / "publications_timeline_events.html"
        fig.write_html(str(html_path))

        try:
            png_path = output_dir / "publications_timeline_events.png"
            fig.write_image(str(png_path), engine="kaleido")
            main_artifact = png_path
        except Exception:
            main_artifact = html_path

        if show_plot:
            fig.show()

        return main_artifact, placed_df

    except Exception:
        # Fallback matplotlib: dibujar cajas de texto sobre una línea
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
        png_path = output_dir / "publications_timeline_events.png"
        fig.savefig(str(png_path), dpi=200)
        if show_plot:
            plt.show()
        plt.close(fig)
        return png_path, placed_df

#Punto 3 Linea temporal de publicaciones por año y por revista


if __name__ == "__main__":
    # Ejecución de ejemplo: generar artefactos rápidos (limit reducido para pruebas)
    try:
        print("Generando mapa de países (limit=200)...")
        artifact, df_countries = plot_institution_countries_heatmap(limit=200, show_plot=False)
        print("Mapa guardado en:", artifact)
    except Exception as e:
        print("Error generando mapa de países:", e)

    try:
        print("Generando nubes de palabras (top_n=15, limit=200)...")
        wc_results = generate_wordclouds_for_fields(top_n=15, show_plot=False, limit=200)
        for k, v in wc_results.items():
            print(f"Nube '{k}' -> {v.get('path')}")
    except Exception as e:
        print("Error generando nubes de palabras:", e)

    try:
        print("Generando línea temporal de publicaciones (events, limit=1000)...")
        timeline_artifact, pivot = plot_publications_timeline_events(limit=1000, max_events=800, show_plot=False)
        print("Línea temporal (events) guardada en:", timeline_artifact)
        print("Resumen (primeras filas):\n", pivot.head())
    except Exception as e:
        print("Error generando línea temporal:", e)
    # Demo reducido completado.
