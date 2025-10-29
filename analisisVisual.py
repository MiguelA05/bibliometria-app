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

#Punto 3 Linea temporal de publicaciones por año y por revista

if __name__ == "__main__":
    # Pequeño demo cuando se ejecuta directamente
    limit = 500
    artifact, counts = plot_institution_countries_heatmap(limit=limit, show_plot=False)
    print(f"Visualización generada: {artifact}")
    print(counts.head(20).to_string(index=False))

    # Generar nubes de palabras para 'abstract' y 'keywords' y una combinada
    try:
        wc_results = generate_wordclouds_for_fields(top_n=15, show_plot=False, limit=limit)
        for key, info in wc_results.items():
            path = info.get("path") if isinstance(info, dict) else None
            if path:
                print(f"Nube generada ({key}): {path}")
    except Exception as exc:
        print(f"No se pudieron generar nubes de palabras: {exc}")
