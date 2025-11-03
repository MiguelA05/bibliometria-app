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
from datetime import datetime

from resultsUtil import read_unified
from contadorPalabras import get_top_words_from_fields
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap


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

# Punto 4: Exportar los resultados en pdf


def export_images_to_pdf(image_paths: list[Path], output_pdf: Path, *, dpi: int = 300, margin_inch: float = 0.3, report_title: str | None = None, index_titles: Optional[Iterable[str]] = None) -> Path:
    """Combina una lista de imágenes en un único PDF de alta calidad.

    Diferencias frente a la versión anterior:
    - Permite escalar las imágenes hacia arriba para que ocupen el área imprimible
      de la página A4 (antes se evitaba el upscaling y quedaban grandes bordes blancos).
    - Añade una página índice con la lista de imágenes.

    image_paths: lista de Path a PNG/JPG existentes (se comprobará su existencia).
    output_pdf: Path de salida
    dpi: resolución usada para calcular tamaño A4 en píxeles
    margin_inch: margen en pulgadas (por defecto 0.5)
    """
    # Página A4 en pulgadas: 210 x 297 mm
    mm_to_in = 1.0 / 25.4
    page_w_in = 210 * mm_to_in
    page_h_in = 297 * mm_to_in
    page_w = int(page_w_in * dpi)
    page_h = int(page_h_in * dpi)
    margin = int(margin_inch * dpi)

    pages: list[Image.Image] = []

    # Página índice
    idx_page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(idx_page)
    # Definir tamaños de fuente en función del DPI para que escalen bien
    try:
        title_font_size = max(18, int(dpi * 0.5))
        body_font_size = max(12, int(dpi * 0.3))
        index_font_size = max(11, int(dpi * 0.3))
        font_title = ImageFont.truetype("arial.ttf", title_font_size)
        font_body = ImageFont.truetype("arial.ttf", body_font_size)
        font_index = ImageFont.truetype("arial.ttf", index_font_size)
    except Exception:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()
        font_index = ImageFont.load_default()

    # Helper para medir texto (compatible con varias versiones de Pillow)
    def _text_size(draw_obj, text, font_obj):
        try:
            bbox = draw_obj.textbbox((0, 0), text, font=font_obj)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])
        except Exception:
            try:
                return font_obj.getsize(text)
            except Exception:
                return (len(text) * 8, int(dpi * 0.04))

    title = "Reportes gráficos"
    # Preparar etiquetas que se mostrarán en el índice. Si el llamador
    # proporciona `index_titles`, se usan esas; si no, se generan
    # etiquetas por defecto del estilo "titulo 1", "titulo 2", ...
    if index_titles is None:
        index_labels = ["Mapa de calor","Nube de palabras abstract","Nube de palabras palabras clave","Nube de palabras combinada","Línea temporal publicaciones"]
    else:
        index_labels = [str(x) for x in index_titles]
        # Alinear longitud: completar con nombres de archivo si faltan,
        # o recortar si hay más etiquetas que imágenes.
        if len(index_labels) < len(image_paths):
            index_labels += [p.name for p in image_paths[len(index_labels):]]
        elif len(index_labels) > len(image_paths):
            index_labels = index_labels[: len(image_paths)]
    # Dibujar título del índice con efecto de negrita (doble pasada)
    draw.text((margin, margin // 2), title, fill="black", font=font_title)
    draw.text((margin + 1, (margin // 2) + 1), title, fill="black", font=font_title)
    y = margin + int(title_font_size * 1.1)
    for i, (p, label) in enumerate(zip(image_paths, index_labels), start=1):
        line = f"{i}. {label}"
        wrapped = textwrap.wrap(line, width=80)
        for wline in wrapped:
            w_w, w_h = _text_size(draw, wline, font_index)
            draw.text((margin, y), wline, fill="black", font=font_index)
            y += w_h + int(dpi * 0.01)
        y += int(dpi * 0.01)
        if y > page_h - margin:
            pages.append(idx_page)
            idx_page = Image.new("RGB", (page_w, page_h), "white")
            draw = ImageDraw.Draw(idx_page)
            y = margin

    pages.append(idx_page)

    imgs: list[Image.Image] = []
    for p in image_paths:
        if not p.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {p}")
        im = Image.open(p)
        if im.mode in ("RGBA", "LA") or im.mode == "P":
            alpha = im.convert("RGBA").split()[-1]
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im.convert("RGBA"), mask=alpha)
            im = bg
        else:
            im = im.convert("RGB")
        imgs.append(im)

    # Escalar cada imagen para que ocupe el área máxima disponible (permitir upscaling)
    # Iteramos las imágenes junto con las etiquetas/etiquetas del índice
    for idx, (p, im, label) in enumerate(zip(image_paths, imgs, index_labels)):
        iw, ih = im.size

        # Calcular altura reservada para título / subtítulo si se proporcionó
        title_block_h = 0
        caption_h = 0
        gap = int(dpi * 0.9)
        if report_title:
            try:
                title_block_h = font_title.getsize(report_title)[1] + gap
            except Exception:
                title_block_h = int(dpi * 0.08)
        # Añadir una pequeña línea con la etiqueta del índice como subtítulo
        try:
            caption_h = font_body.getsize(str(label))[1] + gap
        except Exception:
            caption_h = int(dpi * 0.05)

        top_reserved = margin + title_block_h + caption_h + gap
        max_w = page_w - 2 * margin
        max_h = page_h - top_reserved - margin
        # permitir escalar hacia arriba (no restringir a 1.0)
        scale = min(max_w / iw, max_h / ih)
        if not (scale and scale > 0):
            scale = 1.0
        new_w = max(1, int(iw * scale))
        new_h = max(1, int(ih * scale))
        im_resized = im.resize((new_w, new_h), resample=Image.LANCZOS)

        page = Image.new("RGB", (page_w, page_h), "white")

        # Dibujar título centrado en la parte superior si se proporcionó
        draw_page = ImageDraw.Draw(page)
        def _text_size(draw_obj, text, font_obj):
            try:
                bbox = draw_obj.textbbox((0, 0), text, font=font_obj)
                return (bbox[2] - bbox[0], bbox[3] - bbox[1])
            except Exception:
                try:
                    return font_obj.getsize(text)
                except Exception:
                    return (len(text) * 8, int(dpi * 0.04))

        if report_title:
            w_title, h_title = _text_size(draw_page, report_title, font_title)
            tx = (page_w - w_title) // 2
            ty = margin
            # Dibujar título en negrita (doble pasada)
            draw_page.text((tx, ty), report_title, fill="black", font=font_title)
            draw_page.text((tx + 1, ty + 1), report_title, fill="black", font=font_title)
            # Línea subrayada discreta debajo del título
            underline_y = ty + h_title + int(dpi * 0.11)
            draw_page.line((margin, underline_y, page_w - margin, underline_y), fill=(60, 60, 60), width=1)

        # Dibujar la etiqueta del índice como subtítulo (pequeño) centrado
        w_cap, h_cap = _text_size(draw_page, str(label), font_body)
        cx = (page_w - w_cap) // 2
        cy = margin + (title_block_h if report_title else 0) + gap // 2
        draw_page.text((cx, cy), str(label), fill="black", font=font_body)

        paste_x = (page_w - new_w) // 2
        paste_y = top_reserved + max(0, (max_h - new_h) // 2)
        page.paste(im_resized, (paste_x, paste_y))
        pages.append(page)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    first, rest = pages[0], pages[1:]
    first.save(str(output_pdf), "PDF", resolution=dpi, save_all=True, append_images=rest)
    return output_pdf


if __name__ == "__main__":
    # Ejecutar las tareas principales y recolectar errores
    errors_occurred = False
    error_messages: list[str] = []

    try:
        print("Generando mapa de países (limit=200)...")
        artifact, df_countries = plot_institution_countries_heatmap(limit=200, show_plot=False)
        print("Mapa guardado en:", artifact)
    except Exception as e:
        errors_occurred = True
        error_messages.append(f"Mapa de países: {e}")
        print("Error generando mapa de países:", e)

    try:
        print("Generando nubes de palabras (top_n=15, limit=200)...")
        wc_results = generate_wordclouds_for_fields(top_n=15, show_plot=False, limit=200)
        for k, v in wc_results.items():
            print(f"Nube '{k}' -> {v.get('path')}")
    except Exception as e:
        errors_occurred = True
        error_messages.append(f"Nubes de palabras: {e}")
        print("Error generando nubes de palabras:", e)

    try:
        artifact, pivot = plot_publications_by_year_source(limit=500, top_n_sources=10, show_plot=False)
        print("Artefacto generado:", artifact)
        print(pivot.head())
    except Exception as e:
        errors_occurred = True
        error_messages.append(f"Línea temporal por fuente: {e}")
        print("Error generando línea temporal:", e)

    

    # Si hubo errores, informar y no continuar con la exportación
    report_dir = Path("results") / "reports"
    expected_images = [
        report_dir / "institution_countries_choropleth.png",
        report_dir / "wordcloud_abstract.png",
        report_dir / "wordcloud_keywords.png",
        report_dir / "wordcloud_combined.png",
        report_dir / "publications_timeline_by_source.png",
    ]

    if errors_occurred:
        print("\nHubo errores al generar algunos artefactos. No se exportará el PDF combinado.")
        for msg in error_messages:
            print(" -", msg)
    else:
        # Verificar que los archivos existan
        missing = [str(p) for p in expected_images if not p.exists()]
        if missing:
            print("\nNo se encontraron todos los archivos esperados para exportar:\n", "\n".join(missing))
        else:
            default_pdf = report_dir / "combined_report.pdf"
            try:
                dest = input(f"Ruta de salida para el PDF (Enter para usar '{default_pdf}'): ").strip()
            except Exception:
                dest = ""
            out_pdf = Path(dest) if dest else default_pdf

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            title_input = f"Reporte ({timestamp})"
            report_title = title_input if title_input else None
            try:
                pdf_path = export_images_to_pdf(expected_images, out_pdf, report_title=report_title)
                print("PDF combinado generado en:", pdf_path)
            except Exception as e:
                print("Error exportando imágenes a PDF:", e)

