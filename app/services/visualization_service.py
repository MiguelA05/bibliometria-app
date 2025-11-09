"""
Servicio para análisis visual de producción científica.

Requerimiento 5: Genera visualizaciones incluyendo mapa de calor geográfico,
nubes de palabras, línea temporal de publicaciones y exportación a PDF.
"""

from __future__ import annotations

import json
import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Iterable
from datetime import datetime

import pandas as pd
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

from app.utils.logger import get_logger
from app.utils.csv_reader import read_unified_csv, resolve_column_index, normalize_header
from app.services.word_frequency_service import WordFrequencyService
from app.config import settings


@dataclass
class VisualizationResult:
    """Resultado de visualización."""
    heatmap_path: str
    wordcloud_paths: Dict[str, str]  # field -> path
    timeline_path: str
    pdf_path: Optional[str] = None


class VisualizationService:
    """
    Servicio para análisis visual de producción científica.
    
    Genera visualizaciones interactivas y estáticas: mapas de calor geográficos,
    nubes de palabras, líneas temporales y reportes PDF combinados.
    """
    
    def __init__(self):
        """Inicializar el servicio de visualización."""
        self.logger = get_logger("visualization")
        self.word_frequency_service = WordFrequencyService()
    
    def generate_all_visualizations(
        self,
        csv_path: Optional[str] = None,
        output_dir: Optional[Path] = None,
        limit: Optional[int] = None,
        export_pdf: bool = True
    ) -> VisualizationResult:
        """
        Generar todas las visualizaciones (Requerimiento 5).
        
        Genera mapa de calor geográfico, nubes de palabras, línea temporal
        y opcionalmente exporta todo a un PDF combinado.
        
        Args:
            csv_path: Ruta al CSV unificado (opcional)
            output_dir: Directorio de salida (opcional)
            limit: Límite de artículos a procesar (opcional)
            export_pdf: Si True, exporta todas las visualizaciones a PDF (default: True)
        
        Returns:
            VisualizationResult con rutas de todas las visualizaciones generadas
        """
        if output_dir is None:
            output_dir = Path(settings.results_dir) / "reports" / "visualizations"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Generando mapa de calor geográfico...")
        heatmap_path, _ = self.plot_geographic_heatmap(
            csv_path=csv_path,
            limit=limit,
            output_dir=output_dir
        )
        
        self.logger.info("Generando nubes de palabras...")
        wordcloud_paths = self.generate_wordclouds(
            csv_path=csv_path,
            limit=limit,
            output_dir=output_dir
        )
        
        self.logger.info("Generando línea temporal...")
        timeline_path, _ = self.plot_publications_timeline(
            csv_path=csv_path,
            limit=limit,
            output_dir=output_dir
        )
        
        pdf_path = None
        if export_pdf:
            self.logger.info("Exportando visualizaciones a PDF...")
            image_paths = [
                Path(heatmap_path),
                Path(wordcloud_paths.get("abstract", "")),
                Path(wordcloud_paths.get("keywords", "")),
                Path(wordcloud_paths.get("combined", "")),
                Path(timeline_path)
            ]
            existing_paths = [p for p in image_paths if p.exists()]
            if existing_paths:
                pdf_path = str(self.export_to_pdf(
                    existing_paths,
                    output_dir / "combined_report.pdf",
                    report_title="Análisis Visual de Producción Científica"
                ))
        
        return VisualizationResult(
            heatmap_path=str(heatmap_path),
            wordcloud_paths={k: str(v) for k, v in wordcloud_paths.items()},
            timeline_path=str(timeline_path),
            pdf_path=pdf_path
        )
    
    def plot_geographic_heatmap(
        self,
        csv_path: Optional[str] = None,
        field: str = "institution_countries",
        limit: Optional[int] = None,
        top_n: int = 100,
        output_dir: Optional[Path] = None,
        show_plot: bool = False
    ) -> Tuple[Path, pd.DataFrame]:
        """
        Generar mapa de calor geográfico (Requerimiento 5.1).
        
        Crea un mapa choropleth interactivo (plotly) o gráfico de barras (matplotlib)
        mostrando la distribución geográfica según países de instituciones.
        
        Nota: Se usa institution_countries en lugar del primer autor debido a limitaciones
        en la obtención de datos geográficos del autor durante el web scraping.
        
        Args:
            csv_path: Ruta al CSV unificado (opcional)
            field: Campo con países (default: "institution_countries")
            limit: Límite de filas a procesar (opcional)
            top_n: Top N países a mostrar (default: 100)
            output_dir: Directorio de salida (opcional)
            show_plot: Mostrar gráfico interactivamente (default: False)
        
        Returns:
            Tupla (ruta_archivo, DataFrame con conteos por país)
        
        Raises:
            FileNotFoundError: Si el CSV está vacío o no existe
            ImportError: Si matplotlib no está disponible (fallback)
        """
        if output_dir is None:
            output_dir = Path(settings.results_dir) / "reports" / "visualizations"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        rows = read_unified_csv(csv_path)
        if not rows or len(rows) < 2:
            raise FileNotFoundError("El CSV unificado está vacío o no existe.")
        
        header = normalize_header(rows[0])
        idx, _ = resolve_column_index(header, field)
        
        counter = Counter()
        rows_iter = rows[1:]
        if limit is not None:
            rows_iter = rows_iter[:limit]
        
        for row in rows_iter:
            cell = row[idx] if idx < len(row) else ""
            for country in self._parse_country_cell(cell):
                name = country.strip().rstrip(".")
                if name:
                    counter[name] += 1
        
        df = pd.DataFrame(counter.items(), columns=["country", "count"]).sort_values(
            "count", ascending=False
        )
        
        counts_csv = output_dir / "institution_countries_counts.csv"
        df.to_csv(counts_csv, index=False, encoding="utf-8")
        
        if PLOTLY_AVAILABLE:
            try:
                fig = px.choropleth(
                    df,
                    locations="country",
                    locationmode="country names",
                    color="count",
                    hover_name="country",
                    color_continuous_scale="YlOrRd",
                    title="Distribución geográfica por países de instituciones",
                )
                
                html_path = output_dir / "institution_countries_choropleth.html"
                fig.write_html(str(html_path))
                
                try:
                    import os
                    import sys
                    png_path = output_dir / "institution_countries_choropleth.png"
                    # Suprimir salida de kaleido/chromium
                    with open(os.devnull, 'w') as devnull:
                        old_stderr = sys.stderr
                        sys.stderr = devnull
                        try:
                            fig.write_image(str(png_path), engine="kaleido")
                        finally:
                            sys.stderr = old_stderr
                    main_artifact = png_path
                except Exception:
                    main_artifact = html_path
                
                if show_plot:
                    fig.show()
                
                return main_artifact, df
            except Exception as e:
                self.logger.warning(f"Error con plotly, usando fallback: {e}")
        
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib es requerido para visualizaciones")
        
        top_df = df.head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, max(4, 0.2 * len(top_df))))
        wrapped_names = [textwrap.fill(str(name), width=25) for name in top_df["country"]]
        ax.barh(wrapped_names, top_df["count"], color="#3478b6")
        ax.set_xlabel("Número de instituciones")
        ax.set_title("Distribución geográfica por países de instituciones")
        plt.tight_layout()
        
        png_path = output_dir / "institution_countries_bar.png"
        fig.savefig(str(png_path), dpi=200)
        if show_plot:
            plt.show()
        plt.close(fig)
        
        return png_path, df
    
    def generate_wordclouds(
        self,
        csv_path: Optional[str] = None,
        fields: Iterable[str] = ("abstract", "keywords"),
        top_n: int = 15,
        limit: Optional[int] = None,
        output_dir: Optional[Path] = None,
        show_plot: bool = False
    ) -> Dict[str, Path]:
        """
        Generar nubes de palabras dinámicas (Requerimiento 5.2).
        
        Genera nubes de palabras para campos específicos (abstracts, keywords)
        y una nube combinada usando las palabras más frecuentes.
        
        Args:
            csv_path: Ruta al CSV unificado (opcional)
            fields: Campos a analizar (default: ("abstract", "keywords"))
            top_n: Top N palabras a incluir (default: 15)
            limit: Límite de artículos a procesar (opcional)
            output_dir: Directorio de salida (opcional)
            show_plot: Mostrar gráficos interactivamente (default: False)
        
        Returns:
            Diccionario con rutas de nubes de palabras generadas por campo y "combined"
        
        Raises:
            ImportError: Si wordcloud no está disponible
        """
        if not WORDCLOUD_AVAILABLE:
            raise ImportError("wordcloud es requerido. Instala con: pip install wordcloud")
        
        if output_dir is None:
            output_dir = Path(settings.results_dir) / "reports" / "visualizations"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        combined_freq: Dict[str, int] = {}
        
        for field in fields:
            top = self.word_frequency_service.get_top_words_from_fields(
                field=field,
                top_n=top_n,
                csv_path=csv_path
            )
            freqs = {w: c for w, c in top}
            
            out_path = output_dir / f"wordcloud_{field}.png"
            self._make_wordcloud_from_frequencies(freqs, out_path, show=show_plot)
            results[field] = out_path
            
            for w, c in freqs.items():
                combined_freq[w] = combined_freq.get(w, 0) + c
        
        combined_path = output_dir / "wordcloud_combined.png"
        self._make_wordcloud_from_frequencies(combined_freq, combined_path, show=show_plot)
        results["combined"] = combined_path
        
        return results
    
    def plot_publications_timeline(
        self,
        csv_path: Optional[str] = None,
        date_field: str = "publication_date",
        source_field: str = "journal",
        limit: Optional[int] = None,
        top_n_sources: int = 8,
        output_dir: Optional[Path] = None,
        show_plot: bool = False
    ) -> Tuple[Path, pd.DataFrame]:
        """
        Generar línea temporal de publicaciones (Requerimiento 5.3).
        
        Crea una línea temporal mostrando el número de publicaciones por año
        y por fuente/revista, agrupando las fuentes menos frecuentes como "Other".
        
        Args:
            csv_path: Ruta al CSV unificado (opcional)
            date_field: Campo de fecha (default: "publication_date")
            source_field: Campo de fuente/revista (default: "journal")
            limit: Límite de artículos a procesar (opcional)
            top_n_sources: Top N fuentes a mostrar (default: 8)
            output_dir: Directorio de salida (opcional)
            show_plot: Mostrar gráfico interactivamente (default: False)
        
        Returns:
            Tupla (ruta_archivo, DataFrame pivot con conteos por año y fuente)
        
        Raises:
            FileNotFoundError: Si el CSV está vacío o no existe
            ImportError: Si matplotlib no está disponible (fallback)
        """
        if output_dir is None:
            output_dir = Path(settings.results_dir) / "reports" / "visualizations"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        rows = read_unified_csv(csv_path)
        if not rows or len(rows) < 2:
            raise FileNotFoundError("El CSV unificado está vacío o no existe.")
        
        header = normalize_header(rows[0])
        date_idx, _ = resolve_column_index(header, date_field)
        src_idx, _ = resolve_column_index(header, source_field)
        
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
        df["source_group"] = df["data_source"].where(
            df["data_source"].isin(top_sources), other="Other"
        )
        
        grouped = df.groupby(["year", "source_group"]).size().reset_index(name="count")
        pivot = grouped.pivot(index="year", columns="source_group", values="count").fillna(0).sort_index()
        
        if PLOTLY_AVAILABLE:
            try:
                df_line = pivot.reset_index()
                orig_cols = pivot.columns.tolist()
                wrapped_cols = [textwrap.fill(str(c), width=25) for c in orig_cols]
                rename_map = dict(zip(orig_cols, wrapped_cols))
                df_line = df_line.rename(columns=rename_map)
                fig = px.line(df_line, x="year", y=wrapped_cols, markers=True)
                fig.update_layout(
                    title="Publicaciones por año y fuente",
                    xaxis_title="Año",
                    yaxis_title="Publicaciones"
                )
                
                html_path = output_dir / "publications_timeline_by_source.html"
                fig.write_html(str(html_path))
                
                try:
                    import os
                    import sys
                    png_path = output_dir / "publications_timeline_by_source.png"
                    # Suprimir salida de kaleido/chromium
                    with open(os.devnull, 'w') as devnull:
                        old_stderr = sys.stderr
                        sys.stderr = devnull
                        try:
                            fig.write_image(str(png_path), engine="kaleido")
                        finally:
                            sys.stderr = old_stderr
                    main_artifact = png_path
                except Exception:
                    main_artifact = html_path
                
                if show_plot:
                    fig.show()
                
                counts_csv = output_dir / "publications_by_year_source_counts.csv"
                pivot.to_csv(counts_csv)
                
                return main_artifact, pivot
            except Exception as e:
                self.logger.warning(f"Error con plotly, usando fallback: {e}")
        
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib es requerido para visualizaciones")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for col in pivot.columns:
            label = textwrap.fill(str(col), width=25)
            ax.plot(pivot.index, pivot[col], marker="o", label=label)
        ax.set_xlabel("Año")
        ax.set_ylabel("Publicaciones")
        ax.set_title("Publicaciones por año y fuente")
        ax.legend(loc="best", fontsize="small")
        plt.tight_layout()
        
        png_path = output_dir / "publications_timeline_by_source.png"
        fig.savefig(str(png_path), dpi=200)
        if show_plot:
            plt.show()
        plt.close(fig)
        
        counts_csv = output_dir / "publications_by_year_source_counts.csv"
        pivot.to_csv(counts_csv)
        
        return png_path, pivot
    
    def export_to_pdf(
        self,
        image_paths: List[Path],
        output_pdf: Path,
        dpi: int = 300,
        margin_inch: float = 0.5,
        report_title: Optional[str] = None,
        index_titles: Optional[Iterable[str]] = None
    ) -> Path:
        """
        Exportar visualizaciones a PDF (Requerimiento 5.4).
        
        Combina múltiples imágenes en un PDF con índice, títulos y formato A4.
        Cada imagen se coloca en una página separada con título y caption.
        
        Args:
            image_paths: Lista de rutas a imágenes a incluir
            output_pdf: Ruta de salida del PDF
            dpi: Resolución de imágenes (default: 300)
            margin_inch: Margen en pulgadas (default: 0.5)
            report_title: Título del reporte (opcional)
            index_titles: Títulos personalizados para el índice (opcional)
        
        Returns:
            Path al PDF generado
        
        Raises:
            ImportError: Si Pillow no está disponible
        """
        if not PIL_AVAILABLE:
            raise ImportError("Pillow es requerido para exportar PDF. Instala con: pip install pillow")
        
        mm_to_in = 1.0 / 25.4
        page_w_in = 210 * mm_to_in
        page_h_in = 297 * mm_to_in
        page_w = int(page_w_in * dpi)
        page_h = int(page_h_in * dpi)
        margin = int(margin_inch * dpi)
        
        pages: List[Image.Image] = []
        
        idx_page = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(idx_page)
        
        def _get_font(size_pt: int, bold: bool = False):
            """Obtener fuente con tamaño adecuado."""
            size_px = int(size_pt * dpi / 72)
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "arial.ttf",
                "Arial.ttf"
            ]
            
            for font_path in font_paths:
                try:
                    return ImageFont.truetype(font_path, size_px)
                except Exception:
                    continue
            
            try:
                return ImageFont.load_default()
            except Exception:
                return ImageFont.load_default()
        
        main_title_size = 28
        subtitle_size = 20
        index_item_size = 16
        page_title_size = 24
        caption_size = 14
        
        font_main_title = _get_font(main_title_size, bold=True)
        font_subtitle = _get_font(subtitle_size)
        font_index = _get_font(index_item_size)
        font_page_title = _get_font(page_title_size, bold=True)
        font_caption = _get_font(caption_size)
        
        def _text_size(draw_obj, text, font_obj):
            try:
                bbox = draw_obj.textbbox((0, 0), text, font=font_obj)
                return (bbox[2] - bbox[0], bbox[3] - bbox[1])
            except Exception:
                try:
                    return font_obj.getsize(text)
                except Exception:
                    return (len(text) * 10, int(dpi * 0.06))
        
        title = report_title or "Análisis Visual de Producción Científica"
        subtitle = "Reporte de Visualizaciones - Requerimiento 5"
        
        if index_titles is None:
            index_labels = [
                "Mapa de Calor Geográfico",
                "Nube de Palabras - Abstracts",
                "Nube de Palabras - Keywords",
                "Nube de Palabras - Combinada",
                "Línea Temporal de Publicaciones"
            ]
        else:
            index_labels = [str(x) for x in index_titles]
            if len(index_labels) < len(image_paths):
                index_labels += [p.name for p in image_paths[len(index_labels):]]
            elif len(index_labels) > len(image_paths):
                index_labels = index_labels[:len(image_paths)]
        
        y_start = int(page_h * 0.12)
        
        # Ajustar título si es muy largo - basado en ancho real renderizado
        max_title_width = page_w - 2 * margin - int(dpi * 0.1)  # Margen extra de seguridad
        
        # Verificar ancho real del título
        w_main, h_main = _text_size(draw, title, font_main_title)
        
        if w_main > max_title_width:
            # Título muy largo: dividir en múltiples líneas
            # Calcular cuántos caracteres caben aproximadamente
            avg_char_width = w_main / len(title)
            chars_per_line = int(max_title_width / avg_char_width * 0.9)  # 90% para seguridad
            
            # Dividir el título en palabras y crear líneas
            words = title.split()
            lines = []
            current_line = []
            current_width = 0
            
            for word in words:
                word_with_space = (" " if current_line else "") + word
                test_line = "".join(current_line) + word_with_space
                test_w, _ = _text_size(draw, test_line, font_main_title)
                
                if test_w <= max_title_width:
                    current_line.append(word)
                    current_width = test_w
                else:
                    if current_line:
                        lines.append(" ".join(current_line))
                    current_line = [word]
                    current_width, _ = _text_size(draw, word, font_main_title)
            
            if current_line:
                lines.append(" ".join(current_line))
            
            # Dibujar cada línea del título
            for line in lines:
                w_line, h_line = _text_size(draw, line, font_main_title)
                tx_line = (page_w - w_line) // 2
                draw.text((tx_line, y_start), line, fill="black", font=font_main_title)
                y_start += h_line + int(dpi * 0.08)
        else:
            # Título cabe en una línea
            tx_main = (page_w - w_main) // 2
            draw.text((tx_main, y_start), title, fill="black", font=font_main_title)
            y_start += h_main + int(dpi * 0.12)
        
        # Subtítulo
        w_sub, h_sub = _text_size(draw, subtitle, font_subtitle)
        # Verificar si el subtítulo también necesita ajuste
        if w_sub > max_title_width:
            subtitle_words = subtitle.split()
            sub_lines = []
            current_sub_line = []
            for word in subtitle_words:
                test_sub_line = " ".join(current_sub_line + [word])
                test_w, _ = _text_size(draw, test_sub_line, font_subtitle)
                if test_w <= max_title_width:
                    current_sub_line.append(word)
                else:
                    if current_sub_line:
                        sub_lines.append(" ".join(current_sub_line))
                    current_sub_line = [word]
            if current_sub_line:
                sub_lines.append(" ".join(current_sub_line))
            
            for sub_line in sub_lines:
                w_sub_line, h_sub_line = _text_size(draw, sub_line, font_subtitle)
                tx_sub = (page_w - w_sub_line) // 2
                draw.text((tx_sub, y_start), sub_line, fill=(60, 60, 60), font=font_subtitle)
                y_start += h_sub_line + int(dpi * 0.05)
        else:
            tx_sub = (page_w - w_sub) // 2
            draw.text((tx_sub, y_start), subtitle, fill=(60, 60, 60), font=font_subtitle)
            y_start += h_sub + int(dpi * 0.12)
        
        # Línea separadora
        y_start += int(dpi * 0.08)
        draw.line((margin, y_start, page_w - margin, y_start), fill=(200, 200, 200), width=2)
        y_start += int(dpi * 0.15)
        
        # Título del índice
        index_title = "Índice de Contenidos"
        w_idx_title, h_idx_title = _text_size(draw, index_title, font_subtitle)
        draw.text((margin, y_start), index_title, fill="black", font=font_subtitle)
        y_start += h_idx_title + int(dpi * 0.12)
        
        # Calcular ancho máximo para ítems del índice
        max_index_width = page_w - 2 * margin - int(dpi * 0.2)
        
        for i, (p, label) in enumerate(zip(image_paths, index_labels), start=1):
            line = f"{i}. {label}"
            w_line, h_line = _text_size(draw, line, font_index)
            
            # Si la línea es muy larga, dividirla
            if w_line > max_index_width:
                # Dividir en palabras
                words = line.split()
                index_lines = []
                current_index_line = []
                for word in words:
                    test_index_line = " ".join(current_index_line + [word])
                    test_w, _ = _text_size(draw, test_index_line, font_index)
                    if test_w <= max_index_width:
                        current_index_line.append(word)
                    else:
                        if current_index_line:
                            index_lines.append(" ".join(current_index_line))
                        current_index_line = [word]
                if current_index_line:
                    index_lines.append(" ".join(current_index_line))
                
                # Dibujar cada línea
                for idx_line in index_lines:
                    w_idx_line, h_idx_line = _text_size(draw, idx_line, font_index)
                    draw.text((margin + int(dpi * 0.1), y_start), idx_line, fill="black", font=font_index)
                    y_start += h_idx_line + int(dpi * 0.04)
            else:
                draw.text((margin + int(dpi * 0.1), y_start), line, fill="black", font=font_index)
                y_start += h_line + int(dpi * 0.04)
            
            y_start += int(dpi * 0.06)  # Espacio entre ítems
            
            # Verificar si necesitamos nueva página
            if y_start > page_h - margin - int(dpi * 0.3):
                pages.append(idx_page)
                idx_page = Image.new("RGB", (page_w, page_h), "white")
                draw = ImageDraw.Draw(idx_page)
                y_start = margin + int(dpi * 0.2)
        
        pages.append(idx_page)
        
        imgs: List[Image.Image] = []
        for p in image_paths:
            if not p.exists():
                self.logger.warning(f"Archivo no encontrado: {p}")
                continue
            im = Image.open(p)
            if im.mode in ("RGBA", "LA") or im.mode == "P":
                alpha = im.convert("RGBA").split()[-1]
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im.convert("RGBA"), mask=alpha)
                im = bg
            else:
                im = im.convert("RGB")
            imgs.append(im)
        
        for idx, (p, im, label) in enumerate(zip(image_paths, imgs, index_labels)):
            iw, ih = im.size
            
            page = Image.new("RGB", (page_w, page_h), "white")
            draw_page = ImageDraw.Draw(page)
            
            y_pos = margin
            
            if report_title:
                w_title, h_title = _text_size(draw_page, report_title, font_page_title)
                tx = (page_w - w_title) // 2
                draw_page.text((tx, y_pos), report_title, fill="black", font=font_page_title)
                y_pos += h_title + int(dpi * 0.08)
                
                underline_y = y_pos
                draw_page.line((margin, underline_y, page_w - margin, underline_y), fill=(100, 100, 100), width=3)
                y_pos += int(dpi * 0.12)
            
            w_cap, h_cap = _text_size(draw_page, str(label), font_caption)
            cx = (page_w - w_cap) // 2
            draw_page.text((cx, y_pos), str(label), fill=(40, 40, 40), font=font_caption)
            y_pos += h_cap + int(dpi * 0.1)
            
            draw_page.line((margin, y_pos, page_w - margin, y_pos), fill=(220, 220, 220), width=1)
            y_pos += int(dpi * 0.1)
            
            top_reserved = y_pos
            max_w = page_w - 2 * margin
            max_h = page_h - top_reserved - margin
            scale = min(max_w / iw, max_h / ih)
            if not (scale and scale > 0):
                scale = 1.0
            new_w = max(1, int(iw * scale))
            new_h = max(1, int(ih * scale))
            im_resized = im.resize((new_w, new_h), resample=Image.LANCZOS)
            
            paste_x = (page_w - new_w) // 2
            paste_y = top_reserved + max(0, (max_h - new_h) // 2)
            page.paste(im_resized, (paste_x, paste_y))
            
            footer_text = f"Página {idx + 2} de {len(image_paths) + 1}"
            w_footer, h_footer = _text_size(draw_page, footer_text, font_caption)
            footer_x = (page_w - w_footer) // 2
            footer_y = page_h - margin - h_footer
            draw_page.text((footer_x, footer_y), footer_text, fill=(120, 120, 120), font=font_caption)
            
            pages.append(page)
        
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        first, rest = pages[0], pages[1:]
        first.save(str(output_pdf), "PDF", resolution=dpi, save_all=True, append_images=rest)
        
        return output_pdf
    
    def _parse_country_cell(self, value: str) -> List[str]:
        """Parsear celda con países (soporta JSON, listas separadas por comas/pipes)."""
        if not value:
            return []
        value = value.strip()
        
        if value.startswith("["):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if x]
            except Exception:
                pass
        
        parts = re.split(r"[;,|]\s*|\s+\|\s+", value)
        parts = [p.strip() for p in parts if p and p.strip()]
        if parts:
            return parts
        
        return [value]
    
    def _make_wordcloud_from_frequencies(
        self,
        freqs: Dict[str, int],
        output_path: Path,
        mask_path: Optional[Path] = None,
        show: bool = False
    ) -> Path:
        """Generar nube de palabras desde diccionario de frecuencias."""
        if not WORDCLOUD_AVAILABLE:
            raise ImportError("wordcloud es requerido. Instala con: pip install wordcloud")
        
        mask_array = None
        if mask_path is not None and mask_path.exists():
            try:
                mask_img = Image.open(mask_path).convert("L")
                mask_arr = np.array(mask_img)
                mask_array = (mask_arr > 0).astype(np.uint8) * 255
            except Exception:
                mask_array = None
        
        wc = WordCloud(width=1200, height=600, background_color="white", mask=mask_array)
        wc = wc.generate_from_frequencies(freqs)
        
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib es requerido para visualizaciones")
        
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

