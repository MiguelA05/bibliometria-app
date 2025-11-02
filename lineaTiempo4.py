"""Timeline de estudios científicos estilo Gantt con Matplotlib.

Este módulo genera una línea de tiempo en formato barra horizontal por
estudio, lista para exportar a PDF y PNG en alta resolución. Se pensó
para ~30 estudios, con nombres largos que se dividen en varias líneas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

from resultsUtil import read_unified


@dataclass
class StudyRecord:
	name: str
	start: pd.Timestamp
	end: pd.Timestamp
	group: Optional[str] = None


def _wrap_label(text: str, *, width: int = 45) -> str:
	"""Divide textos largos en múltiples líneas para mejorar legibilidad."""

	if not text:
		return "(Sin título)"

	words = text.split()
	if not words:
		return text

	lines = []
	line = []
	for word in words:
		candidate = " ".join(line + [word]) if line else word
		if len(candidate) <= width:
			line.append(word)
		else:
			if line:
				lines.append(" ".join(line))
			line = [word]
	if line:
		lines.append(" ".join(line))
	return "\n".join(lines)


def _prepare_studies(
	*,
	name_field: str,
	start_field: str,
	end_field: Optional[str],
	duration_days: int,
	group_field: Optional[str],
	limit: Optional[int],
) -> list[StudyRecord]:
	rows = read_unified()
	if not rows or len(rows) < 2:
		raise FileNotFoundError("El CSV unificado está vacío o no existe.")

	header = rows[0]
	lowered = [h.lower() for h in header]

	def _resolve(field: str) -> int:
		if field in header:
			return header.index(field)
		if field.lower() in lowered:
			return lowered.index(field.lower())
		raise ValueError(f"Columna '{field}' no encontrada. Columnas disponibles: {header}")

	name_idx = _resolve(name_field)
	start_idx = _resolve(start_field)
	end_idx = None
	if end_field is not None:
		end_idx = _resolve(end_field)
	group_idx = None
	if group_field is not None:
		try:
			group_idx = _resolve(group_field)
		except ValueError:
			group_idx = None

	studies: list[StudyRecord] = []
	for i, row in enumerate(rows[1:]):
		if limit is not None and i >= limit:
			break
		name_raw = row[name_idx] if name_idx < len(row) else ""
		start_raw = row[start_idx] if start_idx < len(row) else ""
		end_raw = row[end_idx] if (end_idx is not None and end_idx < len(row)) else ""
		group_raw = row[group_idx] if (group_idx is not None and group_idx < len(row)) else None

		start = pd.to_datetime(start_raw, errors="coerce")
		end = pd.to_datetime(end_raw, errors="coerce") if end_idx is not None else pd.NaT
		if pd.isna(start):
			continue
		if pd.isna(end) or end < start:
			end = (start + pd.Timedelta(days=max(1, duration_days))).normalize()

		studies.append(
			StudyRecord(
				name=str(name_raw).strip(),
				start=start.normalize(),
				end=end.normalize(),
				group=str(group_raw).strip() if group_raw else None,
			)
		)

	if not studies:
		raise ValueError("No se encontraron estudios con fechas válidas.")

	# Orden descendente por fecha de inicio (más recientes arriba)
	studies.sort(key=lambda r: (r.start, r.end), reverse=True)
	return studies


def _build_color_palette(groups: list[str]) -> dict[str, Tuple[float, float, float, float]]:
	unique_groups = sorted(set(groups))
	if not unique_groups:
		return {}

	cmap = cm.get_cmap("tab20", len(unique_groups))
	return {group: cmap(i) for i, group in enumerate(unique_groups)}


def plot_scientific_studies_timeline(
	*,
	name_field: str = "title",
	start_field: str = "study_start_date",
	end_field: Optional[str] = "study_end_date",
	group_field: Optional[str] = None,
	limit: Optional[int] = None,
	output_dir: Path | str = Path("results") / "reports",
	pdf_name: str = "timeline.pdf",
	png_name: str = "timeline.png",
	title: str = "Línea de tiempo de estudios científicos (2010–2025)",
	duration_days: int = 90,
	dpi: int = 300,
	show_plot: bool = False,
) -> Tuple[Path, Path, pd.DataFrame]:
	"""Genera una línea de tiempo profesional y la exporta a PDF/PNG.

	Retorna (ruta_pdf, ruta_png, dataframe_usado).
	"""

	studies = _prepare_studies(
		name_field=name_field,
		start_field=start_field,
		end_field=end_field,
		duration_days=duration_days,
		group_field=group_field,
		limit=limit,
	)

	df = pd.DataFrame(
		{
			"name": [s.name for s in studies],
			"start": [s.start for s in studies],
			"end": [s.end for s in studies],
			"group": [s.group for s in studies],
		}
	)

	df["duration_days"] = (df["end"] - df["start"]).dt.days.clip(lower=0) + 1

	num_studies = len(df)
	base_height = 2.5
	height_per_study = 0.45
	fig_height = base_height + num_studies * height_per_study
	fig_width = 14 if num_studies <= 35 else min(22, 14 + (num_studies - 35) * 0.08)

	plt.style.use("seaborn-v0_8-whitegrid")
	fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

	y_positions = np.arange(num_studies)
	start_nums = mdates.date2num(df["start"].tolist())
	durations = (df["end"] - df["start"]).dt.days.clip(lower=0).tolist()
	durations = [max(1, d + 1) for d in durations]  # asegurar mínima longitud visible

	if group_field is not None:
		palette = _build_color_palette([g for g in df["group"] if g])
		colors = [palette.get(g, "#1f77b4") for g in df["group"]]
	else:
		colors = ["#1f77b4"] * num_studies

	bars = []
	for ypos, start_num, duration, color in zip(y_positions, start_nums, durations, colors):
		bars.append(((start_num, duration), (ypos - 0.4, 0.8), color))

	# Dibujar barras usando ax.broken_barh (compatible y eficiente)
	for seg, rect, color in bars:
		start, dur = seg
		y0, height = rect
		ax.broken_barh([(start, dur)], (y0, height), facecolors=color, edgecolor="black", linewidth=0.6)

	ax.set_ylim(-1, num_studies)
	ax.set_xlim(min(start_nums) - 10, max(start_nums + np.array(durations)) + 10)
	ax.set_yticks(y_positions)
	ax.set_yticklabels([_wrap_label(name) for name in df["name"]], fontsize=9)
	ax.invert_yaxis()  # más reciente arriba

	ax.xaxis.set_major_locator(mdates.YearLocator())
	ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
	for label in ax.get_xticklabels():
		label.set_rotation(35)
		label.set_horizontalalignment("right")
		label.set_fontsize(9)

	ax.set_xlabel("Fecha")
	ax.set_title(title, fontsize=14, weight="bold", pad=18)

	# Añadir leyenda si hay grupos
	if group_field is not None and any(df["group"].notna()):
		handles = []
		palette = _build_color_palette([g for g in df["group"] if g])
		for group, color in palette.items():
			patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.6)
			handles.append((patch, group or "Sin grupo"))
		if handles:
			patches, labels = zip(*handles)
			ax.legend(patches, labels, title="Grupo temático", loc="upper right", fontsize=9)

	# Ajustar grid ligero
	ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.6)
	ax.grid(axis="y", linestyle="", alpha=0)

	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	pdf_path = output_dir / pdf_name
	png_path = output_dir / png_name

	fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
	fig.savefig(png_path, dpi=dpi, bbox_inches="tight")

	if show_plot:
		plt.show()

	plt.close(fig)
	return pdf_path, png_path, df


if __name__ == "__main__":
	try:
		pdf, png, df = plot_scientific_studies_timeline(
			name_field="title",
			start_field="publication_date",
			end_field=None,
			group_field="study_group",
			limit=300,
			duration_days=120,
			show_plot=False,
		)
		print("Timeline exportada a:")
		print(" - PDF:", pdf)
		print(" - PNG:", png)
		print(df.head())
	except Exception as exc:
		print("No se pudo generar la línea de tiempo:", exc)

