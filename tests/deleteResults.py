"""
deleteResults.py

Script sencillo para borrar TODOS los archivos contenidos en la carpeta `results`
del repositorio. Pide confirmación por consola para evitar borrados accidentales.

Uso:
  python deleteResults.py          # pregunta confirmación interactiva
  python deleteResults.py --yes    # borra sin preguntar
  python deleteResults.py --dry-run # lista los archivos que se borrarían

Notas de seguridad:
- El script solo borra archivos dentro de la carpeta `results` en la raíz del
  repositorio (la carpeta donde está este script). No intenta borrar archivos
  fuera de esa carpeta.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List


def find_repo_root() -> Path:
	"""Devuelve la ruta del directorio donde está este script (raíz del repo).

	Esto se usa como referencia para asegurar que solo borramos dentro de
	`results` del repositorio.
	"""
	return Path(__file__).resolve().parent


def list_files_in_results(results_dir: Path) -> List[Path]:
	"""Lista todos los archivos (recursivamente) dentro de results_dir.

	No incluye directorios; solo ficheros.
	"""
	if not results_dir.exists():
		return []
	files: List[Path] = [p for p in results_dir.rglob("*") if p.is_file()]
	return files


def confirm_prompt(count: int, path: Path) -> bool:
	resp = input(f"¿Eliminar {count} archivos en '{path}'? [y/N]: ").strip().lower()
	return resp in ("y", "yes")


def safe_check_results_dir(results_dir: Path, repo_root: Path) -> None:
	"""Verifica que results_dir está dentro del repo_root y que su primer
	componente relativo es 'results'. Lanza SystemExit en caso de problema.
	"""
	try:
		rel = results_dir.resolve().relative_to(repo_root.resolve())
	except Exception:
		print("ERROR: La carpeta objetivo no está dentro del repositorio.")
		raise SystemExit(1)

	# Aseguramos que el primer componente del path relativo sea 'results'
	if len(rel.parts) == 0 or rel.parts[0] != "results":
		print("ERROR: Solo está permitido borrar archivos dentro de la carpeta 'results' del repositorio.")
		raise SystemExit(1)


def delete_files(files: List[Path]) -> tuple[int, List[tuple[Path, str]]]:
	"""Intenta borrar los archivos listados.

	Devuelve (borrados, errores) donde errores es lista de (Path, mensaje).
	"""
	deleted = 0
	errors: List[tuple[Path, str]] = []
	for f in files:
		try:
			f.unlink()
			deleted += 1
		except Exception as e:
			errors.append((f, str(e)))
	return deleted, errors


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Borra todos los archivos dentro de la carpeta results del repo.")
	p.add_argument("--yes", "-y", action="store_true", help="Confirmar y borrar sin preguntar")
	p.add_argument("--dry-run", action="store_true", help="Listar los archivos que se borrarían sin borrarlos")
	p.add_argument("--path", type=Path, default=None, help="(Opcional) ruta de la carpeta results. Por defecto se usa ./results del repo.")
	return p.parse_args()


def main() -> int:
	args = parse_args()
	repo_root = find_repo_root()
	results_dir = args.path if args.path is not None else repo_root / "results"

	# comprobación de seguridad: que esté dentro del repo y corresponda a results
	safe_check_results_dir(results_dir, repo_root)

	files = list_files_in_results(results_dir)
	if not files:
		print(f"No se encontraron archivos para borrar en: {results_dir}")
		return 0

	print(f"Se encontraron {len(files)} archivos en: {results_dir}")

	if args.dry_run:
		print("--dry-run activado. Estos archivos se borrarían:")
		for f in files:
			print(f" - {f}")
		return 0

	if not args.yes:
		ok = confirm_prompt(len(files), results_dir)
		if not ok:
			print("Operación cancelada por el usuario.")
			return 0

	deleted, errors = delete_files(files)
	print(f"Archivos borrados: {deleted}")
	if errors:
		print("Algunos archivos no pudieron borrarse:")
		for p, msg in errors:
			print(f" - {p}: {msg}")
		return 2

	print("Operación completada.")
	return 0


if __name__ == "__main__":
	try:
		raise SystemExit(main())
	except KeyboardInterrupt:
		print("Interrumpido por el usuario.")
		raise SystemExit(1)

