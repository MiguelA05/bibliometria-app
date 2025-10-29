("""Utilities para acceder al archivo `unified` dentro de la carpeta `results`.

Funciones añadidas:
- get_unified_path() -> pathlib.Path
- read_unified() -> list[list[str]]  (lee y parsea el CSV usando csv.reader)
- print_fields_per_line() -> None  (imprime cada fila como arreglo)

Por ahora la ruta del archivo está "quemada" (hardcoded). Más adelante
podremos cambiar la lógica de descubrimiento.
""")

from pathlib import Path
import csv
from typing import List, Union, Callable, Any


def get_unified_path() -> Path:
	"""Devuelve la ruta (Path) al fichero `unified` (hardcoded).

	Actualmente esta función devuelve una ruta fija relativa al root del repo.
	En el futuro se puede cambiar para buscar dinámicamente el último fichero.
	"""
	# Ruta quemada según el fichero que hay en el repo actualmente
	return Path("results") / "unified" / "unified_generative_ai_20251029_134723_unified.csv"


def read_unified() -> List[List[str]]:
	"""Lee y parsea el CSV 'unified' y devuelve una lista de filas.

	Cada fila es una lista de strings (campos). Usa la función
	`get_unified_path()` para localizar el fichero.

	Lanza FileNotFoundError si no existe el fichero.
	"""
	path = get_unified_path()
	if not path.exists():
		raise FileNotFoundError(f"Archivo unified no encontrado en: {path}")

	# Abrimos con newline '' para que csv module gestione correctamente las líneas
	with path.open("r", encoding="utf-8", newline="") as fh:
		reader = csv.reader(fh)
		rows = [row for row in reader]

	return rows


def iterate_fields_per_line(processor: Callable[..., Any] = None) -> None:
	"""Separa cada campo de cada línea y lo imprime como arreglo.

	Usa `read_unified()` para obtener las filas.
	Imprime el índice de fila y la lista de campos.
	"""
	def _default_printer(i: int, j: int, v: str) -> None:
		print(f"Linea {i} Campo {j}: {v}")

	try:
		rows = read_unified()
	except FileNotFoundError as exc:
		print(exc)
		return

	if not rows:
		print("El archivo está vacío o no contiene filas.")
		return

	# Procesador opcional: puede tener 1..3 argumentos
	def apply_processor(i: int, j: int, v: str, proc: Callable[..., Any] = None) -> None:
		if proc is None:
			_default_printer(i, j, v)
			return
		try:
			# try (i, j, v)
			proc(i, j, v)
		except TypeError:
			try:
				# try (j, v)
				proc(j, v)
			except TypeError:
				# try (v)
				proc(v)

	for i, row in enumerate(rows):
		for j, val in enumerate(row):
			apply_processor(i, j, val, processor)

"""
Ejemplos de uso de `print_field_records(field)`

Puedes probar la función importándola desde este módulo y pasando
el nombre de la columna (str) o el índice (int).

Ejemplos rápidos:

	from resultsUtil import print_field_records

	# Imprimir todos los títulos usando el nombre de la columna
	print_field_records('title')

	# También puedes pasar el índice de la columna (0 = primera columna)
	print_field_records(0)

	# Probar otras columnas por nombre
	print_field_records('authors')
	print_field_records('publication_year')
	print_field_records('doi')

Desde PowerShell (ejecución directa):

	python -c "from resultsUtil import print_field_records; print_field_records('title')"

Notas:
- La búsqueda por nombre es case-insensitive como fallback (intenta 'Title' también).
- Si la columna no existe se imprimirá un mensaje indicando las columnas disponibles.
"""

def iterate_field_records(field: Union[str, int], processor: Callable[..., Any] = None) -> None:
	"""Imprime (o procesa) los registros de una única columna.

	Parámetros:
	- field: nombre de la columna (str) o índice (int).
	- processor: callable opcional que se aplicará a cada valor de la columna.
	  El callable puede tener la forma `lambda value: ...` o `lambda index, value: ...`.
	  Si `processor` es None, se imprimirá cada valor con su índice.

	La función usa `read_unified()` para obtener las filas, interpreta la
	cabecera (quitando un BOM inicial si existe) y luego aplica `processor`
	a cada valor de la columna (saltando la fila de cabecera).
	"""
	try:
		rows = read_unified()
	except FileNotFoundError as exc:
		print(exc)
		return

	if not rows:
		print("El archivo está vacío o no contiene filas.")
		return

	header = rows[0]
	# Normalizar posible BOM en el primer campo
	header = [h.lstrip('\ufeff') if isinstance(h, str) else h for h in header]

	# Determinar índice de la columna
	if isinstance(field, int):
		idx = field
		if idx < 0 or idx >= len(header):
			print(f"Índice de columna fuera de rango: {idx}")
			return
		col_name = header[idx]
	else:
		# buscar por nombre (case-insensitive fallback)
		if field in header:
			idx = header.index(field)
			col_name = field
		else:
			lowered = [h.lower() for h in header]
			if field.lower() in lowered:
				idx = lowered.index(field.lower())
				col_name = header[idx]
			else:
				print(f"Columna '{field}' no encontrada. Columnas disponibles: {header}")
				return

	# Aplicar processor o imprimir por defecto
	for i, row in enumerate(rows[1:], start=1):
		val = row[idx] if idx < len(row) else ""
		if processor is None:
			print(f"{i}: {val}")
		else:
			# Intentamos pasar (index, value); si falla por aridad, pasamos solo value
			try:
				processor(i, val)
			except TypeError:
				processor(val)


# Nota: el demo anterior se eliminó para centralizar la ejecución en main().



def main():
	# Llamamos a print_fields_per_line pasando la lambda directamente.
	# La lambda puede aceptar (i, j, v), (j, v) o (v).
	iterate_fields_per_line(lambda i, j, v: print(f"Linea {i} Campo {j}: {v}"))


if __name__ == "__main__":
	main()