"""
Utilidades para leer archivos CSV unificados.
Adaptado desde resultsUtil.py para integrarse con la estructura del proyecto.
"""

from pathlib import Path
import csv
from typing import List, Union, Callable, Any, Optional
from app.utils.logger import get_logger
from app.config import settings

logger = get_logger("csv_reader")


def get_unified_path(csv_path: Optional[str] = None) -> Path:
    """
    Obtener la ruta al archivo CSV unificado.
    
    Args:
        csv_path: Ruta específica al CSV (opcional). Si no se proporciona,
                  busca el más reciente en results/unified
    
    Returns:
        Path al archivo CSV unificado
    """
    if csv_path:
        path = Path(csv_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Archivo CSV no encontrado: {csv_path}")
    
    # Buscar el más reciente en results/unified
    unified_dir = Path(settings.results_dir) / "unified"
    if not unified_dir.exists() or not unified_dir.is_dir():
        raise FileNotFoundError(f"Directorio de unificados no encontrado: {unified_dir}")
    
    csv_files = list(unified_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos .csv en: {unified_dir}")
    
    # Elegir el archivo con la fecha de modificación más reciente
    latest = max(csv_files, key=lambda p: p.stat().st_mtime)
    return latest


def read_unified_csv(csv_path: Optional[str] = None) -> List[List[str]]:
    """
    Leer y parsear el CSV unificado.
    
    Args:
        csv_path: Ruta específica al CSV (opcional)
    
    Returns:
        Lista de filas, donde cada fila es una lista de strings (campos)
    """
    path = get_unified_path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo unified no encontrado en: {path}")
    
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh)
            rows = [row for row in reader]
        
        logger.info(f"CSV leído exitosamente: {len(rows)} filas desde {path}")
        return rows
    except Exception as e:
        logger.error(f"Error leyendo CSV: {e}")
        raise


def normalize_header(raw_header: List[str]) -> List[str]:
    """Normalizar cabecera removiendo BOM markers."""
    return [value.lstrip("\ufeff") if isinstance(value, str) else str(value) for value in raw_header]


def resolve_column_index(header: List[str], field: Union[str, int]) -> tuple[int, str]:
    """
    Resolver índice de columna por nombre o índice.
    
    Args:
        header: Lista de nombres de columnas
        field: Nombre de columna (str) o índice (int)
    
    Returns:
        Tupla (índice, nombre_columna)
    """
    lowered = [h.lower() for h in header]
    
    if isinstance(field, int):
        if field < 0 or field >= len(header):
            raise ValueError(f"Índice de columna fuera de rango: {field}")
        return field, header[field]
    
    if field in header:
        return header.index(field), field
    
    if field.lower() in lowered:
        index = lowered.index(field.lower())
        return index, header[index]
    
    raise ValueError(
        f"Columna '{field}' no encontrada. Columnas disponibles: {', '.join(header)}"
    )


def iterate_field_records(
    field: Union[str, int],
    processor: Callable[..., Any],
    csv_path: Optional[str] = None
) -> None:
    """
    Iterar sobre los registros de una columna específica.
    
    Args:
        field: Nombre de columna (str) o índice (int)
        processor: Función que procesa cada valor (puede recibir (index, value) o solo value)
        csv_path: Ruta específica al CSV (opcional)
    """
    try:
        rows = read_unified_csv(csv_path)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return
    
    if not rows:
        logger.warning("El archivo está vacío o no contiene filas.")
        return
    
    header = normalize_header(rows[0])
    
    # Resolver índice de columna
    try:
        idx, col_name = resolve_column_index(header, field)
    except ValueError as e:
        logger.error(str(e))
        return
    
    # Aplicar processor a cada valor
    for i, row in enumerate(rows[1:], start=1):
        val = row[idx] if idx < len(row) else ""
        try:
            processor(i, val)
        except TypeError:
            try:
                processor(val)
            except Exception as e:
                logger.warning(f"Error procesando valor en fila {i}: {e}")

