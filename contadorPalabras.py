"""Pequeño helper para imprimir los valores de un campo del CSV `unified`.

Este archivo importa `print_field_records` desde `resultsUtil` y lo usa para
imprimir aquí (en la salida estándar) los valores de una columna.

Usa desde la línea de comandos:
    python palabrasAbstract.py

O importa y llama a `print_field_in_place` desde otro módulo.
"""

from typing import Optional, Union

from resultsUtil import iterate_field_records, get_unified_path, read_unified
from collections import Counter
import re

# Un conjunto simple de stopwords en inglés (puedes ampliarlo o pasar uno propio)
DEFAULT_STOPWORDS = {
    'the', 'and', 'of', 'in', 'to', 'a', 'is', 'for', 'on', 'that', 'with', 'as',
    'by', 'an', 'are', 'this', 'from', 'be', 'at', 'or', 'we', 'it', 'which',
    'can', 'has', 'have', 'these', 'their', 'our', 'was', 'were', 'will', 'such',
    'but', 'not', 'they', 'its', 'may', 'also', 'more', 'other', 'than'
}


def print_field_in_place(field: str, limit: Optional[int] = None) -> None:
    """Imprime los valores del campo `field` usando `print_field_records`.

    Parámetros:
    - field: nombre o índice (si se pasa como string se buscará por nombre).
    - limit: opcional, número máximo de registros a imprimir. Si es None imprime todos.
    """

    print(f"Imprimiendo campo '{field}' desde: {get_unified_path()}")

    if limit is None:
        # Usamos print_field_records directamente; imprime índice y valor
        iterate_field_records(field, lambda i, v: print(f"{i}: {v}"))
    else:
        # Si se pide un limit, contamos y dejamos de imprimir tras alcanzar el límite
        contador = {"n": 0}

        def processor(i, v):
            if contador["n"] >= limit:
                return
            print(f"{i}: {v}")
            contador["n"] += 1

        iterate_field_records(field, processor)


def get_top_words_from_fields(field: Union[str, int] = 'abstract',
                                 top_n: int = 15,
                                 stopwords: Optional[set] = None,
                                 min_word_length: int = 2) -> list[tuple[str, int]]:
    """Devuelve las top-N palabras más frecuentes en la columna indicada.

    Parámetros:
    - field: nombre de la columna (str) o índice (int). Por defecto 'abstract'.
    - top_n: número de palabras a devolver (por defecto 15).
    - stopwords: conjunto de palabras a ignorar. Si es None se usa DEFAULT_STOPWORDS.
    - min_word_length: longitud mínima para considerar una palabra.

    Retorna una lista de tuplas (palabra, cuenta) ordenada por frecuencia descendente.
    """
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS

    try:
        rows = read_unified()
    except FileNotFoundError as exc:
        print(exc)
        return []

    if not rows or len(rows) < 2:
        return []

    header = rows[0]
    lowered = [h.lower() for h in header]

    # Determinar índice de la columna según parámetro `field` (soporta int o str)
    if isinstance(field, int):
        idx = field
        if idx < 0 or idx >= len(header):
            print(f"Índice de columna fuera de rango: {idx}")
            return []
    else:
        # buscar por nombre (case-insensitive)
        if field in header:
            idx = header.index(field)
        elif field.lower() in lowered:
            idx = lowered.index(field.lower())
        else:
            print(f"Columna '{field}' no encontrada. Columnas disponibles: {header}")
            return []

    counter = Counter()
    word_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")

    for row in rows[1:]:
        text = row[idx] if idx < len(row) else ''
        if not text:
            continue
        for w in word_re.findall(text):
            w_lower = w.lower()
            if len(w_lower) < min_word_length:
                continue
            if w_lower in stopwords:
                continue
            counter[w_lower] += 1

    return counter.most_common(top_n)


if __name__ == "__main__":
    # Demo: imprime los títulos y las top-15 palabras en abstracts
    print('\nTop 15 palabras en abstracts:')
    top = get_top_words_from_fields(top_n=15)
    for word, cnt in top:
        print(f"{word}: {cnt}")
