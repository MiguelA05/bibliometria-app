"""
Configuración de pytest para tests.
Excluye scripts standalone que deben ejecutarse directamente.
"""
import pytest
import os
from pathlib import Path

# Lista de archivos que son scripts standalone y NO deben ejecutarse como tests de pytest
STANDALONE_SCRIPTS = [
    'test_text_similarity.py',
    'test_text_similarity_service.py',
]

def pytest_ignore_collect(path, config):
    """Excluir scripts standalone de la colección de tests."""
    file_name = os.path.basename(str(path))
    if file_name in STANDALONE_SCRIPTS:
        # Estos archivos son scripts standalone con main(), no tests de pytest
        return True
    return False

