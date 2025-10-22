#!/usr/bin/env python3
"""
Script para depurar la extracción de metadatos
"""

import requests
from app.services.openalex_service import OpenAlexService

def debug_extraction():
    """Depurar la extracción de metadatos"""
    
    print("🔍 DEPURANDO EXTRACCIÓN DE METADATOS")
    print("=" * 50)
    
    # Obtener datos de OpenAlex
    response = requests.get('https://api.openalex.org/works?search=machine+learning&per_page=1')
    data = response.json()
    work = data['results'][0]
    
    print("📊 Datos raw de OpenAlex:")
    print(f"  Título: {repr(work.get('title'))}")
    print(f"  Abstract: {repr(work.get('abstract'))}")
    print(f"  Abstract invertido: {len(work.get('abstract_inverted_index', {}))} palabras")
    print()
    
    # Procesar con mi función
    service = OpenAlexService()
    print("🔧 Procesando con mi función:")
    
    try:
        result = service._process_work(work)
        if result:
            print(f"  ✅ Título extraído: {repr(result.title)}")
            print(f"  ✅ Abstract extraído: {repr(result.abstract[:100])}...")
            print(f"  ✅ Autores: {result.authors}")
            print(f"  ✅ Afiliaciones: {result.affiliations}")
        else:
            print("  ❌ Error: No se pudo extraer metadatos")
    except Exception as e:
        print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    debug_extraction()
