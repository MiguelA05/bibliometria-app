#!/usr/bin/env python3
"""
Script para depurar los datos que llegan de OpenAlex
"""

import requests
import json

def debug_openalex_data():
    """Depurar datos de OpenAlex para entender el problema de títulos y abstracts"""
    
    print("🔍 DEPURANDO DATOS DE OPENALEX")
    print("=" * 50)
    
    # Hacer petición a OpenAlex
    url = "https://api.openalex.org/works?search=machine+learning&per_page=3"
    response = requests.get(url)
    data = response.json()
    
    print(f"📊 Total de resultados: {len(data['results'])}")
    print()
    
    for i, work in enumerate(data['results']):
        print(f"--- ARTÍCULO {i+1} ---")
        print(f"OpenAlex ID: {work.get('id')}")
        print(f"DOI: {work.get('doi')}")
        print(f"Título: {repr(work.get('title'))}")
        print(f"Display name: {repr(work.get('display_name'))}")
        print(f"Abstract directo: {repr(work.get('abstract'))}")
        
        # Verificar abstract invertido
        abstract_inverted = work.get('abstract_inverted_index')
        if abstract_inverted:
            print(f"Abstract invertido: {len(abstract_inverted)} palabras")
            # Mostrar primeras 5 palabras
            words = list(abstract_inverted.keys())[:5]
            print(f"Primeras palabras: {words}")
        else:
            print("Abstract invertido: No disponible")
        
        print(f"Tipo: {work.get('type')}")
        print(f"Año: {work.get('publication_year')}")
        primary_location = work.get('primary_location')
        if primary_location and primary_location.get('source'):
            print(f"Fuente: {primary_location['source'].get('display_name', 'N/A')}")
        else:
            print("Fuente: N/A")
        print()

if __name__ == "__main__":
    debug_openalex_data()
