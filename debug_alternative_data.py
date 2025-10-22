#!/usr/bin/env python3
"""
Script para buscar información alternativa de autores
"""

import requests

def debug_alternative_data():
    """Buscar información alternativa de autores"""
    
    print("🔍 BUSCANDO INFORMACIÓN ALTERNATIVA DE AUTORES")
    print("=" * 60)
    
    # Obtener datos de OpenAlex
    response = requests.get('https://api.openalex.org/works?search=machine+learning&per_page=3')
    data = response.json()
    works = data.get('results', [])
    
    for i, work in enumerate(works):
        print(f"\n--- ARTÍCULO {i+1}: {work.get('title', 'N/A')[:50]}... ---")
        
        # Buscar información alternativa
        print(f"🔍 Campos disponibles relacionados con autores:")
        print(f"  - authorships: {len(work.get('authorships', []))}")
        print(f"  - corresponding_author_ids: {work.get('corresponding_author_ids', [])}")
        print(f"  - corresponding_institution_ids: {work.get('corresponding_institution_ids', [])}")
        
        # Buscar en primary_location
        primary_location = work.get('primary_location', {})
        if primary_location and primary_location.get('source'):
            print(f"  - primary_location.source: {primary_location['source'].get('display_name', 'N/A')}")
        else:
            print(f"  - primary_location.source: N/A")
        
        # Buscar en locations
        locations = work.get('locations', [])
        print(f"  - locations: {len(locations)}")
        for j, location in enumerate(locations[:2]):  # Solo primeros 2
            if location and location.get('source'):
                print(f"    Location {j+1}: {location['source'].get('display_name', 'N/A')}")
            else:
                print(f"    Location {j+1}: N/A")
        
        # Buscar en biblio
        biblio = work.get('biblio', {})
        if biblio:
            print(f"  - biblio: {biblio}")
        
        # Buscar en concepts (puede tener información de autores)
        concepts = work.get('concepts', [])
        print(f"  - concepts: {len(concepts)}")
        for j, concept in enumerate(concepts[:3]):  # Solo primeros 3
            print(f"    Concept {j+1}: {concept.get('display_name', 'N/A')}")

if __name__ == "__main__":
    debug_alternative_data()
