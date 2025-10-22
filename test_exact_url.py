#!/usr/bin/env python3
"""
Script para probar la URL exacta que usa el servicio
"""

import requests

def test_exact_url():
    """Probar la URL exacta que usa el servicio"""
    
    print("🔍 PROBANDO URL EXACTA DEL SERVICIO")
    print("=" * 50)
    
    # URL exacta que usa el servicio
    url = "https://api.openalex.org/works?search=machine+learning&per_page=3&sort=cited_by_count%3Adesc"
    
    print(f"🌐 URL: {url}")
    
    response = requests.get(url)
    data = response.json()
    works = data.get('results', [])
    
    print(f"📊 Total de resultados: {len(works)}")
    print()
    
    for i, work in enumerate(works):
        print(f"Artículo {i+1}:")
        print(f"  Título: {repr(work.get('title'))}")
        print(f"  Abstract: {repr(work.get('abstract'))}")
        abstract_inverted = work.get('abstract_inverted_index')
        if abstract_inverted:
            print(f"  Abstract invertido: {len(abstract_inverted)} palabras")
        else:
            print("  Abstract invertido: No disponible")
        print()

if __name__ == "__main__":
    test_exact_url()
