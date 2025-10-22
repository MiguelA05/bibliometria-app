#!/usr/bin/env python3
"""
Script para depurar el flujo completo de la API
"""

import requests
from app.services.openalex_service import OpenAlexService

def debug_api_flow():
    """Depurar el flujo completo de la API"""
    
    print("🔍 DEPURANDO FLUJO COMPLETO DE LA API")
    print("=" * 50)
    
    # Primero verificar qué datos llegan directamente
    print("📊 Verificando datos directos de OpenAlex...")
    response = requests.get('https://api.openalex.org/works?search=machine+learning&per_page=3')
    data = response.json()
    works = data.get('results', [])
    
    for i, work in enumerate(works):
        print(f"  Artículo {i+1}:")
        print(f"    Título: {repr(work.get('title'))}")
        print(f"    Abstract: {repr(work.get('abstract'))}")
        print(f"    Abstract invertido: {len(work.get('abstract_inverted_index', {}))} palabras")
        print()
    
    # Simular exactamente lo que hace la API
    service = OpenAlexService()
    
    print("🔧 Llamando a search_works...")
    articles, csv_path = service.search_works('machine learning', 3)
    
    print(f"\n📊 Resultados:")
    print(f"  Artículos encontrados: {len(articles)}")
    print(f"  Archivo CSV: {csv_path}")
    
    if articles:
        print(f"\n📄 Detalles del primer artículo:")
        article = articles[0]
        print(f"  Título: {repr(article.title)}")
        print(f"  Abstract: {repr(article.abstract[:100])}...")
        print(f"  Autores: {article.authors}")
        print(f"  Afiliaciones: {article.affiliations}")
        print(f"  Fecha: {article.publication_date}")

if __name__ == "__main__":
    debug_api_flow()
