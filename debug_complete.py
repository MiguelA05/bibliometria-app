#!/usr/bin/env python3
"""
Script para depuración completa de la extracción de metadatos
"""

import requests
from app.services.openalex_service import OpenAlexService

def debug_complete():
    """Depuración completa de la extracción"""
    
    print("🔍 DEPURACIÓN COMPLETA DE EXTRACCIÓN DE METADATOS")
    print("=" * 60)
    
    # Obtener datos directamente de OpenAlex
    print("📊 Obteniendo datos directamente de OpenAlex...")
    response = requests.get('https://api.openalex.org/works?search=machine+learning&per_page=3')
    data = response.json()
    works = data.get('results', [])
    
    print(f"📄 Total de artículos obtenidos: {len(works)}")
    print()
    
    # Analizar cada artículo
    for i, work in enumerate(works):
        print(f"--- ANÁLISIS DEL ARTÍCULO {i+1} ---")
        print(f"OpenAlex ID: {work.get('id')}")
        print(f"Título: {repr(work.get('title'))}")
        print(f"DOI: {work.get('doi')}")
        
        # Analizar autores
        print(f"\n🔍 ANÁLISIS DE AUTORES:")
        authorships = work.get('authorships', [])
        print(f"  Authorships encontrados: {len(authorships)}")
        
        for j, authorship in enumerate(authorships):
            print(f"    Authorship {j+1}:")
            author = authorship.get('author', {})
            print(f"      Author: {author.get('display_name', 'N/A')}")
            print(f"      Author ID: {author.get('id', 'N/A')}")
            
            institutions = authorship.get('institutions', [])
            print(f"      Institutions: {len(institutions)}")
            for k, institution in enumerate(institutions):
                print(f"        Institution {k+1}: {institution.get('display_name', 'N/A')}")
                print(f"        Country: {institution.get('country_code', 'N/A')}")
                print(f"        City: {institution.get('city', 'N/A')}")
        
        # Procesar con mi servicio
        print(f"\n🔧 PROCESANDO CON MI SERVICIO:")
        service = OpenAlexService()
        try:
            article = service._process_work(work)
            if article:
                print(f"  ✅ Título: {repr(article.title)}")
                print(f"  ✅ Autores: {article.authors}")
                print(f"  ✅ Afiliaciones: {article.affiliations}")
                print(f"  ✅ Abstract: {repr(article.abstract[:100])}...")
                print(f"  ✅ Fecha: {article.publication_date}")
            else:
                print("  ❌ No se pudo procesar el artículo")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    debug_complete()

