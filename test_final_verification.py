#!/usr/bin/env python3
"""
Script de verificación final de la API
"""

import requests
import json

def test_final_verification():
    """Verificación final de la API"""
    
    print("🔍 VERIFICACIÓN FINAL DE LA API")
    print("=" * 50)
    
    # Probar la API
    url = "http://127.0.0.1:8000/api/v1/fetch-metadata"
    data = {
        "query": "machine learning",
        "max_articles": 3,
        "email": "test@example.com"
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ API respondió correctamente")
        print(f"📊 Total de artículos: {result['total_articles']}")
        
        # Analizar cada artículo
        articles = result['articles']
        for i, article in enumerate(articles):
            print(f"\n--- ARTÍCULO {i+1} ---")
            print(f"Título: {article['title']}")
            print(f"Autores: {article['authors']}")
            print(f"Afiliaciones: {article['affiliations']}")
            print(f"Abstract: {article['abstract'][:100]}...")
            print(f"Fecha: {article['publication_date']}")
            print(f"DOI: {article.get('doi', 'N/A')}")
            print(f"Citado por: {article.get('cited_by_count', 'N/A')} veces")
            
            # Verificar calidad de datos
            quality_issues = []
            if not article['title'] or article['title'] == 'Title not available':
                quality_issues.append("Título no disponible")
            if not article['abstract'] or article['abstract'] == 'Abstract not available':
                quality_issues.append("Abstract no disponible")
            if not article['publication_date'] or article['publication_date'] == 'Date not available':
                quality_issues.append("Fecha no disponible")
            
            if quality_issues:
                print(f"⚠️ Problemas de calidad: {', '.join(quality_issues)}")
            else:
                print(f"✅ Calidad de datos: Excelente")
        
        print(f"\n📊 RESUMEN:")
        print(f"   Total artículos procesados: {len(articles)}")
        print(f"   Artículos con títulos: {sum(1 for a in articles if a['title'] and a['title'] != 'Title not available')}")
        print(f"   Artículos con abstracts: {sum(1 for a in articles if a['abstract'] and a['abstract'] != 'Abstract not available')}")
        print(f"   Artículos con fechas: {sum(1 for a in articles if a['publication_date'] and a['publication_date'] != 'Date not available')}")
        print(f"   Artículos con autores: {sum(1 for a in articles if a['authors'])}")
        print(f"   Artículos con afiliaciones: {sum(1 for a in articles if a['affiliations'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_final_verification()

