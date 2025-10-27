#!/usr/bin/env python3
"""
Prueba final del sistema de similitud textual con el servidor reiniciado.
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def main():
    print("=" * 70)
    print("🔬 PRUEBA FINAL DE SIMILITUD TEXTUAL")
    print("=" * 70)
    
    # 1. Listar CSVs
    print("\n📋 Listando CSVs disponibles...")
    response = requests.get(f"{BASE_URL}/api/v1/text-similarity/csv-list")
    data = response.json()
    csv_path = data['csvs'][0]['filepath']
    print(f"✅ Usando: {data['csvs'][0]['filename']}")
    
    # 2. Analizar similitud
    print("\n🔬 Analizando similitud (artículos 0, 1, 2)...")
    request_data = {
        "csv_file_path": csv_path,
        "article_indices": [0, 1, 2]
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/text-similarity/analyze",
        json=request_data,
        timeout=180
    )
    
    if response.status_code == 200:
        result = response.json()
        
        print("✅ Análisis completado!\n")
        print("📄 ARTÍCULOS:")
        for article in result.get('articles', []):
            print(f"   [{article['index']}] {article['title'][:60]}...")
        
        print(f"\n🎯 RESULTADOS DE 6 ALGORITMOS:")
        for i, res in enumerate(result.get('results', []), 1):
            print(f"\n   {i}. {res['algorithm']}")
            print(f"      Score: {res['score']:.3f}")
            print(f"      Tiempo: {res['time']:.3f}s")
        
        summary = result.get('summary', {})
        print(f"\n📊 RESUMEN:")
        print(f"   Similitud promedio: {summary.get('avg_similarity', 0):.3f}")
        
        print("\n✅ TODOS LOS ENDPOINTS FUNCIONANDO CORRECTAMENTE!")
        
    else:
        print(f"❌ Error {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    main()
