#!/usr/bin/env python3
"""
Script simple para probar el sistema de similitud textual.
Usa el CSV unificado existente.
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def main():
    print("=" * 70)
    print("🔬 PRUEBA DE SIMILITUD TEXTUAL")
    print("=" * 70)
    
    # 1. Listar CSVs disponibles
    print("\n1. Listando CSVs disponibles...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/text-similarity/csv-list", timeout=10)
        if response.status_code == 200:
            data = response.json()
            csvs = data.get('csvs', [])
            if csvs:
                csv_path = csvs[0]['filepath']
                print(f"✅ Usando: {csvs[0]['filename']}")
            else:
                print("❌ No hay CSVs disponibles")
                return
        else:
            print(f"❌ Error: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # 2. Analizar similitud
    print("\n2. Analizando similitud entre artículos 0, 1 y 2...")
    data = {
        "csv_file_path": csv_path,
        "article_indices": [0, 1, 2]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/text-similarity/analyze",
            json=data,
            timeout=180
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ Análisis completado!\n")
            
            # Mostrar artículos
            print("📄 ARTÍCULOS:")
            for article in result.get('articles', []):
                print(f"   [{article['index']}] {article['title'][:60]}...")
            
            # Mostrar resultados
            print("\n🎯 RESULTADOS (6 algoritmos):")
            for i, res in enumerate(result.get('results', []), 1):
                print(f"\n   {i}. {res['algorithm']}")
                print(f"      Score: {res['score']:.3f}")
                print(f"      Tiempo: {res['time']:.3f}s")
            
            # Resumen
            summary = result.get('summary', {})
            print(f"\n📊 RESUMEN:")
            print(f"   Similitud promedio: {summary.get('avg_similarity', 0):.3f}")
            
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
