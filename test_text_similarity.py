#!/usr/bin/env python3
"""
Script de prueba para el sistema de similitud textual.
Prueba los 6 algoritmos de similitud con abstracts de artículos.
"""

import requests
import json
import pandas as pd
import time

BASE_URL = "http://127.0.0.1:8000"

def get_available_csvs():
    """Obtener lista de CSVs unificados disponibles."""
    print("📋 Obteniendo lista de CSVs disponibles...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/text-similarity/csv-list", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            csvs = data.get('csvs', [])
            print(f"✅ Encontrados {len(csvs)} archivos CSV")
            
            for csv_info in csvs[:3]:  # Mostrar primeros 3
                print(f"   📄 {csv_info['filename']}")
                print(f"      Tamaño: {csv_info['size_kb']:.1f} KB")
            
            return csvs
        else:
            print(f"❌ Error: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def test_similarity_analysis(csv_path: str, article_indices: List[int]):
    """Probar análisis de similitud."""
    print(f"\n🔬 ANALIZANDO SIMILITUD TEXTUAL")
    print("=" * 60)
    
    print(f"📄 Archivo: {csv_path}")
    print(f"📊 Artículos a comparar: {article_indices}")
    
    data = {
        "csv_file_path": csv_path,
        "article_indices": article_indices
    }
    
    try:
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/api/v1/text-similarity/analyze",
            json=data,
            timeout=120
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"⏱️ Tiempo total: {elapsed:.2f}s\n")
            
            # Mostrar artículos analizados
            print("📄 ARTÍCULOS ANALIZADOS:")
            for article in result['articles']:
                print(f"   [{article['index']}] {article['title']}")
            
            # Mostrar resultados
            print(f"\n🎯 RESULTADOS DE SIMILITUD:")
            
            for i, res in enumerate(result['results'], 1):
                print(f"\n   {i}. {res['algorithm']}")
                print(f"      Score: {res['score']:.3f}")
                print(f"      Tiempo: {res['time']:.3f}s")
                
                # Mostrar detalles según algoritmo
                details = res.get('details', {})
                
                if 'distance' in details:
                    print(f"      Distancia: {details['distance']}")
                if 'transpositions_count' in details:
                    print(f"      Transposiciones: {details['transpositions_count']}")
                if 'intersection_size' in details:
                    print(f"      Shingles comunes: {details['intersection_size']}")
                if 'top_contributing_terms' in details and details['top_contributing_terms']:
                    terms = [t['term'] for t in details['top_contributing_terms'][:5]]
                    print(f"      Top términos: {', '.join(terms)}")
                if 'interpretation' in details:
                    print(f"      Interpretación: {details['interpretation']}")
            
            # Resumen
            summary = result.get('summary', {})
            print(f"\n📊 RESUMEN:")
            print(f"   Algoritmos usados: {summary.get('algorithms_used', 0)}")
            print(f"   Similitud promedio: {summary.get('avg_similarity', 0):.3f}")
            
            return True
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Función principal."""
    print("=" * 70)
    print("🔬 SISTEMA DE ANÁLISIS DE SIMILITUD TEXTUAL")
    print("=" * 70)
    
    # Obtener CSVs disponibles
    csvs = get_available_csvs()
    
    if not csvs:
        print("\n❌ No hay CSVs disponibles")
        print("   Ejecuta primero el sistema de automatización")
        return
    
    # Usar el CSV más reciente
    latest_csv = csvs[0]
    csv_path = latest_csv['filepath']
    
    print(f"\n✅ Usando CSV: {latest_csv['filename']}")
    
    # Leer CSV para obtener índices
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"   Total artículos: {len(df)}")
        
        # Seleccionar primeros 3 artículos
        indices = [0, 1, 2] if len(df) >= 3 else list(range(len(df)))
        
        print(f"   Comparando artículos: {indices}")
        
        # Ejecutar análisis
        success = test_similarity_analysis(csv_path, indices)
        
        if success:
            print("\n" + "=" * 70)
            print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
            print("=" * 70)
            print("\n📊 Algoritmos ejecutados:")
            print("   1. Levenshtein (Edit Distance)")
            print("   2. Damerau-Levenshtein (with Transposition)")
            print("   3. Jaccard over n-grams")
            print("   4. TF-IDF Cosine Similarity")
            print("   5. Sentence-BERT Semantic Similarity")
            print("   6. LLM-based Similarity (Simulated)")
        else:
            print("\n❌ El análisis falló")
            
    except Exception as e:
        print(f"❌ Error leyendo CSV: {e}")

if __name__ == "__main__":
    main()
