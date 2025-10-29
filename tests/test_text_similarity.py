#!/usr/bin/env python3
"""
Script unificado para probar el sistema de similitud textual.
Prueba los 6 algoritmos de similitud con abstracts de artículos.
"""

import requests
import json
import pandas as pd
import os
import time
from typing import List

BASE_URL = "http://127.0.0.1:8000"

def print_section(title, char="="):
    """Imprimir sección con formato."""
    print(f"\n{char * 70}")
    print(f"{title}")
    print(f"{char * 70}\n")

def test_dependency_check():
    """Verificar dependencias instaladas."""
    print_section("1. VERIFICANDO DEPENDENCIAS")
    
    dependencies = {
        'sklearn': False,
        'nltk': False,
        'sentence_transformers': False,
        'numpy': False,
    }
    
    try:
        import sklearn
        dependencies['sklearn'] = True
        print("[OK] scikit-learn instalado")
    except ImportError:
        print("[ERROR] scikit-learn NO instalado")
    
    try:
        import nltk
        dependencies['nltk'] = True
        print("[OK] nltk instalado")
    except ImportError:
        print("[ERROR] nltk NO instalado")
    
    try:
        import sentence_transformers
        dependencies['sentence_transformers'] = True
        print("[OK] sentence-transformers instalado")
    except ImportError:
        print("[WARNING] sentence-transformers NO instalado (algoritmos de embeddings no funcionarán)")
    
    try:
        import numpy
        dependencies['numpy'] = True
        print("[OK] numpy instalado")
    except ImportError:
        print("[ERROR] numpy NO instalado")
    
    all_critical = dependencies['sklearn'] and dependencies['nltk'] and dependencies['numpy']
    
    if not all_critical:
        print("\n[INFO] Instalar dependencias faltantes con:")
        print("   pip install scikit-learn nltk numpy sentence-transformers")
        print("   python -m nltk.downloader punkt stopwords")
        return False
    
    return True

def test_server_running():
    """Verificar que el servidor esté corriendo."""
    print_section("2. VERIFICANDO SERVIDOR")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("[OK] Servidor corriendo correctamente")
            health = response.json()
            print(f"   Estado: {health.get('status', 'OK')}")
            return True
        else:
            print(f"[ERROR] Servidor respondió con error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("[ERROR] Servidor no está corriendo")
        print("\n[INFO] Para iniciar el servidor:")
        print("   python start.py")
        return False
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False

def test_list_csvs():
    """Obtener lista de CSVs disponibles."""
    print_section("3. LISTANDO CSVs DISPONIBLES")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/text-similarity/csv-list", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            csvs = data.get('csvs', [])
            total = data.get('total', 0)
            
            print(f"[OK] CSVs disponibles: {total}")
            
            if csvs:
                for i, csv_info in enumerate(csvs[:3], 1):
                    print(f"\n   {i}. {csv_info['filename']}")
                    print(f"      Tamaño: {csv_info['size_kb']:.1f} KB")
                    print(f"      Ruta: {csv_info['filepath']}")
                
                return csvs[0]['filepath']
            else:
                print("[WARNING] No hay CSVs disponibles aún")
                print("   Ejecuta primero: python test_system.py")
                return None
        else:
            print(f"[ERROR] Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return None

def test_similarity_analysis(csv_path: str, article_indices: List[int] = None):
    """Probar análisis de similitud textual."""
    print_section("4. ANALIZANDO SIMILITUD TEXTUAL")
    
    try:
        # Leer CSV para obtener índices válidos si no se proporcionan
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        total_articles = len(df)
        
        print(f"Archivo: {os.path.basename(csv_path)}")
        print(f"Total artículos: {total_articles}")
        
        if article_indices is None:
            # Seleccionar primeros 3 artículos por defecto
            article_indices = [0, 1, 2] if total_articles >= 3 else list(range(min(total_articles, 3)))
        
        print(f"Comparando artículos: {article_indices}")
        
        # Preparar petición
        data = {
            "csv_file_path": csv_path,
            "article_indices": article_indices
        }
        
        print(f"\n[INFO] Enviando petición a /api/v1/text-similarity/analyze...")
        
        # Medir tiempo
        start_time = time.time()
        
        # Hacer petición con timeout largo
        response = requests.post(
            f"{BASE_URL}/api/v1/text-similarity/analyze",
            json=data,
            timeout=180
        )
        
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"[OK] Análisis completado en {elapsed_time:.2f}s\n")
            
            # Mostrar artículos analizados
            print("ARTÍCULOS ANALIZADOS:")
            for article in result.get('articles', []):
                title = article.get('title', 'Sin título')
                print(f"   [{article['index']}] {title[:70]}...")
            
            # Mostrar resultados de cada algoritmo
            print(f"\nRESULTADOS DE LOS 6 ALGORITMOS:")
            
            for i, res in enumerate(result.get('results', []), 1):
                print(f"\n   {i}. {res['algorithm']}")
                print(f"      Score: {res['score']:.3f}")
                print(f"      Tiempo: {res['time']:.3f}s")
                
                # Mostrar detalles específicos según algoritmo
                details = res.get('details', {})
                
                if 'distance' in details:
                    print(f"      Distancia de edición: {details['distance']}")
                
                if 'transpositions_count' in details:
                    print(f"      Transposiciones: {details['transpositions_count']}")
                
                if 'intersection_size' in details and 'union_size' in details:
                    inter = details['intersection_size']
                    union = details['union_size']
                    print(f"      Shingles comunes: {inter}/{union}")
                
                if 'top_contributing_terms' in details and details['top_contributing_terms']:
                    terms = details['top_contributing_terms'][:3]
                    if isinstance(terms[0], dict):
                        term_str = ', '.join([t['term'] for t in terms])
                    else:
                        term_str = ', '.join(terms)
                    print(f"      Top términos: {term_str}")
                
                if 'interpretation' in details:
                    print(f"      {details['interpretation']}")
            
            # Resumen general
            summary = result.get('summary', {})
            print(f"\nRESUMEN GENERAL:")
            print(f"   • Algoritmos ejecutados: {summary.get('algorithms_used', 0)}/6")
            print(f"   • Similitud promedio: {summary.get('avg_similarity', 0):.3f}")
            print(f"   • Tiempo total: {elapsed_time:.2f}s")
            
            return True
        else:
            print(f"[ERROR] Error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error durante análisis: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal."""
    print("=" * 70)
    print("SISTEMA DE ANÁLISIS DE SIMILITUD TEXTUAL")
    print("=" * 70)
    
    # 1. Verificar dependencias
    deps_ok = test_dependency_check()
    
    if not deps_ok:
        print("\n[WARNING] Algunas dependencias faltan. Ejecutando de todas formas...\n")
    
    # 2. Verificar servidor
    if not test_server_running():
        print("\n[ERROR] EL SERVIDOR NO ESTÁ CORRIENDO")
        print("\n[INFO] Para iniciar el servidor:")
        print("   python start.py")
        print("\n[INFO] Para poblar datos primero:")
        print("   python test_system.py")
        return
    
    # 3. Listar CSVs
    csv_path = test_list_csvs()
    
    if not csv_path:
        print("\n[WARNING] No hay CSVs para analizar")
        print("[INFO] Ejecuta primero: python test_system.py")
        return
    
    # 4. Probar análisis de similitud
    success = test_similarity_analysis(csv_path)
    
    # Resumen final
    print_section("5. RESUMEN FINAL", "=")
    
    if success:
        print("[OK] TODAS LAS PRUEBAS EXITOSAS")
        print("\nSistema de similitud textual funcionando correctamente")
        print("\nALGORITMOS EJECUTADOS:")
        print("   • Levenshtein (Edit Distance)")
        print("   • Damerau-Levenshtein (with Transposition)")
        print("   • Jaccard over n-grams (Shingling)")
        print("   • TF-IDF Cosine Similarity")
        print("   • Sentence-BERT Semantic Similarity")
        print("   • LLM-based Similarity (Simulated)")
    else:
        print("[ERROR] Algunas pruebas fallaron")
        print("\nVerifica:")
        print("   1. Servidor corriendo (python start.py)")
        print("   2. Dependencias instaladas")
        print("   3. CSV unificado disponible")

if __name__ == "__main__":
    main()
