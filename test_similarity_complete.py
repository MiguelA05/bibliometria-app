#!/usr/bin/env python3
"""
Script completo para probar el sistema de similitud textual.
Ejecuta todas las pruebas y muestra resultados esperados.
"""

import requests
import json
import pandas as pd
import os

BASE_URL = "http://127.0.0.1:8000"

def print_section(title, char="="):
    """Imprimir sección."""
    print(f"\n{char * 70}")
    print(f"{title}")
    print(f"{char * 70}\n")

def test_server_running():
    """Verificar que el servidor esté corriendo."""
    print_section("1. VERIFICANDO SERVIDOR")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Servidor corriendo correctamente")
            health = response.json()
            print(f"   Estado: {health.get('status')}")
            return True
        else:
            print(f"❌ Servidor respondió con error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Servidor no está corriendo")
        print("\n📝 Para iniciar el servidor:")
        print("   python start.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_list_csvs():
    """Probar listado de CSVs."""
    print_section("2. LISTANDO CSVs DISPONIBLES")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/text-similarity/csv-list", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            csvs = data.get('csvs', [])
            total = data.get('total', 0)
            
            print(f"✅ CSVs disponibles: {total}")
            
            if csvs:
                for csv_info in csvs[:3]:
                    print(f"\n   📄 {csv_info['filename']}")
                    print(f"      Tamaño: {csv_info['size_kb']:.1f} KB")
                    print(f"      Ruta: {csv_info['filepath']}")
                
                return csvs[0]['filepath']
            else:
                print("⚠️ No hay CSVs disponibles aún")
                print("   Ejecuta primero: python test_system.py")
                return None
        else:
            print(f"❌ Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_similarity_analysis(csv_path):
    """Probar análisis de similitud."""
    print_section("3. ANALIZANDO SIMILITUD TEXTUAL")
    
    # Leer CSV para obtener índices válidos
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        total_articles = len(df)
        
        print(f"📄 Archivo: {os.path.basename(csv_path)}")
        print(f"📊 Total artículos: {total_articles}")
        
        # Seleccionar primeros 3 artículos
        indices = [0, 1, 2] if total_articles >= 3 else list(range(min(total_articles, 3)))
        print(f"🔬 Comparando artículos: {indices}")
        
        # Preparar petición
        data = {
            "csv_file_path": csv_path,
            "article_indices": indices
        }
        
        print(f"\n📤 Enviando petición a /api/v1/text-similarity/analyze...")
        
        # Hacer petición (timeout largo por posible descarga de modelo)
        response = requests.post(
            f"{BASE_URL}/api/v1/text-similarity/analyze",
            json=data,
            timeout=180
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Análisis completado exitosamente\n")
            
            # Mostrar artículos analizados
            print("📄 ARTÍCULOS ANALIZADOS:")
            for article in result.get('articles', []):
                print(f"   [{article['index']}] {article['title']}")
            
            # Mostrar resultados
            print(f"\n🎯 RESULTADOS DE LOS 6 ALGORITMOS:")
            
            for i, res in enumerate(result.get('results', []), 1):
                print(f"\n   {i}. {res['algorithm']}")
                print(f"      Score: {res['score']:.3f}")
                print(f"      Tiempo: {res['time']:.3f}s")
                
                # Mostrar detalles clave según algoritmo
                details = res.get('details', {})
                
                if 'distance' in details:
                    print(f"      ⚡ Distancia: {details['distance']}")
                    print(f"      ⚡ Max length: {details.get('max_length', 'N/A')}")
                
                if 'transpositions_count' in details:
                    print(f"      🔄 Transposiciones: {details['transpositions_count']}")
                
                if 'intersection_size' in details and 'union_size' in details:
                    inter = details['intersection_size']
                    union = details['union_size']
                    print(f"      📊 Shingles: {inter}/{union} comunes")
                
                if 'top_contributing_terms' in details and details['top_contributing_terms']:
                    terms = [t['term'] for t in details['top_contributing_terms'][:3]]
                    print(f"      🔑 Top términos: {', '.join(terms)}")
                
                if 'interpretation' in details:
                    print(f"      💡 {details['interpretation']}")
            
            # Resumen
            summary = result.get('summary', {})
            print(f"\n📊 RESUMEN GENERAL:")
            print(f"   Algoritmos ejecutados: {summary.get('algorithms_used', 0)}")
            print(f"   Similitud promedio: {summary.get('avg_similarity', 0):.3f}")
            
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error durante análisis: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dependency_check():
    """Verificar dependencias instaladas."""
    print_section("0. VERIFICANDO DEPENDENCIAS")
    
    dependencies = {
        'sklearn': False,
        'nltk': False,
        'sentence_transformers': False,
        'pandas': True,  # Ya está instalado
        'numpy': True
    }
    
    try:
        import sklearn
        dependencies['sklearn'] = True
        print("✅ scikit-learn instalado")
    except:
        print("❌ scikit-learn NO instalado")
    
    try:
        import nltk
        dependencies['nltk'] = True
        print("✅ nltk instalado")
    except:
        print("❌ nltk NO instalado")
    
    try:
        import sentence_transformers
        dependencies['sentence_transformers'] = True
        print("✅ sentence-transformers instalado")
    except:
        print("⚠️ sentence-transformers NO instalado (algunos algoritmos no funcionarán)")
    
    if not all([dependencies['sklearn'], dependencies['nltk']]):
        print("\n📝 Instalar con:")
        print("   pip install scikit-learn nltk sentence-transformers")
        print("   python -m nltk.downloader punkt stopwords")
        return False
    
    return True

def main():
    """Función principal."""
    print("=" * 70)
    print("🧪 PRUEBA COMPLETA - SISTEMA DE SIMILITUD TEXTUAL")
    print("=" * 70)
    
    # Verificar dependencias
    deps_ok = test_dependency_check()
    
    if not deps_ok:
        print("\n⚠️ Falta instalar dependencias. ¿Continuar de todas formas? (s/n)")
        # Continuar por defecto
        print("   Continuando...\n")
    
    # Verificar servidor
    if not test_server_running():
        print("\n❌ EL SERVIDOR NO ESTÁ CORRIENDO")
        print("\n📝 Para iniciar el servidor:")
        print("   python start.py")
        print("\n💡 También puedes ejecutar:")
        print("   python test_system.py  # Para poblar datos primero")
        return
    
    # Listar CSVs
    csv_path = test_list_csvs()
    
    if not csv_path:
        print("\n⚠️ No hay CSVs para analizar")
        print("💡 Ejecuta primero: python test_system.py")
        return
    
    # Probar análisis
    success = test_similarity_analysis(csv_path)
    
    # Resumen final
    print_section("RESUMEN DE PRUEBAS", "=")
    
    if success:
        print("✅ TODAS LAS PRUEBAS EXITOSAS")
        print("\n🎉 El sistema de similitud textual está funcionando correctamente")
        print("\n📋 LO QUE DEBERÍAS VER:")
        print("   • 2 endpoints funcionando")
        print("   • 6 algoritmos ejecutándose")
        print("   • Score de similitud entre 0.0 y 1.0")
        print("   • Tiempos de procesamiento para cada algoritmo")
        print("   • Detalles específicos por algoritmo:")
        print("     - Levenshtein: distancia y operaciones")
        print("     - Damerau: transposiciones detectadas")
        print("     - Jaccard: shingles comunes")
        print("     - TF-IDF: top términos que contribuyen")
        print("     - Sentence-BERT: interpretación semántica")
        print("     - LLM: análisis conceptual")
    else:
        print("❌ Algunas pruebas fallaron")
        print("\n💡 Verifica:")
        print("   1. Servidor corriendo (python start.py)")
        print("   2. Dependencias instaladas (pip install ...)")
        print("   3. CSV unificado disponible")

if __name__ == "__main__":
    main()
