#!/usr/bin/env python3
"""
Script de prueba para el sistema de automatización de descarga y unificación de datos.
Prueba el endpoint de automatización que descarga de múltiples fuentes y elimina duplicados.
"""

import argparse
import requests
import json
import time
import os
import sys
from pathlib import Path

def test_automation_endpoint(base_url: str, results_dir: str, timeout: int = 120) -> bool:
    """Probar el endpoint de automatización."""
    print("🤖 PRUEBA SISTEMA DE AUTOMATIZACIÓN")
    print("=" * 60)
    
    try:
        url = f"{base_url.rstrip('/')}/api/v1/automation/unified-data"
        data = {
            "base_query": "generative artificial intelligence",
            "similarity_threshold": 0.8,
            "max_articles_per_source": 30
        }
        
        print(f"🔍 Enviando petición de automatización...")
        print(f"🌐 URL: {url}")
        print(f"📊 Datos: {json.dumps(data, indent=2)}")
        
        # Medir tiempo de respuesta
        start_time = time.time()
        session = requests.Session()
        response = session.post(url, json=data, timeout=timeout)  # Timeout configurable
        end_time = time.time()

        print(f"⏱️ Tiempo de respuesta: {end_time - start_time:.2f} segundos")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Sistema de automatización respondió correctamente!")
            
            # Mostrar información del proceso
            automation_result = result.get('automation_result', {})
            print(f"\n🤖 RESULTADO DE AUTOMATIZACIÓN:")
            print(f"   Proceso: {automation_result.get('process_type')}")
            print(f"   Consulta base: {automation_result.get('base_query')}")
            print(f"   Umbral de similitud: {automation_result.get('similarity_threshold')}")
            
            # Mostrar estadísticas de datos
            data_stats = result.get('data_statistics', {})
            print(f"\n📊 ESTADÍSTICAS DE DATOS:")
            print(f"   Total artículos descargados: {data_stats.get('total_articles_downloaded')}")
            print(f"   Artículos únicos: {data_stats.get('unique_articles')}")
            print(f"   Duplicados eliminados: {data_stats.get('duplicates_removed')}")
            print(f"   Fuentes procesadas: {data_stats.get('sources_processed')}")
            print(f"   Tasa de duplicación: {data_stats.get('duplication_rate')}")
            
            # Mostrar archivos generados
            generated_files = result.get('generated_files', {})
            print(f"\n📁 ARCHIVOS GENERADOS:")
            print(f"   Archivo unificado: {generated_files.get('unified_file')}")
            print(f"   Archivo de duplicados: {generated_files.get('duplicates_file')}")
            print(f"   Tamaño archivo unificado: {generated_files.get('unified_file_size')}")
            print(f"   Tamaño archivo duplicados: {generated_files.get('duplicates_file_size')}")
            
            # Mostrar rendimiento
            performance = result.get('performance', {})
            print(f"\n⚡ RENDIMIENTO:")
            print(f"   Tiempo de procesamiento: {performance.get('processing_time_seconds')} segundos")
            print(f"   Artículos por segundo: {performance.get('articles_per_second')}")
            
            # Verificar archivos generados (convertir rutas relativas a absolutas respecto a project root)
            unified_file = generated_files.get('unified_file')
            duplicates_file = generated_files.get('duplicates_file')

            # Normalizar paths y buscar en results_dir si no se entregó ruta absoluta
            def _resolve_file(path_val: str) -> str:
                if not path_val:
                    return ""
                p = Path(path_val)
                if p.is_absolute():
                    return str(p)
                # Buscar en results_dir
                candidate = Path(results_dir) / p
                if candidate.exists():
                    return str(candidate)
                # fallback: return original string
                return str(p)

            unified_file = _resolve_file(unified_file)
            duplicates_file = _resolve_file(duplicates_file)
            
            if unified_file and os.path.exists(unified_file):
                print(f"\n✅ Archivo unificado encontrado: {unified_file}")
                
                # Verificar contenido del archivo unificado
                try:
                    import pandas as pd
                    df_unified = pd.read_csv(unified_file, encoding='utf-8-sig')
                    print(f"📊 Contenido del archivo unificado:")
                    print(f"   Filas: {len(df_unified)}")
                    print(f"   Columnas: {list(df_unified.columns)}")
                    
                    # Mostrar distribución por fuente
                    if 'data_source' in df_unified.columns:
                        source_counts = df_unified['data_source'].value_counts()
                        print(f"   Distribución por fuente:")
                        for source, count in source_counts.items():
                            print(f"     {source}: {count}")
                    
                    # Mostrar distribución por tipo
                    if 'type' in df_unified.columns:
                        type_counts = df_unified['type'].value_counts()
                        print(f"   Distribución por tipo:")
                        for type_name, count in type_counts.items():
                            print(f"     {type_name}: {count}")
                    
                except Exception as e:
                    print(f"❌ Error al leer archivo unificado: {e}")
            else:
                print(f"❌ Archivo unificado no encontrado: {unified_file}")
            
            if duplicates_file and os.path.exists(duplicates_file):
                print(f"\n✅ Archivo de duplicados encontrado: {duplicates_file}")
                
                # Verificar contenido del archivo de duplicados
                try:
                    import pandas as pd
                    df_duplicates = pd.read_csv(duplicates_file, encoding='utf-8-sig')
                    print(f"📊 Contenido del archivo de duplicados:")
                    print(f"   Filas: {len(df_duplicates)}")
                    print(f"   Columnas: {list(df_duplicates.columns)}")
                    
                    if len(df_duplicates) > 0:
                        # Mostrar algunos ejemplos de duplicados
                        print(f"   Ejemplos de duplicados eliminados:")
                        for i, row in df_duplicates.head(3).iterrows():
                            print(f"     Duplicado {i+1}:")
                            print(f"       Título original: {row.get('original_title', 'N/A')[:60]}...")
                            print(f"       Título duplicado: {row.get('duplicate_title', 'N/A')[:60]}...")
                            print(f"       Similitud: {row.get('similarity_score', 'N/A')}")
                            print(f"       Fuente duplicado: {row.get('duplicate_source', 'N/A')}")
                    
                except Exception as e:
                    print(f"❌ Error al leer archivo de duplicados: {e}")
            else:
                print(f"❌ Archivo de duplicados no encontrado: {duplicates_file}")
            
            return True
            
        else:
            print(f"❌ Error en el sistema de automatización: {response.status_code}")
            print(f"   Respuesta: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error de conexión: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def test_different_thresholds(base_url: str, timeout: int = 60):
    """Probar diferentes umbrales de similitud."""
    print("\n🔍 PROBANDO DIFERENTES UMBRALES DE SIMILITUD")
    print("=" * 50)
    
    thresholds = [0.6, 0.7, 0.8, 0.9]
    url = f"{base_url.rstrip('/')}/api/v1/automation/unified-data"
    
    for threshold in thresholds:
        try:
            data = {
                "base_query": "generative artificial intelligence",
                "similarity_threshold": threshold,
                "max_articles_per_source": 20
            }
            
            print(f"🎯 Probando umbral: {threshold}")
            response = requests.post(url, json=data, timeout=timeout)
            
            if response.status_code == 200:
                result = response.json()
                data_stats = result.get('data_statistics', {})
                print(f"   ✅ Artículos únicos: {data_stats.get('unique_articles')}")
                print(f"   ✅ Duplicados eliminados: {data_stats.get('duplicates_removed')}")
                print(f"   ✅ Tasa de duplicación: {data_stats.get('duplication_rate')}")
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Script de prueba de automatización")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL de la API")
    parser.add_argument("--results-dir", default="results", help="Directorio donde se esperan/guardan los CSV")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout en segundos para las peticiones")
    args = parser.parse_args()

    print("🤖 PRUEBA COMPLETA - SISTEMA DE AUTOMATIZACIÓN")
    print("Descarga Multi-fuente + Eliminación de Duplicados + Archivos Unificados")
    print("=" * 70)

    # Asegurar directorio de resultados
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # Probar endpoint principal
    main_test_ok = test_automation_endpoint(args.base_url, args.results_dir, timeout=args.timeout)

    # Probar diferentes umbrales
    test_different_thresholds(args.base_url, timeout=min(60, args.timeout))
    
    # Resumen
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE PRUEBAS:")
    print(f"   Sistema de automatización: {'✅ OK' if main_test_ok else '❌ ERROR'}")
    
    if main_test_ok:
        print("🎉 ¡Sistema de automatización funcionando correctamente!")
        print("📋 Características implementadas:")
        print("   ✅ Descarga automática de múltiples fuentes")
        print("   ✅ Detección inteligente de duplicados")
        print("   ✅ Eliminación automática de registros repetidos")
        print("   ✅ Generación de archivo unificado")
        print("   ✅ Generación de archivo de duplicados eliminados")
        print("   ✅ Estadísticas detalladas del proceso")
        print("   ✅ Métricas de rendimiento")
        print("   ✅ Endpoint: /api/v1/automation/unified-data")
    else:
        print("💥 Algunas pruebas fallaron")
        sys.exit(1)

if __name__ == "__main__":
    main()
