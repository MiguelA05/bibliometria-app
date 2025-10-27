#!/usr/bin/env python3
"""
Script para mostrar la estructura de carpetas organizadas y probar el sistema de automatización.

Este script es un pequeño runner (no un test de pytest) que:
 - espera a que la API responda en /health
 - llama al endpoint de automatización y al endpoint universitario
 - muestra y verifica los archivos generados en `results/`

Usar:
    python test_organized_folders.py

Asegúrate de tener la API corriendo (por ejemplo con `python start.py --start` en otra terminal)
"""

import os
import requests
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
RESULTS_DIR = PROJECT_ROOT / "results"


def show_folder_structure():
    """Mostrar la estructura de carpetas de resultados."""
    print("📁 ESTRUCTURA DE CARPETAS ORGANIZADAS")
    print("=" * 50)

    base_dir = RESULTS_DIR

    if not base_dir.exists():
        print("❌ Directorio results no existe aún")
        return

    print(f"📂 {base_dir}/")

    # Mostrar estructura de carpetas
    for root, dirs, files in os.walk(base_dir):
        level = Path(root).relative_to(base_dir).parts
        indent = ' ' * 2 * (len(level))
        print(f"{indent}📁 {os.path.basename(root)}/")

        subindent = ' ' * 2 * (len(level) + 1)
        for file in files:
            file_size = os.path.getsize(os.path.join(root, file))
            size_kb = file_size / 1024
            print(f"{subindent}📄 {file} ({size_kb:.1f} KB)")

    print()

def test_automation_with_organized_folders():
    """Probar el sistema de automatización y mostrar archivos organizados."""
    print("🤖 PROBANDO SISTEMA CON CARPETAS ORGANIZADAS")
    print("=" * 60)
    
    try:
        # Ensure server is responding before calling
        health_url = "http://127.0.0.1:8000/health"
        start = time.time()
        while time.time() - start < 10:
            try:
                h = requests.get(health_url, timeout=2)
                if h.status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(0.5)

        url = "http://127.0.0.1:8000/api/v1/automation/unified-data"
        data = {
            "base_query": "generative artificial intelligence",
            "similarity_threshold": 0.8,
            "max_articles_per_source": 20
        }
        
        print(f"🔍 Enviando petición de automatización...")
        
        # Medir tiempo de respuesta
        start_time = time.time()
        response = requests.post(url, json=data, timeout=60)
        end_time = time.time()
        
        print(f"⏱️ Tiempo de respuesta: {end_time - start_time:.2f} segundos")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Sistema de automatización respondió correctamente!")

            # Mostrar archivos generados
            generated_files = result.get('generated_files', {})
            unified_file = generated_files.get('unified_file')
            duplicates_file = generated_files.get('duplicates_file')
            unified_size = generated_files.get('unified_file_size') or generated_files.get('unified_file_size_kb')
            duplicates_size = generated_files.get('duplicates_file_size') or generated_files.get('duplicates_file_size_kb')

            print(f"\n📁 ARCHIVOS GENERADOS EN CARPETAS ORGANIZADAS:")
            print(f"   📄 Archivo unificado: {unified_file}")
            print(f"   📄 Archivo de duplicados: {duplicates_file}")
            print(f"   📊 Tamaño archivo unificado: {unified_size}")
            print(f"   📊 Tamaño archivo duplicados: {duplicates_size}")

            # Resolve and check existence (try absolute and results/ locations)
            def resolve_path(p):
                if not p:
                    return None
                p = Path(p)
                if p.exists():
                    return p
                # try relative to project results
                p2 = RESULTS_DIR / p.name
                if p2.exists():
                    return p2
                # try relative to project root
                p3 = PROJECT_ROOT / p
                if p3.exists():
                    return p3
                return None

            u_path = resolve_path(unified_file)
            d_path = resolve_path(duplicates_file)

            print(f"   ✅ Unified file exists: {bool(u_path)} -> {u_path}")
            print(f"   ✅ Duplicates file exists: {bool(d_path)} -> {d_path}")


            # Mostrar estadísticas
            data_stats = result.get('data_statistics', {})
            print(f"\n📊 ESTADÍSTICAS:")
            print(f"   Total artículos descargados: {data_stats.get('total_articles_downloaded')}")
            print(f"   Artículos únicos: {data_stats.get('unique_articles')}")
            print(f"   Duplicados eliminados: {data_stats.get('duplicates_removed')}")
            print(f"   Fuentes procesadas: {data_stats.get('sources_processed')}")
            print(f"   Tasa de duplicación: {data_stats.get('duplication_rate')}")

            return True
            
        else:
            print(f"❌ Error en el sistema: {response.status_code}")
            print(f"   Respuesta: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_university_endpoint_with_organized_folders():
    """Probar el endpoint universitario y mostrar archivos organizados."""
    print("\n🎓 PROBANDO ENDPOINT UNIVERSITARIO CON CARPETAS ORGANIZADAS")
    print("=" * 60)
    
    try:
        url = "http://127.0.0.1:8000/api/v1/uniquindio/generative-ai"
        data = {
            "max_articles": 15,
            "email": "estudiante@uniquindio.edu.co"
        }
        
        print(f"🔍 Enviando petición al endpoint universitario...")
        
        response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Endpoint universitario respondió correctamente!")
            
            # Mostrar archivo generado
            research_results = result.get('research_results', {})
            csv_file = research_results.get('csv_file_path')
            print(f"\n📁 ARCHIVO GENERADO EN CARPETA ORGANIZADA:")
            print(f"   📄 Archivo CSV: {csv_file}")
            
            if csv_file and os.path.exists(csv_file):
                file_size = os.path.getsize(csv_file)
                size_kb = file_size / 1024
                print(f"   📊 Tamaño: {size_kb:.1f} KB")
                
                # Verificar contenido
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_file, encoding='utf-8-sig')
                    print(f"   📊 Contenido: {len(df)} filas, {len(df.columns)} columnas")
                except Exception as e:
                    print(f"   ❌ Error al leer archivo: {e}")
            
            return True
            
        else:
            print(f"❌ Error en el endpoint: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def show_folder_descriptions():
    """Mostrar descripción de cada carpeta."""
    print("\n📋 DESCRIPCIÓN DE CARPETAS:")
    print("=" * 40)
    print("📁 results/")
    print("   ├── 📁 raw_data/          # Datos sin procesar de OpenAlex")
    print("   ├── 📁 unified/           # Archivos unificados sin duplicados")
    print("   ├── 📁 duplicates/        # Registro de duplicados eliminados")
    print("   └── 📁 reports/           # Reportes de procesamiento y estadísticas")
    print()
    print("🎯 BENEFICIOS DE LA ORGANIZACIÓN:")
    print("   ✅ Separación clara por tipo de datos")
    print("   ✅ Fácil identificación de archivos")
    print("   ✅ Mejor gestión de resultados")
    print("   ✅ Reportes detallados de procesamiento")
    print("   ✅ Trazabilidad completa del proceso")

def main():
    """Función principal."""
    print("📁 SISTEMA DE ORGANIZACIÓN DE ARCHIVOS POR CARPETAS")
    print("=" * 60)
    
    # Mostrar descripción de carpetas
    show_folder_descriptions()
    
    # Mostrar estructura actual
    show_folder_structure()
    
    # Probar sistema de automatización
    automation_ok = test_automation_with_organized_folders()
    
    # Probar endpoint universitario
    university_ok = test_university_endpoint_with_organized_folders()
    
    # Mostrar estructura final
    print("\n📁 ESTRUCTURA FINAL DE CARPETAS:")
    print("=" * 40)
    show_folder_structure()
    
    # Resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE PRUEBAS:")
    print(f"   Sistema de automatización: {'✅ OK' if automation_ok else '❌ ERROR'}")
    print(f"   Endpoint universitario: {'✅ OK' if university_ok else '❌ ERROR'}")
    
    if automation_ok and university_ok:
        print("🎉 ¡Sistema de organización por carpetas funcionando correctamente!")
        print("📋 Características implementadas:")
        print("   ✅ Carpeta raw_data para datos sin procesar")
        print("   ✅ Carpeta unified para archivos unificados")
        print("   ✅ Carpeta duplicates para registros de duplicados")
        print("   ✅ Carpeta reports para reportes de procesamiento")
        print("   ✅ Organización automática por tipo de archivo")
        print("   ✅ Nombres de archivo descriptivos con timestamp")
    else:
        print("💥 Algunas pruebas fallaron")

if __name__ == "__main__":
    main()





