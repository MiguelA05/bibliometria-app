#!/usr/bin/env python3
"""
Script para mostrar la estructura de carpetas organizadas y probar el sistema de automatización.
"""

import os
import requests
import json
import time

def show_folder_structure():
    """Mostrar la estructura de carpetas de resultados."""
    print("📁 ESTRUCTURA DE CARPETAS ORGANIZADAS")
    print("=" * 50)
    
    base_dir = "results"
    
    if not os.path.exists(base_dir):
        print("❌ Directorio results no existe aún")
        return
    
    print(f"📂 {base_dir}/")
    
    # Mostrar estructura de carpetas
    for root, dirs, files in os.walk(base_dir):
        level = root.replace(base_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}📁 {os.path.basename(root)}/")
        
        subindent = ' ' * 2 * (level + 1)
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
            print(f"\n📁 ARCHIVOS GENERADOS EN CARPETAS ORGANIZADAS:")
            print(f"   📄 Archivo unificado: {generated_files.get('unified_file')}")
            print(f"   📄 Archivo de duplicados: {generated_files.get('duplicates_file')}")
            print(f"   📊 Tamaño archivo unificado: {generated_files.get('unified_file_size')}")
            print(f"   📊 Tamaño archivo duplicados: {generated_files.get('duplicates_file_size')}")
            
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





