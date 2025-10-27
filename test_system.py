#!/usr/bin/env python3
"""
Script central de pruebas - Sistema Bibliométrico
Verifica: automatización, datos geográficos integrados, y archivos generados.
"""

import requests
import pandas as pd
import os
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def check_server():
    """Verificar que el servidor esté corriendo."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_automation():
    """Probar el endpoint de automatización."""
    print("\n🤖 PRUEBA: Sistema de Automatización")
    print("-" * 50)
    
    try:
        url = f"{BASE_URL}/api/v1/automation/unified-data"
        data = {
            "base_query": "generative artificial intelligence",
            "similarity_threshold": 0.8,
            "max_articles_per_source": 30
        }
        
        print(f"   Consulta: {data['base_query']}")
        start = time.time()
        response = requests.post(url, json=data, timeout=120)
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            stats = result.get('data_statistics', {})
            
            print(f"   ✅ Tiempo: {elapsed:.1f}s")
            print(f"   ✅ Total artículos: {stats.get('total_articles_downloaded')}")
            print(f"   ✅ Únicos: {stats.get('unique_articles')}")
            print(f"   ✅ Duplicados eliminados: {stats.get('duplicates_removed')}")
            print(f"   ✅ Tasa duplicación: {stats.get('duplication_rate')}")
            
            files = result.get('generated_files', {})
            unified = files.get('unified_file', '')
            
            if unified and os.path.exists(unified):
                print(f"   ✅ Archivo: {os.path.basename(unified)}")
                return True, unified
            else:
                print(f"   ❌ Archivo no encontrado")
                return True, None
        else:
            print(f"   ❌ Error {response.status_code}")
            return False, None
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, None

def verify_geographic_data(csv_file):
    """Verificar que el CSV incluye datos geográficos."""
    print("\n🌍 PRUEBA: Datos Geográficos Integrados")
    print("-" * 50)
    
    if not csv_file or not os.path.exists(csv_file):
        print("   ❌ No hay CSV")
        return False
    
    try:
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        
        geo_columns = [
            'author_countries', 'author_cities',
            'institution_countries', 'institution_cities',
            'geographic_coordinates'
        ]
        
        print(f"   Registros: {len(df)}")
        
        all_present = True
        for col in geo_columns:
            if col in df.columns:
                count = df[col].notna().sum()
                print(f"   ✅ {col}: {count} registros")
            else:
                print(f"   ❌ {col}: FALTA")
                all_present = False
        
        if all_present:
            print(f"   ✅ Campos geográficos integrados correctamente")
            
            # Mostrar ejemplos
            print(f"\n   Ejemplos:")
            for i, row in df.head(3).iterrows():
                inst = row.get('institution_countries', '')
                if pd.notna(inst) and inst:
                    print(f"      • {str(row['title'])[:50]}... → {inst}")
            
            return True
        else:
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def show_structure():
    """Mostrar estructura de archivos."""
    print("\n📁 ARCHIVOS GENERADOS:")
    print("-" * 50)
    
    if not os.path.exists("results"):
        print("   No hay archivos")
        return
    
    for root, dirs, files in os.walk("results"):
        level = root.replace("results", '').count(os.sep)
        indent = '   ' + '  ' * level
        
        folder = os.path.basename(root)
        if folder == "results":
            print(f"📂 results/")
        else:
            print(f"{indent}📁 {folder}/")
            
            for file in files:
                try:
                    size = os.path.getsize(os.path.join(root, file)) / 1024
                    print(f"{indent}📄 {file} ({size:.1f} KB)")
                except:
                    pass

def main():
    """Función principal."""
    print("=" * 70)
    print("🚀 SISTEMA DE PRUEBAS BIBLIOMÉTRICO")
    print("=" * 70)
    
    # Verificar servidor
    print("\n🔍 Verificando servidor...")
    if not check_server():
        print("   ❌ Servidor no está corriendo")
        print("   Ejecuta: python start.py")
        return
    print("   ✅ Servidor corriendo")
    
    # Ejecutar pruebas
    automation_ok, csv_file = test_automation()
    
    geographic_ok = False
    if csv_file:
        geographic_ok = verify_geographic_data(csv_file)
    
    # Mostrar estructura
    show_structure()
    
    # Resumen
    print("\n" + "=" * 70)
    print("📊 RESUMEN:")
    print(f"   Servidor: ✅")
    print(f"   Automatización: {'✅' if automation_ok else '❌'}")
    print(f"   Datos geográficos: {'✅' if geographic_ok else '❌'}")
    
    if automation_ok and geographic_ok:
        print("\n🎉 ¡TODAS LAS PRUEBAS EXITOSAS!")
        print("\n✅ Sistema funcionando:")
        print("   • Descarga multi-fuente automática")
        print("   • Eliminación de duplicados")
        print("   • Datos geográficos integrados en CSV")
        print("   • Archivos organizados por tipo")
    else:
        print("\n❌ Algunas pruebas fallaron")

if __name__ == "__main__":
    main()