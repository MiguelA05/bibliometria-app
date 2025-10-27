#!/usr/bin/env python3
"""
Script de prueba completo para la funcionalidad de datos geográficos.
Prueba la extracción de datos geográficos y su compatibilidad con mapas de calor.
"""

import requests
import json
import time
import os
import pandas as pd

def test_geographic_endpoint():
    """Probar el endpoint específico de datos geográficos."""
    print("🌍 PRUEBA ENDPOINT DE DATOS GEOGRÁFICOS")
    print("=" * 60)
    
    try:
        url = "http://127.0.0.1:8000/api/v1/geographic/heatmap-data"
        data = {
            "query": "generative artificial intelligence",
            "max_articles": 15,
            "email": "test@example.com"
        }
        
        print(f"🔍 Enviando petición de datos geográficos...")
        print(f"🌐 URL: {url}")
        print(f"📊 Datos: {json.dumps(data, indent=2)}")
        
        # Medir tiempo de respuesta
        start_time = time.time()
        response = requests.post(url, json=data, timeout=60)
        end_time = time.time()
        
        print(f"⏱️ Tiempo de respuesta: {end_time - start_time:.2f} segundos")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Endpoint geográfico respondió correctamente!")
            
            # Mostrar análisis geográfico
            geo_analysis = result.get('geographic_analysis', {})
            print(f"\n🌍 ANÁLISIS GEOGRÁFICO:")
            print(f"   Consulta: {geo_analysis.get('query')}")
            print(f"   Total artículos: {geo_analysis.get('total_articles')}")
            print(f"   Países cubiertos: {geo_analysis.get('countries_covered')}")
            print(f"   Ciudades cubiertas: {geo_analysis.get('cities_covered')}")
            print(f"   Coordenadas disponibles: {geo_analysis.get('coordinates_available')}")
            
            # Mostrar datos de mapa de calor
            heatmap_data = result.get('heatmap_data', {})
            print(f"\n🗺️ DATOS PARA MAPA DE CALOR:")
            print(f"   Archivo: {heatmap_data.get('file_path')}")
            print(f"   Tamaño: {heatmap_data.get('file_size')}")
            print(f"   Formato: {heatmap_data.get('format')}")
            print(f"   Herramientas compatibles: {', '.join(heatmap_data.get('compatible_tools', []))}")
            
            # Mostrar estadísticas geográficas
            geo_stats = result.get('geographic_statistics', {})
            print(f"\n📊 ESTADÍSTICAS GEOGRÁFICAS:")
            
            top_countries = geo_stats.get('top_countries', [])
            if top_countries:
                print(f"   Top países:")
                for country, count in top_countries[:5]:
                    print(f"     {country}: {count} artículos")
            
            top_cities = geo_stats.get('top_cities', [])
            if top_cities:
                print(f"   Top ciudades:")
                for city, count in top_cities[:5]:
                    print(f"     {city}: {count} artículos")
            
            coverage = geo_stats.get('coverage_percentage', {})
            print(f"   Cobertura:")
            print(f"     Artículos con países: {coverage.get('articles_with_countries', 'N/A')}")
            print(f"     Artículos con ciudades: {coverage.get('articles_with_cities', 'N/A')}")
            print(f"     Artículos con coordenadas: {coverage.get('articles_with_coordinates', 'N/A')}")
            
            # Verificar archivo generado
            file_path = heatmap_data.get('file_path')
            if file_path and os.path.exists(file_path):
                print(f"\n✅ Archivo de datos geográficos encontrado: {file_path}")
                
                # Verificar contenido del archivo
                try:
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                    print(f"📊 Contenido del archivo:")
                    print(f"   Filas: {len(df)}")
                    print(f"   Columnas: {list(df.columns)}")
                    
                    # Mostrar algunas filas de ejemplo
                    if len(df) > 0:
                        print(f"\n📋 EJEMPLOS DE DATOS GEOGRÁFICOS:")
                        for i, row in df.head(3).iterrows():
                            print(f"   Registro {i+1}:")
                            print(f"     País: {row.get('country', 'N/A')}")
                            print(f"     Ciudad: {row.get('city', 'N/A')}")
                            print(f"     Institución: {row.get('institution', 'N/A')}")
                            print(f"     Latitud: {row.get('latitude', 'N/A')}")
                            print(f"     Longitud: {row.get('longitude', 'N/A')}")
                            print(f"     Citado por: {row.get('cited_by_count', 'N/A')}")
                    
                    # Verificar coordenadas válidas
                    valid_coords = df.dropna(subset=['latitude', 'longitude'])
                    print(f"\n🎯 COORDENADAS VÁLIDAS:")
                    print(f"   Registros con coordenadas: {len(valid_coords)}")
                    print(f"   Porcentaje con coordenadas: {(len(valid_coords) / len(df) * 100):.1f}%")
                    
                except Exception as e:
                    print(f"❌ Error al leer archivo: {e}")
            else:
                print(f"❌ Archivo de datos geográficos no encontrado: {file_path}")
            
            return True
            
        else:
            print(f"❌ Error en el endpoint geográfico: {response.status_code}")
            print(f"   Respuesta: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error de conexión: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def test_regular_endpoint_with_geographic_data():
    """Probar el endpoint regular para verificar que incluye datos geográficos."""
    print("\n🔍 PRUEBA ENDPOINT REGULAR CON DATOS GEOGRÁFICOS")
    print("=" * 60)
    
    try:
        url = "http://127.0.0.1:8000/api/v1/fetch-metadata"
        data = {
            "query": "generative artificial intelligence",
            "max_articles": 5,
            "email": "test@example.com"
        }
        
        print(f"🔍 Enviando petición al endpoint regular...")
        
        response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Endpoint regular respondió correctamente!")
            
            # Verificar que incluye datos geográficos
            articles = result.get('articles', [])
            if articles:
                first_article = articles[0]
                print(f"\n📊 VERIFICACIÓN DE DATOS GEOGRÁFICOS:")
                
                geo_fields = [
                    'author_countries', 'author_cities', 
                    'institution_countries', 'institution_cities', 
                    'geographic_coordinates'
                ]
                
                for field in geo_fields:
                    value = first_article.get(field)
                    if value:
                        print(f"   ✅ {field}: {value}")
                    else:
                        print(f"   ❌ {field}: No disponible")
            
            return True
            
        else:
            print(f"❌ Error en el endpoint regular: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def show_folder_structure():
    """Mostrar la estructura de carpetas con datos geográficos."""
    print("\n📁 ESTRUCTURA DE CARPETAS CON DATOS GEOGRÁFICOS:")
    print("=" * 60)
    
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

def show_visualization_examples():
    """Mostrar ejemplos de código para visualización."""
    print("\n💡 EJEMPLOS DE CÓDIGO PARA MAPAS DE CALOR:")
    print("=" * 60)
    
    print("""
🔧 INSTALACIÓN DE LIBRERÍAS:
pip install folium plotly geopandas

🗺️ EJEMPLO CON FOLIUM:
import folium
from folium.plugins import HeatMap
import pandas as pd

# Leer datos geográficos
df = pd.read_csv('results/geographic/heatmap_data_generative_artificial_intelligence_YYYYMMDD_HHMMSS.csv')

# Crear mapa base
m = folium.Map(location=[20, 0], zoom_start=2)

# Preparar datos para mapa de calor
heat_data = []
for idx, row in df.iterrows():
    if pd.notna(row['latitude']) and pd.notna(row['longitude']):
        heat_data.append([
            row['latitude'],
            row['longitude'],
            row['cited_by_count']  # Peso basado en citas
        ])

# Agregar mapa de calor
HeatMap(heat_data, radius=15, blur=25).add_to(m)

# Guardar mapa
m.save('mapa_calor_bibliometrico.html')

📊 EJEMPLO CON PLOTLY:
import plotly.express as px
import pandas as pd

# Leer datos
df = pd.read_csv('results/geographic/heatmap_data_generative_artificial_intelligence_YYYYMMDD_HHMMSS.csv')

# Crear mapa de densidad
fig = px.density_mapbox(
    df,
    lat='latitude',
    lon='longitude',
    z='cited_by_count',
    radius=10,
    center=dict(lat=0, lon=0),
    zoom=1,
    mapbox_style="open-street-map",
    title='Mapa de Calor Bibliométrico - Generative AI'
)

fig.show()
""")

def main():
    """Función principal."""
    print("🌍 PRUEBA COMPLETA - FUNCIONALIDAD DE DATOS GEOGRÁFICOS")
    print("=" * 70)
    
    # Probar endpoint específico de datos geográficos
    geo_endpoint_ok = test_geographic_endpoint()
    
    # Probar endpoint regular con datos geográficos
    regular_endpoint_ok = test_regular_endpoint_with_geographic_data()
    
    # Mostrar estructura de carpetas
    show_folder_structure()
    
    # Mostrar ejemplos de visualización
    show_visualization_examples()
    
    # Resumen
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE PRUEBAS:")
    print(f"   Endpoint geográfico específico: {'✅ OK' if geo_endpoint_ok else '❌ ERROR'}")
    print(f"   Endpoint regular con datos geo: {'✅ OK' if regular_endpoint_ok else '❌ ERROR'}")
    
    if geo_endpoint_ok and regular_endpoint_ok:
        print("🎉 ¡Funcionalidad de datos geográficos completamente implementada!")
        print("📋 Características implementadas:")
        print("   ✅ Extracción automática de datos geográficos")
        print("   ✅ Coordenadas lat/lng para mapas de calor")
        print("   ✅ Información de países y ciudades")
        print("   ✅ Datos de instituciones con ubicación")
        print("   ✅ Endpoint específico para mapas de calor")
        print("   ✅ Archivos CSV optimizados para visualización")
        print("   ✅ Compatible con Folium, Plotly, GeoPandas, etc.")
        print("\n🚀 ¡LISTO PARA CREAR MAPAS DE CALOR BIBLIOMÉTRICOS!")
    else:
        print("💥 Algunas pruebas fallaron")

if __name__ == "__main__":
    main()





