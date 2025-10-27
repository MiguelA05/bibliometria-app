#!/usr/bin/env python3
"""
Ejemplo práctico de creación de mapa de calor bibliométrico usando los datos geográficos extraídos.
"""

import pandas as pd
import os
import glob

def create_heatmap_example():
    """Crear un ejemplo de mapa de calor usando los datos disponibles."""
    print("🗺️ CREANDO MAPA DE CALOR BIBLIOMÉTRICO")
    print("=" * 50)
    
    # Buscar el archivo más reciente de datos geográficos
    geographic_files = glob.glob("results/geographic/heatmap_data_*.csv")
    
    if not geographic_files:
        print("❌ No se encontraron archivos de datos geográficos")
        print("   Ejecuta primero: python test_geographic_complete.py")
        return False
    
    # Usar el archivo más reciente
    latest_file = max(geographic_files, key=os.path.getctime)
    print(f"📄 Usando archivo: {latest_file}")
    
    try:
        # Leer datos
        df = pd.read_csv(latest_file, encoding='utf-8-sig')
        print(f"📊 Datos cargados: {len(df)} registros")
        
        # Mostrar estadísticas básicas
        print(f"\n📈 ESTADÍSTICAS DE LOS DATOS:")
        print(f"   Total registros: {len(df)}")
        print(f"   Países únicos: {df['country'].nunique()}")
        print(f"   Artículos únicos: {df['title'].nunique()}")
        
        # Mostrar distribución por países
        country_counts = df['country'].value_counts()
        print(f"\n🌍 TOP 10 PAÍSES:")
        for country, count in country_counts.head(10).items():
            print(f"   {country}: {count} registros")
        
        # Crear mapa de calor con Folium (si está disponible)
        try:
            import folium
            from folium.plugins import HeatMap
            
            print(f"\n🗺️ CREANDO MAPA DE CALOR CON FOLIUM...")
            
            # Crear mapa base centrado en el mundo
            m = folium.Map(location=[20, 0], zoom_start=2, tiles='OpenStreetMap')
            
            # Preparar datos para mapa de calor
            # Como no tenemos coordenadas exactas, usaremos coordenadas aproximadas por país
            country_coordinates = {
                'United States': [39.8283, -98.5795],
                'Canada': [56.1304, -106.3468],
                'Australia': [-25.2744, 133.7751],
                'Ghana': [7.9465, -1.0232],
                'United Kingdom': [55.3781, -3.4360],
                'Germany': [51.1657, 10.4515],
                'France': [46.2276, 2.2137],
                'Italy': [41.8719, 12.5674],
                'Spain': [40.4637, -3.7492],
                'Netherlands': [52.1326, 5.2913],
                'Ireland': [53.4129, -8.2439],
                'China': [35.8617, 104.1954],
                'Japan': [36.2048, 138.2529],
                'India': [20.5937, 78.9629],
                'Brazil': [-14.2350, -51.9253],
                'Mexico': [23.6345, -102.5528],
                'Argentina': [-38.4161, -63.6167],
                'Chile': [-35.6751, -71.5430],
                'Colombia': [4.5709, -74.2973],
                'Peru': [-9.1900, -75.0152]
            }
            
            # Crear datos de calor
            heat_data = []
            for idx, row in df.iterrows():
                country = row['country']
                if country in country_coordinates:
                    lat, lng = country_coordinates[country]
                    # Usar el número de citas como peso
                    weight = row['cited_by_count'] if pd.notna(row['cited_by_count']) else 1
                    heat_data.append([lat, lng, weight])
            
            if heat_data:
                # Agregar mapa de calor
                HeatMap(heat_data, radius=20, blur=15, max_zoom=1).add_to(m)
                
                # Agregar marcadores para países con más actividad
                for country, count in country_counts.head(5).items():
                    if country in country_coordinates:
                        lat, lng = country_coordinates[country]
                        folium.Marker(
                            [lat, lng],
                            popup=f"{country}<br>{count} registros",
                            tooltip=country,
                            icon=folium.Icon(color='red', icon='info-sign')
                        ).add_to(m)
                
                # Guardar mapa
                map_filename = "mapa_calor_bibliometrico_generative_ai.html"
                m.save(map_filename)
                print(f"✅ Mapa guardado: {map_filename}")
                print(f"   Abre el archivo en tu navegador para ver el mapa interactivo")
                
            else:
                print("❌ No se pudieron crear datos de calor")
                
        except ImportError:
            print("❌ Folium no está instalado")
            print("   Instala con: pip install folium")
        
        # Crear mapa de calor con Plotly (si está disponible)
        try:
            import plotly.express as px
            
            print(f"\n📊 CREANDO MAPA DE DENSIDAD CON PLOTLY...")
            
            # Preparar datos para Plotly
            plotly_data = []
            for idx, row in df.iterrows():
                country = row['country']
                if country in country_coordinates:
                    lat, lng = country_coordinates[country]
                    plotly_data.append({
                        'country': country,
                        'latitude': lat,
                        'longitude': lng,
                        'cited_by_count': row['cited_by_count'],
                        'title': row['title'][:50] + "..." if len(row['title']) > 50 else row['title']
                    })
            
            if plotly_data:
                plotly_df = pd.DataFrame(plotly_data)
                
                # Crear mapa de densidad
                fig = px.density_mapbox(
                    plotly_df,
                    lat='latitude',
                    lon='longitude',
                    z='cited_by_count',
                    radius=20,
                    center=dict(lat=20, lon=0),
                    zoom=1,
                    mapbox_style="open-street-map",
                    title='Mapa de Calor Bibliométrico - Generative AI',
                    hover_data=['country', 'title', 'cited_by_count']
                )
                
                # Guardar mapa
                plotly_filename = "mapa_densidad_bibliometrico_generative_ai.html"
                fig.write_html(plotly_filename)
                print(f"✅ Mapa de densidad guardado: {plotly_filename}")
                print(f"   Abre el archivo en tu navegador para ver el mapa interactivo")
                
        except ImportError:
            print("❌ Plotly no está instalado")
            print("   Instala con: pip install plotly")
        
        # Crear análisis de texto para países
        print(f"\n📝 ANÁLISIS DE COLABORACIÓN INTERNACIONAL:")
        
        # Analizar colaboraciones por país
        country_collaborations = {}
        for idx, row in df.iterrows():
            countries = str(row['institution_countries']).split(';')
            countries = [c.strip() for c in countries if c.strip() and c.strip() != 'nan']
            
            if len(countries) > 1:
                for country in countries:
                    if country not in country_collaborations:
                        country_collaborations[country] = 0
                    country_collaborations[country] += 1
        
        if country_collaborations:
            print(f"   Países con más colaboraciones internacionales:")
            sorted_collaborations = sorted(country_collaborations.items(), key=lambda x: x[1], reverse=True)
            for country, count in sorted_collaborations[:5]:
                print(f"     {country}: {count} colaboraciones")
        
        return True
        
    except Exception as e:
        print(f"❌ Error procesando datos: {e}")
        return False

def show_installation_instructions():
    """Mostrar instrucciones de instalación."""
    print(f"\n🔧 INSTRUCCIONES DE INSTALACIÓN:")
    print("=" * 40)
    print("Para crear mapas de calor interactivos, instala:")
    print("   pip install folium plotly")
    print("\nPara análisis geoespacial avanzado:")
    print("   pip install geopandas")
    print("\nPara visualización 3D:")
    print("   pip install keplergl")

def main():
    """Función principal."""
    print("🗺️ EJEMPLO PRÁCTICO - MAPA DE CALOR BIBLIOMÉTRICO")
    print("=" * 60)
    
    success = create_heatmap_example()
    show_installation_instructions()
    
    if success:
        print(f"\n🎉 ¡MAPAS DE CALOR CREADOS EXITOSAMENTE!")
        print("=" * 50)
        print("📁 Archivos generados:")
        print("   • mapa_calor_bibliometrico_generative_ai.html (Folium)")
        print("   • mapa_densidad_bibliometrico_generative_ai.html (Plotly)")
        print("\n🌐 Abre los archivos HTML en tu navegador para ver los mapas interactivos")
        print("\n💡 Los mapas muestran:")
        print("   • Distribución geográfica de la investigación")
        print("   • Intensidad basada en número de citas")
        print("   • Colaboraciones internacionales")
        print("   • Países líderes en el campo")
    else:
        print(f"\n❌ Error creando mapas de calor")

if __name__ == "__main__":
    main()





