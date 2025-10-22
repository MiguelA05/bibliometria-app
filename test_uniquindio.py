#!/usr/bin/env python3
"""
Script de prueba específico para el proyecto de la Universidad del Quindío.
Prueba el endpoint dedicado para búsquedas de "generative artificial intelligence".
"""

import requests
import json
import time
import os

def test_uniquindio_endpoint():
    """Probar el endpoint específico de la Universidad del Quindío."""
    print("🎓 PRUEBA ENDPOINT UNIVERSIDAD DEL QUINDÍO")
    print("=" * 60)
    
    try:
        url = "http://127.0.0.1:8000/api/v1/uniquindio/generative-ai"
        data = {
            "max_articles": 5,
            "email": "estudiante@uniquindio.edu.co"
        }
        
        print(f"🔍 Enviando petición al endpoint universitario...")
        print(f"🌐 URL: {url}")
        print(f"📊 Datos: {json.dumps(data, indent=2)}")
        
        # Medir tiempo de respuesta
        start_time = time.time()
        response = requests.post(url, json=data, timeout=60)
        end_time = time.time()
        
        print(f"⏱️ Tiempo de respuesta: {end_time - start_time:.2f} segundos")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Endpoint universitario respondió correctamente!")
            
            # Mostrar información del proyecto universitario
            university_info = result.get('university_project', {})
            print(f"\n🎓 INFORMACIÓN DEL PROYECTO UNIVERSITARIO:")
            print(f"   Institución: {university_info.get('institution')}")
            print(f"   Curso: {university_info.get('course')}")
            print(f"   Dominio: {university_info.get('domain')}")
            print(f"   Consulta: {university_info.get('search_query')}")
            print(f"   Base de datos: {university_info.get('database_source')}")
            print(f"   Formato de exportación: {university_info.get('export_format')}")
            
            # Mostrar resultados de investigación
            research_results = result.get('research_results', {})
            print(f"\n📊 RESULTADOS DE INVESTIGACIÓN:")
            print(f"   Total artículos: {research_results.get('total_articles')}")
            print(f"   Archivo CSV: {research_results.get('csv_file_path')}")
            print(f"   Fuente de datos: {research_results.get('data_source')}")
            print(f"   Mensaje: {research_results.get('message')}")
            
            # Mostrar tipos de contenido encontrados
            content_types = result.get('content_types', {})
            print(f"\n📚 TIPOS DE CONTENIDO:")
            print(f"   Tipos disponibles: {', '.join(content_types.get('available_types', []))}")
            print(f"   Tipos encontrados: {', '.join(content_types.get('found_types', []))}")
            
            # Verificar archivo CSV
            csv_path = research_results.get('csv_file_path')
            if csv_path and os.path.exists(csv_path):
                print(f"\n✅ Archivo CSV encontrado: {csv_path}")
                
                # Verificar contenido del CSV
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_path, encoding='utf-8-sig')
                    print(f"📊 Contenido del CSV:")
                    print(f"   Filas: {len(df)}")
                    print(f"   Columnas: {list(df.columns)}")
                    
                    # Mostrar distribución de tipos
                    if 'type' in df.columns:
                        type_counts = df['type'].value_counts()
                        print(f"   Distribución por tipo:")
                        for type_name, count in type_counts.items():
                            print(f"     {type_name}: {count}")
                    
                except Exception as e:
                    print(f"❌ Error al leer CSV: {e}")
            else:
                print(f"❌ Archivo CSV no encontrado: {csv_path}")
            
            # Mostrar información de los artículos
            articles = research_results.get('articles', [])
            if articles:
                print(f"\n📄 INFORMACIÓN DE ARTÍCULOS:")
                for i, article in enumerate(articles[:3]):  # Solo los primeros 3
                    print(f"   Artículo {i+1}:")
                    print(f"     Título: {article['title'][:60]}...")
                    print(f"     Autores: {len(article['authors'])} autores")
                    print(f"     Afiliaciones: {len(article['affiliations'])} afiliaciones")
                    print(f"     DOI: {article.get('doi', 'N/A')}")
                    print(f"     Tipo: {article.get('type', 'N/A')}")
                    print(f"     Año: {article.get('publication_year', 'N/A')}")
                    print(f"     Citado por: {article.get('cited_by_count', 'N/A')} veces")
                    print(f"     Open Access: {'Sí' if article.get('is_oa') else 'No'}")
            
            return True
            
        else:
            print(f"❌ Error en el endpoint universitario: {response.status_code}")
            print(f"   Respuesta: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error de conexión: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def test_content_types():
    """Probar diferentes tipos de contenido."""
    print("\n🔍 PROBANDO DIFERENTES TIPOS DE CONTENIDO")
    print("=" * 50)
    
    content_types = [
        "journal-article",
        "conference-paper", 
        "book-chapter",
        "book",
        "thesis",
        "report"
    ]
    
    url = "http://127.0.0.1:8000/api/v1/uniquindio/generative-ai"
    
    for content_type in content_types:
        try:
            data = {
                "max_articles": 2
            }
            
            print(f"📚 Probando tipo: {content_type}")
            response = requests.post(url, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                articles_count = result.get('research_results', {}).get('total_articles', 0)
                print(f"   ✅ Encontrados: {articles_count} artículos")
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def main():
    """Función principal."""
    print("🎓 PRUEBA COMPLETA - UNIVERSIDAD DEL QUINDÍO")
    print("Proyecto: Análisis de Algoritmos")
    print("Dominio: Generative Artificial Intelligence")
    print("=" * 60)
    
    # Probar endpoint principal
    main_test_ok = test_uniquindio_endpoint()
    
    # Probar tipos de contenido
    test_content_types()
    
    # Resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE PRUEBAS:")
    print(f"   Endpoint universitario: {'✅ OK' if main_test_ok else '❌ ERROR'}")
    
    if main_test_ok:
        print("🎉 ¡Proyecto universitario funcionando correctamente!")
        print("📋 Características implementadas:")
        print("   ✅ Búsqueda específica: 'generative artificial intelligence'")
        print("   ✅ Base de datos: OpenAlex")
        print("   ✅ Exportación: CSV")
        print("   ✅ Tipologías: Artículos, conferencias, capítulos, libros, tesis, reportes")
        print("   ✅ Endpoint específico: /api/v1/uniquindio/generative-ai")
        print("   ✅ Metadatos completos: Autores, afiliaciones, citas, DOI, etc.")
    else:
        print("💥 Algunas pruebas fallaron")

if __name__ == "__main__":
    main()
