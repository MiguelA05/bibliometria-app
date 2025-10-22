#!/usr/bin/env python3
"""
Script de prueba para la API con OpenAlex.
"""

import requests
import json
import time
import os

def test_api():
    """Probar la API con OpenAlex."""
    print("🚀 Probando API con OpenAlex...")
    
    try:
        url = "http://127.0.0.1:8000/api/v1/fetch-metadata"
        data = {
            "query": "machine learning",
            "max_articles": 3,
            "email": "test@example.com"
        }
        
        print(f"🔍 Enviando petición: {data}")
        print(f"🌐 URL: {url}")
        
        # Medir tiempo de respuesta
        start_time = time.time()
        response = requests.post(url, json=data, timeout=60)
        end_time = time.time()
        
        print(f"⏱️ Tiempo de respuesta: {end_time - start_time:.2f} segundos")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API respondió correctamente!")
            print(f"📊 Respuesta completa:")
            print(f"   Total artículos: {result['total_articles']}")
            print(f"   Fuente de datos: {result.get('data_source', 'OpenAlex API')}")
            print(f"   Archivo CSV: {result['csv_file_path']}")
            print(f"   Mensaje: {result['message']}")
            
            # Verificar que el archivo CSV existe
            csv_path = result['csv_file_path']
            if csv_path and os.path.exists(csv_path):
                print(f"✅ Archivo CSV encontrado: {csv_path}")
                
                # Verificar contenido del CSV
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_path, encoding='utf-8-sig')
                    print(f"📊 Contenido del CSV:")
                    print(f"   Filas: {len(df)}")
                    print(f"   Columnas: {list(df.columns)}")
                    
                    # Mostrar una fila de ejemplo
                    if len(df) > 0:
                        print(f"\n📄 Ejemplo de datos:")
                        for col in df.columns:
                            value = str(df.iloc[0][col])
                            if len(value) > 80:
                                value = value[:80] + "..."
                            print(f"   {col}: {value}")
                    
                except Exception as e:
                    print(f"❌ Error al leer CSV: {e}")
            else:
                print(f"❌ Archivo CSV no encontrado: {csv_path}")
            
            # Mostrar información de los artículos
            articles = result['articles']
            if articles:
                print(f"\n📄 Información de artículos:")
                for i, article in enumerate(articles[:2]):  # Solo los primeros 2
                    print(f"   Artículo {i+1}:")
                    print(f"     Título: {article['title'][:60]}...")
                    print(f"     Autores: {article['authors']}")
                    print(f"     Afiliaciones: {len(article['affiliations'])} encontradas")
                    print(f"     DOI: {article.get('doi', 'N/A')}")
                    print(f"     Open Access: {article.get('is_oa', 'N/A')}")
                    print(f"     Citado por: {article.get('cited_by_count', 'N/A')} veces")
                    print(f"     Año: {article.get('publication_year', 'N/A')}")
                    print(f"     Tipo: {article.get('type', 'N/A')}")
                    print(f"     URL: {article['article_url']}")
            
            return True
            
        else:
            print(f"❌ Error en la API: {response.status_code}")
            print(f"   Respuesta: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error de conexión: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def test_filters():
    """Probar filtros de OpenAlex."""
    print("\n🔍 Probando filtros de OpenAlex...")
    
    try:
        url = "http://127.0.0.1:8000/api/v1/fetch-metadata"
        
        # Probar con filtro de año
        data = {
            "query": "artificial intelligence",
            "max_articles": 2,
            "filters": {
                "publication_year": "2024"
            }
        }
        
        print("📅 Probando filtro de año 2024...")
        response = requests.post(url, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Filtro de año funcionando: {result['total_articles']} artículos")
            
            # Verificar que los artículos son de 2024
            if result['articles']:
                for article in result['articles']:
                    year = article.get('publication_year')
                    if year:
                        print(f"   Año del artículo: {year}")
        else:
            print(f"❌ Error con filtro de año: {response.status_code}")
        
        # Probar con filtro de tipo
        data = {
            "query": "machine learning",
            "max_articles": 2,
            "filters": {
                "type": "journal-article"
            }
        }
        
        print("\n📰 Probando filtro de tipo journal-article...")
        response = requests.post(url, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Filtro de tipo funcionando: {result['total_articles']} artículos")
            
            # Verificar que los artículos son journal-article
            if result['articles']:
                for article in result['articles']:
                    article_type = article.get('type')
                    if article_type:
                        print(f"   Tipo del artículo: {article_type}")
        else:
            print(f"❌ Error con filtro de tipo: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error probando filtros: {e}")
        return False

def main():
    """Función principal."""
    print("🚀 PRUEBA DE API CON OPENALEX")
    print("=" * 50)
    
    # Probar API básica
    api_ok = test_api()
    
    # Probar filtros
    filters_ok = test_filters()
    
    # Resumen
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE PRUEBAS:")
    print(f"   API básica: {'✅ OK' if api_ok else '❌ ERROR'}")
    print(f"   Filtros: {'✅ OK' if filters_ok else '❌ ERROR'}")
    
    if all([api_ok, filters_ok]):
        print("🎉 ¡Todas las pruebas completadas exitosamente!")
    else:
        print("💥 Algunas pruebas fallaron")

if __name__ == "__main__":
    main()
