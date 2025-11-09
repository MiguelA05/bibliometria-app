#!/usr/bin/env python3
"""
Punto de entrada principal para Bibliometría App
Combina el menú interactivo y el servidor FastAPI en una sola aplicación.
"""

import os
import sys
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.data_unification_service import DataUnificationService
from app.services.text_similarity_service import TextSimilarityService
from app.services.word_frequency_service import WordFrequencyService
from app.services.hierarchical_clustering_service import HierarchicalClusteringService
from app.services.visualization_service import VisualizationService
from app.utils.text_extractor import TextExtractor, get_unified_csv_list
from app.config import settings

# Intentar importar helpers del servidor
try:
    from app.utils.server_helper import (
        check_server_running,
        start_server,
        ensure_server_ready,
        get_server_status
    )
    SERVER_HELPER_AVAILABLE = True
except ImportError:
    SERVER_HELPER_AVAILABLE = False
    check_server_running = None
    start_server = None
    ensure_server_ready = None
    get_server_status = None


class MenuPrincipal:
    """Menú principal interactivo."""
    
    def __init__(self):
        self.unification_service = DataUnificationService()
        self.similarity_service = TextSimilarityService()
        self.word_frequency_service = WordFrequencyService()
        self.clustering_service = HierarchicalClusteringService()
        self.visualization_service = VisualizationService()
        self.text_extractor = TextExtractor()
        
        # Iniciar servidor FastAPI automáticamente
        self._iniciar_servidor_automatico()
    
    def _iniciar_servidor_automatico(self):
        """Iniciar el servidor FastAPI automáticamente al iniciar el menú."""
        if not SERVER_HELPER_AVAILABLE:
            print("\n[WARNING] Helper del servidor no disponible")
            print("[INFO] El servidor FastAPI no se iniciará automáticamente")
            return
        
        print("\n" + "="*70)
        print("INICIANDO SERVIDOR FASTAPI")
        print("="*70)
        
        if check_server_running():
            status = get_server_status() if get_server_status else None
            if status:
                print(f"[OK] Servidor FastAPI ya está corriendo en {status.get('url', 'N/A')}")
            else:
                print("[OK] Servidor FastAPI ya está corriendo")
        else:
            print(f"[INFO] Iniciando servidor FastAPI en http://{settings.api_host}:{settings.api_port}...")
            if ensure_server_ready():
                status = get_server_status() if get_server_status else None
                if status:
                    print(f"[OK] Servidor FastAPI iniciado exitosamente")
                    print(f"[INFO] URL: {status.get('url', 'N/A')}")
                    print(f"[INFO] Documentación: {status.get('url', 'N/A')}/docs")
                else:
                    print("[OK] Servidor FastAPI iniciado exitosamente")
            else:
                print("[WARNING] No se pudo iniciar el servidor FastAPI automáticamente")
                print("[INFO] Puedes iniciarlo manualmente con: python start.py")
        
        print("="*70)
        time.sleep(1)  # Pausa breve para que el usuario vea el mensaje
        
    def limpiar_pantalla(self):
        """Limpiar la pantalla."""
        os.system('clear' if os.name != 'nt' else 'cls')
    
    def mostrar_banner(self):
        """Mostrar banner del menú."""
        print("\n" + "="*70)
        print(" " * 15 + "BIBLIOMETRÍA APP - PUNTO DE ENTRADA UNIFICADO")
        print("="*70)
        print("\nRequerimiento 1: Automatización de descarga y unificación de datos")
        print("Requerimiento 2: Análisis de similitud textual con 6 algoritmos")
        print("Requerimiento 3: Análisis de frecuencia de palabras")
        print("Requerimiento 4: Agrupamiento jerárquico de abstracts")
        print("Requerimiento 5: Análisis visual (mapas, nubes, líneas temporales)")
        print("API REST: Servidor FastAPI con endpoints para todos los servicios")
        print("="*70 + "\n")
    
    def mostrar_menu_principal(self):
        """Mostrar menú principal."""
        print("\n[MENÚ PRINCIPAL]")
        print("-" * 70)
        print("1. Probar Web Scraping y Generar Resultados (Requerimiento 1)")
        print("2. Evaluar Algoritmos de Similitud Textual (Requerimiento 2)")
        print("3. Análisis de Frecuencia de Palabras (Requerimiento 3)")
        print("4. Agrupamiento Jerárquico de Abstracts (Requerimiento 4)")
        print("5. Análisis Visual (Requerimiento 5)")
        print("6. Salir")
        print("-" * 70)
    
    def mostrar_submenu_scraping(self):
        """Mostrar submenú de scraping."""
        print("\n[WEB SCRAPING Y GENERACIÓN DE RESULTADOS]")
        print("-" * 70)
        print("1. Ejecutar proceso completo de automatización")
        print("2. Ver resultados generados")
        print("3. Volver al menú principal")
        print("-" * 70)
    
    def mostrar_submenu_similitud(self):
        """Mostrar submenú de similitud."""
        print("\n[ANÁLISIS DE SIMILITUD TEXTUAL]")
        print("-" * 70)
        print("1. Seleccionar archivo CSV unificado")
        print("2. Seleccionar artículos y analizar")
        print("3. Ver algoritmos disponibles")
        print("4. Volver al menú principal")
        print("-" * 70)
    
    def ejecutar_proceso_automatizacion(self):
        """Ejecutar el proceso completo de automatización."""
        print("\n" + "="*70)
        print("PROCESO DE AUTOMATIZACIÓN - REQUERIMIENTO 1")
        print("="*70)
        
        # Solicitar parámetros
        print("\nParámetros de configuración:")
        query = input("Consulta de búsqueda [generative artificial intelligence]: ").strip()
        if not query:
            query = "generative artificial intelligence"
        
        try:
            max_articles = input("Artículos por fuente [350]: ").strip()
            max_articles = int(max_articles) if max_articles else 350
        except ValueError:
            max_articles = 350
        
        try:
            threshold = input("Umbral de similitud para duplicados [0.75]: ").strip()
            threshold = float(threshold) if threshold else 0.75
        except ValueError:
            threshold = 0.75
        
        print(f"\n[INFO] Iniciando proceso de automatización...")
        print(f"  - Consulta: {query}")
        print(f"  - Artículos por fuente: {max_articles}")
        print(f"  - Umbral de similitud: {threshold}")
        print(f"  - Fuentes: OpenAlex, PubMed, ArXiv")
        print("\n[INFO] Esto puede tardar varios minutos...")
        
        # Ejecutar proceso
        resultado = self.unification_service.run_automated_process(
            base_query=query,
            similarity_threshold=threshold,
            max_articles_per_source=max_articles
        )
        
        if resultado['success']:
            print("\n" + "="*70)
            print("PROCESO COMPLETADO EXITOSAMENTE")
            print("="*70)
            print(f"\nEstadísticas:")
            print(f"  - Total descargados: {resultado['total_articles_downloaded']}")
            print(f"  - Artículos únicos: {resultado['unique_articles']}")
            print(f"  - Duplicados eliminados: {resultado['duplicates_removed']}")
            print(f"  - Tiempo de procesamiento: {resultado['processing_time_seconds']:.2f} segundos")
            
            print(f"\nArchivos generados:")
            print(f"  - Archivo unificado: {resultado['unified_file']}")
            print(f"  - Archivo de duplicados: {resultado['duplicates_file']}")
            
            # Verificar estructura de directorios
            print(f"\nEstructura de archivos:")
            self.mostrar_estructura_resultados()
        else:
            print(f"\n[ERROR] Error en el proceso: {resultado.get('error', 'Error desconocido')}")
        
        input("\nPresiona Enter para continuar...")
    
    def mostrar_estructura_resultados(self):
        """Mostrar estructura de directorios de resultados."""
        base_dir = Path(settings.results_dir)
        
        print(f"\n  {base_dir}/")
        
        # Raw data
        raw_dir = base_dir / "raw_data"
        if raw_dir.exists():
            csv_files = list(raw_dir.glob("*.csv"))
            print(f"  ├── raw_data/ ({len(csv_files)} archivos)")
            for csv in csv_files[-3:]:  # Mostrar últimos 3
                print(f"  │   └── {csv.name}")
        
        # Unified
        unified_dir = base_dir / "unified"
        if unified_dir.exists():
            csv_files = list(unified_dir.glob("*.csv"))
            print(f"  ├── unified/ ({len(csv_files)} archivos)")
            for csv in csv_files[-3:]:  # Mostrar últimos 3
                size_kb = csv.stat().st_size / 1024
                print(f"  │   └── {csv.name} ({size_kb:.1f} KB)")
        
        # Duplicates
        duplicates_dir = base_dir / "duplicates"
        if duplicates_dir.exists():
            csv_files = list(duplicates_dir.glob("*.csv"))
            print(f"  ├── duplicates/ ({len(csv_files)} archivos)")
            for csv in csv_files[-3:]:  # Mostrar últimos 3
                size_kb = csv.stat().st_size / 1024
                print(f"  │   └── {csv.name} ({size_kb:.1f} KB)")
        
        # Reports
        reports_dir = base_dir / "reports"
        if reports_dir.exists():
            csv_files = list(reports_dir.glob("*.csv"))
            print(f"  └── reports/ ({len(csv_files)} archivos)")
            for csv in csv_files[-3:]:  # Mostrar últimos 3
                size_kb = csv.stat().st_size / 1024
                print(f"      └── {csv.name} ({size_kb:.1f} KB)")
    
    def listar_csvs_unificados(self) -> List[Dict[str, Any]]:
        """Listar CSVs unificados disponibles."""
        csvs = get_unified_csv_list()
        return csvs
    
    def seleccionar_csv(self) -> Optional[str]:
        """Permitir al usuario seleccionar un CSV."""
        csvs = self.listar_csvs_unificados()
        
        if not csvs:
            print("\n[ERROR] No se encontraron archivos CSV unificados.")
            print("[INFO] Ejecuta primero el proceso de automatización.")
            input("\nPresiona Enter para continuar...")
            return None
        
        print("\n" + "="*70)
        print("ARCHIVOS CSV UNIFICADOS DISPONIBLES")
        print("="*70)
        
        for i, csv_info in enumerate(csvs, 1):
            from datetime import datetime
            fecha = datetime.fromtimestamp(csv_info['modified']).strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n{i}. {csv_info['filename']}")
            print(f"   Ruta: {csv_info['filepath']}")
            print(f"   Tamaño: {csv_info['size_kb']:.1f} KB")
            print(f"   Modificado: {fecha}")
        
        try:
            opcion = input(f"\nSelecciona un archivo (1-{len(csvs)}): ").strip()
            idx = int(opcion) - 1
            if 0 <= idx < len(csvs):
                return csvs[idx]['filepath']
            else:
                print("[ERROR] Opción inválida")
                return None
        except (ValueError, IndexError):
            print("[ERROR] Opción inválida")
            return None
    
    def seleccionar_articulos(self, csv_path: str) -> List[int]:
        """Permitir al usuario seleccionar artículos del CSV."""
        try:
            df = self.text_extractor.read_unified_csv(csv_path)
            
            print("\n" + "="*70)
            print(f"ARTÍCULOS DISPONIBLES ({len(df)} total)")
            print("="*70)
            
            # Mostrar primeros 20 artículos
            mostrar = min(20, len(df))
            for i in range(mostrar):
                titulo = df.iloc[i]['title'][:60] + "..." if len(df.iloc[i]['title']) > 60 else df.iloc[i]['title']
                print(f"{i+1:3d}. {titulo}")
            
            if len(df) > 20:
                print(f"\n... y {len(df) - 20} artículos más")
            
            print("\nSelecciona 2 o más artículos (ej: 1,2,3 o 1-5):")
            seleccion = input("Artículos: ").strip()
            
            indices = self._parse_seleccion(seleccion, len(df))
            
            if len(indices) < 2:
                print("[ERROR] Debes seleccionar al menos 2 artículos")
                return []
            
            print(f"\n[OK] Seleccionados {len(indices)} artículos: {[i+1 for i in indices]}")
            return indices
            
        except Exception as e:
            print(f"[ERROR] Error leyendo CSV: {e}")
            return []
    
    def _parse_seleccion(self, seleccion: str, max_indices: int) -> List[int]:
        """Parsear string de selección (ej: "1,2,3" o "1-5")."""
        indices = set()
        
        for part in seleccion.split(','):
            part = part.strip()
            if '-' in part:
                # Rango
                inicio, fin = part.split('-', 1)
                try:
                    inicio_idx = int(inicio.strip()) - 1
                    fin_idx = int(fin.strip()) - 1
                    for i in range(min(inicio_idx, fin_idx), max(inicio_idx, fin_idx) + 1):
                        if 0 <= i < max_indices:
                            indices.add(i)
                except ValueError:
                    pass
            else:
                # Índice individual
                try:
                    idx = int(part) - 1
                    if 0 <= idx < max_indices:
                        indices.add(idx)
                except ValueError:
                    pass
        
        return sorted(list(indices))
    
    def mostrar_algoritmos_disponibles(self):
        """Mostrar información sobre los algoritmos disponibles."""
        print("\n" + "="*70)
        print("ALGORITMOS DE SIMILITUD TEXTUAL DISPONIBLES")
        print("="*70)
        
        algoritmos = [
            {
                'nombre': '1. Levenshtein (Distancia de Edición)',
                'tipo': 'Clásico - Distancia de edición',
                'descripcion': 'Calcula el número mínimo de operaciones (inserción, eliminación, sustitución) necesarias para convertir un texto en otro.'
            },
            {
                'nombre': '2. Damerau-Levenshtein',
                'tipo': 'Clásico - Distancia de edición',
                'descripcion': 'Similar a Levenshtein pero incluye transposición de caracteres adyacentes como operación adicional.'
            },
            {
                'nombre': '3. Jaccard',
                'tipo': 'Clásico - Vectorización estadística',
                'descripcion': 'Mide la similitud entre dos conjuntos usando la intersección sobre la unión de n-gramas.'
            },
            {
                'nombre': '4. TF-IDF Cosine Similarity',
                'tipo': 'Clásico - Vectorización estadística',
                'descripcion': 'Usa Term Frequency-Inverse Document Frequency para vectorizar textos y calcula similitud del coseno.'
            },
            {
                'nombre': '5. Sentence-BERT',
                'tipo': 'IA - Embeddings semánticos',
                'descripcion': 'Usa modelos transformer pre-entrenados para generar embeddings semánticos y calcular similitud.'
            },
            {
                'nombre': '6. LLM-based Similarity',
                'tipo': 'IA - Similitud contextual',
                'descripcion': 'Simula análisis basado en modelos de lenguaje grandes para capturar similitud semántica profunda.'
            }
        ]
        
        for algo in algoritmos:
            print(f"\n{algo['nombre']}")
            print(f"  Tipo: {algo['tipo']}")
            print(f"  Descripción: {algo['descripcion']}")
        
        print("\n" + "="*70)
        print("\n[INFO] Todos los algoritmos proporcionan explicación detallada paso a paso")
        print("      con detalles matemáticos y algorítmicos.")
        input("\nPresiona Enter para continuar...")
    
    def analizar_similitud_articulos(self, csv_path: str, indices: List[int]):
        """Analizar similitud entre artículos seleccionados."""
        try:
            df = self.text_extractor.read_unified_csv(csv_path)
            articles_data = self.text_extractor.extract_abstracts(df, indices)
            
            if len(articles_data) < 2:
                print("[ERROR] Necesitas al menos 2 artículos para comparar")
                return
            
            print("\n" + "="*70)
            print("ANÁLISIS DE SIMILITUD TEXTUAL")
            print("="*70)
            
            # Mostrar artículos seleccionados
            print("\nArtículos seleccionados:")
            for art in articles_data:
                print(f"\n  Artículo {art['index']+1}:")
                print(f"    Título: {art['title'][:70]}...")
                print(f"    Abstract (primeros 100 chars): {art['abstract'][:100]}...")
            
            # Menú de algoritmos
            print("\n" + "="*70)
            print("Selecciona algoritmo(s) a ejecutar:")
            print("  1. Todos los algoritmos")
            print("  2. Solo algoritmos clásicos (1-4)")
            print("  3. Solo algoritmos de IA (5-6)")
            print("  4. Levenshtein")
            print("  5. Damerau-Levenshtein")
            print("  6. Jaccard")
            print("  7. TF-IDF Cosine")
            print("  8. Sentence-BERT")
            print("  9. LLM-based")
            print("  0. Volver")
            
            opcion = input("\nOpción: ").strip()
            
            textos = [art['abstract'] for art in articles_data]
            
            if opcion == "1":
                self._ejecutar_todos_algoritmos(textos, articles_data)
            elif opcion == "2":
                self._ejecutar_algoritmos_clasicos(textos, articles_data)
            elif opcion == "3":
                self._ejecutar_algoritmos_ia(textos, articles_data)
            elif opcion == "4":
                self._ejecutar_algoritmo_individual("levenshtein", textos, articles_data)
            elif opcion == "5":
                self._ejecutar_algoritmo_individual("damerau", textos, articles_data)
            elif opcion == "6":
                self._ejecutar_algoritmo_individual("jaccard", textos, articles_data)
            elif opcion == "7":
                self._ejecutar_algoritmo_individual("tfidf", textos, articles_data)
            elif opcion == "8":
                self._ejecutar_algoritmo_individual("sbert", textos, articles_data)
            elif opcion == "9":
                self._ejecutar_algoritmo_individual("llm", textos, articles_data)
            else:
                return
            
            input("\nPresiona Enter para continuar...")
            
        except Exception as e:
            print(f"[ERROR] Error en análisis: {e}")
            import traceback
            traceback.print_exc()
            input("\nPresiona Enter para continuar...")
    
    def _ejecutar_todos_algoritmos(self, textos: List[str], articles_data: List[Dict]):
        """Ejecutar todos los algoritmos."""
        algoritmos = [
            ("Levenshtein", self.similarity_service.levenshtein_similarity),
            ("Damerau-Levenshtein", self.similarity_service.damerau_levenshtein_similarity),
            ("Jaccard", lambda t1, t2: self.similarity_service.jaccard_similarity(t1, t2, n=3)),
            ("TF-IDF Cosine", self.similarity_service.tfidf_cosine_similarity),
            ("Sentence-BERT", self.similarity_service.sentence_bert_similarity),
            ("LLM-based", self.similarity_service.llm_based_similarity),
        ]
        
        for nombre, algoritmo in algoritmos:
            print(f"\n\n{'='*70}")
            print(f"ALGORITMO: {nombre.upper()}")
            print("="*70)
            
            # Comparar cada par
            for i in range(len(textos)):
                for j in range(i + 1, len(textos)):
                    resultado = algoritmo(textos[i], textos[j])
                    self._mostrar_resultado_detallado(resultado, articles_data[i], articles_data[j])
    
    def _ejecutar_algoritmos_clasicos(self, textos: List[str], articles_data: List[Dict]):
        """Ejecutar solo algoritmos clásicos."""
        algoritmos = [
            ("Levenshtein", self.similarity_service.levenshtein_similarity),
            ("Damerau-Levenshtein", self.similarity_service.damerau_levenshtein_similarity),
            ("Jaccard", lambda t1, t2: self.similarity_service.jaccard_similarity(t1, t2, n=3)),
            ("TF-IDF Cosine", self.similarity_service.tfidf_cosine_similarity),
        ]
        
        for nombre, algoritmo in algoritmos:
            print(f"\n\n{'='*70}")
            print(f"ALGORITMO: {nombre.upper()}")
            print("="*70)
            
            for i in range(len(textos)):
                for j in range(i + 1, len(textos)):
                    resultado = algoritmo(textos[i], textos[j])
                    self._mostrar_resultado_detallado(resultado, articles_data[i], articles_data[j])
    
    def _ejecutar_algoritmos_ia(self, textos: List[str], articles_data: List[Dict]):
        """Ejecutar solo algoritmos de IA."""
        algoritmos = [
            ("Sentence-BERT", self.similarity_service.sentence_bert_similarity),
            ("LLM-based", self.similarity_service.llm_based_similarity),
        ]
        
        for nombre, algoritmo in algoritmos:
            print(f"\n\n{'='*70}")
            print(f"ALGORITMO: {nombre.upper()}")
            print("="*70)
            
            for i in range(len(textos)):
                for j in range(i + 1, len(textos)):
                    resultado = algoritmo(textos[i], textos[j])
                    self._mostrar_resultado_detallado(resultado, articles_data[i], articles_data[j])
    
    def _ejecutar_algoritmo_individual(self, tipo: str, textos: List[str], articles_data: List[Dict]):
        """Ejecutar un algoritmo individual."""
        algoritmos_map = {
            "levenshtein": ("Levenshtein", self.similarity_service.levenshtein_similarity),
            "damerau": ("Damerau-Levenshtein", self.similarity_service.damerau_levenshtein_similarity),
            "jaccard": ("Jaccard", lambda t1, t2: self.similarity_service.jaccard_similarity(t1, t2, n=3)),
            "tfidf": ("TF-IDF Cosine", self.similarity_service.tfidf_cosine_similarity),
            "sbert": ("Sentence-BERT", self.similarity_service.sentence_bert_similarity),
            "llm": ("LLM-based", self.similarity_service.llm_based_similarity),
        }
        
        if tipo not in algoritmos_map:
            print(f"[ERROR] Algoritmo desconocido: {tipo}")
            return
        
        nombre, algoritmo = algoritmos_map[tipo]
        
        print(f"\n\n{'='*70}")
        print(f"ALGORITMO: {nombre.upper()}")
        print("="*70)
        
        for i in range(len(textos)):
            for j in range(i + 1, len(textos)):
                resultado = algoritmo(textos[i], textos[j])
                self._mostrar_resultado_detallado(resultado, articles_data[i], articles_data[j])
    
    def _mostrar_resultado_detallado(self, resultado, art1: Dict, art2: Dict):
        """Mostrar resultado detallado de un algoritmo."""
        print(f"\n{'-'*70}")
        print(f"Comparación: Artículo {art1['index']+1} vs Artículo {art2['index']+1}")
        print(f"{'-'*70}")
        print(f"\n📊 Score de Similitud: {resultado.similarity_score:.4f}")
        print(f"⏱️  Tiempo de procesamiento: {resultado.processing_time:.4f} segundos")
        
        print(f"\n📝 Explicación Detallada:")
        print(f"{'='*70}")
        print(resultado.explanation)
        
        print(f"\n🔍 Detalles Adicionales:")
        print(f"{'='*70}")
        for key, value in resultado.details.items():
            if value is not None:
                if isinstance(value, list) and len(value) > 10:
                    print(f"  {key}: [Lista con {len(value)} elementos - mostrando primeros 10]")
                    print(f"    {value[:10]}")
                elif isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in list(value.items())[:5]:  # Mostrar primeros 5
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
        
        print(f"\n{'-'*70}\n")
    
    def ejecutar_analisis_frecuencia(self):
        """Ejecutar análisis de frecuencia de palabras (Requerimiento 3)."""
        print("\n" + "="*70)
        print("ANÁLISIS DE FRECUENCIA DE PALABRAS - REQUERIMIENTO 3")
        print("="*70)
        
        # Seleccionar CSV
        csv_path = self.seleccionar_csv()
        if not csv_path:
            input("\nPresiona Enter para continuar...")
            return
        
        print("\n[INFO] Analizando frecuencia de palabras...")
        print("[INFO] Esto puede tardar unos momentos...")
        
        try:
            # Solicitar parámetros
            category = input("\nCategoría de análisis [Generative AI in Education]: ").strip()
            if not category:
                category = "Generative AI in Education"
            
            try:
                max_words = input("Máximo de palabras asociadas [15]: ").strip()
                max_words = int(max_words) if max_words else 15
            except ValueError:
                max_words = 15
            
            # Ejecutar análisis
            resultado = self.word_frequency_service.analyze_word_frequency(
                csv_path=csv_path,
                category=category,
                max_associated_words=max_words
            )
            
            # Mostrar resultados
            print("\n" + "="*70)
            print("RESULTADOS DEL ANÁLISIS DE FRECUENCIA")
            print("="*70)
            print(f"\n📊 Categoría: {resultado.category}")
            print(f"📄 Total de artículos analizados: {resultado.total_articles}")
            print(f"📝 Total de palabras analizadas: {resultado.total_words_analyzed}")
            
            print(f"\n🔤 Palabras de la categoría ({len(resultado.category_words)}):")
            print("-" * 70)
            for word in sorted(resultado.category_words):
                freq = resultado.category_frequencies.get(word, 0)
                print(f"  • {word}: {freq} apariciones")
            
            print(f"\n🔗 Palabras asociadas (Top {len(resultado.associated_words)}):")
            print("-" * 70)
            for word, freq, precision in resultado.associated_words:
                print(f"  • {word}: {freq} apariciones (precisión: {precision:.2%})")
            
            # Obtener top palabras de abstracts
            print(f"\n📈 Top palabras en abstracts:")
            print("-" * 70)
            top_abstracts = self.word_frequency_service.get_top_words_from_fields(
                field="abstract",
                top_n=15,
                csv_path=csv_path
            )
            for word, count in top_abstracts:
                print(f"  • {word}: {count} apariciones")
            
            print("\n" + "="*70)
            print("[OK] Análisis completado exitosamente")
            print("="*70)
            
        except Exception as e:
            print(f"\n[ERROR] Error durante el análisis: {e}")
            import traceback
            traceback.print_exc()
        
        input("\nPresiona Enter para continuar...")
    
    def ejecutar_agrupamiento_jerarquico(self):
        """Ejecutar agrupamiento jerárquico (Requerimiento 4)."""
        print("\n" + "="*70)
        print("AGRUPAMIENTO JERÁRQUICO DE ABSTRACTS - REQUERIMIENTO 4")
        print("="*70)
        
        # Seleccionar CSV
        csv_path = self.seleccionar_csv()
        if not csv_path:
            input("\nPresiona Enter para continuar...")
            return
        
        print("\n[INFO] Configurando parámetros de clustering...")
        
        try:
            # Solicitar parámetros
            try:
                limit = input("Límite de documentos [None = todos]: ").strip()
                limit = int(limit) if limit else None
            except ValueError:
                limit = None
            
            try:
                max_features = input("Máximo de características TF-IDF [1500]: ").strip()
                max_features = int(max_features) if max_features else 1500
            except ValueError:
                max_features = 1500
            
            methods_input = input("Métodos de linkage [single,complete,average]: ").strip()
            if methods_input:
                methods = [m.strip() for m in methods_input.split(",")]
            else:
                methods = ["single", "complete", "average"]
            
            try:
                distance_threshold = input("Umbral de distancia [1.0]: ").strip()
                distance_threshold = float(distance_threshold) if distance_threshold else 1.0
            except ValueError:
                distance_threshold = 1.0
            
            print("\n[INFO] Ejecutando agrupamiento jerárquico...")
            print("[INFO] Esto puede tardar varios minutos dependiendo del tamaño del dataset...")
            
            # Ejecutar clustering
            resultados = self.clustering_service.perform_hierarchical_clustering(
                csv_path=csv_path,
                limit=limit,
                max_features=max_features,
                methods=methods,
                distance_threshold=distance_threshold
            )
            
            # Mostrar resultados
            print("\n" + "="*70)
            print("RESULTADOS DEL AGRUPAMIENTO JERÁRQUICO")
            print("="*70)
            
            best_method = None
            best_correlation = float("-inf")
            
            for method, resultado in resultados.items():
                print(f"\n📊 Método: {method.upper()} (métrica: {resultado.metric})")
                print("-" * 70)
                print(f"  • Dendrograma: {resultado.dendrogram_path}")
                print(f"  • Número de clusters: {resultado.cluster_count}")
                print(f"  • Correlación cophenética: {resultado.cophenetic_correlation:.4f}")
                
                if resultado.cophenetic_correlation > best_correlation:
                    best_correlation = resultado.cophenetic_correlation
                    best_method = method
            
            if best_method:
                print(f"\n🏆 Mejor método: {best_method.upper()} (correlación: {best_correlation:.4f})")
                print(f"   Dendrograma: {resultados[best_method].dendrogram_path}")
            
            print("\n" + "="*70)
            print("[OK] Agrupamiento completado exitosamente")
            print("="*70)
            
        except Exception as e:
            print(f"\n[ERROR] Error durante el agrupamiento: {e}")
            import traceback
            traceback.print_exc()
        
        input("\nPresiona Enter para continuar...")
    
    def ejecutar_analisis_visual(self):
        """Ejecutar análisis visual completo (Requerimiento 5)."""
        print("\n" + "="*70)
        print("ANÁLISIS VISUAL - REQUERIMIENTO 5")
        print("="*70)
        
        # Seleccionar CSV
        csv_path = self.seleccionar_csv()
        if not csv_path:
            input("\nPresiona Enter para continuar...")
            return
        
        print("\n[INFO] Configurando análisis visual...")
        
        try:
            # Solicitar parámetros
            try:
                limit = input("Límite de artículos [None = todos]: ").strip()
                limit = int(limit) if limit else None
            except ValueError:
                limit = None
            
            export_pdf_input = input("Exportar a PDF [S/n]: ").strip().lower()
            export_pdf = export_pdf_input != 'n'
            
            print("\n[INFO] Generando visualizaciones...")
            print("[INFO] Esto incluye:")
            print("  • Mapa de calor geográfico")
            print("  • Nubes de palabras (abstracts, keywords, combinada)")
            print("  • Línea temporal de publicaciones")
            if export_pdf:
                print("  • Exportación a PDF")
            print("\n[INFO] Esto puede tardar varios minutos...")
            
            # Suprimir logs JSON durante la ejecución
            import logging
            import structlog
            original_level = logging.getLogger().level
            logging.getLogger().setLevel(logging.WARNING)
            
            try:
                # Ejecutar visualizaciones
                resultado = self.visualization_service.generate_all_visualizations(
                    csv_path=csv_path,
                    limit=limit,
                    export_pdf=export_pdf
                )
            finally:
                # Restaurar nivel de logging
                logging.getLogger().setLevel(original_level)
            
            # Mostrar resultados
            print("\n" + "="*70)
            print("RESULTADOS DEL ANÁLISIS VISUAL")
            print("="*70)
            print(f"\n🗺️  Mapa de calor geográfico:")
            print(f"   {resultado.heatmap_path}")
            
            print(f"\n☁️  Nubes de palabras:")
            for field, path in resultado.wordcloud_paths.items():
                print(f"   • {field}: {path}")
            
            print(f"\n📈 Línea temporal:")
            print(f"   {resultado.timeline_path}")
            
            if resultado.pdf_path:
                print(f"\n📄 Reporte PDF combinado:")
                print(f"   {resultado.pdf_path}")
            
            print("\n" + "="*70)
            print("[OK] Análisis visual completado exitosamente")
            print("="*70)
            
        except Exception as e:
            print(f"\n[ERROR] Error durante el análisis visual: {e}")
            import traceback
            traceback.print_exc()
        
        input("\nPresiona Enter para continuar...")
    
    def ejecutar(self):
        """Ejecutar el menú principal."""
        csv_seleccionado = None
        
        while True:
            self.limpiar_pantalla()
            self.mostrar_banner()
            self.mostrar_menu_principal()
            
            opcion = input("\nSelecciona una opción: ").strip()
            
            if opcion == "1":
                # Submenú de scraping
                while True:
                    self.limpiar_pantalla()
                    self.mostrar_submenu_scraping()
                    sub_opcion = input("\nOpción: ").strip()
                    
                    if sub_opcion == "1":
                        self.ejecutar_proceso_automatizacion()
                        csv_seleccionado = None  # Resetear selección
                    elif sub_opcion == "2":
                        self.mostrar_estructura_resultados()
                        input("\nPresiona Enter para continuar...")
                    elif sub_opcion == "3":
                        break
                    else:
                        print("[ERROR] Opción inválida")
            
            elif opcion == "2":
                # Submenú de similitud
                while True:
                    self.limpiar_pantalla()
                    self.mostrar_submenu_similitud()
                    
                    if csv_seleccionado:
                        print(f"\n[INFO] CSV seleccionado: {Path(csv_seleccionado).name}")
                    
                    sub_opcion = input("\nOpción: ").strip()
                    
                    if sub_opcion == "1":
                        csv_seleccionado = self.seleccionar_csv()
                    elif sub_opcion == "2":
                        if not csv_seleccionado:
                            csv_seleccionado = self.seleccionar_csv()
                        if csv_seleccionado:
                            indices = self.seleccionar_articulos(csv_seleccionado)
                            if indices:
                                self.analizar_similitud_articulos(csv_seleccionado, indices)
                    elif sub_opcion == "3":
                        self.mostrar_algoritmos_disponibles()
                    elif sub_opcion == "4":
                        break
                    else:
                        print("[ERROR] Opción inválida")
            
            elif opcion == "3":
                # Análisis de frecuencia de palabras
                self.ejecutar_analisis_frecuencia()
            
            elif opcion == "4":
                # Agrupamiento jerárquico
                self.ejecutar_agrupamiento_jerarquico()
            
            elif opcion == "5":
                # Análisis visual
                self.ejecutar_analisis_visual()
            
            elif opcion == "6":
                print("\n[INFO] Saliendo...")
                break
            else:
                print("[ERROR] Opción inválida")
                input("\nPresiona Enter para continuar...")


def main():
    """Función principal."""
    try:
        menu = MenuPrincipal()
        menu.ejecutar()
    except KeyboardInterrupt:
        print("\n\n[INFO] Programa interrumpido por el usuario")
    except Exception as e:
        print(f"\n[ERROR] Error fatal: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

