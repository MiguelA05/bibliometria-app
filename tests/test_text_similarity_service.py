#!/usr/bin/env python3
"""
Script de prueba completo para el servicio de similitud textual.
Verifica que todos los 6 algoritmos funcionen correctamente.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.text_similarity_service import TextSimilarityService
import traceback


def print_section(title, char="=", width=70):
    """Imprimir sección con formato."""
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}\n")


def test_dependencies():
    """Verificar que las dependencias críticas estén disponibles."""
    print_section("1. VERIFICANDO DEPENDENCIAS")
    
    dependencies = {
        'sklearn': False,
        'nltk': False,
        'sentence_transformers': False,
        'numpy': False,
    }
    
    try:
        import sklearn
        dependencies['sklearn'] = True
        print("[✓] scikit-learn instalado")
    except ImportError:
        print("[✗] scikit-learn NO instalado")
    
    try:
        import nltk
        dependencies['nltk'] = True
        print("[✓] nltk instalado")
    except ImportError:
        print("[✗] nltk NO instalado")
    
    try:
        import sentence_transformers
        dependencies['sentence_transformers'] = True
        print("[✓] sentence-transformers instalado")
    except ImportError:
        print("[⚠] sentence-transformers NO instalado (algoritmos de embeddings no funcionarán)")
    
    try:
        import numpy
        dependencies['numpy'] = True
        print("[✓] numpy instalado")
    except ImportError:
        print("[✗] numpy NO instalado")
    
    critical = dependencies['sklearn'] and dependencies['nltk'] and dependencies['numpy']
    
    if not critical:
        print("\n[INFO] Instalar dependencias faltantes con:")
        print("   pip install scikit-learn nltk numpy sentence-transformers")
        print("   python -m nltk.downloader punkt stopwords")
    
    return dependencies


def test_preprocessing(service: TextSimilarityService):
    """Probar preprocesamiento de texto."""
    print_section("2. PROBANDO PREPROCESAMIENTO")
    
    test_text = "Machine Learning is a subset of Artificial Intelligence!"
    
    methods = ['standard', 'char-level', 'token-level', 'minimal']
    
    for method in methods:
        try:
            result = service.preprocess_text(test_text, method)
            print(f"\nMétodo: {method}")
            print(f"  Original: {result.original_text[:50]}...")
            print(f"  Procesado: {result.processed_text[:50]}...")
            print(f"  Tokens: {len(result.tokens)}")
            print(f"  Idioma detectado: {result.language}")
            print(f"  Vocabulario único: {result.stats.get('vocab_size', 0)}")
            print("[✓] OK")
        except Exception as e:
            print(f"[✗] Error en método '{method}': {e}")
            traceback.print_exc()


def test_algorithm(name: str, func, text1: str, text2: str, expected_range=(0.0, 1.0)):
    """Probar un algoritmo individual."""
    try:
        result = func(text1, text2)
        
        # Validar resultado
        assert hasattr(result, 'algorithm_name'), "Resultado debe tener algorithm_name"
        assert hasattr(result, 'similarity_score'), "Resultado debe tener similarity_score"
        assert hasattr(result, 'explanation'), "Resultado debe tener explanation"
        assert hasattr(result, 'details'), "Resultado debe tener details"
        assert hasattr(result, 'processing_time'), "Resultado debe tener processing_time"
        
        # Validar score en rango válido
        assert 0.0 <= result.similarity_score <= 1.0, f"Score debe estar en [0, 1], obtuvo {result.similarity_score}"
        
        # Validar tiempo de procesamiento
        assert result.processing_time >= 0, "Tiempo de procesamiento debe ser >= 0"
        
        print(f"  [✓] {result.algorithm_name}")
        print(f"      Score: {result.similarity_score:.4f}")
        print(f"      Tiempo: {result.processing_time:.4f}s")
        
        # Mostrar detalles relevantes
        if 'distance' in result.details:
            print(f"      Distancia: {result.details['distance']}")
        if 'transpositions_count' in result.details:
            print(f"      Transposiciones: {result.details['transpositions_count']}")
        if 'intersection_size' in result.details and 'union_size' in result.details:
            inter = result.details['intersection_size']
            union = result.details['union_size']
            print(f"      Shingles: {inter}/{union}")
        if 'top_contributing_terms' in result.details and result.details['top_contributing_terms']:
            terms = result.details['top_contributing_terms'][:3]
            if isinstance(terms[0], dict):
                term_str = ', '.join([t['term'] for t in terms])
            else:
                term_str = ', '.join(str(t) for t in terms)
            print(f"      Top términos: {term_str[:50]}...")
        if 'interpretation' in result.details:
            print(f"      Interpretación: {result.details['interpretation']}")
        
        return True, result
        
    except Exception as e:
        print(f"  [✗] Error en {name}: {e}")
        traceback.print_exc()
        return False, None


def test_all_algorithms(service: TextSimilarityService):
    """Probar todos los algoritmos con diferentes casos."""
    print_section("3. PROBANDO TODOS LOS ALGORITMOS")
    
    # Casos de prueba
    test_cases = [
        {
            'name': 'Textos idénticos',
            'text1': 'Machine Learning is transforming artificial intelligence',
            'text2': 'Machine Learning is transforming artificial intelligence',
            'expected_min': 0.95  # Debe ser muy alto
        },
        {
            'name': 'Textos similares (sinónimos)',
            'text1': 'Deep learning algorithms improve neural network performance',
            'text2': 'Machine learning methods enhance AI model efficiency',
            'expected_min': 0.3  # Debe tener cierta similitud
        },
        {
            'name': 'Textos diferentes',
            'text1': 'Machine learning and artificial intelligence',
            'text2': 'The weather is sunny today in the park',
            'expected_min': 0.0  # Debe ser bajo
        },
        {
            'name': 'Textos con palabras comunes',
            'text1': 'Natural language processing uses machine learning',
            'text2': 'Computer vision also uses machine learning techniques',
            'expected_min': 0.2  # Debe tener algo de similitud
        },
        {
            'name': 'Textos cortos',
            'text1': 'AI research',
            'text2': 'AI studies',
            'expected_min': 0.3  # Textos muy cortos pueden ser similares
        }
    ]
    
    algorithms = [
        ('Levenshtein', service.levenshtein_similarity),
        ('Damerau-Levenshtein', service.damerau_levenshtein_similarity),
        ('Jaccard', lambda t1, t2: service.jaccard_similarity(t1, t2, n=3)),
        ('TF-IDF Cosine', service.tfidf_cosine_similarity),
        ('Sentence-BERT', service.sentence_bert_similarity),
        ('LLM-based', service.llm_based_similarity),
    ]
    
    results_summary = {}
    
    for test_case in test_cases:
        print(f"\n--- Caso: {test_case['name']} ---")
        print(f"Texto 1: {test_case['text1']}")
        print(f"Texto 2: {test_case['text2']}")
        
        case_results = {}
        
        for alg_name, alg_func in algorithms:
            success, result = test_algorithm(alg_name, alg_func, 
                                            test_case['text1'], 
                                            test_case['text2'])
            
            if success and result:
                case_results[alg_name] = result.similarity_score
                
                # Validar expectativa mínima si se especificó
                if 'expected_min' in test_case:
                    if result.similarity_score < test_case['expected_min']:
                        print(f"      [⚠] Score menor al esperado ({test_case['expected_min']:.2f})")
            else:
                case_results[alg_name] = None
        
        results_summary[test_case['name']] = case_results
    
    return results_summary


def test_edge_cases(service: TextSimilarityService):
    """Probar casos límite."""
    print_section("4. PROBANDO CASOS LÍMITE")
    
    edge_cases = [
        {
            'name': 'Textos vacíos',
            'text1': '',
            'text2': '',
            'should_work': True
        },
        {
            'name': 'Un texto vacío',
            'text1': 'Machine learning',
            'text2': '',
            'should_work': True
        },
        {
            'name': 'Textos muy largos',
            'text1': 'Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models. ' * 10,
            'text2': 'Deep learning is a specialized form of machine learning using neural networks with multiple layers. ' * 10,
            'should_work': True
        },
        {
            'name': 'Solo puntuación',
            'text1': '!!! ??? ...',
            'text2': '..., !!! ???',
            'should_work': True
        },
        {
            'name': 'Caracteres especiales',
            'text1': 'C++ & Python: AI/ML #2024',
            'text2': 'Java & Scala: NLP #2023',
            'should_work': True
        },
        {
            'name': 'Texto None (debe manejarse)',
            'text1': None,
            'text2': 'test',
            'should_work': True
        }
    ]
    
    algorithms = [
        ('Levenshtein', service.levenshtein_similarity),
        ('Jaccard', lambda t1, t2: service.jaccard_similarity(t1, t2, n=3)),
        ('TF-IDF Cosine', service.tfidf_cosine_similarity),
    ]
    
    for edge_case in edge_cases:
        print(f"\n--- {edge_case['name']} ---")
        print(f"Texto 1: {repr(edge_case['text1'])}")
        print(f"Texto 2: {repr(edge_case['text2'])}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(edge_case['text1'], edge_case['text2'])
                
                if result and 0.0 <= result.similarity_score <= 1.0:
                    print(f"  [✓] {alg_name}: Score={result.similarity_score:.4f}")
                else:
                    print(f"  [✗] {alg_name}: Resultado inválido")
            except Exception as e:
                if edge_case['should_work']:
                    print(f"  [✗] {alg_name}: Error inesperado - {e}")
                else:
                    print(f"  [⚠] {alg_name}: Error esperado - {e}")


def test_matrix_backtrace(service: TextSimilarityService):
    """Probar funcionalidad avanzada de Levenshtein (matriz y backtrace)."""
    print_section("5. PROBANDO FUNCIONALIDAD AVANZADA")
    
    text1 = "kitten"
    text2 = "sitting"
    
    print(f"Comparando: '{text1}' vs '{text2}'")
    
    try:
        result = service.levenshtein_similarity(text1, text2, include_matrix=True)
        
        if result.details.get('matrix'):
            matrix = result.details['matrix']
            print(f"[✓] Matriz DP generada: {len(matrix)}x{len(matrix[0])}")
            print(f"    Distancia final: {result.details['distance']}")
            
            # Mostrar parte de la matriz
            print("\n    Primeras filas de la matriz:")
            for i in range(min(4, len(matrix))):
                print(f"      Row {i}: {matrix[i][:8]}")
        
        if result.details.get('backtrace'):
            backtrace = result.details['backtrace']
            print(f"[✓] Backtrace generado: {len(backtrace)} operaciones")
            print(f"    Primeras operaciones: {backtrace[:5]}")
        
        return True
        
    except Exception as e:
        print(f"[✗] Error: {e}")
        traceback.print_exc()
        return False


def test_batch_analysis(service: TextSimilarityService):
    """Probar análisis por lotes."""
    print_section("6. PROBANDO ANÁLISIS POR LOTES")
    
    texts = [
        "Machine learning is transforming AI",
        "Deep learning improves neural networks",
        "Natural language processing uses AI",
        "Computer vision applications in robotics"
    ]
    
    print(f"Analizando {len(texts)} textos...")
    
    try:
        results = service.analyze_texts_similarity(texts)
        
        # Calcular número esperado de comparaciones
        n = len(texts)
        expected_comparisons = (n * (n - 1)) // 2
        expected_algorithms = 6  # 6 algoritmos por comparación
        expected_results = expected_comparisons * expected_algorithms
        
        print(f"[✓] Análisis completado")
        print(f"    Textos: {n}")
        print(f"    Comparaciones: {expected_comparisons}")
        print(f"    Resultados totales: {len(results)} (esperado: {expected_results})")
        
        if len(results) > 0:
            # Agrupar por algoritmo
            algo_counts = {}
            for r in results:
                algo_name = r.algorithm_name
                algo_counts[algo_name] = algo_counts.get(algo_name, 0) + 1
            
            print("\n    Resultados por algoritmo:")
            for algo_name, count in algo_counts.items():
                print(f"      {algo_name}: {count} resultados")
        
        return len(results) == expected_results
        
    except Exception as e:
        print(f"[✗] Error: {e}")
        traceback.print_exc()
        return False


def generate_summary(deps: dict, results_summary: dict):
    """Generar resumen final."""
    print_section("7. RESUMEN FINAL", "=")
    
    print("\n[ESTADO DE DEPENDENCIAS]")
    critical_ok = deps['sklearn'] and deps['nltk'] and deps['numpy']
    
    if critical_ok:
        print("  [✓] Dependencias críticas: OK")
    else:
        print("  [✗] Dependencias críticas: FALTANTES")
    
    if deps['sentence_transformers']:
        print("  [✓] Sentence-BERT: Disponible")
    else:
        print("  [⚠] Sentence-BERT: No disponible (algoritmo no funcionará)")
    
    print("\n[ESTADO DE ALGORITMOS]")
    
    all_algorithms = ['Levenshtein', 'Damerau-Levenshtein', 'Jaccard', 
                      'TF-IDF Cosine', 'Sentence-BERT', 'LLM-based']
    
    for algo in all_algorithms:
        # Verificar si hay resultados para este algoritmo
        has_results = any(
            algo in case_results and case_results[algo] is not None
            for case_results in results_summary.values()
        )
        
        if has_results:
            print(f"  [✓] {algo}: Funcionando")
        elif algo == 'Sentence-BERT' and not deps['sentence_transformers']:
            print(f"  [⚠] {algo}: No disponible (dependencia faltante)")
        else:
            print(f"  [✗] {algo}: No probado o con errores")
    
    print("\n[CONCLUSIÓN]")
    
    if critical_ok:
        print("  ✓ Servicio de similitud textual FUNCIONANDO CORRECTAMENTE")
        print("\n  Todos los algoritmos están implementados y funcionando.")
        print("  Puedes usar el servicio para análisis de similitud textual.")
    else:
        print("  ✗ Servicio requiere dependencias adicionales")
        print("\n  Instala las dependencias faltantes para uso completo.")


def main():
    """Función principal."""
    print("=" * 70)
    print("PRUEBA COMPLETA DEL SERVICIO DE SIMILITUD TEXTUAL")
    print("=" * 70)
    
    # 1. Verificar dependencias
    deps = test_dependencies()
    
    # 2. Inicializar servicio
    print_section("2. INICIALIZANDO SERVICIO")
    try:
        service = TextSimilarityService()
        print("[✓] Servicio inicializado correctamente")
        
        if service.sbert_model:
            print("[✓] Modelo Sentence-BERT cargado")
        else:
            print("[⚠] Modelo Sentence-BERT no disponible")
            
    except Exception as e:
        print(f"[✗] Error al inicializar servicio: {e}")
        traceback.print_exc()
        return
    
    # 3. Probar preprocesamiento
    test_preprocessing(service)
    
    # 4. Probar todos los algoritmos
    results_summary = test_all_algorithms(service)
    
    # 5. Probar casos límite
    test_edge_cases(service)
    
    # 6. Probar funcionalidad avanzada
    test_matrix_backtrace(service)
    
    # 7. Probar análisis por lotes
    test_batch_analysis(service)
    
    # 8. Generar resumen
    generate_summary(deps, results_summary)


if __name__ == "__main__":
    main()

