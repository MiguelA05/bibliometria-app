#!/usr/bin/env python3
"""
Pruebas unitarias para el servicio de OpenAlex.
Este es el único servicio de extracción de metadatos académicos.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Agregar el directorio raíz al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.openalex_service import OpenAlexService, fetch_articles_metadata_openalex
from app.models.article import ArticleMetadata

class TestOpenAlexService:
    """Clase de pruebas para el servicio de OpenAlex."""
    
    def test_openalex_service_init(self):
        """Probar inicialización del servicio."""
        service = OpenAlexService(email="test@example.com")
        assert service.base_url == "https://api.openalex.org"
        assert "test@example.com" in service.headers['User-Agent']
    
    def test_openalex_service_init_no_email(self):
        """Probar inicialización sin email."""
        service = OpenAlexService()
        assert service.base_url == "https://api.openalex.org"
        assert "BibliometriaApp/1.0" in service.headers['User-Agent']
    
    @patch('app.services.openalex_service.requests.Session.get')
    def test_search_works_success(self, mock_get):
        """Probar búsqueda exitosa de trabajos."""
        # Mock de respuesta de OpenAlex
        mock_response = Mock()
        mock_response.json.return_value = {
            'results': [
                {
                    'id': 'https://openalex.org/W1234567890',
                    'title': 'Test Article',
                    'abstract_inverted_index': {'This': [0], 'is': [1], 'a': [2], 'test': [3]},
                    'authorships': [
                        {
                            'author': {'display_name': 'John Doe'},
                            'institutions': [{'display_name': 'Test University', 'country_code': 'US'}]
                        }
                    ],
                    'publication_date': '2024-01-01',
                    'publication_year': 2024,
                    'type': 'journal-article',
                    'doi': '10.1000/test',
                    'open_access': {'is_oa': True, 'oa_status': 'gold'},
                    'cited_by_count': 10,
                    'concepts': [{'display_name': 'Machine Learning', 'score': 0.8}],
                    'primary_location': {
                        'source': {
                            'display_name': 'Test Journal',
                            'type': 'journal',
                            'homepage_url': 'https://testjournal.com'
                        },
                        'publisher': 'Test Publisher'
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        service = OpenAlexService()
        articles, csv_path = service.search_works("test query", max_articles=1)
        
        assert len(articles) == 1
        assert articles[0].title == "Test Article"
        assert articles[0].openalex_id == "https://openalex.org/W1234567890"
        assert articles[0].doi == "10.1000/test"
        assert articles[0].is_oa == True
        assert articles[0].cited_by_count == 10
        assert "John Doe" in articles[0].authors
        assert "Test University, US" in articles[0].affiliations
    
    @patch('app.services.openalex_service.requests.Session.get')
    def test_search_works_no_results(self, mock_get):
        """Probar búsqueda sin resultados."""
        mock_response = Mock()
        mock_response.json.return_value = {'results': []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        service = OpenAlexService()
        articles, csv_path = service.search_works("nonexistent query", max_articles=1)
        
        assert len(articles) == 0
        assert csv_path == ""
    
    @patch('app.services.openalex_service.requests.Session.get')
    def test_search_works_with_filters(self, mock_get):
        """Probar búsqueda con filtros."""
        mock_response = Mock()
        mock_response.json.return_value = {'results': []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        service = OpenAlexService()
        filters = {'publication_year': '2024', 'type': 'journal-article'}
        service.search_works("test", max_articles=1, filters=filters)
        
        # Verificar que los filtros se pasaron correctamente
        call_args = mock_get.call_args
        assert 'publication_year' in call_args[1]['params']
        assert 'type' in call_args[1]['params']
    
    def test_extract_abstract(self):
        """Probar extracción de abstract."""
        service = OpenAlexService()
        
        # Test con índice invertido
        work = {
            'abstract_inverted_index': {
                'This': [0], 'is': [1], 'a': [2], 'test': [3], 'abstract': [4]
            }
        }
        abstract = service._extract_abstract(work)
        assert abstract == "This is a test abstract"
        
        # Test con abstract directo
        work = {'abstract': 'Direct abstract text'}
        abstract = service._extract_abstract(work)
        assert abstract == "Direct abstract text"
        
        # Test sin abstract
        work = {}
        abstract = service._extract_abstract(work)
        assert abstract == "Abstract not available"
    
    def test_extract_authors_and_affiliations(self):
        """Probar extracción de autores y afiliaciones."""
        service = OpenAlexService()
        
        work = {
            'authorships': [
                {
                    'author': {'display_name': 'John Doe'},
                    'institutions': [
                        {'display_name': 'Harvard University', 'country_code': 'US'},
                        {'display_name': 'MIT', 'country_code': 'US'}
                    ]
                },
                {
                    'author': {'display_name': 'Jane Smith'},
                    'institutions': [
                        {'display_name': 'Oxford University', 'country_code': 'GB'}
                    ]
                }
            ]
        }
        
        authors, affiliations = service._extract_authors_and_affiliations(work)
        
        assert len(authors) == 2
        assert "John Doe" in authors
        assert "Jane Smith" in authors
        
        assert len(affiliations) == 3
        assert "Harvard University, US" in affiliations
        assert "MIT, US" in affiliations
        assert "Oxford University, GB" in affiliations
    
    def test_extract_publication_date(self):
        """Probar extracción de fecha de publicación."""
        service = OpenAlexService()
        
        # Test con fecha completa
        work = {'publication_date': '2024-01-15'}
        date = service._extract_publication_date(work)
        assert date == "2024-01-15"
        
        # Test con componentes individuales
        work = {
            'publication_year': 2024,
            'publication_month': 1,
            'publication_day': 15
        }
        date = service._extract_publication_date(work)
        assert date == "2024-01-15"
        
        # Test solo con año
        work = {'publication_year': 2024}
        date = service._extract_publication_date(work)
        assert date == "2024"
    
    def test_extract_article_url(self):
        """Probar extracción de URL del artículo."""
        service = OpenAlexService()
        
        # Test con URL de Open Access
        work = {
            'open_access': {'oa_url': 'https://example.com/oa-paper'},
            'primary_location': {'landing_page_url': 'https://example.com/paper'}
        }
        url = service._extract_article_url(work)
        assert url == "https://example.com/oa-paper"
        
        # Test con URL primaria
        work = {
            'primary_location': {'landing_page_url': 'https://example.com/paper'}
        }
        url = service._extract_article_url(work)
        assert url == "https://example.com/paper"
        
        # Test con ID de OpenAlex
        work = {'id': 'https://openalex.org/W1234567890'}
        url = service._extract_article_url(work)
        assert url == "https://openalex.org/W1234567890"
    
    def test_extract_concepts_and_topics(self):
        """Probar extracción de conceptos y temas."""
        service = OpenAlexService()
        
        work = {
            'concepts': [
                {'display_name': 'Machine Learning', 'score': 0.8},
                {'display_name': 'Artificial Intelligence', 'score': 0.7}
            ]
        }
        
        concepts, topics = service._extract_concepts_and_topics(work)
        
        assert len(concepts) == 2
        assert len(topics) == 2
        assert "Machine Learning" in topics
        assert "Artificial Intelligence" in topics
    
    def test_export_to_csv(self):
        """Probar exportación a CSV."""
        service = OpenAlexService()
        
        # Crear artículos de prueba
        articles = [
            ArticleMetadata(
                title="Test Article 1",
                authors=["John Doe"],
                affiliations=["Test University"],
                abstract="Test abstract 1",
                publication_date="2024-01-01",
                article_url="https://example.com/1",
                openalex_id="https://openalex.org/W1",
                doi="10.1000/test1",
                publication_year=2024,
                type="journal-article",
                is_oa=True,
                cited_by_count=10
            ),
            ArticleMetadata(
                title="Test Article 2",
                authors=["Jane Smith"],
                affiliations=["Another University"],
                abstract="Test abstract 2",
                publication_date="2024-01-02",
                article_url="https://example.com/2",
                openalex_id="https://openalex.org/W2",
                doi="10.1000/test2",
                publication_year=2024,
                type="conference-paper",
                is_oa=False,
                cited_by_count=5
            )
        ]
        
        csv_path = service._export_to_csv(articles, "test query")
        
        # Verificar que se creó el archivo
        assert os.path.exists(csv_path)
        assert "openalex" in csv_path
        assert "test_query" in csv_path
        
        # Verificar contenido del CSV
        import pandas as pd
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        assert len(df) == 2
        # Columnas esperadas (sin is_oa, source_title, source_type, publisher, cited_by_count)
        # topics fue renombrado a keywords
        expected_columns = [
            'title', 'authors', 'affiliations', 'abstract', 'publication_date',
            'article_url', 'doi', 'publication_year',
            'type', 'language', 'keywords', 'license', 'journal'
        ]
        # The service may include geographic fields appended to the CSV. Accept
        # both the original expected columns and the extended set with geographic
        # columns (author/institution countries/cities and geographic coordinates) and data_source.
        extended_columns = expected_columns + [
            'author_countries', 'author_cities', 'institution_countries',
            'institution_cities', 'geographic_coordinates', 'data_source'
        ]

        # The DataFrame should match the base schema and optionally include geographic and data_source fields
        cols = list(df.columns)
        
        # Check that all expected base columns are present
        missing_base_cols = [col for col in expected_columns if col not in cols]
        assert len(missing_base_cols) == 0, (
            f"Missing base columns: {missing_base_cols}. Got: {cols}"
        )
        
        # Check that extended columns are optional
        has_extended = all(col in cols for col in ['author_countries', 'author_cities', 'institution_countries', 'institution_cities', 'geographic_coordinates', 'data_source'])
        
        # Limpiar archivo de prueba
        os.remove(csv_path)
    
    def test_fetch_articles_metadata_openalex_function(self):
        """Probar función de conveniencia."""
        with patch('app.services.openalex_service.OpenAlexService.search_works') as mock_search:
            mock_search.return_value = ([], "")
            
            articles, csv_path = fetch_articles_metadata_openalex("test", 5)
            
            assert articles == []
            assert csv_path == ""
            mock_search.assert_called_once_with("test", 5, None)

if __name__ == "__main__":
    # Ejecutar pruebas si se ejecuta directamente
    pytest.main([__file__, "-v"])
