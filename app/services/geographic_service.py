"""
Servicio para extracción y procesamiento de datos geográficos de artículos académicos.
Proporciona información geográfica compatible con herramientas de visualización de mapas de calor.
"""

import requests
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from app.utils.logger import get_logger
from app.config import settings


class GeographicDataService:
    """Servicio para extraer y procesar datos geográficos de OpenAlex."""
    
    def __init__(self):
        self.logger = get_logger("geographic_service")
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': settings.openalex_user_agent
        })
        
        # Cache para coordenadas geográficas
        self.coordinates_cache = {}
        
    def extract_geographic_data(self, work: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extraer datos geográficos de un trabajo de OpenAlex.
        
        Args:
            work: Datos del trabajo de OpenAlex
            
        Returns:
            Diccionario con información geográfica estructurada
        """
        try:
            geographic_data = {
                'author_countries': [],
                'author_cities': [],
                'institution_countries': [],
                'institution_cities': [],
                'geographic_coordinates': []
            }
            
            # Extraer información de autores e instituciones
            authorships = work.get('authorships', [])
            
            for authorship in authorships:
                # Información del autor
                author = authorship.get('author', {})
                if author:
                    # Obtener información geográfica del autor
                    author_geo = self._get_author_geographic_info(author)
                    if author_geo:
                        geographic_data['author_countries'].extend(author_geo.get('countries', []))
                        geographic_data['author_cities'].extend(author_geo.get('cities', []))
                
                # Información de instituciones
                institutions = authorship.get('institutions', [])
                for institution in institutions:
                    if institution:
                        inst_geo = self._get_institution_geographic_info(institution)
                        if inst_geo:
                            geographic_data['institution_countries'].extend(inst_geo.get('countries', []))
                            geographic_data['institution_cities'].extend(inst_geo.get('cities', []))
                            
                            # Agregar coordenadas si están disponibles
                            coords = inst_geo.get('coordinates')
                            if coords:
                                geographic_data['geographic_coordinates'].append({
                                    'institution': institution.get('display_name', ''),
                                    'country': inst_geo.get('countries', [''])[0],
                                    'city': inst_geo.get('cities', [''])[0],
                                    'latitude': coords[0],
                                    'longitude': coords[1]
                                })
            
            # Eliminar duplicados y limpiar datos
            geographic_data = self._clean_geographic_data(geographic_data)
            
            return geographic_data
            
        except Exception as e:
            self.logger.error(f"Error extracting geographic data: {e}")
            return {
                'author_countries': [],
                'author_cities': [],
                'institution_countries': [],
                'institution_cities': [],
                'geographic_coordinates': []
            }
    
    def _get_author_geographic_info(self, author: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Obtener información geográfica de un autor."""
        try:
            # Si el autor tiene información geográfica directa
            if 'last_known_institution' in author:
                institution = author['last_known_institution']
                if institution:
                    return self._get_institution_geographic_info(institution)
            
            # Si no, intentar obtener información del autor completo
            author_id = author.get('id')
            if author_id:
                return self._fetch_author_geographic_data(author_id)
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error getting author geographic info: {e}")
            return None
    
    def _get_institution_geographic_info(self, institution: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Obtener información geográfica de una institución."""
        try:
            geo_data = {
                'countries': [],
                'cities': [],
                'coordinates': None
            }
            
            # Extraer país
            country_code = institution.get('country_code')
            if country_code:
                country_name = self._get_country_name(country_code)
                if country_name:
                    geo_data['countries'].append(country_name)
            
            # Extraer ciudad
            city = institution.get('city')
            if city:
                geo_data['cities'].append(city)
            
            # Extraer coordenadas geográficas
            geo = institution.get('geo')
            if geo:
                lat = geo.get('lat')
                lng = geo.get('lng')
                if lat is not None and lng is not None:
                    geo_data['coordinates'] = [lat, lng]
            
            return geo_data
            
        except Exception as e:
            self.logger.debug(f"Error getting institution geographic info: {e}")
            return None
    
    def _fetch_author_geographic_data(self, author_id: str) -> Optional[Dict[str, Any]]:
        """Obtener datos geográficos completos de un autor desde la API."""
        try:
            # Construir URL para obtener datos completos del autor
            url = f"{settings.openalex_base_url}/authors/{author_id.split('/')[-1]}"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            author_data = response.json()
            
            # Extraer información geográfica de la última institución conocida
            last_institution = author_data.get('last_known_institution')
            if last_institution:
                return self._get_institution_geographic_info(last_institution)
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error fetching author geographic data: {e}")
            return None
    
    def _get_country_name(self, country_code: str) -> Optional[str]:
        """Convertir código de país a nombre completo."""
        country_mapping = {
            'US': 'United States',
            'GB': 'United Kingdom',
            'DE': 'Germany',
            'FR': 'France',
            'IT': 'Italy',
            'ES': 'Spain',
            'CA': 'Canada',
            'AU': 'Australia',
            'JP': 'Japan',
            'CN': 'China',
            'IN': 'India',
            'BR': 'Brazil',
            'MX': 'Mexico',
            'AR': 'Argentina',
            'CL': 'Chile',
            'CO': 'Colombia',
            'PE': 'Peru',
            'VE': 'Venezuela',
            'EC': 'Ecuador',
            'UY': 'Uruguay',
            'PY': 'Paraguay',
            'BO': 'Bolivia',
            'CR': 'Costa Rica',
            'PA': 'Panama',
            'GT': 'Guatemala',
            'HN': 'Honduras',
            'SV': 'El Salvador',
            'NI': 'Nicaragua',
            'CU': 'Cuba',
            'DO': 'Dominican Republic',
            'HT': 'Haiti',
            'JM': 'Jamaica',
            'TT': 'Trinidad and Tobago',
            'RU': 'Russia',
            'NL': 'Netherlands',
            'BE': 'Belgium',
            'CH': 'Switzerland',
            'AT': 'Austria',
            'SE': 'Sweden',
            'NO': 'Norway',
            'DK': 'Denmark',
            'FI': 'Finland',
            'PL': 'Poland',
            'CZ': 'Czech Republic',
            'HU': 'Hungary',
            'RO': 'Romania',
            'BG': 'Bulgaria',
            'GR': 'Greece',
            'PT': 'Portugal',
            'IE': 'Ireland',
            'LU': 'Luxembourg',
            'MT': 'Malta',
            'CY': 'Cyprus',
            'EE': 'Estonia',
            'LV': 'Latvia',
            'LT': 'Lithuania',
            'SK': 'Slovakia',
            'SI': 'Slovenia',
            'HR': 'Croatia',
            'BA': 'Bosnia and Herzegovina',
            'RS': 'Serbia',
            'ME': 'Montenegro',
            'MK': 'North Macedonia',
            'AL': 'Albania',
            'XK': 'Kosovo',
            'MD': 'Moldova',
            'UA': 'Ukraine',
            'BY': 'Belarus',
            'GE': 'Georgia',
            'AM': 'Armenia',
            'AZ': 'Azerbaijan',
            'KZ': 'Kazakhstan',
            'UZ': 'Uzbekistan',
            'KG': 'Kyrgyzstan',
            'TJ': 'Tajikistan',
            'TM': 'Turkmenistan',
            'MN': 'Mongolia',
            'KR': 'South Korea',
            'KP': 'North Korea',
            'TW': 'Taiwan',
            'HK': 'Hong Kong',
            'SG': 'Singapore',
            'MY': 'Malaysia',
            'TH': 'Thailand',
            'VN': 'Vietnam',
            'PH': 'Philippines',
            'ID': 'Indonesia',
            'BN': 'Brunei',
            'KH': 'Cambodia',
            'LA': 'Laos',
            'MM': 'Myanmar',
            'BD': 'Bangladesh',
            'LK': 'Sri Lanka',
            'MV': 'Maldives',
            'NP': 'Nepal',
            'BT': 'Bhutan',
            'PK': 'Pakistan',
            'AF': 'Afghanistan',
            'IR': 'Iran',
            'IQ': 'Iraq',
            'SY': 'Syria',
            'LB': 'Lebanon',
            'JO': 'Jordan',
            'IL': 'Israel',
            'PS': 'Palestine',
            'SA': 'Saudi Arabia',
            'AE': 'United Arab Emirates',
            'QA': 'Qatar',
            'BH': 'Bahrain',
            'KW': 'Kuwait',
            'OM': 'Oman',
            'YE': 'Yemen',
            'TR': 'Turkey',
            'EG': 'Egypt',
            'LY': 'Libya',
            'TN': 'Tunisia',
            'DZ': 'Algeria',
            'MA': 'Morocco',
            'SD': 'Sudan',
            'SS': 'South Sudan',
            'ET': 'Ethiopia',
            'ER': 'Eritrea',
            'DJ': 'Djibouti',
            'SO': 'Somalia',
            'KE': 'Kenya',
            'UG': 'Uganda',
            'TZ': 'Tanzania',
            'RW': 'Rwanda',
            'BI': 'Burundi',
            'MW': 'Malawi',
            'ZM': 'Zambia',
            'ZW': 'Zimbabwe',
            'BW': 'Botswana',
            'NA': 'Namibia',
            'ZA': 'South Africa',
            'LS': 'Lesotho',
            'SZ': 'Eswatini',
            'MG': 'Madagascar',
            'MU': 'Mauritius',
            'SC': 'Seychelles',
            'KM': 'Comoros',
            'CV': 'Cape Verde',
            'ST': 'São Tomé and Príncipe',
            'GW': 'Guinea-Bissau',
            'GN': 'Guinea',
            'SL': 'Sierra Leone',
            'LR': 'Liberia',
            'CI': 'Ivory Coast',
            'GH': 'Ghana',
            'TG': 'Togo',
            'BJ': 'Benin',
            'NE': 'Niger',
            'BF': 'Burkina Faso',
            'ML': 'Mali',
            'SN': 'Senegal',
            'GM': 'Gambia',
            'MR': 'Mauritania',
            'TD': 'Chad',
            'CF': 'Central African Republic',
            'CM': 'Cameroon',
            'GQ': 'Equatorial Guinea',
            'GA': 'Gabon',
            'CG': 'Republic of the Congo',
            'CD': 'Democratic Republic of the Congo',
            'AO': 'Angola',
            'MZ': 'Mozambique',
            'MG': 'Madagascar',
            'NZ': 'New Zealand',
            'FJ': 'Fiji',
            'PG': 'Papua New Guinea',
            'SB': 'Solomon Islands',
            'VU': 'Vanuatu',
            'NC': 'New Caledonia',
            'PF': 'French Polynesia',
            'WS': 'Samoa',
            'TO': 'Tonga',
            'KI': 'Kiribati',
            'TV': 'Tuvalu',
            'NR': 'Nauru',
            'PW': 'Palau',
            'FM': 'Micronesia',
            'MH': 'Marshall Islands'
        }
        
        return country_mapping.get(country_code.upper())
    
    def _clean_geographic_data(self, geographic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Limpiar y deduplicar datos geográficos."""
        try:
            # Eliminar duplicados y valores vacíos
            for key in ['author_countries', 'author_cities', 'institution_countries', 'institution_cities']:
                if geographic_data[key]:
                    # Eliminar duplicados manteniendo el orden
                    seen = set()
                    unique_list = []
                    for item in geographic_data[key]:
                        if item and item not in seen:
                            seen.add(item)
                            unique_list.append(item)
                    geographic_data[key] = unique_list
            
            # Limpiar coordenadas duplicadas
            if geographic_data['geographic_coordinates']:
                seen_coords = set()
                unique_coords = []
                for coord in geographic_data['geographic_coordinates']:
                    coord_key = (coord.get('latitude'), coord.get('longitude'))
                    if coord_key not in seen_coords and coord_key != (None, None):
                        seen_coords.add(coord_key)
                        unique_coords.append(coord)
                geographic_data['geographic_coordinates'] = unique_coords
            
            return geographic_data
            
        except Exception as e:
            self.logger.error(f"Error cleaning geographic data: {e}")
            return geographic_data
    
    def get_geographic_summary(self, articles: List[Any]) -> Dict[str, Any]:
        """
        Generar resumen geográfico de una lista de artículos.
        
        Args:
            articles: Lista de artículos con información geográfica
            
        Returns:
            Diccionario con estadísticas geográficas
        """
        try:
            summary = {
                'total_articles': len(articles),
                'countries_count': 0,
                'cities_count': 0,
                'coordinates_count': 0,
                'top_countries': [],
                'top_cities': [],
                'geographic_coverage': {}
            }
            
            all_countries = []
            all_cities = []
            all_coordinates = []
            
            for article in articles:
                # Recopilar países
                if hasattr(article, 'author_countries') and article.author_countries:
                    all_countries.extend(article.author_countries)
                if hasattr(article, 'institution_countries') and article.institution_countries:
                    all_countries.extend(article.institution_countries)
                
                # Recopilar ciudades
                if hasattr(article, 'author_cities') and article.author_cities:
                    all_cities.extend(article.author_cities)
                if hasattr(article, 'institution_cities') and article.institution_cities:
                    all_cities.extend(article.institution_cities)
                
                # Recopilar coordenadas
                if hasattr(article, 'geographic_coordinates') and article.geographic_coordinates:
                    all_coordinates.extend(article.geographic_coordinates)
            
            # Calcular estadísticas
            from collections import Counter
            
            country_counts = Counter(all_countries)
            city_counts = Counter(all_cities)
            
            summary['countries_count'] = len(country_counts)
            summary['cities_count'] = len(city_counts)
            summary['coordinates_count'] = len(all_coordinates)
            summary['top_countries'] = country_counts.most_common(10)
            summary['top_cities'] = city_counts.most_common(10)
            
            # Calcular cobertura geográfica
            summary['geographic_coverage'] = {
                'articles_with_countries': sum(1 for a in articles if 
                    (hasattr(a, 'author_countries') and a.author_countries) or 
                    (hasattr(a, 'institution_countries') and a.institution_countries)),
                'articles_with_cities': sum(1 for a in articles if 
                    (hasattr(a, 'author_cities') and a.author_cities) or 
                    (hasattr(a, 'institution_cities') and a.institution_cities)),
                'articles_with_coordinates': sum(1 for a in articles if 
                    hasattr(a, 'geographic_coordinates') and a.geographic_coordinates)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating geographic summary: {e}")
            return {
                'total_articles': len(articles),
                'countries_count': 0,
                'cities_count': 0,
                'coordinates_count': 0,
                'top_countries': [],
                'top_cities': [],
                'geographic_coverage': {}
            }
    
    def export_geographic_data(self, articles: List[Any], file_path: str):
        """
        Exportar datos geográficos a CSV para análisis de mapas de calor.
        
        Args:
            articles: Lista de artículos con información geográfica
            file_path: Ruta del archivo CSV
        """
        try:
            import pandas as pd
            
            geographic_data = []
            
            for article in articles:
                # Datos básicos del artículo
                base_data = {
                    'title': getattr(article, 'title', ''),
                    'doi': getattr(article, 'doi', ''),
                    'publication_year': getattr(article, 'publication_year', ''),
                    'cited_by_count': getattr(article, 'cited_by_count', 0)
                }
                
                # Datos geográficos de autores
                author_countries = getattr(article, 'author_countries', []) or []
                author_cities = getattr(article, 'author_cities', []) or []
                
                # Datos geográficos de instituciones
                institution_countries = getattr(article, 'institution_countries', []) or []
                institution_cities = getattr(article, 'institution_cities', []) or []
                
                # Coordenadas geográficas
                coordinates = getattr(article, 'geographic_coordinates', []) or []
                
                # Crear registro para cada combinación país-ciudad-coordenada
                if coordinates:
                    for coord in coordinates:
                        geographic_data.append({
                            **base_data,
                            'country': coord.get('country', ''),
                            'city': coord.get('city', ''),
                            'institution': coord.get('institution', ''),
                            'latitude': coord.get('latitude', ''),
                            'longitude': coord.get('longitude', ''),
                            'author_countries': '; '.join(author_countries),
                            'author_cities': '; '.join(author_cities),
                            'institution_countries': '; '.join(institution_countries),
                            'institution_cities': '; '.join(institution_cities)
                        })
                else:
                    # Si no hay coordenadas, crear registro con países disponibles
                    countries = list(set(author_countries + institution_countries))
                    cities = list(set(author_cities + institution_cities))
                    
                    for country in countries:
                        geographic_data.append({
                            **base_data,
                            'country': country,
                            'city': cities[0] if cities else '',
                            'institution': '',
                            'latitude': '',
                            'longitude': '',
                            'author_countries': '; '.join(author_countries),
                            'author_cities': '; '.join(author_cities),
                            'institution_countries': '; '.join(institution_countries),
                            'institution_cities': '; '.join(institution_cities)
                        })
            
            # Exportar a CSV
            df = pd.DataFrame(geographic_data)
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"Exported geographic data for {len(articles)} articles to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting geographic data: {e}")
            raise
    
    def extract_geographic_data_from_affiliation_text(self, affiliations: List[str]) -> Dict[str, List[str]]:
        """
        Extraer datos geográficos de texto libre de afiliaciones (PubMed, etc.).
        
        Args:
            affiliations: Lista de afiliaciones en texto libre
            
        Returns:
            Diccionario con países extraídos
        """
        if not affiliations:
            return {
                'author_countries': [],
                'author_cities': [],
                'institution_countries': [],
                'institution_cities': [],
                'geographic_coordinates': []
            }
        
        countries = set()
        cities = set()
        
        # Lista de países comunes
        known_countries = {
            'United States', 'USA', 'US', 'United Kingdom', 'UK',
            'Canada', 'Australia', 'Germany', 'France', 'Italy', 'Spain',
            'Netherlands', 'Sweden', 'Switzerland', 'Norway', 'Denmark',
            'Finland', 'Belgium', 'Austria', 'Poland', 'Czech Republic',
            'Portugal', 'Ireland', 'Greece', 'Israel', 'Japan', 'China',
            'India', 'South Korea', 'Singapore', 'Taiwan', 'Hong Kong',
            'Thailand', 'Vietnam', 'Malaysia', 'Indonesia', 'Philippines',
            'Pakistan', 'Bangladesh', 'Saudi Arabia', 'United Arab Emirates',
            'Turkey', 'Brazil', 'Mexico', 'Argentina', 'Chile', 'Colombia',
            'South Africa', 'Egypt', 'Nigeria', 'Kenya', 'Ghana', 'Russia',
            'New Zealand', 'Taiwan', 'Belarus', 'Ukraine', 'Romania',
            'Bulgaria', 'Hungary', 'Serbia', 'Croatia', 'Slovenia'
        }
        
        for affiliation in affiliations:
            if not affiliation:
                continue
            
            # Normalizar texto
            text = affiliation.strip()
            
            # Buscar países al final de afiliaciones
            # Patrón: ...University, City, Country o ...University, Country
            country_end_pattern = r',\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\.?\s*$'
            matches = re.findall(country_end_pattern, text)
            
            for match in matches:
                potential_country = match.strip().rstrip('.')
                
                # Verificar si es un país conocido
                for country in known_countries:
                    if potential_country.lower() == country.lower() or \
                       potential_country.lower() in country.lower() or \
                       country.lower() in potential_country.lower():
                        countries.add(country)
                        break
            
            # Buscar países en cualquier parte del texto
            text_upper = text.upper()
            for country in known_countries:
                country_escaped = re.escape(country)
                if re.search(r'\b' + country_escaped + r'\b', text, re.IGNORECASE):
                    countries.add(country)
            
            # Extraer ciudades (patrón simple: City, STATE/Country)
            # Ej: "Harvard Medical School, Boston, MA, USA"
            city_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s+(?:[A-Z]{2}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)$'
            city_matches = re.findall(city_pattern, text)
            for city_match in city_matches:
                city = city_match.strip()
                if city and len(city) < 50:  # Filtrar ciudades muy largas (probablemente países)
                    cities.add(city)
        
        return {
            'author_countries': sorted(list(countries)),
            'author_cities': sorted(list(cities)),
            'institution_countries': sorted(list(countries)),  # En PubMed, mismo origen
            'institution_cities': sorted(list(cities)),
            'geographic_coordinates': []  # No tenemos coordenadas de texto libre
        }





