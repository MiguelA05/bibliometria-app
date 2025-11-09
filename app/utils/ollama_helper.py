"""
Utilidades para manejar Ollama: verificación, inicio del servidor y comunicación con modelos.
"""

import subprocess
import requests
import time
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from app.utils.logger import get_logger

logger = get_logger("ollama_helper")

# URL por defecto de Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api"

# Flag para verificar disponibilidad
OLLAMA_AVAILABLE = True  # Se establecerá dinámicamente


def check_ollama_installed() -> bool:
    """Verificar si Ollama está instalado en el sistema."""
    try:
        result = subprocess.run(
            ["which", "ollama"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception as e:
        logger.warning(f"Error verificando instalación de Ollama: {e}")
        return False


def check_ollama_server_running() -> bool:
    """Verificar si el servidor de Ollama está corriendo."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def start_ollama_server() -> bool:
    """Iniciar el servidor de Ollama en background."""
    if check_ollama_server_running():
        logger.info("Servidor Ollama ya está corriendo")
        return True
    
    if not check_ollama_installed():
        logger.error("Ollama no está instalado. Ejecuta: bash scripts/install_ollama.sh")
        return False
    
    try:
        logger.info("Iniciando servidor Ollama...")
        # Iniciar servidor en background
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        # Esperar a que el servidor esté listo (máximo 30 segundos)
        for _ in range(30):
            time.sleep(1)
            if check_ollama_server_running():
                logger.info("Servidor Ollama iniciado correctamente")
                return True
        
        logger.error("Timeout esperando que el servidor Ollama inicie")
        return False
        
    except Exception as e:
        logger.error(f"Error iniciando servidor Ollama: {e}")
        return False


def ensure_ollama_ready() -> bool:
    """Asegurar que Ollama esté instalado y el servidor corriendo."""
    if not check_ollama_installed():
        logger.error("Ollama no está instalado")
        logger.info("Para instalar Ollama, ejecuta:")
        logger.info("  bash scripts/install_ollama.sh")
        return False
    
    if not check_ollama_server_running():
        logger.info("Servidor Ollama no está corriendo, intentando iniciarlo...")
        if not start_ollama_server():
            logger.error("No se pudo iniciar el servidor Ollama")
            logger.info("Intenta iniciarlo manualmente con: ollama serve")
            return False
    
    return True


def list_available_models() -> List[str]:
    """Listar modelos disponibles en Ollama."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return models
        return []
    except Exception as e:
        logger.warning(f"Error listando modelos: {e}")
        return []


def check_model_available(model_name: str) -> bool:
    """Verificar si un modelo específico está disponible."""
    models = list_available_models()
    return model_name in models


def pull_model(model_name: str) -> bool:
    """Descargar un modelo de Ollama."""
    if not ensure_ollama_ready():
        return False
    
    try:
        logger.info(f"Descargando modelo {model_name}...")
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutos máximo
        )
        
        if result.returncode == 0:
            logger.info(f"Modelo {model_name} descargado correctamente")
            return True
        else:
            logger.error(f"Error descargando modelo: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout descargando modelo {model_name}")
        return False
    except Exception as e:
        logger.error(f"Error descargando modelo: {e}")
        return False


def generate_with_ollama(
    prompt: str,
    model: str = "llama3.2:3b",
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 500,
    seed: Optional[int] = None
) -> Optional[str]:
    """
    Generar texto usando Ollama.
    
    Args:
        prompt: Prompt del usuario
        model: Nombre del modelo (default: llama3.2:3b)
        system_prompt: Prompt del sistema (opcional)
        temperature: Temperatura para la generación (0.0-1.0). 0.0 = determinístico
        max_tokens: Máximo número de tokens a generar
        seed: Seed para reproducibilidad (opcional). Si se proporciona, hace la generación determinística
        
    Returns:
        Texto generado o None si hay error
    """
    if not ensure_ollama_ready():
        return None
    
    # Verificar que el modelo esté disponible
    if not check_model_available(model):
        logger.warning(f"Modelo {model} no está disponible, intentando descargarlo...")
        if not pull_model(model):
            logger.error(f"No se pudo descargar el modelo {model}")
            return None
    
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        # Agregar seed si se proporciona (para reproducibilidad)
        if seed is not None:
            payload["options"]["seed"] = seed
        
        if system_prompt:
            payload["system"] = system_prompt
        
        response = requests.post(
            f"{OLLAMA_API_URL}/generate",
            json=payload,
            timeout=120  # 2 minutos máximo
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "").strip()
        else:
            logger.error(f"Error en la API de Ollama: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        logger.error("Timeout esperando respuesta de Ollama")
        return None
    except Exception as e:
        logger.error(f"Error generando con Ollama: {e}")
        return None


def analyze_similarity_with_llm(text1: str, text2: str, model: str = "llama3.2:3b") -> Dict[str, Any]:
    """
    Analizar similitud entre dos textos usando un LLM local.
    
    Args:
        text1: Primer texto
        text2: Segundo texto
        model: Modelo de Ollama a usar
        
    Returns:
        Diccionario con análisis de similitud
    """
    # Prompt más estructurado y determinístico
    system_prompt = """Eres un experto en análisis de similitud textual. 
Tu tarea es analizar dos textos y determinar su similitud semántica y temática de manera consistente y objetiva.

INSTRUCCIONES:
1. Evalúa la similitud considerando: temas comunes, vocabulario compartido, estructura semántica
2. Asigna un score entre 0.0 y 1.0 donde:
   - 0.0-0.2: Textos completamente diferentes
   - 0.2-0.4: Textos con algunos temas comunes pero diferentes enfoques
   - 0.4-0.6: Textos similares con temas relacionados
   - 0.6-0.8: Textos muy similares con temas y enfoques cercanos
   - 0.8-1.0: Textos casi idénticos o muy relacionados
3. Responde EXACTAMENTE en el formato: SCORE: X.XX - JUSTIFICACIÓN (máximo 2 frases)
4. Sé consistente: textos similares deben recibir scores similares"""
    
    # Crear hash de los textos para usar como seed adicional (más determinístico)
    import hashlib
    text_hash = hashlib.md5(f"{text1[:1000]}{text2[:1000]}".encode()).hexdigest()
    # Usar los primeros 8 caracteres del hash como seed adicional
    text_seed = int(text_hash[:8], 16) % (2**31)  # Limitar a rango de int32
    
    user_prompt = f"""Analiza la similitud entre estos dos textos de manera objetiva y consistente:

TEXTO 1:
{text1[:1000]}

TEXTO 2:
{text2[:1000]}

Evalúa la similitud semántica y temática. Responde EXACTAMENTE en el formato:
SCORE: X.XX - JUSTIFICACIÓN"""
    
    # Usar temperatura muy baja y seed basado en el contenido para consistencia
    # El seed se basa en el hash de los textos, así que textos idénticos siempre dan el mismo resultado
    response = generate_with_ollama(
        prompt=user_prompt,
        model=model,
        system_prompt=system_prompt,
        temperature=0.0,  # Temperatura 0 para máxima consistencia (determinístico)
        max_tokens=200,
        seed=text_seed  # Seed basado en el contenido de los textos para reproducibilidad
    )
    
    if not response:
        return {
            "score": 0.0,
            "justification": "Error al comunicarse con el modelo LLM",
            "raw_response": None
        }
    
    # Parsear respuesta
    score = 0.0
    justification = response.strip()
    
    # Intentar extraer el score y justificación
    import re
    
    # Buscar patrón "SCORE: X.XX - JUSTIFICACIÓN"
    score_pattern = re.search(r'SCORE:\s*([0-9]*\.?[0-9]+)', response, re.IGNORECASE)
    if score_pattern:
        try:
            score = float(score_pattern.group(1))
            score = max(0.0, min(1.0, score))  # Asegurar rango [0, 1]
            
            # Extraer justificación (todo después del guion o después del score)
            # Buscar el guion después del score
            dash_match = re.search(r'SCORE:\s*[0-9]*\.?[0-9]+\s*-\s*(.+)', response, re.IGNORECASE | re.DOTALL)
            if dash_match:
                justification = dash_match.group(1).strip()
            else:
                # Si no hay guion, tomar todo después del score
                score_end = score_pattern.end()
                if score_end < len(response):
                    justification = response[score_end:].strip()
                    # Limpiar posibles prefijos como "-" o ":"
                    justification = re.sub(r'^[-:\s]+', '', justification)
        except ValueError:
            # Si falla el parsing, intentar método alternativo
            pass
    
    # Método alternativo: buscar número decimal al inicio
    if score == 0.0:
        match = re.search(r'\b([01]\.\d+|\d+\.\d+)\b', response)
        if match:
            try:
                score = float(match.group(1))
                score = max(0.0, min(1.0, score))
                # Intentar extraer justificación después del número
                num_end = match.end()
                if num_end < len(response):
                    remaining = response[num_end:].strip()
                    # Buscar guion o dos puntos
                    if re.match(r'^[-:\s]+', remaining):
                        justification = re.sub(r'^[-:\s]+', '', remaining).strip()
            except ValueError:
                pass
    
    return {
        "score": score,
        "justification": justification,
        "raw_response": response
    }

