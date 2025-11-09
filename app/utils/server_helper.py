"""
Utilidades para manejar el servidor FastAPI: verificación, inicio y configuración.
"""

import subprocess
import requests
import time
import os
import sys
from pathlib import Path
from typing import Optional
from app.utils.logger import get_logger
from app.config import settings

logger = get_logger("server_helper")

# URL por defecto del servidor
SERVER_BASE_URL = f"http://{settings.api_host}:{settings.api_port}"


def check_server_running() -> bool:
    """Verificar si el servidor FastAPI está corriendo."""
    try:
        response = requests.get(f"{SERVER_BASE_URL}/health", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def start_server(host: Optional[str] = None, port: Optional[int] = None, 
                 background: bool = True) -> bool:
    """
    Iniciar el servidor FastAPI en background.
    
    Args:
        host: Host del servidor (default: desde settings)
        port: Puerto del servidor (default: desde settings)
        background: Si True, inicia en background. Si False, bloquea.
        
    Returns:
        True si se inició correctamente, False en caso contrario
    """
    if check_server_running():
        logger.info("Servidor FastAPI ya está corriendo")
        return True
    
    host = host or settings.api_host
    port = port or settings.api_port
    
    try:
        logger.info(f"Iniciando servidor FastAPI en {host}:{port}...")
        
        cmd = [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", host,
            "--port", str(port)
        ]
        
        if background:
            # Iniciar servidor en background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            
            # Esperar a que el servidor esté listo (máximo 30 segundos)
            for _ in range(30):
                time.sleep(1)
                if check_server_running():
                    logger.info(f"Servidor FastAPI iniciado correctamente en {host}:{port}")
                    return True
            
            logger.error("Timeout esperando que el servidor FastAPI inicie")
            return False
        else:
            # Modo bloqueante (no se usa normalmente desde el menú)
            subprocess.run(cmd)
            return True
            
    except Exception as e:
        logger.error(f"Error iniciando servidor FastAPI: {e}")
        return False


def ensure_server_ready(host: Optional[str] = None, port: Optional[int] = None) -> bool:
    """
    Asegurar que el servidor FastAPI esté corriendo.
    
    Args:
        host: Host del servidor (default: desde settings)
        port: Puerto del servidor (default: desde settings)
        
    Returns:
        True si el servidor está listo, False en caso contrario
    """
    if check_server_running():
        return True
    
    logger.info("Servidor FastAPI no está corriendo, intentando iniciarlo...")
    return start_server(host, port, background=True)


def get_server_status() -> dict:
    """
    Obtener el estado del servidor.
    
    Returns:
        Diccionario con información del estado del servidor
    """
    status = {
        "running": False,
        "url": SERVER_BASE_URL,
        "host": settings.api_host,
        "port": settings.api_port
    }
    
    if check_server_running():
        try:
            response = requests.get(f"{SERVER_BASE_URL}/health", timeout=3)
            if response.status_code == 200:
                status["running"] = True
                status["health"] = response.json()
        except Exception as e:
            logger.warning(f"Error obteniendo estado del servidor: {e}")
    
    return status

