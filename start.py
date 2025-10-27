#!/usr/bin/env python3
"""
Script de inicio para Bibliometría App.
Configura el entorno y ejecuta la aplicación con las mejores prácticas.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_requirements():
    """Verificar que las dependencias estén instaladas."""
    try:
        import fastapi
        import uvicorn
        import requests
        import pandas
        import pydantic
        print("[OK] Todas las dependencias están instaladas")
        return True
    except ImportError as e:
        print(f"[ERROR] Dependencia faltante: {e}")
        print("[INFO] Ejecuta: pip install -r requirements.txt")
        return False

def create_env_file():
    """Crear archivo .env si no existe."""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists() and env_example.exists():
        print("[INFO] Creando archivo .env desde env.example...")
        env_file.write_text(env_example.read_text())
        print("[OK] Archivo .env creado")
    elif not env_file.exists():
        print("[WARNING] No se encontró env.example, creando .env básico...")
        env_file.write_text("""# Configuración básica
API_TITLE="Bibliometría App"
API_DESCRIPTION="API para extracción de metadatos de artículos académicos"
API_VERSION="1.0.0"
API_HOST="0.0.0.0"
API_PORT=8000
LOG_LEVEL="INFO"
DEBUG=false
""")
        print("[OK] Archivo .env básico creado")

def create_directories():
    """Crear directorios necesarios."""
    directories = ["results", "logs"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"[INFO] Directorio {directory} creado/verificado")

def run_tests():
    """Ejecutar pruebas unitarias."""
    print("[TEST] Ejecutando pruebas unitarias...")
    try:
        result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("[OK] Todas las pruebas pasaron")
            return True
        else:
            print("[ERROR] Algunas pruebas fallaron:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"[ERROR] Error ejecutando pruebas: {e}")
        return False

def start_server(host="0.0.0.0", port=8000, reload=False, workers=1):
    """Iniciar el servidor de desarrollo."""
    print(f"[INFO] Iniciando servidor en http://{host}:{port}")
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
        print("[INFO] Modo de recarga automática activado")
    
    if workers > 1:
        cmd.extend(["--workers", str(workers)])
        print(f"[INFO] Ejecutando con {workers} workers")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[INFO] Servidor detenido por el usuario")
    except Exception as e:
        print(f"[ERROR] Error iniciando servidor: {e}")

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Bibliometría App - Script de inicio")
    parser.add_argument("--host", default="0.0.0.0", help="Host del servidor")
    parser.add_argument("--port", type=int, default=8000, help="Puerto del servidor")
    parser.add_argument("--reload", action="store_true", help="Activar recarga automática")
    parser.add_argument("--workers", type=int, default=1, help="Número de workers")
    parser.add_argument("--test", action="store_true", help="Solo ejecutar pruebas")
    parser.add_argument("--setup", action="store_true", help="Solo configurar entorno")
    
    args = parser.parse_args()
    
    print("BIBLIOMETRIA APP - CONFIGURACION INICIAL")
    print("=" * 50)
    
    # Verificar dependencias
    if not check_requirements():
        sys.exit(1)
    
    # Crear archivos de configuración
    create_env_file()
    
    # Crear directorios
    create_directories()
    
    if args.setup:
        print("[OK] Configuración completada")
        return
    
    # Ejecutar pruebas si se solicita
    if args.test:
        if not run_tests():
            sys.exit(1)
        return
    
    # Ejecutar pruebas antes de iniciar
    print("[TEST] Verificando pruebas antes del inicio...")
    if not run_tests():
        print("[ERROR] Pruebas fallaron, abortando.")
        sys.exit(1)
    
    # Iniciar servidor
    print("\nINICIANDO SERVIDOR")
    print("=" * 30)
    start_server(args.host, args.port, args.reload, args.workers)

if __name__ == "__main__":
    main()






