#!/usr/bin/env python3
"""
automation_runner.py

Script simple para pruebas de automatización:
- Carga variables de entorno (soporta .env)
- Inicia el servidor local (opcional) y espera a que /health responda
- Ejecuta llamadas a los endpoints:
  - /api/v1/fetch-metadata (por cada query en AUTOMATION_QUERIES o una lista por defecto)
  - /api/v1/uniquindio/generative-ai
  - /api/v1/automation/unified-data
- Muestra un resumen y comprueba que los CSV reportados existen

Uso:
  # desde el venv del proyecto
  python automation_runner.py --start-server
  python automation_runner.py            # asume servidor ya en marcha

Opciones principales:
  --start-server    Inicia uvicorn en background usando el python activo
  --stop-server     Detiene el servidor si fue iniciado por el script
  --queries         Coma-separadas, anulan AUTOMATION_QUERIES en .env
  --timeout         Tiempo máximo (s) para esperar al endpoint /health

"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()  # carga .env si existe

# Configuración desde entorno
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', '8000'))
BASE_URL = f'http://{API_HOST}:{API_PORT}'
RESULTS_DIR = os.getenv('RESULTS_DIR', 'results')
CSV_ENCODING = os.getenv('CSV_ENCODING', 'utf-8-sig')
MAX_ARTICLES_DEFAULT = int(os.getenv('MAX_ARTICLES_DEFAULT', '10'))
AUTOMATION_QUERIES = os.getenv('AUTOMATION_QUERIES', '')

DEFAULT_QUERIES = [q.strip() for q in AUTOMATION_QUERIES.split(',') if q.strip()] or [
    'machine learning',
    'natural language processing',
    'quantum computing'
]

HEALTH_URL = f'{BASE_URL}/health'
FETCH_METADATA_URL = f'{BASE_URL}/api/v1/fetch-metadata'
UNIQUINDIO_URL = f'{BASE_URL}/api/v1/uniquindio/generative-ai'
AUTOMATION_URL = f'{BASE_URL}/api/v1/automation/unified-data'


def start_uvicorn(host: str, port: int, reload: bool = False):
    """Start uvicorn as a background process using the active Python interpreter."""
    py = sys.executable
    args = [py, '-m', 'uvicorn', 'app.main:app', '--host', host, '--port', str(port)]
    if reload:
        args.append('--reload')
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc


def wait_for_health(url: str, timeout: int = 30, interval: float = 0.5):
    """Poll /health until ready or timeout."""
    start = time.time()
    while True:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        if time.time() - start > timeout:
            return False
        time.sleep(interval)


def post_json(url: str, payload: dict, timeout: int = 60):
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Request to {url} failed: {e}")
        return None


def ensure_results_dir():
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)


def run_queries(queries, max_articles=MAX_ARTICLES_DEFAULT):
    results = []
    for q in queries:
        print(f"\n--- Running query: {q} (max_articles={max_articles})")
        payload = {
            'query': q,
            'max_articles': max_articles,
            'email': None,
            'filters': {}
        }
        data = post_json(FETCH_METADATA_URL, payload)
        if not data:
            print(f"Failed to get data for query: {q}")
            continue
        total = data.get('total_articles', 0)
        csv_path = data.get('csv_file_path')
        print(f"Found {total} articles. CSV: {csv_path}")
        # If CSV path is provided, check existence (relative path)
        if csv_path:
            p = Path(csv_path)
            if not p.exists():
                # Try relative to project results dir
                p2 = Path(RESULTS_DIR) / p.name
                exists = p2.exists()
                print(f"CSV exists: {exists} (checked {p} and {p2})")
            else:
                print(f"CSV exists: True ({p})")
        results.append({'query': q, 'total': total, 'csv': csv_path})
    return results


def run_uniquindio(max_articles=10):
    print('\n--- Running University-specific endpoint (uniquindio)')
    payload = {'max_articles': max_articles, 'email': None, 'filters': {}}
    data = post_json(UNIQUINDIO_URL, payload)
    if data:
        info = data.get('research_results', {})
        total = info.get('total_articles', 'N/A') if isinstance(info, dict) else 'N/A'
        csv = info.get('csv_file_path') if isinstance(info, dict) else None
        print(f"University endpoint: total_articles={total}, csv={csv}")
    return data


def run_automation_unified(base_query='generative artificial intelligence', similarity_threshold=0.8, max_articles_per_source=50):
    print('\n--- Running automation unified endpoint')
    payload = {
        'base_query': base_query,
        'similarity_threshold': similarity_threshold,
        'max_articles_per_source': max_articles_per_source
    }
    data = post_json(AUTOMATION_URL, payload, timeout=300)
    if data:
        res = data.get('automation_result', {})
        stats = data.get('data_statistics', {})
        files = data.get('generated_files', {})
        print('Automation result:', res.get('success', True))
        print('Data statistics:', stats)
        print('Generated files:', files)
    return data


def main():
    parser = argparse.ArgumentParser(description='Automation runner for Bibliometria App')
    parser.add_argument('--start-server', action='store_true', help='Start uvicorn server in background')
    parser.add_argument('--stop-server', action='store_true', help='Stop the server started by this script')
    parser.add_argument('--queries', type=str, help='Comma-separated queries to run')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout for health check (seconds)')
    parser.add_argument('--max-articles', type=int, default=MAX_ARTICLES_DEFAULT, help='Max articles per query')
    parser.add_argument('--base-query', type=str, default='generative artificial intelligence')
    parser.add_argument('--similarity', type=float, default=0.8)
    parser.add_argument('--max-per-source', type=int, default=50)
    args = parser.parse_args()

    ensure_results_dir()

    server_proc = None
    started_by_script = False

    if args.start_server:
        print('Starting uvicorn server...')
        server_proc = start_uvicorn(API_HOST, API_PORT, reload=False)
        started_by_script = True
        print('Waiting for /health...')
        ok = wait_for_health(HEALTH_URL, timeout=args.timeout)
        if not ok:
            print('Server did not respond to /health in time. Stderr output:')
            if server_proc and server_proc.stderr:
                try:
                    print(server_proc.stderr.read().decode(errors='replace'))
                except Exception:
                    pass
            if started_by_script and server_proc:
                server_proc.terminate()
            sys.exit(1)
        print('Server healthy.')
    else:
        # if no start_server, still check health before running
        print('Checking /health...')
        if not wait_for_health(HEALTH_URL, timeout=5):
            print(f"No server responding at {HEALTH_URL}. Start the server or run with --start-server")
            sys.exit(1)

    queries = [q.strip() for q in (args.queries.split(',') if args.queries else DEFAULT_QUERIES) if q.strip()]

    # Run tests
    fetch_results = run_queries(queries, max_articles=args.max_articles)
    uni_res = run_uniquindio(max_articles=args.max_articles)
    auto_res = run_automation_unified(base_query=args.base_query, similarity_threshold=args.similarity, max_articles_per_source=args.max_per_source)

    # Summary
    print('\n=== SUMMARY ===')
    for r in fetch_results:
        print(f"Query={r['query']}: total={r['total']}, csv={r['csv']}")
    if uni_res:
        print('Uniquindio endpoint returned data (see above)')
    if auto_res:
        print('Automation unified endpoint returned data (see above)')

    if args.stop_server and started_by_script and server_proc:
        print('Stopping server...')
        server_proc.terminate()
        server_proc.wait(timeout=5)
        print('Server stopped.')

    if started_by_script and server_proc:
        print('\nNote: server was started by this script. Use --stop-server to stop it, or Ctrl+C in the server terminal if run manually.')


if __name__ == '__main__':
    main()
