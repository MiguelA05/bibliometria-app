#!/usr/bin/env python3
"""Menú interactivo por consola para ejecutar las principales tareas del repo.

Opciones:
1. Descargar articulos cientificos
2. Similitud textual
3. Frecuencia de aparición de palabras en abstract
4. Agrupamiento jerárquico
5. Análisis visual

El script intenta reutilizar las funciones/servicios existentes cuando es
posible; si falta alguna dependencia, avisa y vuelve al menú.
"""
from __future__ import annotations

import sys
from typing import Iterable
import runpy

try:
    import colorama
    colorama.init()
    RESET = colorama.Style.RESET_ALL
    BOLD = colorama.Style.BRIGHT
    RED = colorama.Fore.RED
    GREEN = colorama.Fore.GREEN
    YELLOW = colorama.Fore.YELLOW
    CYAN = colorama.Fore.CYAN
except Exception:
    # Fallback a secuencias ANSI (puede funcionar en consolas modernas)
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"


def colored(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def prompt_choice(prompt: str) -> str:
    try:
        return input(prompt)
    except (KeyboardInterrupt, EOFError):
        print()
        return ""


def handle_download() -> None:
    # Ejecutar el antiguo menú de descarga (menu.py) como módulo para reutilizar su flujo
    print(colored("\n== Abrir menú de descarga integrado ==" , YELLOW))
    try:
        runpy.run_module("menu", run_name="__main__")
    except Exception as e:
        print(colored(f"Error ejecutando menu: {e}", RED))


def handle_similarity() -> None:
    print(colored("\n== Similitud textual ==", YELLOW))
    a = prompt_choice("Texto A: ").strip()
    if not a:
        print("Texto A vacío. Volviendo al menú.")
        return
    b = prompt_choice("Texto B: ").strip()
    if not b:
        print("Texto B vacío. Volviendo al menú.")
        return

    try:
        from app.services.text_similarity_service import TextSimilarityService

        svc = TextSimilarityService()
        # Usar Levenshtein por defecto (disponible sin scikit)
        res = svc.levenshtein_similarity(a, b, include_matrix=False)
        print(colored(f"Algoritmo: {res.algorithm_name}", CYAN))
        print(f"Similarity score: {res.similarity_score:.4f}")
        print(res.explanation)
    except Exception as e:
        print(colored(f"Error ejecutando similitud textual: {e}", RED))


def handle_frequency() -> None:
    # Ejecutar el módulo contadorPalabras como script (llama a su __main__)
    print(colored("\n== Frecuencia de palabras en abstract ==", YELLOW))
    try:
        runpy.run_module("contadorPalabras", run_name="__main__")
    except Exception as e:
        print(colored(f"Error ejecutando contadorPalabras: {e}", RED))


def handle_clustering() -> None:
    # Ejecutar el módulo agrupamientoJerárquico como script (su main)
    print(colored("\n== Agrupamiento jerárquico ==", YELLOW))
    try:
        runpy.run_module("agrupamientoJerárquico", run_name="__main__")
    except Exception as e:
        print(colored(f"Error ejecutando agrupamiento: {e}", RED))


def handle_visual() -> None:
    # Ejecutar el módulo analisisVisual como script (su __main__)
    print(colored("\n== Análisis visual ==", YELLOW))
    try:
        runpy.run_module("analisisVisual", run_name="__main__")
    except Exception as e:
        print(colored(f"Error en análisis visual: {e}", RED))


def run_menu() -> None:
    title = colored(BOLD + "BIBLIOMETRÍA APP - MENÚ" + RESET, GREEN)
    while True:
        print("\n" + title)
        print(colored("Selecciona una opción:", BOLD))
        print(colored("1.", CYAN), "Descargar artículos científicos y similitudes de texto")
        print(colored("2.", CYAN), "Frecuencia de aparición de palabras en abstract")
        print(colored("3.", CYAN), "Agrupamiento jerárquico")
        print(colored("4.", CYAN), "Análisis visual")
        print(colored("q.", YELLOW), "Salir")

        choice = prompt_choice("Opción: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            print("Saliendo...")
            return
        if choice == "1":
            handle_download()
        elif choice == "2":
            handle_frequency()
        elif choice == "3":
            handle_clustering()
        elif choice == "4":
            handle_visual()
        else:
            print(colored("Opción no válida. Intenta de nuevo.", RED))


if __name__ == "__main__":
    try:
        run_menu()
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario. Adiós.")
