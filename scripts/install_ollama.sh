#!/bin/bash
# Script de instalación de Ollama y modelo para Bibliometría App

set -e

echo "=========================================="
echo "Instalación de Ollama para Bibliometría App"
echo "=========================================="
echo ""

# Verificar si Ollama ya está instalado
if command -v ollama &> /dev/null; then
    echo "✅ Ollama ya está instalado"
    ollama --version
else
    echo "📥 Instalando Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✅ Ollama instalado correctamente"
fi

echo ""
echo "=========================================="
echo "Iniciando servidor Ollama..."
echo "=========================================="

# Iniciar servidor Ollama en background si no está corriendo
if ! pgrep -x "ollama" > /dev/null; then
    echo "🚀 Iniciando servidor Ollama..."
    ollama serve &
    sleep 3
    echo "✅ Servidor Ollama iniciado"
else
    echo "✅ Servidor Ollama ya está corriendo"
fi

echo ""
echo "=========================================="
echo "Descargando modelos..."
echo "=========================================="

# Descargar Llama 3.2 3B (más ligero y rápido)
echo "📥 Descargando Llama 3.2 3B (esto puede tardar varios minutos)..."
ollama pull llama3.2:3b

# Opcional: Descargar Mistral 7B (más pesado pero mejor calidad)
read -p "¿Deseas también descargar Mistral 7B? (s/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
    echo "📥 Descargando Mistral 7B (esto puede tardar varios minutos)..."
    ollama pull mistral:7b
fi

echo ""
echo "=========================================="
echo "Verificando instalación..."
echo "=========================================="

# Verificar que los modelos están disponibles
echo "📋 Modelos disponibles:"
ollama list

echo ""
echo "✅ Instalación completada!"
echo ""
echo "Para usar el modelo en el código, usa:"
echo "  - llama3.2:3b (recomendado para velocidad)"
echo "  - mistral:7b (recomendado para calidad)"

