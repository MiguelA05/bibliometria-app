# Script de instalación de Ollama y modelo para Bibliometría App (Windows PowerShell)
# Ejecutar con: powershell -ExecutionPolicy Bypass -File scripts/install_ollama.ps1

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Instalación de Ollama para Bibliometría App" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar si Ollama ya está instalado
$ollamaInstalled = $false
try {
    $ollamaVersion = ollama --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $ollamaInstalled = $true
        Write-Host "✅ Ollama ya está instalado" -ForegroundColor Green
        Write-Host $ollamaVersion
    }
} catch {
    $ollamaInstalled = $false
}

if (-not $ollamaInstalled) {
    Write-Host "📥 Ollama no está instalado" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Para instalar Ollama en Windows:" -ForegroundColor Yellow
    Write-Host "1. Descarga el instalador desde: https://ollama.com/download" -ForegroundColor Yellow
    Write-Host "2. Ejecuta el instalador (OllamaSetup.exe)" -ForegroundColor Yellow
    Write-Host "3. Sigue las instrucciones del instalador" -ForegroundColor Yellow
    Write-Host "4. Reinicia este script después de la instalación" -ForegroundColor Yellow
    Write-Host ""
    
    $continue = Read-Host "¿Deseas abrir la página de descarga ahora? (S/N)"
    if ($continue -match "^[Ss]$") {
        Start-Process "https://ollama.com/download"
    }
    
    Write-Host ""
    Write-Host "Presiona cualquier tecla cuando hayas completado la instalación..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    
    # Verificar nuevamente después de la instalación
    try {
        $ollamaVersion = ollama --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Ollama instalado correctamente" -ForegroundColor Green
            Write-Host $ollamaVersion
        } else {
            Write-Host "❌ Ollama aún no está instalado. Por favor, instálalo manualmente." -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "❌ Ollama aún no está instalado. Por favor, instálalo manualmente." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Iniciando servidor Ollama..." -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Verificar si el servidor Ollama está corriendo
$serverRunning = $false
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        $serverRunning = $true
    }
} catch {
    $serverRunning = $false
}

if (-not $serverRunning) {
    Write-Host "🚀 Iniciando servidor Ollama..." -ForegroundColor Yellow
    
    # En Windows, Ollama generalmente se ejecuta como servicio o aplicación
    # Intentar iniciar el servidor
    try {
        Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 5
        
        # Verificar si se inició correctamente
        $maxAttempts = 10
        $attempt = 0
        while ($attempt -lt $maxAttempts) {
            try {
                $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 2 -ErrorAction SilentlyContinue
                if ($response.StatusCode -eq 200) {
                    $serverRunning = $true
                    break
                }
            } catch {
                Start-Sleep -Seconds 2
                $attempt++
            }
        }
        
        if ($serverRunning) {
            Write-Host "✅ Servidor Ollama iniciado" -ForegroundColor Green
        } else {
            Write-Host "⚠️  No se pudo verificar el servidor automáticamente" -ForegroundColor Yellow
            Write-Host "   Asegúrate de que Ollama esté corriendo (debería iniciarse automáticamente)" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "⚠️  No se pudo iniciar el servidor automáticamente" -ForegroundColor Yellow
        Write-Host "   En Windows, Ollama generalmente se inicia automáticamente como servicio" -ForegroundColor Yellow
        Write-Host "   Si no está corriendo, inícialo manualmente desde el menú de inicio" -ForegroundColor Yellow
    }
} else {
    Write-Host "✅ Servidor Ollama ya está corriendo" -ForegroundColor Green
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Descargando modelos..." -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Descargar Llama 3.2 3B (más ligero y rápido)
Write-Host "📥 Descargando Llama 3.2 3B (esto puede tardar varios minutos)..." -ForegroundColor Yellow
try {
    ollama pull llama3.2:3b
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Llama 3.2 3B descargado correctamente" -ForegroundColor Green
    } else {
        Write-Host "❌ Error al descargar Llama 3.2 3B" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Error al descargar Llama 3.2 3B: $_" -ForegroundColor Red
}

# Opcional: Descargar Mistral 7B
Write-Host ""
$downloadMistral = Read-Host "¿Deseas también descargar Mistral 7B? (s/N)"
if ($downloadMistral -match "^[Ss]$") {
    Write-Host "📥 Descargando Mistral 7B (esto puede tardar varios minutos)..." -ForegroundColor Yellow
    try {
        ollama pull mistral:7b
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Mistral 7B descargado correctamente" -ForegroundColor Green
        } else {
            Write-Host "❌ Error al descargar Mistral 7B" -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ Error al descargar Mistral 7B: $_" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Verificando instalación..." -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Verificar que los modelos están disponibles
Write-Host "📋 Modelos disponibles:" -ForegroundColor Yellow
try {
    ollama list
} catch {
    Write-Host "❌ Error al listar modelos: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "✅ Instalación completada!" -ForegroundColor Green
Write-Host ""
Write-Host "Para usar el modelo en el código, usa:" -ForegroundColor Cyan
Write-Host "  - llama3.2:3b (recomendado para velocidad)" -ForegroundColor White
Write-Host "  - mistral:7b (recomendado para calidad)" -ForegroundColor White
Write-Host ""

