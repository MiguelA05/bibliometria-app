@echo off
REM Script de instalación de Ollama y modelo para Bibliometría App (Windows Batch)
REM Ejecutar haciendo doble clic o desde la línea de comandos

echo ==========================================
echo Instalación de Ollama para Bibliometría App
echo ==========================================
echo.

REM Verificar si Ollama ya está instalado
where ollama >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Ollama ya está instalado
    ollama --version
    goto :check_server
) else (
    echo 📥 Ollama no está instalado
    echo.
    echo Para instalar Ollama en Windows:
    echo 1. Descarga el instalador desde: https://ollama.com/download
    echo 2. Ejecuta el instalador (OllamaSetup.exe)
    echo 3. Sigue las instrucciones del instalador
    echo 4. Reinicia este script después de la instalación
    echo.
    
    set /p OPEN_DOWNLOAD="¿Deseas abrir la página de descarga ahora? (S/N): "
    if /i "%OPEN_DOWNLOAD%"=="S" (
        start https://ollama.com/download
    )
    
    echo.
    echo Presiona cualquier tecla cuando hayas completado la instalación...
    pause >nul
    
    REM Verificar nuevamente
    where ollama >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Ollama instalado correctamente
        ollama --version
    ) else (
        echo ❌ Ollama aún no está instalado. Por favor, instálalo manualmente.
        pause
        exit /b 1
    )
)

:check_server
echo.
echo ==========================================
echo Iniciando servidor Ollama...
echo ==========================================

REM Verificar si el servidor está corriendo (intento simple)
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Servidor Ollama ya está corriendo
    goto :download_models
) else (
    echo 🚀 Iniciando servidor Ollama...
    echo    (En Windows, Ollama generalmente se inicia automáticamente como servicio)
    start /B ollama serve
    timeout /t 5 /nobreak >nul
    
    REM Verificar nuevamente
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Servidor Ollama iniciado
    ) else (
        echo ⚠️  No se pudo verificar el servidor automáticamente
        echo    Asegúrate de que Ollama esté corriendo
    )
)

:download_models
echo.
echo ==========================================
echo Descargando modelos...
echo ==========================================

REM Descargar Llama 3.2 3B
echo 📥 Descargando Llama 3.2 3B (esto puede tardar varios minutos)...
ollama pull llama3.2:3b
if %ERRORLEVEL% EQU 0 (
    echo ✅ Llama 3.2 3B descargado correctamente
) else (
    echo ❌ Error al descargar Llama 3.2 3B
)

REM Opcional: Descargar Mistral 7B
echo.
set /p DOWNLOAD_MISTRAL="¿Deseas también descargar Mistral 7B? (s/N): "
if /i "%DOWNLOAD_MISTRAL%"=="S" (
    echo 📥 Descargando Mistral 7B (esto puede tardar varios minutos)...
    ollama pull mistral:7b
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Mistral 7B descargado correctamente
    ) else (
        echo ❌ Error al descargar Mistral 7B
    )
)

echo.
echo ==========================================
echo Verificando instalación...
echo ==========================================

echo 📋 Modelos disponibles:
ollama list

echo.
echo ✅ Instalación completada!
echo.
echo Para usar el modelo en el código, usa:
echo   - llama3.2:3b (recomendado para velocidad)
echo   - mistral:7b (recomendado para calidad)
echo.
pause

