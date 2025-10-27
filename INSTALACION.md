# 🐍 Guía de Instalación - Entorno Python

## 📋 Pre-requisitos

### **Verificar Python Instalado:**
```bash
python --version
# Debe mostrar Python 3.9+ o 3.10+
```

### **Verificar pip:**
```bash
pip --version
```

---

## 🔧 OPCIÓN 1: Entorno Virtual (RECOMENDADO)

### **Crear y Activar Entorno Virtual:**

```bash
# 1. Navegar al directorio del proyecto
cd /home/miguel/Documentos/GitHub/bibliometria-app

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno virtual (Linux/Mac)
source venv/bin/activate

# En Windows usaría:
# venv\Scripts\activate

# 4. Verificar que estás en el entorno virtual
which python  # Debe mostrar: .../venv/bin/python
```

### **Instalar Dependencias en el Entorno Virtual:**

```bash
# Una vez activado el entorno virtual
pip install -r requirements.txt

# Descargar datos de NLTK
python -m nltk.downloader punkt stopwords

# Verificar instalación
python -c "import fastapi, sklearn, nltk; print('✅ OK')"
```

---

## 🔧 OPCIÓN 2: Instalación Global

### **⚠️ No recomendado (pero funciona):**

```bash
# Instalar directamente en Python global
pip install -r requirements.txt

# Descargar NLTK
python -m nltk.downloader punkt stopwords
```

**Problema:** Puede causar conflictos con otros proyectos

---

## 🔧 OPCIÓN 3: Con Anaconda/Conda

### **Crear Entorno con Conda:**

```bash
# Crear entorno virtual con conda
conda create -n bibliometria-app python=3.10
conda activate bibliometria-app

# Instalar dependencias
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords
```

---

## 📝 Proceso Completo (Desde Cero)

### **Paso a Paso:**

```bash
# 1. Navegar al directorio
cd /home/miguel/Documentos/GitHub/bibliometria-app

# 2. Crear entorno virtual
python3 -m venv venv

# 3. Activar entorno virtual
source venv/bin/activate

# 4. Actualizar pip (opcional pero recomendado)
pip install --upgrade pip

# 5. Instalar dependencias
pip install -r requirements.txt

# 6. Descargar datos de NLTK
python -m nltk.downloader punkt stopwords

# 7. Verificar instalación
python test_similarity_complete.py --help
```

---

## 🎯 Comandos Esenciales

### **Activar Entorno Virtual:**
```bash
# En Linux/Mac
source venv/bin/activate

# En Windows
venv\Scripts\activate
```

### **Desactivar Entorno Virtual:**
```bash
deactivate
```

### **Ver Instalaciones:**
```bash
pip list
```

### **Verificar Instalación de Librerías Específicas:**
```bash
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import sklearn; print('sklearn:', sklearn.__version__)"
python -c "import nltk; print('nltk:', nltk.__version__)"
python -c "import sentence_transformers; print('SBERT: OK')"
```

---

## ⚠️ Problemas Comunes

### **Error: "command not found: python"**
```bash
# Usar python3
python3 -m venv venv
source venv/bin/activate
```

### **Error: "pip not found"**
```bash
# Instalar pip
sudo apt install python3-pip  # Linux
# o
brew install python  # Mac
```

### **Error: "No module named 'nltk'"**
```bash
# Reinstalar
pip install --upgrade nltk
python -m nltk.downloader punkt stopwords
```

### **Error al activar venv**
```bash
# Verificar que el archivo de activación existe
ls venv/bin/activate

# Si no existe, recrear entorno virtual
rm -rf venv
python3 -m venv venv
source venv/bin/activate
```

---

## ✅ Verificación Final

### **Script de Verificación Rápido:**

```bash
#!/bin/bash
echo "Verificando instalación..."

echo "1. Python:"
python --version

echo "2. Dependencias básicas:"
python -c "import fastapi, uvicorn; print('✅ fastapi, uvicorn')"

echo "3. Datos:"
python -c "import pandas, requests; print('✅ pandas, requests')"

echo "4. Similitud textual:"
python -c "import numpy, sklearn; print('✅ numpy, sklearn')"
python -c "import nltk; print('✅ nltk')"

echo "5. IA (opcional):"
python -c "import sentence_transformers; print('✅ sentence-transformers')" 2>/dev/null || echo "⚠️ sentence-transformers no instalado"

echo "✅ Verificación completa"
```

---

## 🚀 Siguiente Paso

Una vez instalado todo:

```bash
# 1. Activar entorno virtual (si usaste uno)
source venv/bin/activate

# 2. Iniciar servidor
python start.py

# 3. En otra terminal, ejecutar pruebas
python test_similarity_complete.py
```

---

## 💡 Tip: Automatizar con Script

Crear `setup.sh`:

```bash
#!/bin/bash
echo "🚀 Configurando entorno..."

# Crear venv si no existe
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activar
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords

echo "✅ Configuración completa"
echo "   Activa el entorno con: source venv/bin/activate"
```

Ejecutar:
```bash
chmod +x setup.sh
./setup.sh
```
