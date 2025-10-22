from fastapi import FastAPI
from app.api.endpoints import router as api_router

app = FastAPI(
    title="Bibliometría App",
    description="API para extracción de metadatos de artículos académicos",
    version="1.0.0"
)

# Incluir las rutas de la API
app.include_router(api_router)

@app.get("/")
async def root():
    return {"message": "Bibliometría App - API para extracción de metadatos de artículos académicos"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
