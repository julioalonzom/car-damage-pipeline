from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import get_settings
from src.api.endpoints import damage_router

settings = get_settings()

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="API for car damage detection and cost estimation",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json"
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with actual frontend URL
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(
        damage_router,
        prefix=settings.API_V1_STR + "/damage",
        tags=["damage"]
    )

    return app

app = create_app()

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy"}