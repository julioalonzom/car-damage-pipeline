import uvicorn
from src.config import get_settings

settings = get_settings()

if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Enable auto-reload during development
    )