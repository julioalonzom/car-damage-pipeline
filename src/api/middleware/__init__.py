from fastapi import Request
import time
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def log_requests(request: Request, call_next):
    """Log request/response timing and status."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        f"{request.method} {request.url.path} "
        f"Status: {response.status_code} "
        f"Duration: {process_time:.3f}s"
    )
    
    return response