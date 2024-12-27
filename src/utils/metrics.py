import time
import functools
from src.utils.logger import get_logger

logger = get_logger(__name__)

def log_performance(component: str):
    """Decorator to log function execution time and metrics."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                logger.info(
                    f"{component} Performance Metrics | "
                    f"Function: {func.__name__} | "
                    f"Execution Time: {execution_time:.3f}s"
                )
                return result
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                logger.error(
                    f"{component} Failed | "
                    f"Function: {func.__name__} | "
                    f"Execution Time: {execution_time:.3f}s | "
                    f"Error: {str(e)}"
                )
                raise
        return wrapper
    return decorator