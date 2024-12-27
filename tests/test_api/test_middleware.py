# tests/test_api/test_middleware.py
from fastapi.testclient import TestClient
from src.api.app import create_app
import pytest

def test_request_logging(caplog):
    """Test request logging middleware."""
    client = TestClient(create_app())
    response = client.get("/health")
    assert "Status: 200" in caplog.text