.PHONY: install format lint test clean

install:
	pip install -r requirements.txt

format:
	black src tests
	isort src tests

lint:
	flake8 src tests
	mypy src tests
	black --check src tests
	isort --check src tests

test:
	pytest --cov=src --cov-report=term-missing tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.pyc" -delete
	find . -type d -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +