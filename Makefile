.PHONY: install lint format test pipeline docker-build docker-run clean

install:
	pip install --upgrade pip
	pip install -r requirements-dev.txt
	pip install -e .

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest -v

pipeline:
	python -m fraud_rings.pipeline --accounts 3000 --rings 12 --out artifacts/

docker-build:
	docker build -t fraud-rings:latest .

docker-run:
	docker run --rm -p 8000:8000 --name fraud-api fraud-rings:latest

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache artifacts/
	find . -type d -name __pycache__ -exec rm -rf {} +
