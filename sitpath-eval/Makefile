install:
	pip install -e .

lint:
	ruff check .
	black --check .

test:
	pytest -q

clean:
	rm -rf .pytest_cache build dist *.egg-info
