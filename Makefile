.PHONY: install data train reports examples lint test validate api

install:
	python3 -m venv .venv
	.venv/bin/python -m pip install -e '.[dev]'

data:
	.venv/bin/python scripts/generate_sample_data.py

train:
	LOKY_MAX_CPU_COUNT=8 .venv/bin/f1-podium train

reports:
	MPLCONFIGDIR=/tmp/f1-matplotlib .venv/bin/python scripts/generate_reports.py

examples:
	LOKY_MAX_CPU_COUNT=8 .venv/bin/python scripts/generate_examples.py

lint:
	.venv/bin/ruff check src tests scripts

test:
	LOKY_MAX_CPU_COUNT=8 .venv/bin/python -m pytest --cov=f1_podium --cov-report=term-missing

validate: lint test train reports examples

api:
	.venv/bin/uvicorn f1_podium.api:app --reload

