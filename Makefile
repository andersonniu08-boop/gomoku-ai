.PHONY: install install-dev test ui train worker benchmarks

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v --timeout=60

ui:
	python -m ui.server

train:
	python -m selfplay.train

worker:
	python -m selfplay.worker

benchmarks:
	python -m tools.benchmark_runner
	python -m tools.bench_nn_architecture
