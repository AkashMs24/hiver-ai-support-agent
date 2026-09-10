# Makefile for Hiver AI Support Agent

.PHONY: help setup test eval demo reproduce clean

help:
	@echo "Available commands:"
	@echo "  make setup      Install dependencies and create data dirs"
	@echo "  make test       Run unit tests"
	@echo "  make eval       Run the comprehensive evaluation harness"
	@echo "  make demo       Launch the interactive CLI demo"
	@echo "  make reproduce  End-to-end setup and evaluation run (< 15 mins)"
	@echo "  make clean      Clean temporary cache and pycache files"

setup:
	pip install -e .
	python data/golden/create_golden_set.py

test:
	pytest tests/ -v

eval:
	python -m eval.harness

demo:
	python -m src.demo

reproduce: setup eval demo

clean:
	rm -rf __pycache__ .pytest_cache *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
