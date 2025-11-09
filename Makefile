# Makefile for gym-tracker

PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help install run test coverage lint format typecheck precommit hooks

help:
	@echo "Targets: install, run, test, coverage, lint, format, typecheck, precommit, hooks"

install:
	python -m venv .venv || true
	$(PIP) install -r requirements.txt

run:
	shiny run --reload app.py

test:
	$(PYTHON) -m pytest -q

coverage:
	$(PYTHON) -m pytest --cov=gymtracker --cov-report=term-missing

lint:
	ruff check . || true

format:
	ruff format . || true

typecheck:
	mypy gymtracker || true

precommit:
	pre-commit run --all-files || true

hooks:
	pre-commit install
