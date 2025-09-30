# simple make tasks for local dev

SHELL := /bin/sh

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# defaults (override like: make train SYMBOLS="AAPL MSFT" DAYS=365)
SYMBOLS ?= AAPL MSFT
DAYS ?= 365
OUT ?= artifacts/models/tiny.json
PROFILE ?= balanced
SYMBOL ?= AAPL

.PHONY: help venv install upgrade-pip test cli train clean print-python

help:
	@echo "make venv           # create venv (.venv)"
	@echo "make install        # install project (editable) + pytest"
	@echo "make test           # run tests"
	@echo "make cli            # run single-symbol analysis"
	@echo "make train          # train tiny model (stub if no torch)"
	@echo "make backtest       # run multi-symbol backtest"
	@echo "make sp500          # download and save S&P 500 universe"
	@echo "make backtest-sp500 # run backtest over S&P 500 (subset)"
	@echo "make daily          # run daily simulation step and append equity point"
	@echo "make predict        # rank a universe and write picks CSV"
	@echo "make daily-autopilot# generate picks then rebalance once and log equity/trades"
	@echo "make plot-equity    # render artifacts/equity.jsonl to artifacts/equity.png (matplotlib optional)"
	@echo "make print-python   # show which python"
	@echo "make clean          # remove caches/artifacts"

venv:
	python3 -m venv $(VENV)

upgrade-pip: venv
	$(PY) -m pip install --upgrade pip

install: venv upgrade-pip
	$(PY) -m pip install -e .
	$(PY) -m pip install -q pytest

test:
	$(PY) -m pytest -q

cli:
	$(PY) -m src.cli

train:
	$(PY) -m src.train --symbols $(SYMBOLS) --days $(DAYS) --out $(OUT)

backtest:
	$(PY) -m src.backtest_cli --symbols $(SYMBOLS) --cash 10000 --profile $(PROFILE)

sp500:
	$(PY) -m src.tools.fetch_sp500

backtest-sp500:
	$(PY) -m src.backtest_cli --universe sp500 --max-symbols 50 --cash 10000 --profile $(PROFILE) --interval 1d --period 5y

daily:
	$(PY) -m src.runner_daily

predict:
	$(PY) -m src.predict_cli --universe sp500 --max-symbols 200 --top-n 20 --weights score --profile $(PROFILE) --cash 10000

daily-autopilot:
	$(PY) -m src.runner_daily --autopilot --use-picks

plot-equity:
	$(PY) -m src.tools.plot_equity

print-python:
	@echo "Python: $$($(PY) --version)"
	@echo "Path:   $(PY)"

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage* dist build *.egg-info
	rm -rf data/cache
