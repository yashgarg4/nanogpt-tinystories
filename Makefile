# Makefile — convenience shortcuts. On Colab/Linux/macOS `make <target>` works.
#
# On Windows, `make` usually isn't installed; either run the commands shown under
# each target directly, or use the PowerShell venv python:
#     .venv\Scripts\python.exe -m src.data
#
# PY points at the venv's Python. Override on the command line if needed, e.g.:
#     make data PY=python
PY ?= .venv/Scripts/python.exe          # Windows venv layout
# For Linux/macOS/Colab, override with:  make data PY=.venv/bin/python

.PHONY: help setup data data-small verify train generate clean

help:
	@echo "Targets:"
	@echo "  setup       create .venv and install pinned requirements"
	@echo "  data        download + tokenize ALL of TinyStories -> data/*.bin"
	@echo "  data-small  quick subset (20k train / 2k val) to test the pipeline"
	@echo "  verify      decode a random chunk from train.bin (round-trip check)"
	@echo "  train       run the training loop            (Phase 3+)"
	@echo "  generate    sample a story from a checkpoint  (Phase 4+)"
	@echo "  clean       remove generated .bin files and __pycache__"

setup:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.data

data-small:
	$(PY) -m src.data --max-train-docs 20000 --max-val-docs 2000

verify:
	$(PY) -m src.data --verify-only

train:
	$(PY) -m src.train

generate:
	$(PY) -m src.generate --prompt "Once upon a time"

clean:
	rm -f data/*.bin
	rm -rf src/__pycache__ __pycache__
