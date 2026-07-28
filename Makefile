.PHONY: help install install-dev test lint single replicas aggregate all clean

PY ?= python3
REPS ?= 30

help:                ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:             ## Instala las dependencias de ejecución
	$(PY) -m pip install -r requirements.txt

install-dev:         ## Instala dependencias de ejecución y desarrollo
	$(PY) -m pip install -r requirements-dev.txt

test:                ## Ejecuta la batería de pruebas
	$(PY) -m pytest

lint:                ## Verifica el estilo del código
	$(PY) -m ruff check src tests scripts

single:              ## Corrida única con la semilla de referencia (~30 s)
	$(PY) scripts/run_single.py --config configs/default.yaml

replicas:            ## Ejecuta las REPS réplicas (resumible)
	$(PY) scripts/run_replicas.py --config configs/multiseed.yaml --reps $(REPS)

aggregate:           ## Agrega réplicas, escribe tablas y genera figuras
	$(PY) scripts/aggregate.py --config configs/multiseed.yaml

all: replicas aggregate  ## Reproduce el experimento completo del artículo

clean:               ## Elimina salidas regenerables
	rm -rf results/*.json results/*.csv results/*.log results/replicas figures/*.png figures/*.pdf
	find . -name '__pycache__' -type d -exec rm -rf {} +
