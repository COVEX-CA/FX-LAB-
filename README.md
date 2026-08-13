# fx-research-lab

Laboratorio de investigación cuantitativa sobre Forex. No es un bot de trading:
no ejecuta órdenes ni se conecta a ningún bróker. El objetivo es evaluar
hipótesis de trading (por ejemplo, cruces de medias móviles) con el rigor
estadístico suficiente para distinguir una ventaja real del ruido.

Ver [`AGENTS.md`](./AGENTS.md) para las reglas del proyecto.

## Instalación

Requiere Python 3.11 y [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Comandos

```bash
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # formato
uv run mypy src/         # tipos
```

## Descargar datos

```bash
uv run python scripts/download_data.py --symbol EUR/USD \
    --start 2024-01-01 --end 2024-03-01 --interval 1MIN --offer-side bid
```

Los datos descargados se cachean en `data/` (particionado por símbolo,
intervalo y lado) y no se versionan en git.

## Ejecutar un barrido de parámetros

```bash
uv run python scripts/run_sweep.py configs/pullback_eurusd_h1.yaml
```

Opera siempre sobre la partición de desarrollo (2004–2019) salvo que se pida
explícitamente `--partition holdout`. Cada combinación evaluada, gane o
pierda, queda registrada en un SQLite (`data/sweep_trials.db` por defecto);
el script no imprime rankings ni "mejores resultados".
