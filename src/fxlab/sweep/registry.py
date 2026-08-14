"""Registro de pruebas del barrido de parámetros, sobre SQLite.

Cada combinación de parámetros evaluada se escribe como una fila —
**gane o pierda, incluso sin operaciones**. Nada se descarta. El recuento
total de filas por experimento (`count_trials`) es el dato central para la
fase de validación estadística: sin saber cuántas combinaciones se
probaron no se puede corregir el sesgo de selección por pruebas múltiples.

Por eso cada llamada a `record_trial` hace commit inmediatamente (la
conexión abre en modo autocommit, con journal WAL): si el barrido se
interrumpe a mitad, las filas ya escritas quedan intactas y el recuento
sigue siendo exacto — no hay una transacción larga que perder ni un fichero
de resultados a medio escribir.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    partition TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    data_hash TEXT NOT NULL,
    code_version TEXT,
    params_json TEXT NOT NULL,
    n_trades INTEGER NOT NULL,
    total_return REAL,
    sharpe_annualized REAL,
    max_drawdown REAL,
    profit_factor REAL,
    win_rate REAL,
    expectancy REAL,
    note TEXT
);
CREATE INDEX IF NOT EXISTS idx_trials_experiment ON trials(experiment_id);
"""


def current_code_version() -> str | None:
    """Hash del commit actual (`git rev-parse HEAD`), o `None` si no está disponible."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def hash_dataset(*frames: pd.DataFrame) -> str:
    """Hash estable del contenido (índice + valores) de uno o más DataFrames.

    Sirve para saber si dos filas del registro son comparables: si el
    `data_hash` difiere, no se generaron con exactamente los mismos datos
    (aunque el símbolo, intervalo y rango de fechas coincidan — por
    ejemplo, si se volvió a descargar un mes que antes tenía un hueco).
    """
    hasher = hashlib.sha256()
    for frame in frames:
        hashed = pd.util.hash_pandas_object(frame, index=True)
        hasher.update(hashed.to_numpy().tobytes())
    return hasher.hexdigest()


class TrialRegistry:
    """Conexión al registro SQLite de un fichero. Ver docstring del módulo."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: autocommit, cada execute() se confirma solo.
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._code_version = current_code_version()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> TrialRegistry:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def record_trial(
        self,
        *,
        experiment_id: str,
        symbol: str,
        interval: str,
        partition: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        data_hash: str,
        params: Mapping[str, object],
        n_trades: int,
        total_return: float | None,
        sharpe_annualized: float | None,
        max_drawdown: float | None,
        profit_factor: float | None,
        win_rate: float | None,
        expectancy: float | None,
        note: str | None,
    ) -> None:
        """Escribe una fila para una combinación evaluada. Commit inmediato.

        `n_trades=0` y el resto de métricas en `None` (con `note` indicando
        el motivo) es un resultado válido que se registra igual que
        cualquier otro — nunca se omite.
        """
        self._conn.execute(
            """
            INSERT INTO trials (
                experiment_id, created_at, symbol, interval, partition,
                start_date, end_date, data_hash, code_version, params_json,
                n_trades, total_return, sharpe_annualized, max_drawdown, profit_factor,
                win_rate, expectancy, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                datetime.now(UTC).isoformat(),
                symbol,
                interval,
                partition,
                start_date.isoformat(),
                end_date.isoformat(),
                data_hash,
                self._code_version,
                json.dumps(dict(params), sort_keys=True, default=str),
                n_trades,
                total_return,
                sharpe_annualized,
                max_drawdown,
                profit_factor,
                win_rate,
                expectancy,
                note,
            ),
        )

    def count_trials(self, experiment_id: str) -> int:
        """Recuento fiable de filas registradas para `experiment_id`.

        Consulta directa a la tabla (no un contador en memoria que se
        pueda desincronizar): siempre refleja lo que hay realmente en disco.
        """
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM trials WHERE experiment_id = ?", (experiment_id,)
        )
        row = cursor.fetchone()
        return int(row[0])

    def load_experiment(self, experiment_id: str) -> pd.DataFrame:
        """Resultados de `experiment_id` como DataFrame, uno por fila registrada.

        `params_json` se expande en columnas `param_<nombre>` para que sea
        cómodo filtrar/agrupar por parámetro sin parsear JSON a mano.
        """
        df = pd.read_sql_query(
            "SELECT * FROM trials WHERE experiment_id = ? ORDER BY id",
            self._conn,
            params=(experiment_id,),
        )
        if not df.empty:
            expanded = pd.json_normalize(df["params_json"].apply(json.loads))
            expanded.columns = [f"param_{c}" for c in expanded.columns]
            df = pd.concat([df.drop(columns=["params_json"]), expanded], axis=1)
        return df
