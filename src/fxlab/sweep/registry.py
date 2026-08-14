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

import numpy as np
import pandas as pd

# `returns_values` va en la MISMA fila que las métricas y es NOT NULL: es lo
# que hace estructuralmente imposible tener un trial sin su serie de retornos
# (la BD rechaza el INSERT), o retornos sin su trial (viven en la fila). Los
# timestamps del índice, en cambio, son idénticos para todos los trials de un
# mismo barrido, así que se guardan una sola vez por (experiment_id,
# data_hash) en `returns_index` en vez de repetirlos en cada fila.
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
    note TEXT,
    returns_values BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trials_experiment ON trials(experiment_id);
CREATE TABLE IF NOT EXISTS returns_index (
    experiment_id TEXT NOT NULL,
    data_hash TEXT NOT NULL,
    n INTEGER NOT NULL,
    index_ns BLOB NOT NULL,
    PRIMARY KEY (experiment_id, data_hash)
);
"""


def _encode_index(index: pd.DatetimeIndex) -> bytes:
    """Serializa un `DatetimeIndex` UTC como nanosegundos int64."""
    naive = index.tz_convert("UTC").tz_localize(None)
    return naive.to_numpy(dtype="datetime64[ns]").view("int64").tobytes()


def _decode_index(blob: bytes) -> pd.DatetimeIndex:
    """Reconstruye el `DatetimeIndex` UTC desde los nanosegundos int64."""
    ns = np.frombuffer(blob, dtype="int64")
    return pd.DatetimeIndex(pd.to_datetime(ns, utc=True))


def _encode_returns(returns: pd.Series) -> bytes:
    """Serializa los valores de una serie de retornos como float64."""
    return returns.to_numpy(dtype="float64").tobytes()


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
        returns: pd.Series,
    ) -> int:
        """Escribe una fila para una combinación evaluada. Commit inmediato.

        `n_trades=0` y el resto de métricas en `None` (con `note` indicando
        el motivo) es un resultado válido que se registra igual que
        cualquier otro — nunca se omite. Incluso sin operaciones hay una
        serie de retornos (todo ceros): `returns` es obligatoria y se guarda
        en la misma fila, así que nunca hay un trial sin sus retornos.

        `returns` debe ser exactamente la serie que produjo las métricas de
        arriba (no un recálculo): una sola fuente de verdad. Su índice tiene
        que ser el mismo para todos los trials del experimento —lo garantiza
        que compartan `data_hash`— y se guarda una única vez.

        Args:
            returns: serie de retornos barra a barra, con `DatetimeIndex`
                timezone-aware en UTC. Sus valores se guardan en la fila del
                trial; su índice, una vez por (experiment_id, data_hash).

        Returns:
            El `id` autoincremental asignado a la fila. Es lo que liga el
            trial con su columna en `load_returns_matrix`.

        Raises:
            ValueError: si `returns` no tiene un `DatetimeIndex` UTC, o si su
                longitud no casa con la del índice ya registrado para el
                mismo (experiment_id, data_hash).
        """
        index = returns.index
        if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
            raise ValueError("returns necesita un DatetimeIndex timezone-aware en UTC")

        # El índice es idéntico para todos los trials del mismo barrido, así
        # que se escribe una sola vez: INSERT OR IGNORE es idempotente (lo
        # fija el primer trial, el resto es no-op). Va ANTES del INSERT del
        # trial para que una interrupción nunca deje una fila de trial cuyo
        # índice no esté todavía en disco.
        self._conn.execute(
            "INSERT OR IGNORE INTO returns_index (experiment_id, data_hash, n, index_ns) "
            "VALUES (?, ?, ?, ?)",
            (experiment_id, data_hash, len(returns), _encode_index(index)),
        )
        stored_n = self._conn.execute(
            "SELECT n FROM returns_index WHERE experiment_id = ? AND data_hash = ?",
            (experiment_id, data_hash),
        ).fetchone()[0]
        if int(stored_n) != len(returns):
            raise ValueError(
                f"la serie de retornos tiene {len(returns)} barras pero el índice ya "
                f"registrado para el experimento {experiment_id!r} (data_hash "
                f"{data_hash}) tiene {int(stored_n)}: no casan"
            )

        cursor = self._conn.execute(
            """
            INSERT INTO trials (
                experiment_id, created_at, symbol, interval, partition,
                start_date, end_date, data_hash, code_version, params_json,
                n_trades, total_return, sharpe_annualized, max_drawdown, profit_factor,
                win_rate, expectancy, note, returns_values
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                _encode_returns(returns),
            ),
        )
        row_id = cursor.lastrowid
        if row_id is None:  # pragma: no cover - sqlite siempre asigna rowid en un INSERT
            raise RuntimeError("el INSERT no devolvió un rowid")
        return int(row_id)

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
        cómodo filtrar/agrupar por parámetro sin parsear JSON a mano. El blob
        `returns_values` se descarta: esta es la vista de métricas; la matriz
        de retornos barra a barra se obtiene con `load_returns_matrix`.
        """
        df = pd.read_sql_query(
            "SELECT * FROM trials WHERE experiment_id = ? ORDER BY id",
            self._conn,
            params=(experiment_id,),
        )
        df = df.drop(columns=["returns_values"], errors="ignore")
        if not df.empty:
            expanded = pd.json_normalize(df["params_json"].apply(json.loads))
            expanded.columns = [f"param_{c}" for c in expanded.columns]
            df = pd.concat([df.drop(columns=["params_json"]), expanded], axis=1)
        return df

    def load_returns_matrix(self, experiment_id: str) -> pd.DataFrame:
        """Matriz T×N de retornos barra a barra de `experiment_id`.

        Cada columna es el `id` de trial (como cadena) y va en el mismo orden
        que `load_experiment`; el índice es el `DatetimeIndex` UTC común a
        todos los trials del barrido. Es exactamente el contrato que espera
        `fxlab.reporting`: liga cada configuración con su curva de equity por
        `id`, sin emparejar por posición.

        Raises:
            ValueError: si el experimento no tiene pruebas, si mezcla varios
                `data_hash` (no se puede formar una única matriz coherente),
                o si algún trial guarda un número de retornos que no casa con
                el índice del experimento.
        """
        rows = self._conn.execute(
            "SELECT id, data_hash, returns_values FROM trials WHERE experiment_id = ? ORDER BY id",
            (experiment_id,),
        ).fetchall()
        if not rows:
            raise ValueError(f"el experimento {experiment_id!r} no tiene ninguna prueba registrada")

        data_hashes = {row[1] for row in rows}
        if len(data_hashes) != 1:
            raise ValueError(
                f"el experimento {experiment_id!r} mezcla {len(data_hashes)} conjuntos de "
                "datos distintos (data_hash): no se puede formar una única matriz de retornos"
            )
        data_hash = next(iter(data_hashes))

        index_row = self._conn.execute(
            "SELECT index_ns FROM returns_index WHERE experiment_id = ? AND data_hash = ?",
            (experiment_id, data_hash),
        ).fetchone()
        if index_row is None:  # pragma: no cover - la invariante lo impide
            raise ValueError(
                f"faltan los timestamps del experimento {experiment_id!r} (data_hash {data_hash})"
            )
        index = _decode_index(index_row[0])

        columns: dict[str, np.ndarray] = {}
        for trial_id, _, returns_blob in rows:
            values = np.frombuffer(returns_blob, dtype="float64")
            if len(values) != len(index):
                raise ValueError(
                    f"el trial {trial_id} guarda {len(values)} retornos pero el índice del "
                    f"experimento tiene {len(index)}: no casan"
                )
            columns[str(trial_id)] = values
        return pd.DataFrame(columns, index=index)
