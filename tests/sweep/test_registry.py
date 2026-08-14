from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from fxlab.sweep.registry import TrialRegistry, current_code_version, hash_dataset

# Todos los trials de un mismo experimento comparten el índice de retornos
# (comparten data_hash), así que las series de prueba tienen longitud fija.
_RETURNS_INDEX = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")


def _returns(*, first: float = 0.0) -> pd.Series:
    """Serie de retornos de prueba, longitud fija. `first` la hace distinguible."""
    return pd.Series([first, 0.01, -0.01, 0.02], index=_RETURNS_INDEX)


def _open_with_retry(db_path: Path, timeout: float = 15.0) -> TrialRegistry:
    """Abre el registro reintentando ante `OperationalError` transitorio.

    Por qué hace falta: en Windows, `Popen.kill()` es `TerminateProcess`, que
    libera los handles de fichero del hijo de forma **asíncrona**. Si el padre
    reabre la base microsegundos después, la apertura del `-wal`/`-shm` puede
    chocar con una sharing violation, que SQLite traduce a `SQLITE_IOERR`
    ("disk I/O error"). Es una carrera de liberación de handles del sistema de
    ficheros, no un fallo de recuperación: el estado en disco es recuperable
    (lo demuestra `test_registry_recovers_from_a_wal_without_a_valid_shm`, que
    no depende de ningún proceso). En Linux la ventana no existe.

    Esto NO es un `skip` disfrazado: si se agota el plazo, el test falla. La
    garantía se sigue verificando en ambos sistemas; lo único que se concede
    es el tiempo que el sistema operativo tarda en soltar los ficheros.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            return TrialRegistry(db_path)
        except sqlite3.OperationalError as exc:
            if time.monotonic() >= deadline:
                raise AssertionError(
                    f"el registro no se pudo reabrir en {timeout}s tras el kill "
                    f"({exc}). Agotar este plazo significa que la recuperación WAL "
                    "está realmente rota, no que el sistema de ficheros tardara en "
                    "soltar los handles."
                ) from exc
            time.sleep(0.1)


def _record(reg: TrialRegistry, experiment_id: str, i: int, *, n_trades: int = 0) -> int:
    idx = pd.Timestamp("2024-01-01", tz="UTC")
    return reg.record_trial(
        experiment_id=experiment_id,
        symbol="EUR/USD",
        interval="1HOUR",
        partition="development",
        start_date=idx,
        end_date=idx,
        data_hash="deadbeef",
        params={"i": i, "name": "sma"},
        n_trades=n_trades,
        total_return=0.01 if n_trades else None,
        sharpe_annualized=1.0 if n_trades else None,
        max_drawdown=-0.01 if n_trades else None,
        profit_factor=1.2 if n_trades else None,
        win_rate=0.5 if n_trades else None,
        expectancy=0.001 if n_trades else None,
        note=None if n_trades else "sin operaciones",
        returns=_returns(first=0.001 * i),
    )


def test_count_trials_matches_number_of_recorded_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "trials.db"
    with TrialRegistry(db_path) as reg:
        for i in range(7):
            _record(reg, "exp1", i)
        assert reg.count_trials("exp1") == 7
        # otro experimento no debe contaminar el recuento
        _record(reg, "exp2", 0)
        assert reg.count_trials("exp1") == 7
        assert reg.count_trials("exp2") == 1


def test_zero_trade_combination_is_recorded_not_omitted(tmp_path: Path) -> None:
    db_path = tmp_path / "trials.db"
    with TrialRegistry(db_path) as reg:
        _record(reg, "exp", 0, n_trades=0)
        df = reg.load_experiment("exp")

    assert len(df) == 1
    assert df.iloc[0]["n_trades"] == 0
    assert df.iloc[0]["note"] == "sin operaciones"
    assert pd.isna(df.iloc[0]["total_return"])
    assert pd.isna(df.iloc[0]["sharpe_annualized"])


def test_load_experiment_returns_dataframe_with_expanded_params(tmp_path: Path) -> None:
    db_path = tmp_path / "trials.db"
    with TrialRegistry(db_path) as reg:
        _record(reg, "exp", 0, n_trades=3)
        _record(reg, "exp", 1, n_trades=0)
        df = reg.load_experiment("exp")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "param_i" in df.columns
    assert "param_name" in df.columns
    assert sorted(df["param_i"].tolist()) == [0, 1]
    assert df["code_version"].iloc[0] == current_code_version()


def test_registry_survives_unclean_disconnect_without_losing_committed_rows(
    tmp_path: Path,
) -> None:
    # cada record_trial hace commit inmediato (autocommit); simular una
    # caída abandonando la conexión sin cerrarla no debe perder ni
    # corromper lo ya escrito.
    db_path = tmp_path / "trials.db"
    reg = TrialRegistry(db_path)
    for i in range(5):
        _record(reg, "exp", i)
    del reg  # sin close(): simula un proceso que muere sin apagado limpio

    reopened = TrialRegistry(db_path)
    assert reopened.count_trials("exp") == 5
    assert len(reopened.load_experiment("exp")) == 5
    reopened.close()


def test_registry_survives_hard_kill_mid_write(tmp_path: Path) -> None:
    # interrupción real: un proceso hijo escribe filas una a una y se mata
    # a mitad del barrido (SIGKILL, sin oportunidad de limpieza). Lo que ya
    # estaba comprometido en disco debe seguir intacto y el recuento debe
    # ser exacto para lo efectivamente escrito.
    db_path = tmp_path / "trials.db"
    script = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from fxlab.sweep.registry import TrialRegistry\n"
        "import pandas as pd\n"
        f"reg = TrialRegistry(Path({str(db_path)!r}))\n"
        "idx = pd.Timestamp('2024-01-01', tz='UTC')\n"
        "ret = pd.Series([0.0, 0.01, -0.01], "
        "index=pd.date_range('2024-01-01', periods=3, freq='1h', tz='UTC'))\n"
        "for i in range(20):\n"
        "    reg.record_trial(\n"
        "        experiment_id='crash', symbol='EUR/USD', interval='1HOUR',\n"
        "        partition='development', start_date=idx, end_date=idx,\n"
        "        data_hash='h', params={'i': i}, n_trades=0, total_return=None,\n"
        "        sharpe_annualized=None, max_drawdown=None, profit_factor=None,\n"
        "        win_rate=None, expectancy=None, note='sin operaciones', returns=ret,\n"
        "    )\n"
        "    print(f'WROTE {i}', flush=True)\n"
        "    time.sleep(0.03)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdout=subprocess.PIPE,
        text=True,
    )
    written = 0
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            if line.startswith("WROTE"):
                written = int(line.split()[1]) + 1
                if written >= 4:
                    proc.kill()
                    break
    finally:
        proc.stdout.close()  # suelta el pipe antes de esperar al hijo
    proc.wait(timeout=10)

    assert written >= 4  # el hijo llegó a escribir al menos las filas esperadas

    # Lo escrito vive en el -wal: el fichero principal ni siquiera tiene la
    # tabla todavía. Reabrir aquí ejercita la recuperación WAL de verdad.
    reg = _open_with_retry(db_path)
    count = reg.count_trials("crash")
    reg.close()

    assert count == written


def test_registry_recovers_from_a_wal_without_a_valid_shm(tmp_path: Path) -> None:
    """La garantía de recuperación, sin depender de matar ningún proceso.

    Reproduce a nivel de ficheros el estado que deja una interrupción dura —
    un `-wal` con transacciones confirmadas y un `-shm` inservible — y
    comprueba que al reabrir se recuperan todas las filas. El `-shm` no
    guarda datos: es un índice compartido que SQLite reconstruye desde el
    `-wal`, y es justo el fichero que Windows deja en mal estado tras un
    kill duro. Este test es determinista y se comporta igual en todos los
    sistemas, así que si algún día la recuperación se rompe de verdad,
    falla aquí y no en una carrera de handles.
    """
    source = tmp_path / "live"
    db_path = source / "trials.db"
    registry = TrialRegistry(db_path)
    for i in range(5):
        _record(registry, "crash", i)

    # Copia del estado en disco CON la conexión aún abierta: es la foto que
    # habría quedado si el proceso muriera en este instante. No se cierra la
    # conexión antes de copiar porque `close()` hace checkpoint y volcaría el
    # -wal al fichero principal, que es exactamente lo que no queremos probar.
    crashed = tmp_path / "crashed"
    crashed.mkdir()
    for suffix in ("", "-wal", "-shm"):
        origin = Path(str(db_path) + suffix)
        if origin.exists():
            shutil.copy2(origin, crashed / origin.name)
    registry.close()

    copied_db = crashed / "trials.db"
    assert (crashed / "trials.db-wal").stat().st_size > 0, (
        "el -wal copiado está vacío: la prueba no estaría ejercitando la recuperación"
    )
    # Sin el -shm, que es lo que Windows deja inservible.
    (crashed / "trials.db-shm").unlink(missing_ok=True)

    recovered = TrialRegistry(copied_db)
    assert recovered.count_trials("crash") == 5
    recovered.close()


def test_hash_dataset_is_stable_and_sensitive_to_content(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    df_a = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    df_b = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    df_c = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.1]}, index=idx)

    assert hash_dataset(df_a) == hash_dataset(df_b)
    assert hash_dataset(df_a) != hash_dataset(df_c)


def test_a_trial_row_cannot_exist_without_its_returns_series(tmp_path: Path) -> None:
    """La invariante que justifica meter los retornos en la misma base: no
    puede colarse una fila de trial sin su serie de retornos. Se comprueba en
    los dos únicos puntos por donde podría pasar."""
    db_path = tmp_path / "trials.db"
    idx = pd.Timestamp("2024-01-01", tz="UTC")
    with TrialRegistry(db_path) as reg:
        # 1) A nivel de API: omitir `returns` es un error, no una fila a medias.
        with pytest.raises(TypeError):
            reg.record_trial(  # type: ignore[call-arg]
                experiment_id="e",
                symbol="EUR/USD",
                interval="1HOUR",
                partition="development",
                start_date=idx,
                end_date=idx,
                data_hash="h",
                params={"i": 0},
                n_trades=0,
                total_return=None,
                sharpe_annualized=None,
                max_drawdown=None,
                profit_factor=None,
                win_rate=None,
                expectancy=None,
                note="sin operaciones",
            )
        # 2) A nivel de esquema: `returns_values` es NOT NULL, así que un
        #    INSERT directo sin retornos lo rechaza la propia base de datos.
        #    La invariante es estructural, no una convención del código.
        with pytest.raises(sqlite3.IntegrityError):
            reg._conn.execute(
                "INSERT INTO trials (experiment_id, created_at, symbol, interval, "
                "partition, start_date, end_date, data_hash, params_json, n_trades) "
                "VALUES ('e', 'now', 's', 'i', 'p', 'd', 'd', 'h', '{}', 0)"
            )


def test_record_trial_returns_the_assigned_autoincrement_id(tmp_path: Path) -> None:
    db_path = tmp_path / "trials.db"
    with TrialRegistry(db_path) as reg:
        first = _record(reg, "exp", 0)
        second = _record(reg, "exp", 1)

    assert isinstance(first, int)
    assert second == first + 1


def test_load_returns_matrix_columns_are_trial_ids_and_round_trip_values(tmp_path: Path) -> None:
    db_path = tmp_path / "trials.db"
    with TrialRegistry(db_path) as reg:
        ids = [_record(reg, "exp", i, n_trades=3) for i in range(3)]
        trials = reg.load_experiment("exp")
        matrix = reg.load_returns_matrix("exp")

    # columnas = id de trial (como cadena), en el mismo orden que load_experiment
    assert list(matrix.columns) == [str(i) for i in trials["id"]]
    assert list(matrix.columns) == [str(i) for i in ids]
    # el índice UTC común se conserva intacto
    assert matrix.index.equals(_RETURNS_INDEX)
    # valores exactos, sin transformación: una sola fuente de verdad
    import numpy as np

    for i, trial_id in enumerate(ids):
        np.testing.assert_array_equal(
            matrix[str(trial_id)].to_numpy(), _returns(first=0.001 * i).to_numpy()
        )


def test_returns_stay_coherent_with_trials_after_unclean_disconnect(tmp_path: Path) -> None:
    # cada record_trial confirma trial+retornos de forma atómica; abandonar la
    # conexión sin cerrarla no debe dejar trials sin su matriz de retornos.
    db_path = tmp_path / "trials.db"
    reg = TrialRegistry(db_path)
    for i in range(5):
        _record(reg, "exp", i, n_trades=3)
    del reg  # sin close(): simula un proceso que muere sin apagado limpio

    reopened = TrialRegistry(db_path)
    trials = reopened.load_experiment("exp")
    matrix = reopened.load_returns_matrix("exp")
    reopened.close()

    assert len(trials) == 5
    assert list(matrix.columns) == [str(i) for i in trials["id"]]


def test_record_trial_rejects_returns_length_inconsistent_with_experiment(tmp_path: Path) -> None:
    db_path = tmp_path / "trials.db"
    idx = pd.Timestamp("2024-01-01", tz="UTC")
    with TrialRegistry(db_path) as reg:
        _record(reg, "exp", 0, n_trades=3)  # fija el índice del experimento a 4 barras
        wrong = pd.Series(
            [0.0, 0.0],
            index=pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC"),
        )
        with pytest.raises(ValueError, match="no casan"):
            reg.record_trial(
                experiment_id="exp",
                symbol="EUR/USD",
                interval="1HOUR",
                partition="development",
                start_date=idx,
                end_date=idx,
                data_hash="deadbeef",  # mismo data_hash, longitud incoherente
                params={"i": 99},
                n_trades=0,
                total_return=None,
                sharpe_annualized=None,
                max_drawdown=None,
                profit_factor=None,
                win_rate=None,
                expectancy=None,
                note="sin operaciones",
                returns=wrong,
            )
        # la fila incoherente no llegó a escribirse
        assert reg.count_trials("exp") == 1


def test_load_returns_matrix_rejects_mixed_data_hash(tmp_path: Path) -> None:
    db_path = tmp_path / "trials.db"
    idx = pd.Timestamp("2024-01-01", tz="UTC")
    with TrialRegistry(db_path) as reg:
        _record(reg, "exp", 0, n_trades=3)  # data_hash "deadbeef"
        reg.record_trial(
            experiment_id="exp",
            symbol="EUR/USD",
            interval="1HOUR",
            partition="development",
            start_date=idx,
            end_date=idx,
            data_hash="otro",  # datos distintos: no forman una única matriz
            params={"i": 1},
            n_trades=0,
            total_return=None,
            sharpe_annualized=None,
            max_drawdown=None,
            profit_factor=None,
            win_rate=None,
            expectancy=None,
            note="sin operaciones",
            returns=_returns(),
        )
        with pytest.raises(ValueError, match="data_hash"):
            reg.load_returns_matrix("exp")
