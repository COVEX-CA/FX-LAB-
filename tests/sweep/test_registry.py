from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from fxlab.sweep.registry import TrialRegistry, current_code_version, hash_dataset


def _record(reg: TrialRegistry, experiment_id: str, i: int, *, n_trades: int = 0) -> None:
    idx = pd.Timestamp("2024-01-01", tz="UTC")
    reg.record_trial(
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
        "for i in range(20):\n"
        "    reg.record_trial(\n"
        "        experiment_id='crash', symbol='EUR/USD', interval='1HOUR',\n"
        "        partition='development', start_date=idx, end_date=idx,\n"
        "        data_hash='h', params={'i': i}, n_trades=0, total_return=None,\n"
        "        sharpe_annualized=None, max_drawdown=None, profit_factor=None,\n"
        "        win_rate=None, expectancy=None, note='sin operaciones',\n"
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
    for line in proc.stdout:
        if line.startswith("WROTE"):
            written = int(line.split()[1]) + 1
            if written >= 4:
                proc.kill()
                break
    proc.wait(timeout=10)

    assert written >= 4  # el hijo llegó a escribir al menos las filas esperadas

    reg = TrialRegistry(db_path)
    count = reg.count_trials("crash")
    reg.close()

    assert count == written


def test_hash_dataset_is_stable_and_sensitive_to_content(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    df_a = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    df_b = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    df_c = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.1]}, index=idx)

    assert hash_dataset(df_a) == hash_dataset(df_b)
    assert hash_dataset(df_a) != hash_dataset(df_c)
