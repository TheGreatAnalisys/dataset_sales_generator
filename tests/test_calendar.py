import numpy as np
import pandas as pd

from src.calendar_mx import (
    MX_HOLIDAYS,
    QUINCENA_STRENGTH,
    holiday_factor_series,
    quincena_factor_series,
    quincena_mask,
)


def _dates(start="2022-01-01", end="2022-12-31"):
    return pd.date_range(start, end, freq="D")


def test_quincena_mask_hits_paydays():
    d = _dates()
    mask = quincena_mask(d)
    m = pd.Series(mask, index=d)
    assert m["2022-01-15"]  # mediados
    assert m["2022-01-31"]  # fin de mes
    assert m["2022-02-01"]  # inicio (tras pago de fin de mes)
    assert not m["2022-01-10"]  # día normal


def test_quincena_factor_values():
    d = _dates()
    f = quincena_factor_series(d)
    assert set(np.round(np.unique(f), 6)) == {1.0, round(1.0 + QUINCENA_STRENGTH, 6)}


def test_holiday_factor_marks_holiday_and_prewindow():
    d = _dates()
    mult, names = holiday_factor_series(d)
    s = pd.Series(mult, index=d)
    ns = pd.Series(names, index=d)
    # Independencia 16 sep con boost, y nombre en el día exacto
    assert s["2022-09-16"] > 1.0
    assert ns["2022-09-16"] == "Independencia"
    # ventana pre-festivo (2 días antes) también recibe boost
    assert s["2022-09-14"] > 1.0
    # un día lejano no
    assert s["2022-07-10"] == 1.0


def test_holidays_are_fixed_dates():
    assert len(MX_HOLIDAYS) >= 6
    assert all(1 <= h.month <= 12 and 1 <= h.day <= 31 for h in MX_HOLIDAYS)
