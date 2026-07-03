import numpy as np

from src.events import get_event
from src.seasonality import (
    MONTHLY_INDEX,
    WEEKDAY_FACTOR,
    monthly_factor,
    weekday_factor,
)


def test_monthly_factor_neutral_without_strength():
    for m in range(1, 13):
        assert monthly_factor(m, 0.0) == 1.0


def test_monthly_factor_recovers_index_at_full_strength():
    for m in range(1, 13):
        assert np.isclose(monthly_factor(m, 1.0), MONTHLY_INDEX[m - 1])


def test_weekday_factor_matches_table():
    for d in range(7):
        assert weekday_factor(d) == WEEKDAY_FACTOR[d]


def test_get_event_hit():
    ev = get_event(11, 16)  # Buen Fin (15–18 nov)
    assert ev is not None
    assert ev.name == "Buen Fin"


def test_get_event_miss():
    assert get_event(7, 15) is None


def test_get_event_boundaries():
    assert get_event(1, 4).name == "Reyes"  # primer día del rango
    assert get_event(1, 6).name == "Reyes"  # último día del rango
    assert get_event(1, 7) is None  # fuera del rango
