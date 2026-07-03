import numpy as np
import pytest

from src.metrics import bias, fva, mae, mape, mase, smape, wape


def test_perfect_forecast_is_zero_error():
    a = [10, 20, 30, 40]
    assert mae(a, a) == 0.0
    assert wape(a, a) == 0.0
    assert bias(a, a) == 0.0


def test_mae_known_value():
    assert mae([10, 20], [12, 18]) == 2.0


def test_wape_known_value():
    # |10-12|+|20-18| = 4 ; Σ|real| = 30 → 4/30
    assert wape([10, 20], [12, 18]) == pytest.approx(4 / 30)


def test_bias_sign():
    assert bias([10, 10], [12, 12]) > 0  # sobre-pronóstico
    assert bias([10, 10], [8, 8]) < 0  # sub-pronóstico


def test_mase_scaling():
    # train con paso m=1: diffs = [10,10,10] → denom=10
    train = [0, 10, 20, 30]
    # forecast con MAE=5 → MASE = 0.5
    assert mase([100, 100], [95, 105], train, m=1) == pytest.approx(0.5)


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        mae([1, 2, 3], [1, 2])


def test_mape_known_value():
    # |10-12|/10 + |20-18|/20 = 0.2 + 0.1 → media 0.15
    assert mape([10, 20], [12, 18]) == pytest.approx(0.15)


def test_mape_explodes_with_zeros():
    assert not np.isfinite(mape([0, 10], [5, 10]))  # división por cero → inf


def test_smape_bounded_and_handles_zero():
    assert smape([0, 10], [5, 10]) < 2.0  # acotado, no explota
    assert smape([0, 0], [0, 0]) == 0.0


def test_fva_positive_when_better_than_baseline():
    actual = [10, 20, 30, 40]
    good = [11, 19, 31, 39]  # cerca
    baseline = [20, 20, 20, 20]  # malo
    assert fva(actual, good, baseline) > 0


def test_fva_nonpositive_when_not_better():
    actual = [10, 20, 30, 40]
    same = [20, 20, 20, 20]
    assert fva(actual, same, same) == pytest.approx(0.0)
