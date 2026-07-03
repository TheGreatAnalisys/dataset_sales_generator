import numpy as np

from src.intermittent import adida, croston, tsb


def test_croston_constant_demand():
    # Demanda en todos los períodos (intervalo=1) → tasa ≈ nivel
    rate = croston([5, 5, 5, 5, 5, 5], alpha=0.2)
    assert 4.0 < rate < 6.0


def test_croston_rate_on_sparse_series():
    # 10 unidades cada 3 períodos → tasa ≈ 10/3
    y = [0, 0, 10, 0, 0, 10, 0, 0, 10, 0, 0, 10]
    rate = croston(y, alpha=0.1)
    assert rate == float(rate)  # es un número
    assert 2.0 < rate < 5.0


def test_croston_all_zeros():
    assert croston([0, 0, 0, 0]) == 0.0


def test_sba_below_classic():
    y = [0, 0, 8, 0, 0, 6, 0, 0, 9, 0]
    assert croston(y, variant="sba") < croston(y, variant="classic")


def test_tsb_handles_obsolescence():
    # Demanda al inicio y luego muere: el pronóstico TSB debe decaer
    y = [4, 5, 4, 6, 5] + [0] * 20
    fitted = tsb(y, return_fitted=True)
    assert fitted[5] > fitted[-1]  # decae con los ceros finales
    assert fitted[-1] < 1.0


def test_croston_stays_flat_after_death():
    # Croston NO decae con ceros finales (solo se actualiza en demandas)
    y = [4, 5, 4, 6, 5] + [0] * 20
    fitted = croston(y, return_fitted=True)
    assert np.isclose(fitted[6], fitted[-1])  # constante tras la última venta


def test_adida_positive_rate():
    y = [0, 0, 3, 0, 0, 0, 2, 0, 0, 5, 0, 0, 0, 4] * 3
    r = adida(y, aggregation=7)
    assert r > 0
