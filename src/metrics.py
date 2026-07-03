"""Métricas de forecasting reutilizables.

Funciones puras sobre array-likes (listas, np.ndarray o pd.Series). Se usan en los
notebooks de modelado (Fase 3) y se estudian a fondo en el Video 18.
"""

import numpy as np


def _align(actual, forecast):
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if a.shape != f.shape:
        raise ValueError(
            f"actual y forecast deben tener la misma forma: {a.shape} vs {f.shape}"
        )
    return a, f


def mae(actual, forecast) -> float:
    """Error absoluto medio (en unidades de la serie)."""
    a, f = _align(actual, forecast)
    return float(np.mean(np.abs(a - f)))


def wape(actual, forecast) -> float:
    """Weighted Absolute Percentage Error: Σ|error| / Σ|real|.

    Robusto ante ceros (a diferencia del MAPE) y expresado como fracción.
    """
    a, f = _align(actual, forecast)
    denom = np.sum(np.abs(a))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(a - f)) / denom)


def bias(actual, forecast) -> float:
    """Sesgo relativo: Σ(forecast − real) / Σ|real|.

    Positivo = el modelo sobre-pronostica; negativo = se queda corto.
    """
    a, f = _align(actual, forecast)
    denom = np.sum(np.abs(a))
    if denom == 0:
        return float("nan")
    return float(np.sum(f - a) / denom)


def mase(actual, forecast, train, m: int = 1) -> float:
    """Mean Absolute Scaled Error, escalado por el naive estacional en train.

    denom = MAE del naive estacional (paso `m`) sobre `train`.
    MASE < 1 le gana a ese baseline; MASE > 1 no lo supera.
    """
    a, f = _align(actual, forecast)
    tr = np.asarray(train, dtype=float)
    if len(tr) <= m:
        raise ValueError("train debe tener más de `m` observaciones")
    denom = np.mean(np.abs(tr[m:] - tr[:-m]))
    if denom == 0:
        return float("nan")
    return float(np.mean(np.abs(a - f)) / denom)
