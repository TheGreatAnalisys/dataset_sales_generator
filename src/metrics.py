"""Métricas de forecasting reutilizables.

Funciones puras sobre array-likes (listas, np.ndarray o pd.Series). Se usan en los
notebooks de modelado (Fase 3) y se estudian a fondo en el Video 19.
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


def mape(actual, forecast) -> float:
    """Mean Absolute Percentage Error: mean(|a−f| / |a|).

    ¡Cuidado! Es indefinido/explota cuando hay ceros o valores muy pequeños en
    `actual` — justo la demanda intermitente. Se incluye para mostrar sus peligros.
    """
    a, f = _align(actual, forecast)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.abs(a - f) / np.abs(a)
    return float(np.mean(ratios))


def smape(actual, forecast) -> float:
    """Symmetric MAPE: mean(2|a−f| / (|a|+|f|)). Acotado en [0, 2], evita el
    infinito del MAPE cuando a=0 (salvo que a=f=0, que se trata como 0)."""
    a, f = _align(actual, forecast)
    denom = np.abs(a) + np.abs(f)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(denom == 0, 0.0, 2 * np.abs(a - f) / denom)
    return float(np.mean(terms))


def fva(actual, forecast, baseline, metric=wape) -> float:
    """Forecast Value Added: mejora relativa del modelo sobre un baseline.

    FVA = (error_baseline − error_modelo) / error_baseline con la métrica dada
    (WAPE por defecto). Positivo = el modelo AÑADE valor; ≤ 0 = no aporta.
    """
    err_model = metric(actual, forecast)
    err_base = metric(actual, baseline)
    if err_base == 0:
        return float("nan")
    return float((err_base - err_model) / err_base)
