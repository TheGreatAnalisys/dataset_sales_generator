"""Validación cruzada temporal para series de tiempo.

Genera particiones que respetan el orden del tiempo (nunca entrenar con el futuro):
esquema **expanding** (ventana de entrenamiento que crece) y **sliding** (ventana de
tamaño fijo que se desliza). Base del walk-forward. Se usa en el notebook 19.
"""

from typing import Iterator, List, Tuple

import numpy as np


def temporal_splits(
    n_obs: int,
    initial: int,
    horizon: int,
    step: int | None = None,
    mode: str = "expanding",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Particiones temporales (train_idx, test_idx).

    - `initial`: tamaño del primer train (y tamaño fijo de la ventana en 'sliding').
    - `horizon`: cuántos períodos futuros evalúa cada fold.
    - `step`: cuánto avanza el origen entre folds (por defecto = horizon).
    - `mode`: 'expanding' (train crece) o 'sliding' (ventana fija).

    Los índices de test siempre van DESPUÉS de los de train (sin fuga temporal).
    """
    if mode not in ("expanding", "sliding"):
        raise ValueError("mode debe ser 'expanding' o 'sliding'")
    if initial < 1 or horizon < 1 or initial + horizon > n_obs:
        raise ValueError("parámetros incompatibles con n_obs")
    step = horizon if step is None else step

    folds = []
    train_end = initial
    while train_end + horizon <= n_obs:
        if mode == "expanding":
            train = np.arange(0, train_end)
        else:  # sliding: ventana de tamaño `initial`
            train = np.arange(train_end - initial, train_end)
        test = np.arange(train_end, train_end + horizon)
        folds.append((train, test))
        train_end += step
    return folds


def walk_forward(
    y,
    forecaster,
    initial: int,
    horizon: int,
    step: int | None = None,
    mode: str = "expanding",
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Itera (y_real, y_pred) por cada fold temporal.

    `forecaster(train_array, horizon)` debe devolver `horizon` pronósticos. Permite
    evaluar cualquier modelo en walk-forward y obtener una DISTRIBUCIÓN de error,
    no un solo número.
    """
    y = np.asarray(y, dtype=float)
    for train_idx, test_idx in temporal_splits(len(y), initial, horizon, step, mode):
        pred = np.asarray(forecaster(y[train_idx], horizon), dtype=float)
        yield y[test_idx], pred
