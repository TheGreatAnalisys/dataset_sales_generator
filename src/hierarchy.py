"""Reconciliación de pronósticos jerárquicos.

Total → categorías → SKUs. Dado un conjunto de pronósticos base (uno por nodo,
posiblemente incoherentes entre niveles), produce pronósticos **coherentes** que
suman correctamente. Implementa bottom-up, top-down y reconciliación óptima
(MinT: OLS y WLS). Sin dependencias externas — solo NumPy. Se usa en el notebook 16.
"""

import numpy as np


def summing_matrix(bottom_labels, category_of):
    """Matriz de suma S (n_nodos × n_bottom).

    Mapea el nivel base (SKUs) a todos los nodos: [Total, categorías…, SKUs…].
    `category_of` es un dict sku_id -> categoría. Devuelve (S, labels).
    """
    cats = sorted({category_of[b] for b in bottom_labels})
    n = len(bottom_labels)
    rows = [np.ones(n)]  # Total
    for c in cats:  # una fila por categoría
        rows.append(
            np.array([1.0 if category_of[b] == c else 0.0 for b in bottom_labels])
        )
    rows.extend(np.eye(n))  # una fila por SKU
    S = np.vstack(rows)
    labels = ["Total"] + [f"cat:{c}" for c in cats] + list(bottom_labels)
    return S, labels


def _G(S, method, weights=None):
    n_all, n_bottom = S.shape
    if method == "bottom_up":
        return np.hstack([np.zeros((n_bottom, n_all - n_bottom)), np.eye(n_bottom)])
    if method == "top_down":
        p = np.asarray(weights, dtype=float)
        p = p / p.sum()
        G = np.zeros((n_bottom, n_all))
        G[:, 0] = p  # reparte el Total (fila 0) por proporciones
        return G
    if method == "ols":  # MinT con W = I
        return np.linalg.inv(S.T @ S) @ S.T
    if method == "wls":  # MinT con W diagonal (estructural o por varianzas)
        w = np.asarray(S.sum(axis=1) if weights is None else weights, dtype=float)
        Wi = np.diag(1.0 / w)
        return np.linalg.inv(S.T @ Wi @ S) @ S.T @ Wi
    raise ValueError(f"método desconocido: {method}")


def reconcile(base, S, method="ols", weights=None):
    """Reconcilia pronósticos base a coherentes.

    `base` es (n_nodos,) o (n_nodos, H) en el orden de las filas de S.
    Devuelve pronósticos coherentes de la misma forma. `ỹ = S G ŷ`.
    - bottom_up: usa solo el nivel base y agrega.
    - top_down: reparte el Total por `weights` (proporciones de los SKUs).
    - ols / wls: reconciliación óptima (MinT). En wls, `weights` = varianzas por
      nodo (o estructural si None).
    """
    S = np.asarray(S, dtype=float)
    base = np.asarray(base, dtype=float)
    G = _G(S, method, weights)
    bottom = G @ base
    return S @ bottom
