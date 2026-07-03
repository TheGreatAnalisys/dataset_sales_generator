"""Métodos para demanda intermitente (slow-movers con muchos ceros).

Croston, su corrección SBA (Syntetos-Boylan) y TSB (Teunter-Syntetos-Babai).
Todos estiman una **tasa de demanda** por período — la cantidad relevante para
inventario — en vez de intentar acertar en qué período exacto ocurre la venta.
Se estudian en el notebook 14.
"""

import numpy as np


def croston(
    y, alpha: float = 0.1, variant: str = "classic", return_fitted: bool = False
):
    """Método de Croston.

    Separa la demanda en tamaño (z) e intervalo entre demandas (p), suaviza cada
    uno con SES y pronostica la tasa z/p. `variant='sba'` aplica la corrección de
    sesgo de Syntetos-Boylan (× (1 − alpha/2)).

    Devuelve la tasa pronosticada (float) o, si `return_fitted`, la serie de
    pronósticos un-paso-adelante.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    fitted = np.full(n, np.nan)
    corr = (1 - alpha / 2) if variant == "sba" else 1.0

    nz = np.where(y > 0)[0]
    if len(nz) == 0:
        return np.zeros(n) if return_fitted else 0.0

    z = y[nz[0]]
    p = float(nz[0] + 1)
    last = nz[0]
    fitted[nz[0] + 1 :] = corr * z / p
    for i in nz[1:]:
        z = alpha * y[i] + (1 - alpha) * z
        p = alpha * (i - last) + (1 - alpha) * p
        last = i
        fitted[i + 1 :] = corr * z / p

    rate = corr * z / p
    return fitted if return_fitted else rate


def tsb(y, alpha: float = 0.1, beta: float = 0.05, return_fitted: bool = False):
    """Método TSB (Teunter-Syntetos-Babai).

    Actualiza la **probabilidad de demanda** en cada período (haya venta o no), en
    lugar del intervalo. Por eso maneja la **obsolescencia**: si el ítem deja de
    venderse, la probabilidad (y el pronóstico) decaen hacia 0.

    Pronóstico = probabilidad × tamaño. Devuelve la tasa final o la serie ajustada.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    fitted = np.full(n, np.nan)

    z = y[y > 0].mean() if (y > 0).any() else 0.0
    prob = float((y > 0).mean())
    for t in range(n):
        fitted[t] = prob * z  # pronóstico para t con la info hasta t-1
        if y[t] > 0:
            z = alpha * y[t] + (1 - alpha) * z
            prob = beta * 1.0 + (1 - beta) * prob
        else:
            prob = (1 - beta) * prob

    rate = prob * z
    return fitted if return_fitted else rate


def adida(y, aggregation: int = 7, alpha: float = 0.1):
    """ADIDA: agrega en bloques de `aggregation` períodos para reducir la
    intermitencia, pronostica el bloque con SES y desagrega por igual.

    Devuelve la tasa pronosticada **por período original**.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < aggregation:
        return float(y.mean()) if n else 0.0
    n_blocks = n // aggregation
    blocks = y[-n_blocks * aggregation :].reshape(n_blocks, aggregation).sum(axis=1)

    level = blocks[0]
    for b in blocks[1:]:
        level = alpha * b + (1 - alpha) * level
    return float(level / aggregation)
