"""Framework de forecasting de punta a punta: del dato crudo a la decisión.

Integra todo lo construido en la serie en un solo flujo, model-agnostic y sin
dependencias pesadas (solo NumPy/pandas + módulos propios), para que la app y el
notebook del Video 21 corran rápido y en cualquier entorno:

    ventas → serie por SKU → pronóstico por tier → reconciliación
           → métricas → recomendación de inventario (unidades y pesos)

El pronóstico por defecto es tiered: demanda regular con estacional + deriva y
slow-movers con Croston. Cualquier modelo de la serie puede sustituirlo.
"""

import math

import numpy as np
import pandas as pd

from src.hierarchy import reconcile, summing_matrix
from src.intermittent import croston
from src.metrics import bias, mase, wape

# z para niveles de servicio típicos (aprox. de la normal estándar)
_Z = {0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}


def _z_score(service_level: float) -> float:
    best = min(_Z, key=lambda s: abs(s - service_level))
    return _Z[best]


def weekly_by_sku(sales: pd.DataFrame) -> pd.DataFrame:
    """Matriz semanal (semanas × SKUs) de unidades; quita semanas parciales de borde."""
    s = sales.copy()
    s["date"] = pd.to_datetime(s["date"])
    w = (
        s.groupby(["sku_id", pd.Grouper(key="date", freq="W-MON")])["units_sold"]
        .sum()
        .unstack("sku_id")
        .sort_index()
        .fillna(0.0)
    )
    return w.iloc[1:-1]


def _drift_snaive(y: np.ndarray, horizon: int, m: int = 52) -> np.ndarray:
    """Forma estacional del último año + deriva lineal estimada en la ventana."""
    y = np.asarray(y, dtype=float)
    if len(y) >= m:
        shape = np.resize(y[-m:], horizon)
    else:
        shape = np.full(horizon, y.mean() if len(y) else 0.0)
    slope = np.polyfit(np.arange(len(y)), y, 1)[0] if len(y) > 2 else 0.0
    return np.clip(shape + slope * np.arange(1, horizon + 1), 0, None)


def forecast_sku(train: np.ndarray, horizon: int, intermittent: bool) -> np.ndarray:
    """Pronóstico por tier: Croston (intermitente) o estacional+deriva (regular)."""
    if intermittent:
        return np.full(horizon, max(croston(train), 0.0))
    return _drift_snaive(train, horizon)


def run_framework(
    sales: pd.DataFrame,
    catalog: pd.DataFrame,
    horizon: int = 13,
    service_level: float = 0.95,
) -> dict:
    """Ejecuta el framework completo y devuelve pronósticos, métricas y decisión.

    Planea el **último `horizon`** de la historia (para poder comparar con lo real)
    y recomienda el inventario para ese horizonte por SKU.
    """
    weekly = weekly_by_sku(sales)
    skus = list(weekly.columns)
    if len(weekly) <= horizon + 4:
        raise ValueError("historia insuficiente para el horizonte pedido")

    train_df, actual_df = weekly.iloc[:-horizon], weekly.iloc[-horizon:]
    profile = catalog["demand_profile"].reindex(skus).fillna("Regular")
    price = sales.groupby("sku_id")["unit_price"].mean().reindex(skus).fillna(0.0)
    z = _z_score(service_level)

    fc = {}
    for s in skus:
        tr = train_df[s].values
        fc[s] = forecast_sku(tr, horizon, profile[s] == "Intermitente")
    forecast_df = pd.DataFrame(fc, index=actual_df.index)

    # Recomendación de inventario por SKU
    rows = []
    for s in skus:
        pred = forecast_df[s].values
        act = actual_df[s].values
        demand_std = float(train_df[s].std())
        safety = z * demand_std * math.sqrt(horizon)
        reco_units = float(pred.sum() + safety)
        rows.append(
            {
                "sku_id": s,
                "category": catalog["category"].get(s, "?"),
                "tier": (
                    "C · Intermitente"
                    if profile[s] == "Intermitente"
                    else "A/B · Regular"
                ),
                "pred_units": float(pred.sum()),
                "actual_units": float(act.sum()),
                "safety_stock": float(safety),
                "reco_units": reco_units,
                "unit_price": float(price[s]),
                "reco_pesos": reco_units * float(price[s]),
            }
        )
    reco = pd.DataFrame(rows).set_index("sku_id")

    # Reconciliación jerárquica de los pronósticos (por período del horizonte)
    category_of = catalog["category"].reindex(skus).fillna("?").to_dict()
    S, labels = summing_matrix(skus, category_of)
    base = np.vstack([forecast_df[s].values for s in skus])  # SKUs × horizonte
    base_all = S @ base  # todos los nodos (ya coherente: es suma directa)
    coherent = reconcile(base_all, S, method="bottom_up")
    coherence_ok = bool(np.allclose(base_all, coherent))

    # Métricas agregadas (total) sobre el horizonte planeado
    total_pred = forecast_df.sum(axis=1).values
    total_act = actual_df.sum(axis=1).values
    total_train = train_df.sum(axis=1).values
    metrics = {
        "wape": wape(total_act, total_pred),
        "bias": bias(total_act, total_pred),
        "mase": mase(total_act, total_pred, total_train, min(52, len(total_train) - 1)),
    }

    return {
        "forecast": forecast_df,
        "actual": actual_df,
        "train": train_df,
        "reco": reco.sort_values("reco_pesos", ascending=False),
        "metrics": metrics,
        "horizon": horizon,
        "service_level": service_level,
        "total_investment": float(reco["reco_pesos"].sum()),
        "coherent": coherence_ok,
    }
