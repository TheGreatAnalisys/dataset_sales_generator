"""Construcción de la matriz supervisada para forecasting con ML.

Convierte el histórico de ventas en un dataset tabular (una fila por SKU-período)
con features de calendario, precio/promoción y memoria (lags + ventanas móviles).
Reutiliza el calendario del generador (src.events / src.calendar_mx) como fuente
única de verdad. Se usa en el notebook 16 (XGBoost/LightGBM).
"""

import numpy as np
import pandas as pd

from src.calendar_mx import holiday_factor_series, quincena_factor_series
from src.events import get_event


def _calendar_daily(dmin, dmax) -> pd.DataFrame:
    dates = pd.date_range(dmin, dmax, freq="D")
    ev = np.array(
        [
            (get_event(d.month, d.day).mult if get_event(d.month, d.day) else 1.0)
            for d in dates
        ]
    )
    return pd.DataFrame(
        {
            "cal_event": ev,
            "cal_quincena": quincena_factor_series(dates),
            "cal_holiday": holiday_factor_series(dates)[0],
        },
        index=dates,
    )


def make_supervised(
    sales: pd.DataFrame,
    catalog: pd.DataFrame,
    freq: str = "W-MON",
    lags=(1, 4, 52),
    windows=(4, 12),
) -> pd.DataFrame:
    """Devuelve un DataFrame tabular (una fila por SKU-período).

    Columnas: sku_id, date, y (unidades), promo, log_price, features de calendario
    (mes, semana del año, event/quincena/holiday), lags y ventanas móviles por SKU
    (siempre con shift → sin fuga de datos), y estáticas (category, base_demand).
    """
    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"])  # robusto a datetime.date u str
    grp = sales.groupby(["sku_id", pd.Grouper(key="date", freq=freq)])
    agg = grp.agg(
        y=("units_sold", "sum"),
        promo=("discount", "mean"),
        price=("unit_price", "mean"),
    ).reset_index()
    agg["log_price"] = np.log(agg["price"])

    # Calendario a la frecuencia objetivo (media de los factores diarios del período)
    cal = (
        _calendar_daily(sales["date"].min(), sales["date"].max()).resample(freq).mean()
    )
    cal["month"] = cal.index.month
    cal["weekofyear"] = cal.index.isocalendar().week.astype(int)
    agg = agg.merge(cal, left_on="date", right_index=True, how="left")

    # Memoria por SKU: lags y ventanas móviles, siempre desplazadas (sin fuga)
    agg = agg.sort_values(["sku_id", "date"])
    gy = agg.groupby("sku_id")["y"]
    for L in lags:
        agg[f"lag_{L}"] = gy.shift(L)
    for w in windows:
        agg[f"rollmean_{w}"] = gy.transform(lambda s: s.shift(1).rolling(w).mean())
        agg[f"rollstd_{w}"] = gy.transform(lambda s: s.shift(1).rolling(w).std())

    # Estáticas del SKU
    static = catalog[["category", "base_demand"]]
    agg = agg.merge(static, left_on="sku_id", right_index=True, how="left")
    return agg.reset_index(drop=True)
