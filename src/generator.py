import numpy as np
import pandas as pd

from src.calendar_mx import holiday_factor_series, quincena_factor_series
from src.config import Config
from src.events import get_event
from src.seasonality import MONTHLY_INDEX, WEEKDAY_FACTOR


def generate_sales(sku_catalog: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Genera el histórico de ventas de forma vectorizada.

    El modelo de demanda es idéntico al conceptual (base × tendencia ×
    estacionalidad mensual × día de semana × evento, con ruido log-normal
    por celda), pero se calcula con NumPy en lugar de bucles fila a fila.
    """
    rng = np.random.default_rng(cfg.random_seed)

    dates = pd.date_range(cfg.start_date, cfg.end_date, freq="D")
    n_days = len(dates)
    n_sku = len(sku_catalog)
    channels = list(cfg.channels.items())
    n_ch = len(channels)

    # ── Factores dependientes de la fecha (vector de longitud n_days) ──────────
    days_in_year = np.where(dates.is_leap_year, 366, 365)
    year_frac = (
        dates.year.to_numpy() - cfg.start_date.year
    ) + dates.dayofyear.to_numpy() / days_in_year
    wday_mult = WEEKDAY_FACTOR[dates.dayofweek.values]
    monthly_index = MONTHLY_INDEX[dates.month.values - 1]

    event_mult = np.ones(n_days)
    event_name = np.array(["Regular"] * n_days, dtype=object)
    for j, dt in enumerate(dates):
        ev = get_event(dt.month, dt.day)
        if ev is not None:
            event_mult[j] = ev.mult
            event_name[j] = ev.name

    # Calendario mexicano: quincenas (días de pago) y festivos de fecha fija
    quincena_mult = quincena_factor_series(dates)
    holiday_mult, _ = holiday_factor_series(dates)

    # ── Factores por SKU (vector de longitud n_sku) ────────────────────────────
    base_demand = sku_catalog["base_demand"].to_numpy()
    base_price = sku_catalog["base_price"].to_numpy()
    annual_trend = sku_catalog["annual_trend"].to_numpy()
    seas_strength = sku_catalog["seas_strength"].to_numpy()

    # ── Matrices (n_sku × n_days) ──────────────────────────────────────────────
    trend_mult = (1 + annual_trend)[:, None] ** year_frac[None, :]
    seas_mult = 1 + seas_strength[:, None] * (monthly_index[None, :] - 1)
    daily_demand = (
        base_demand[:, None]
        * trend_mult
        * seas_mult
        * wday_mult[None, :]
        * event_mult[None, :]
        * quincena_mult[None, :]
        * holiday_mult[None, :]
    )

    # Intermitencia: en los slow-movers, cada día tiene demanda solo con prob.
    # `active_prob` (misma decisión para todos los canales de ese SKU-día, para
    # que los ceros sean coherentes al agregar canales). Para SKUs regulares
    # active_prob = 1.0 → siempre activos.
    active_prob = sku_catalog["active_prob"].to_numpy()
    active = rng.random((n_sku, n_days)) < active_prob[:, None]
    daily_demand = daily_demand * active

    # ── Índices del grid (sku, día, canal) en orden C: sku lento, canal rápido ─
    sku_idx = np.repeat(np.arange(n_sku), n_days * n_ch)
    day_idx = np.tile(np.repeat(np.arange(n_days), n_ch), n_sku)
    ch_idx = np.tile(np.arange(n_ch), n_sku * n_days)
    total = n_sku * n_days * n_ch

    ch_price_mult = np.array([c[1]["price_mult"] for c in channels])
    ch_demand_share = np.array([c[1]["demand_share"] for c in channels])
    ch_names = np.array([c[0] for c in channels], dtype=object)

    # ── Ruido y variación de precio (una extracción por celda del grid) ────────
    noise = rng.lognormal(0.0, cfg.noise_sigma, size=total)
    pv = cfg.price_variation
    price_var = rng.uniform(1 - pv, 1 + pv, size=total)

    demand_flat = daily_demand[sku_idx, day_idx]
    units = np.maximum(
        0, np.rint(demand_flat * ch_demand_share[ch_idx] * noise)
    ).astype(int)
    price = base_price[sku_idx] * ch_price_mult[ch_idx] * price_var
    revenue = np.round(units * price, 2)

    isocal = dates.isocalendar()
    df = pd.DataFrame(
        {
            "date": dates.date[day_idx],
            "year": dates.year.values[day_idx],
            "month": dates.month.values[day_idx],
            "week": isocal["week"].to_numpy()[day_idx],
            "weekday": dates.day_name().to_numpy()[day_idx],
            "sku_id": sku_catalog["sku_id"].to_numpy()[sku_idx],
            "category": sku_catalog["category"].to_numpy()[sku_idx],
            "channel": ch_names[ch_idx],
            "units_sold": units,
            "unit_price": np.round(price, 2),
            "revenue": revenue,
            "event": event_name[day_idx],
            "trend_mult": np.round(trend_mult[sku_idx, day_idx], 4),
        }
    )
    return df
