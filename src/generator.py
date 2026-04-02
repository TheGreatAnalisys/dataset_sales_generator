import numpy as np
import pandas as pd

from src.config import Config
from src.events import get_event
from src.seasonality import monthly_factor, weekday_factor


def generate_sales(sku_catalog: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    np.random.seed(cfg.random_seed)
    dates = pd.date_range(cfg.start_date, cfg.end_date, freq="D")
    records = []

    for _, sku in sku_catalog.iterrows():
        for dt in dates:
            year_frac = (dt.year - cfg.start_date.year) + dt.dayofyear / 365
            trend_mult = (1 + sku["annual_trend"]) ** year_frac
            seas_mult = monthly_factor(dt.month, sku["seas_strength"])
            wday_mult = weekday_factor(dt.dayofweek)
            event = get_event(dt.month, dt.day)
            event_mult = event.mult if event else 1.0
            event_name = event.name if event else "Regular"

            daily_demand = (
                sku["base_demand"] * trend_mult * seas_mult * wday_mult * event_mult
            )

            for channel, ch_cfg in cfg.channels.items():
                noise = np.random.lognormal(0, cfg.noise_sigma)
                units = max(
                    0, int(round(daily_demand * ch_cfg["demand_share"] * noise))
                )
                pv = cfg.price_variation
                price = (
                    sku["base_price"]
                    * ch_cfg["price_mult"]
                    * np.random.uniform(1 - pv, 1 + pv)
                )

                records.append(
                    {
                        "date": dt.date(),
                        "year": dt.year,
                        "month": dt.month,
                        "week": dt.isocalendar().week,
                        "weekday": dt.day_name(),
                        "sku_id": sku["sku_id"],
                        "category": sku["category"],
                        "channel": channel,
                        "units_sold": units,
                        "unit_price": round(price, 2),
                        "revenue": round(units * price, 2),
                        "event": event_name,
                        "trend_mult": round(trend_mult, 4),
                    }
                )

    return pd.DataFrame(records)
