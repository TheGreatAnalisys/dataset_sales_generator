import numpy as np
import pandas as pd

from src.config import Config

CATEGORIES = {
    "Electrónica": {"base_price_range": (800, 5000), "base_demand_range": (5, 25)},
    "Ropa": {"base_price_range": (150, 800), "base_demand_range": (10, 50)},
    "Alimentos": {"base_price_range": (20, 200), "base_demand_range": (30, 120)},
    "Hogar": {"base_price_range": (100, 1500), "base_demand_range": (5, 30)},
    "Deportes": {"base_price_range": (200, 2000), "base_demand_range": (5, 35)},
}


def build_sku_catalog(cfg: Config) -> pd.DataFrame:
    np.random.seed(cfg.random_seed)
    cat_names = list(CATEGORIES.keys())
    rows = []
    for i in range(1, cfg.n_skus + 1):
        cat = cat_names[i % len(cat_names)]
        spec = CATEGORIES[cat]
        bp = np.random.uniform(*spec["base_price_range"])
        bd = np.random.uniform(*spec["base_demand_range"])
        trend = np.random.uniform(cfg.trend_min, cfg.trend_max)
        seas = np.random.uniform(0.5, 1.5)
        rows.append(
            {
                "sku_id": f"SKU-{i:03d}",
                "category": cat,
                "base_price": round(bp, 2),
                "base_demand": round(bd, 2),
                "annual_trend": round(trend, 4),
                "seas_strength": round(seas, 4),
            }
        )
    return pd.DataFrame(rows)
