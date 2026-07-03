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

# Fracción de SKUs que son de demanda intermitente (slow-movers con días en cero).
# Genera un catálogo realista con tiers de forecasting distintos (ver notebook 07).
INTERMITTENT_SHARE = 0.30


def build_sku_catalog(cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed)
    cat_names = list(CATEGORIES.keys())
    rows = []
    for i in range(1, cfg.n_skus + 1):
        cat = cat_names[i % len(cat_names)]
        spec = CATEGORIES[cat]
        bp = rng.uniform(*spec["base_price_range"])
        bd = rng.uniform(*spec["base_demand_range"])
        trend = rng.uniform(cfg.trend_min, cfg.trend_max)
        seas = rng.uniform(0.5, 1.5)

        # Perfil de demanda: una fracción son slow-movers intermitentes
        is_intermittent = rng.random() < INTERMITTENT_SHARE
        if is_intermittent:
            bd = bd * rng.uniform(0.05, 0.15)  # demanda base mucho más baja
            active_prob = float(rng.uniform(0.10, 0.30))  # días con demanda > 0
            profile = "Intermitente"
        else:
            active_prob = 1.0
            profile = "Regular"

        rows.append(
            {
                "sku_id": f"SKU-{i:03d}",
                "category": cat,
                "base_price": round(bp, 2),
                "base_demand": round(bd, 2),
                "annual_trend": round(trend, 4),
                "seas_strength": round(seas, 4),
                "demand_profile": profile,
                "active_prob": round(active_prob, 3),
            }
        )
    return pd.DataFrame(rows)
