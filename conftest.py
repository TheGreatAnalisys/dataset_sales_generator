import sys
from datetime import date
from pathlib import Path

import pytest

# Asegura que `import src...` funcione al correr pytest desde la raíz del repo
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config


@pytest.fixture
def make_config(tmp_path):
    """Fábrica de Config para tests, sin depender de variables de entorno."""

    def _make(**overrides):
        params = dict(
            start_date=date(2022, 1, 1),
            end_date=date(2022, 1, 31),
            n_skus=4,
            random_seed=7,
            trend_min=-0.05,
            trend_max=0.20,
            noise_sigma=0.15,
            price_variation=0.02,
            channels={
                "Retail": {"price_mult": 1.0, "demand_share": 0.6},
                "Online": {"price_mult": 0.9, "demand_share": 0.4},
            },
            output_dir=tmp_path,
            sales_filename="sales.csv",
            catalog_filename="catalog.csv",
        )
        params.update(overrides)
        return Config(**params)

    return _make


@pytest.fixture
def cfg(make_config):
    return make_config()
