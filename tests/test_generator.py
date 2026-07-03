from pandas.testing import assert_frame_equal

from src.catalog import build_sku_catalog
from src.generator import generate_sales

CATALOG_COLUMNS = [
    "sku_id",
    "category",
    "base_price",
    "base_demand",
    "annual_trend",
    "seas_strength",
]


def test_catalog_shape_and_columns(make_config):
    cfg = make_config(n_skus=5)
    cat = build_sku_catalog(cfg)
    assert len(cat) == 5
    assert list(cat.columns) == CATALOG_COLUMNS
    assert cat["base_price"].gt(0).all()
    assert cat["base_demand"].gt(0).all()
    assert cat["annual_trend"].between(cfg.trend_min, cfg.trend_max).all()


def test_generate_shape_and_columns(make_config):
    cfg = make_config(n_skus=3)  # enero 2022 = 31 días, 2 canales
    cat = build_sku_catalog(cfg)
    df = generate_sales(cat, cfg)
    assert len(df) == 3 * 31 * len(cfg.channels)
    assert set(df["channel"]) == set(cfg.channels)
    assert set(df["sku_id"]) == set(cat["sku_id"])


def test_no_negative_values(make_config):
    cfg = make_config()
    df = generate_sales(build_sku_catalog(cfg), cfg)
    assert (df["units_sold"] >= 0).all()
    assert (df["revenue"] >= 0).all()
    assert (df["unit_price"] > 0).all()


def test_revenue_zero_when_units_zero(make_config):
    cfg = make_config()
    df = generate_sales(build_sku_catalog(cfg), cfg)
    zero_units = df["units_sold"] == 0
    assert (df.loc[zero_units, "revenue"] == 0).all()


def test_determinism(make_config):
    cfg = make_config()
    df1 = generate_sales(build_sku_catalog(cfg), cfg)
    df2 = generate_sales(build_sku_catalog(cfg), cfg)
    assert_frame_equal(df1, df2)


def test_event_column(make_config):
    cfg = make_config()  # enero incluye Reyes (4–6 ene)
    df = generate_sales(build_sku_catalog(cfg), cfg)
    events = set(df["event"])
    assert "Reyes" in events
    assert "Regular" in events
