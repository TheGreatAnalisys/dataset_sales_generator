import numpy as np
import pandas as pd

from src.catalog import build_sku_catalog
from src.features import make_supervised
from src.generator import generate_sales


def _data(make_config):
    cfg = make_config(n_skus=4)
    cat = build_sku_catalog(cfg).set_index("sku_id")
    sales = generate_sales(cat.reset_index(), cfg)
    return sales, cat


def test_make_supervised_columns(make_config):
    sales, cat = _data(make_config)
    sup = make_supervised(sales, cat, freq="W-MON", lags=(1, 4), windows=(4,))
    for col in [
        "sku_id",
        "date",
        "y",
        "promo",
        "log_price",
        "month",
        "weekofyear",
        "cal_event",
        "cal_quincena",
        "cal_holiday",
        "lag_1",
        "lag_4",
        "rollmean_4",
        "rollstd_4",
        "category",
        "base_demand",
    ]:
        assert col in sup.columns, col


def test_lags_have_no_leakage(make_config):
    # lag_1 de la fila t debe ser y de la fila t-1 del MISMO sku (sin mirar el futuro)
    sales, cat = _data(make_config)
    sup = make_supervised(sales, cat, lags=(1,), windows=(4,)).sort_values(
        ["sku_id", "date"]
    )
    one = sup[sup["sku_id"] == sup["sku_id"].iloc[0]].reset_index(drop=True)
    assert np.isnan(one["lag_1"].iloc[0])  # primer período no tiene pasado
    assert one["lag_1"].iloc[1] == one["y"].iloc[0]


def test_one_row_per_sku_period(make_config):
    sales, cat = _data(make_config)
    sup = make_supervised(sales, cat)
    assert not sup.duplicated(subset=["sku_id", "date"]).any()
