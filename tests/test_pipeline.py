import pandas as pd

from src.catalog import build_sku_catalog
from src.generator import generate_sales
from src.pipeline import run_framework, weekly_by_sku


def _data(make_config):
    # 3 años para tener historia > horizonte
    from datetime import date

    cfg = make_config(
        n_skus=8, start_date=date(2021, 1, 1), end_date=date(2023, 12, 31)
    )
    cat = build_sku_catalog(cfg).set_index("sku_id")
    sales = generate_sales(cat.reset_index(), cfg)
    return sales, cat


def test_weekly_by_sku_shape(make_config):
    sales, cat = _data(make_config)
    w = weekly_by_sku(sales)
    assert set(w.columns) == set(cat.index)
    assert (w >= 0).all().all()


def test_run_framework_outputs(make_config):
    sales, cat = _data(make_config)
    out = run_framework(sales, cat, horizon=8, service_level=0.95)
    assert (
        set(["forecast", "reco", "metrics", "total_investment", "coherent"])
        <= out.keys()
    )
    assert out["forecast"].shape[0] == 8
    assert out["coherent"] is True
    assert out["total_investment"] > 0
    assert 0 <= out["metrics"]["wape"]


def test_reco_columns_and_positive(make_config):
    sales, cat = _data(make_config)
    out = run_framework(sales, cat, horizon=8)
    reco = out["reco"]
    for c in ["reco_units", "reco_pesos", "safety_stock", "tier"]:
        assert c in reco.columns
    # el stock recomendado nunca es menor que el pronóstico (incluye seguridad)
    assert (reco["reco_units"] >= reco["pred_units"] - 1e-6).all()


def test_insufficient_history_raises(make_config):
    sales, cat = _data(make_config)
    import pytest

    with pytest.raises(ValueError):
        run_framework(sales, cat, horizon=1000)
