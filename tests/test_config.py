from datetime import date

import pytest

from src.config import _parse_channels, _parse_date


def test_parse_channels_valid():
    ch = _parse_channels("A|1.0|0.5, B|0.9|0.5")
    assert set(ch) == {"A", "B"}
    assert ch["A"]["price_mult"] == 1.0
    assert ch["B"]["demand_share"] == 0.5


def test_parse_channels_shares_must_sum_to_one():
    with pytest.raises(ValueError):
        _parse_channels("A|1.0|0.5, B|0.9|0.2")


def test_parse_channels_bad_format():
    with pytest.raises(ValueError):
        _parse_channels("A|1.0")


def test_parse_date():
    assert _parse_date("2021-03-04") == date(2021, 3, 4)


def test_config_rejects_inverted_dates(make_config):
    with pytest.raises(ValueError):
        make_config(start_date=date(2022, 2, 1), end_date=date(2022, 1, 1))


def test_config_rejects_bad_trend(make_config):
    with pytest.raises(ValueError):
        make_config(trend_min=0.3, trend_max=0.1)


def test_config_rejects_zero_skus(make_config):
    with pytest.raises(ValueError):
        make_config(n_skus=0)


def test_config_derived_paths(make_config, tmp_path):
    cfg = make_config()
    assert cfg.sales_path == tmp_path / "sales.csv"
    assert cfg.catalog_path == tmp_path / "catalog.csv"
