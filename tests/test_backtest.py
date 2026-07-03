import numpy as np
import pytest

from src.backtest import temporal_splits, walk_forward


def test_expanding_grows_and_no_leakage():
    folds = temporal_splits(n_obs=20, initial=10, horizon=2, step=2, mode="expanding")
    assert len(folds) == 5
    prev_len = 0
    for train, test in folds:
        assert train.max() < test.min()  # test siempre después de train
        assert len(train) >= prev_len  # expanding: no decrece
        prev_len = len(train)
        assert len(test) == 2


def test_sliding_constant_window():
    folds = temporal_splits(n_obs=20, initial=8, horizon=2, step=2, mode="sliding")
    for train, test in folds:
        assert len(train) == 8  # ventana fija
        assert train.max() < test.min()


def test_no_overlap_train_test():
    for train, test in temporal_splits(30, 12, 3, 3):
        assert len(set(train) & set(test)) == 0


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        temporal_splits(10, 9, 5)  # initial+horizon > n
    with pytest.raises(ValueError):
        temporal_splits(20, 10, 2, mode="random")


def test_walk_forward_yields_pairs():
    y = np.arange(20.0)
    # forecaster naive: repite el último valor
    preds = list(
        walk_forward(y, lambda tr, h: np.full(h, tr[-1]), initial=10, horizon=2)
    )
    assert len(preds) == 5
    real, pred = preds[0]
    assert real.shape == pred.shape == (2,)
    assert np.allclose(pred, 9.0)  # último de train[0:10] = 9
