import numpy as np

from src.hierarchy import reconcile, summing_matrix

CATEGORY_OF = {"A1": "A", "A2": "A", "B1": "B"}
BOTTOM = ["A1", "A2", "B1"]


def test_summing_matrix_shape_and_labels():
    S, labels = summing_matrix(BOTTOM, CATEGORY_OF)
    # nodos: Total, cat:A, cat:B, A1, A2, B1 = 6 ; bottom = 3
    assert S.shape == (6, 3)
    assert labels[0] == "Total"
    assert labels[1:3] == ["cat:A", "cat:B"]
    # el Total suma los 3 SKUs
    assert np.array_equal(S[0], np.ones(3))


def _is_coherent(y_all, S):
    bottom = y_all[-S.shape[1] :]
    return np.allclose(y_all, S @ bottom)


def test_bottom_up_is_coherent():
    S, _ = summing_matrix(BOTTOM, CATEGORY_OF)
    base = np.array([100.0, 40, 20, 30, 25, 18])  # incoherente a propósito
    rec = reconcile(base, S, method="bottom_up")
    assert _is_coherent(rec, S)
    # bottom-up conserva el nivel base
    assert np.allclose(rec[-3:], base[-3:])


def test_ols_reconciliation_is_coherent():
    S, _ = summing_matrix(BOTTOM, CATEGORY_OF)
    base = np.array([100.0, 40, 20, 30, 25, 18])
    rec = reconcile(base, S, method="ols")
    assert _is_coherent(rec, S)


def test_reconcile_preserves_already_coherent():
    S, _ = summing_matrix(BOTTOM, CATEGORY_OF)
    bottom = np.array([30.0, 25, 18])
    coherent = S @ bottom
    for method in ("ols", "wls", "bottom_up"):
        rec = reconcile(coherent, S, method=method)
        assert np.allclose(rec, coherent), method


def test_wls_and_topdown_coherent():
    S, _ = summing_matrix(BOTTOM, CATEGORY_OF)
    base = np.array([100.0, 40, 20, 30, 25, 18])
    assert _is_coherent(reconcile(base, S, method="wls"), S)
    props = np.array([0.5, 0.3, 0.2])
    assert _is_coherent(reconcile(base, S, method="top_down", weights=props), S)


def test_matrix_input_multihorizon():
    S, _ = summing_matrix(BOTTOM, CATEGORY_OF)
    base = np.random.default_rng(0).uniform(1, 50, size=(6, 4))
    rec = reconcile(base, S, method="ols")
    assert rec.shape == (6, 4)
    for h in range(4):
        assert _is_coherent(rec[:, h], S)
