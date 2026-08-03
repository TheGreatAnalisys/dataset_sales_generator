import pytest

from src.io import find_csv


def test_find_csv_localiza_en_output(tmp_path, monkeypatch):
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "ventas.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert find_csv("ventas.csv") == tmp_path / "output" / "ventas.csv"


def test_find_csv_sube_niveles(tmp_path, monkeypatch):
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "ventas.csv").write_text("x", encoding="utf-8")
    sub = tmp_path / "notebooks"
    sub.mkdir()
    monkeypatch.chdir(sub)  # como correr un notebook desde notebooks/
    assert find_csv("ventas.csv").parent.name == "output"


def test_find_csv_lanza_si_no_existe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="Corre main.py"):
        find_csv("no_existe.csv")
