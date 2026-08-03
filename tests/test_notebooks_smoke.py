"""Smoke-test de ejecución de los notebooks.

Ejecuta las celdas de CÓDIGO de cada notebook como script (sin kernel Jupyter,
para no depender de ZeroMQ) con backend matplotlib Agg. Verifica que el código
corre de punta a punta contra el dataset generado.

- Si a un notebook le falta una dependencia de análisis (p. ej. `statsmodels`),
  el test se **salta** en vez de fallar → en el CI ligero (solo `requirements.txt`)
  casi todos se saltan; en el job `notebooks` (con `requirements-dev.txt`) corren.
- El notebook de Foundation Models se salta siempre: descarga pesos de HuggingFace.
"""

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
NB_IDS = [p.stem for p in NOTEBOOKS]

# Notebooks que no se ejecutan en automático (y por qué).
SKIP = {
    "18_Foundation_Models": "descarga pesos de Chronos desde HuggingFace (requiere red)"
}

# Módulos que asumimos siempre presentes (stdlib + runtime del generador).
_BASE = {
    "warnings",
    "sys",
    "os",
    "re",
    "json",
    "math",
    "itertools",
    "collections",
    "datetime",
    "pathlib",
    "numpy",
    "pandas",
    "src",
    "matplotlib",
}


def _ensure_dataset():
    if not (ROOT / "output" / "sales_history.csv").exists():
        subprocess.run([sys.executable, str(ROOT / "main.py")], cwd=ROOT, check=True)


def _notebook_script(nb):
    lines = [
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as _plt",
        "_plt.show = lambda *a, **k: None",
    ]
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        for line in c["source"]:
            s = line.rstrip("\n")
            if s.lstrip().startswith(("%", "!")):  # saltar magics / shell
                continue
            lines.append(s)
        lines.append("")
    return "\n".join(lines)


def _missing_dependency(code):
    mods = set(re.findall(r"^\s*(?:from|import)\s+([a-zA-Z0-9_]+)", code, re.M))
    for mod in mods - _BASE:
        try:
            importlib.import_module(mod)
        except Exception:
            return mod
    return None


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NB_IDS)
def test_notebook_ejecuta(path):
    if path.stem in SKIP:
        pytest.skip(SKIP[path.stem])
    pytest.importorskip("matplotlib", reason="notebooks requieren matplotlib")

    nb = json.loads(path.read_text(encoding="utf-8"))
    code = _notebook_script(nb)

    missing = _missing_dependency(code)
    if missing:
        pytest.skip(f"falta dependencia de análisis: {missing}")

    _ensure_dataset()
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        exec(compile(code, str(path), "exec"), {"__name__": "__main__"})
    finally:
        os.chdir(cwd)
        import matplotlib.pyplot as plt

        plt.close("all")
