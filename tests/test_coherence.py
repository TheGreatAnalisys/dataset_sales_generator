"""Guardrails de coherencia de la serie de notebooks.

Chequeos ESTÁTICOS (sin kernel ni dependencias pesadas) que fijan lo que
auditamos a mano: numeración de videos, fases, cadena "Próximo video",
referencias cruzadas a modelos y consistencia README ↔ código.

Corre en el CI normal (`pytest`). Si un reordenamiento futuro rompe algo,
esto lo marca antes del merge.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
NB_IDS = [p.stem for p in NOTEBOOKS]

# Fase esperada por número de video (fuente de verdad del CURRICULUM).
PHASE_OF = {
    **{n: 1 for n in range(2, 8)},  # Fundamentos y Diagnóstico
    **{n: 2 for n in range(8, 11)},  # Feature Engineering
    **{n: 3 for n in range(11, 19)},  # Modelado
    **{n: 4 for n in range(19, 23)},  # Rigor y Producción
}
LAST_VIDEO = max(PHASE_OF)

# Cada modelo/tema vive en UN video: una referencia "Modelo (Vn)" debe usar ese n.
KEYWORD_VIDEO = {
    "SARIMAX": 13,
    "Prophet": 14,
    "Croston": 15,
    "XGBoost": 16,
    "LightGBM": 16,
    "Chronos": 18,
    "Holt-Winters": 12,
    "Optuna": 21,
}


def _markdown(nb):
    return "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"
    )


def _all_text(nb):
    return "\n".join("".join(c["source"]) for c in nb["cells"])


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _num(path):
    return int(path.name[:2])


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NB_IDS)
def test_header_matches_filename_and_phase(path):
    """El header `Fase X · Video Y` coincide con el número de archivo y su fase."""
    nb = _load(path)
    m = re.search(r"Fase\s*(\d)\s*·\s*Video\s*(\d+)", _markdown(nb))
    assert m, f"{path.name}: falta el header 'Fase X · Video Y'"
    phase, video = int(m.group(1)), int(m.group(2))
    num = _num(path)
    assert video == num, f"{path.name}: header dice Video {video}, el archivo es {num}"
    assert (
        phase == PHASE_OF[num]
    ), f"{path.name}: header dice Fase {phase}, esperado Fase {PHASE_OF[num]}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NB_IDS)
def test_proximo_video_apunta_al_siguiente(path):
    """La sección 'Próximo video' apunta a Video (N+1), salvo el último."""
    num = _num(path)
    if num == LAST_VIDEO:
        return
    md = _markdown(_load(path))
    idx = md.find("Próximo video")
    if idx == -1:
        pytest.skip("sin sección 'Próximo video'")
    m = re.search(r"Video\s*(\d+)", md[idx:])
    assert m, f"{path.name}: 'Próximo video' sin número"
    assert (
        int(m.group(1)) == num + 1
    ), f"{path.name}: 'Próximo video' apunta a V{m.group(1)}, esperado V{num + 1}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NB_IDS)
def test_referencias_a_modelos_usan_el_video_correcto(path):
    """Una referencia 'Modelo (Vn)' / 'Modelo (Video n)' debe apuntar al video real."""
    text = _all_text(_load(path))
    problemas = []
    for kw, expected in KEYWORD_VIDEO.items():
        for m in re.finditer(
            rf"{re.escape(kw)}[^)\n.]{{0,15}}\((?:V|Video )(\d+)\)", text
        ):
            got = int(m.group(1))
            if got != expected:
                problemas.append(f"'{kw} (V{got})' debería ser V{expected}")
    assert not problemas, f"{path.name}: refs cruzadas incorrectas: {problemas}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NB_IDS)
def test_referencias_de_video_en_rango(path):
    """Ninguna referencia 'Video N' apunta fuera de [1, LAST_VIDEO]."""
    text = _all_text(_load(path))
    for n in re.findall(r"Video\s*(\d+)", text):
        assert 1 <= int(n) <= LAST_VIDEO, f"{path.name}: 'Video {n}' fuera de rango"


# ─────────────────────────── README ↔ código ───────────────────────────


def _read(p):
    return (ROOT / p).read_text(encoding="utf-8")


def test_readme_lista_las_categorias():
    readme = _read("README.md")
    cats = re.findall(r'"([^"]+)":\s*\{"base_price_range"', _read("src/catalog.py"))
    assert len(cats) == 5
    faltan = [c for c in cats if c not in readme]
    assert not faltan, f"README no menciona categorías: {faltan}"


def test_readme_lista_eventos_con_multiplicadores():
    readme = _read("README.md")
    eventos = re.findall(
        r'SpecialEvent\("([^"]+)",\s*\d+,\s*\d+,\s*\d+,\s*([\d.]+)\)',
        _read("src/events.py"),
    )
    assert len(eventos) == 8
    faltan = [f"{n}×{mult}" for n, mult in eventos if mult not in readme]
    assert not faltan, f"README no refleja multiplicadores de eventos: {faltan}"


def test_readme_refleja_defaults_de_config():
    readme = _read("README.md")
    cfg = _read("src/config.py")
    defaults = dict(re.findall(r'_get(?:_int|_float)?\("(\w+)",\s*"?([^")]+)"?\)', cfg))
    faltan = []
    for key in [
        "START_DATE",
        "END_DATE",
        "N_SKUS",
        "RANDOM_SEED",
        "TREND_MIN",
        "TREND_MAX",
        "NOISE_SIGMA",
        "PRICE_VARIATION",
    ]:
        val = defaults[key].strip().strip('"')
        if val not in readme:
            faltan.append(f"{key}={val}")
    assert not faltan, f"README no refleja defaults: {faltan}"


def test_readme_lista_las_columnas_del_dataset():
    readme = _read("README.md")
    gen = _read("src/generator.py")
    cols = list(
        dict.fromkeys(re.findall(r'"(\w+)":\s', gen.split("pd.DataFrame(")[-1]))
    )
    assert len(cols) >= 14
    faltan = [c for c in cols if c not in readme]
    assert not faltan, f"README no lista columnas: {faltan}"


def test_readme_lista_los_archivos_de_salida():
    readme = _read("README.md")
    for f in ["sales_history.csv", "sku_catalog.csv", "sku_tiers.csv"]:
        assert f in readme, f"README no menciona la salida {f}"


# Paquete → video donde se usa (para comentarios de los archivos de dependencias).
PKG_VIDEO = {
    "prophet": 14,
    "xgboost": 16,
    "lightgbm": 16,
    "pytorch": 18,
    "torch": 18,
    "chronos": 18,
    "optuna": 21,
    "streamlit": 22,
}


@pytest.mark.parametrize("depfile", ["requirements-dev.txt", "environment.yml"])
def test_comentarios_de_dependencias_citan_el_video_correcto(depfile):
    """Un comentario 'paquete ... (Video N)' debe citar el video donde se usa."""
    problemas = []
    for line in _read(depfile).splitlines():
        m = re.search(r"Video\s*(\d+)", line)
        if not m:
            continue
        video = int(m.group(1))
        low = line.lower()
        for pkg, expected in PKG_VIDEO.items():
            if pkg in low and video != expected:
                problemas.append(f"'{pkg}' cita Video {video}, esperado {expected}")
    assert not problemas, f"{depfile}: {problemas}"
