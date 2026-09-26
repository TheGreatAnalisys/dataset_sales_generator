import locale

import pytest

from src.plot_style import palette, recolor, use_brand_plotly, use_brand_style


def test_palette_expone_los_dos_temas():
    oscuro, claro = palette("dark"), palette("light")
    assert oscuro.bg == "#08090D"
    assert claro.bg == "#FFFFFF"
    assert len(oscuro.series) == len(claro.series) == 8
    assert oscuro.series[0] == oscuro.cyan


def test_palette_rechaza_tema_desconocido():
    with pytest.raises(ValueError, match="theme"):
        palette("sepia")


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_paleta_coincide_con_el_ciclo_del_mplstyle(theme):
    mpl = pytest.importorskip("matplotlib")
    from src.plot_style import _STYLE_FILES, _STYLES_DIR

    rc = mpl.rc_params_from_file(
        _STYLES_DIR / _STYLE_FILES[theme], use_default_template=False
    )
    ciclo = [c.upper() for c in rc["axes.prop_cycle"].by_key()["color"]]
    assert ciclo == [c.upper() for c in palette(theme).series]
    assert rc["axes.facecolor"].upper() == palette(theme).bg


def test_use_brand_style_aplica_el_tema_y_restaura_con_rc_context():
    mpl = pytest.importorskip("matplotlib")
    fechas = locale.setlocale(locale.LC_TIME)
    try:
        with mpl.rc_context():
            pal = use_brand_style("dark")  # sin el locale es_MX no debe fallar
            assert mpl.rcParams["axes.facecolor"].upper() == pal.bg
            assert mpl.rcParams["axes.titlelocation"] == "left"
            # la monoespaciada va primero; la de emoji (si existe) solo es respaldo
            assert mpl.rcParams["font.family"][0] == "monospace"
    finally:
        locale.setlocale(locale.LC_TIME, fechas)


def test_recolor_pinta_lineas_y_bandas():
    pytest.importorskip("matplotlib")
    from matplotlib.colors import to_hex
    from matplotlib.figure import Figure

    fig = Figure()
    ax = fig.subplots()
    (linea,) = ax.plot([0, 1], [0, 1], color="#0072B2")
    banda = ax.fill_between([0, 1], [0, 0], [1, 1], color="#0072B2", alpha=0.2)
    recolor(fig, "#FF176B")
    assert to_hex(linea.get_color()) == "#ff176b"
    assert to_hex(banda.get_facecolor()[0]) == "#ff176b"


def test_use_brand_plotly_deja_la_plantilla_por_defecto():
    pio = pytest.importorskip("plotly.io")
    previa = pio.templates.default
    try:
        pal = use_brand_plotly("light")
        assert pio.templates.default == "analytical-eye-light"
        assert pio.templates["analytical-eye-light"].layout.paper_bgcolor == pal.bg
    finally:
        pio.templates.default = previa
