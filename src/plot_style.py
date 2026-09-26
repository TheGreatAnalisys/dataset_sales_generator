"""Estilo de gráficas de la marca "The Analytical Eye" para los notebooks.

Fuente única de colores y tipografía, igual que el resto de `src/`: cada notebook
llama a `use_brand_style()` en su primera celda y pinta con la paleta que devuelve
(`pal.cyan`, `pal.magenta`, `pal.ink`, …) en vez de códigos de color fijos. Así
toda la serie se ve igual y pasar del tema oscuro al claro es una sola línea.

- `use_brand_style()`: matplotlib. Tema oscuro por defecto (el de los videos).
- `use_brand_plotly()`: la plantilla equivalente para Plotly.
- `recolor()`: lleva a la paleta las figuras que otra librería dibuja con sus
  propios colores (por ejemplo, los componentes de Prophet).
"""

from __future__ import annotations

import locale
import logging
from dataclasses import dataclass
from pathlib import Path

_STYLES_DIR = Path(__file__).parent / "styles"
_STYLE_FILES = {
    "dark": "analytical-eye.mplstyle",
    "light": "analytical-eye-light.mplstyle",
}

# Mismos valores que los .mplstyle; las series siguen el orden del ciclo C0…C7.
_COLORS = {
    "dark": {
        "cyan": "#00E5FF",
        "magenta": "#FF176B",
        "orange": "#FF6A00",
        "lavender": "#B9A8E8",
        "mint": "#3DDC97",
        "blue": "#5B8CFF",
        "amber": "#FFD166",
        "neutral": "#8B7FB0",
        "ink": "#E6E3F0",
        "bg": "#08090D",
        "faint": "#4A4166",
        "grid": "#1E1638",
    },
    "light": {
        "cyan": "#0097B2",
        "magenta": "#FF176B",
        "orange": "#E65C00",
        "lavender": "#6D4FC2",
        "mint": "#0E9F6E",
        "blue": "#3B6FE0",
        "amber": "#B7791F",
        "neutral": "#857D9C",
        "ink": "#170B2C",
        "bg": "#FFFFFF",
        "faint": "#D5D0E3",
        "grid": "#EEEBF4",
    },
}

# Colores de interfaz que solo usa la plantilla de Plotly.
_UI = {
    "dark": {
        "spine": "#2A2140",
        "label": "#C9C3DC",
        "panel": "#0C0A16",
        "active": "#2A1A4A",
        "scale": ["#170B2C", "#FF176B", "#FF6A00", "#FFD166"],
    },
    "light": {
        "spine": "#D9D4E6",
        "label": "#3D3452",
        "panel": "#F6F4FA",
        "active": "#E4DEF0",
        "scale": ["#F3F0F8", "#B9A8E8", "#FF176B", "#170B2C"],
    },
}

_FONT_STACK = "JetBrains Mono, Cascadia Mono, Consolas, monospace"

# Fuentes de emoji con contornos: matplotlib no dibuja las de mapa de bits a color.
_EMOJI_FONTS = ("Segoe UI Emoji", "Noto Emoji")


@dataclass(frozen=True)
class Palette:
    """Colores del tema activo.

    Series (C0…C6 del ciclo): cyan, magenta, orange, lavender, mint, blue, amber.
    Apoyo:
    - neutral: referencias suaves y series de comparación (C7).
    - ink: texto y la serie "real"; va donde antes iba "black".
    - bg: el fondo; va donde antes iba "white" (bordes que separan barras o
      puntos, texto sobre un relleno de color).
    - faint: series de contexto, muy tenues.
    - grid: la cuadrícula.
    """

    theme: str
    cyan: str
    magenta: str
    orange: str
    lavender: str
    mint: str
    blue: str
    amber: str
    neutral: str
    ink: str
    bg: str
    faint: str
    grid: str

    @property
    def series(self) -> list[str]:
        """Los 8 colores del ciclo, en orden (C0…C7)."""
        return [
            self.cyan,
            self.magenta,
            self.orange,
            self.lavender,
            self.mint,
            self.blue,
            self.amber,
            self.neutral,
        ]

    @property
    def cmap(self):
        """Mapa de color secuencial del fondo al cian (matrices y heatmaps simples)."""
        from matplotlib.colors import LinearSegmentedColormap

        return LinearSegmentedColormap.from_list(
            f"analytical-eye-{self.theme}", [self.bg, self.cyan]
        )


def palette(theme: str = "dark") -> Palette:
    """Devuelve la paleta de un tema sin tocar la configuración de matplotlib."""
    if theme not in _COLORS:
        raise ValueError(f"theme debe ser 'dark' o 'light', no {theme!r}")
    return Palette(theme=theme, **_COLORS[theme])


def use_brand_style(theme: str = "dark", spanish_dates: bool = True) -> Palette:
    """Aplica el estilo de matplotlib de la marca y devuelve su paleta.

    `theme`: "dark" (fondo #08090D, el mismo del perfil de grabación) o "light"
    (fondo blanco). Con `spanish_dates`, los ejes de fecha salen en español
    (ene., abr., jul.) si el sistema tiene ese locale; si no, quedan como estén.
    """
    import matplotlib.pyplot as plt

    pal = palette(theme)
    plt.style.use(str(_STYLES_DIR / _STYLE_FILES[theme]))
    _add_emoji_fallback()
    if spanish_dates:
        _use_spanish_month_names()
    return pal


def use_brand_plotly(theme: str = "dark") -> Palette:
    """Registra la plantilla de Plotly de la marca, la deja por defecto y devuelve la paleta.

    Las figuras que no piden otra plantilla (`template=...`) usan esta.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    pal = palette(theme)
    ui = _UI[theme]
    scale = [[i / (len(ui["scale"]) - 1), c] for i, c in enumerate(ui["scale"])]
    axis = dict(
        gridcolor=pal.grid,
        linecolor=ui["spine"],
        zerolinecolor=ui["spine"],
        tickfont=dict(color=pal.neutral),
        title=dict(font=dict(color=ui["label"])),
    )
    # El selector de rango y el slider solo existen en el eje X.
    xaxis = dict(
        axis,
        rangeselector=dict(
            bgcolor=ui["panel"],
            activecolor=ui["active"],
            bordercolor=ui["spine"],
            font=dict(color=pal.ink),
        ),
        rangeslider=dict(bgcolor=ui["panel"], bordercolor=ui["spine"]),
    )
    colorbar = dict(outlinewidth=0, tickfont=dict(color=pal.neutral))
    template = go.layout.Template(
        layout=dict(
            paper_bgcolor=pal.bg,
            plot_bgcolor=pal.bg,
            font=dict(family=_FONT_STACK, color=pal.ink, size=12),
            title=dict(x=0.01, xanchor="left", font=dict(size=17, color=pal.ink)),
            colorway=pal.series,
            colorscale=dict(sequential=scale),
            coloraxis=dict(colorbar=colorbar),
            xaxis=xaxis,
            yaxis=axis,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=ui["label"])),
            hoverlabel=dict(
                bgcolor=ui["panel"],
                bordercolor=ui["spine"],
                font=dict(family=_FONT_STACK, color=pal.ink),
            ),
            annotationdefaults=dict(font=dict(color=ui["label"])),
        ),
        data=dict(
            heatmap=[go.Heatmap(colorscale=scale, colorbar=colorbar)],
            treemap=[go.Treemap(marker=dict(line=dict(color=pal.bg, width=1)))],
        ),
    )
    name = "analytical-eye" if theme == "dark" else "analytical-eye-light"
    pio.templates[name] = template
    pio.templates.default = name
    return pal


def recolor(fig, color: str) -> None:
    """Pinta con `color` las líneas y bandas de una figura que dibujó otra librería."""
    for ax in fig.axes:
        for line in ax.get_lines():
            line.set_color(color)
        for band in ax.collections:
            band.set_facecolor(color)


class _EmojiWeightFilter(logging.Filter):
    """Calla el aviso "Failed to find font weight" de la fuente de emoji.

    La fuente de emoji solo trae peso regular; en títulos seminegritos matplotlib
    avisa en cada notebook aunque el emoji se dibuje bien.
    """

    def __init__(self, family: str) -> None:
        super().__init__()
        self.family = family

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not ("Failed to find font weight" in message and self.family in message)


def _add_emoji_fallback() -> None:
    """Respaldo para los emojis de títulos y etiquetas (🟣, ✅, ❌), si hay fuente."""
    from matplotlib import font_manager, rcParams

    for family in _EMOJI_FONTS:
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        rcParams["font.family"] = [*rcParams["font.family"], family]
        logger = logging.getLogger("matplotlib.font_manager")
        if not any(isinstance(f, _EmojiWeightFilter) for f in logger.filters):
            logger.addFilter(_EmojiWeightFilter(family))
        return


def _use_spanish_month_names() -> None:
    """Meses en español en los ejes de fecha, con el primer locale disponible."""
    for name in ("es_MX.UTF-8", "es_ES.UTF-8", "es_MX", "Spanish_Mexico.1252"):
        try:
            locale.setlocale(locale.LC_TIME, name)
            return
        except locale.Error:
            continue
