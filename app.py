"""Dashboard del framework de forecasting — Video 21.

Del dato crudo a la decisión de negocio: ingesta → pronóstico por tier →
reconciliación → métricas → recomendación de inventario en pesos.

Ejecutar con:  streamlit run app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.pipeline import run_framework

st.set_page_config(page_title="Forecasting Framework", page_icon="📦", layout="wide")

BLUE, GREEN, ORANGE, RED = "#2563EB", "#16A34A", "#EA580C", "#DC2626"


def _find(filename: str) -> Path:
    base = Path(__file__).resolve().parent
    for _ in range(4):
        cand = base / "output" / filename
        if cand.exists():
            return cand
        base = base.parent
    raise FileNotFoundError(filename)


@st.cache_data(show_spinner=False)
def load_data():
    sales = pd.read_csv(_find("sales_history.csv"), parse_dates=["date"])
    catalog = pd.read_csv(_find("sku_catalog.csv")).set_index("sku_id")
    return sales, catalog


@st.cache_data(show_spinner=True)
def compute(horizon: int, service_level: float):
    sales, catalog = load_data()
    return run_framework(sales, catalog, horizon=horizon, service_level=service_level)


st.title("📦 Framework de Forecasting — del dato a la decisión")
st.caption(
    "Ingesta → pronóstico por tier (regular / intermitente) → reconciliación → "
    "métricas → recomendación de inventario en pesos."
)

# ── Controles ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parámetros de planeación")
    horizon = st.slider("Horizonte (semanas)", 4, 26, 13)
    service = st.select_slider(
        "Nivel de servicio", options=[0.90, 0.95, 0.975, 0.99], value=0.95
    )
    st.markdown("---")
    st.markdown(
        "El framework planea el **último tramo** de la historia y lo compara con lo "
        "realmente ocurrido, así que muestra a la vez la **recomendación** y su **precisión**."
    )

try:
    res = compute(horizon, service)
except FileNotFoundError:
    st.error(
        "No se encontró `output/sales_history.csv`. Corre `python main.py` primero."
    )
    st.stop()

reco, metrics = res["reco"], res["metrics"]

# ── KPIs ─────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Inversión recomendada", f"${res['total_investment'] / 1e6:,.1f}M")
c2.metric("WAPE (total)", f"{metrics['wape']:.1%}")
c3.metric("Sesgo (BIAS)", f"{metrics['bias']:+.1%}")
c4.metric("SKUs planeados", f"{len(reco):,}")

st.markdown("---")
left, right = st.columns([3, 2])

# ── Pronóstico total vs real ─────────────────────────────────────────────────
with left:
    st.subheader("Pronóstico del total vs. realidad (horizonte de planeación)")
    total_fc = res["forecast"].sum(axis=1)
    total_act = res["actual"].sum(axis=1)
    train_tail = res["train"].sum(axis=1).iloc[-52:]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(
        train_tail.index, train_tail.values, color=BLUE, alpha=0.5, label="histórico"
    )
    ax.plot(
        total_act.index, total_act.values, color="black", linewidth=2.2, label="real"
    )
    ax.plot(
        total_fc.index,
        total_fc.values,
        color=ORANGE,
        linewidth=2,
        linestyle="--",
        label="pronóstico",
    )
    ax.set_ylabel("unidades / semana")
    ax.legend(loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)
    coherence = (
        "✅ coherente (SKU→categoría→total cuadran)"
        if res["coherent"]
        else "⚠️ incoherente"
    )
    st.caption(f"Reconciliación jerárquica: {coherence}")

# ── Inversión por tier ───────────────────────────────────────────────────────
with right:
    st.subheader("Inversión recomendada por tier")
    by_tier = reco.groupby("tier")["reco_pesos"].sum().sort_values()
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.barh(by_tier.index, by_tier.values, color=[GREEN, ORANGE])
    ax2.set_xlabel("pesos")
    fig2.tight_layout()
    st.pyplot(fig2)
    st.caption(
        "Los slow-movers (tier C) concentran muchos SKUs pero poca inversión: "
        "el grueso del capital va a los regulares de alto volumen."
    )

# ── Tabla de recomendación ───────────────────────────────────────────────────
st.subheader("Recomendación de inventario por SKU")
show = reco.reset_index()[
    [
        "sku_id",
        "category",
        "tier",
        "pred_units",
        "safety_stock",
        "reco_units",
        "reco_pesos",
    ]
].head(25)
st.dataframe(
    show.style.format(
        {
            "pred_units": "{:,.0f}",
            "safety_stock": "{:,.0f}",
            "reco_units": "{:,.0f}",
            "reco_pesos": "${:,.0f}",
        }
    ),
    width="stretch",
    hide_index=True,
)
st.caption(
    f"Stock recomendado = pronóstico del horizonte + stock de seguridad "
    f"(nivel de servicio {service:.0%}). Top 25 SKUs por inversión."
)
