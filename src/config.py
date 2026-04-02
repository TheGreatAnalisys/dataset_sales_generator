"""
Carga, parsea y valida todas las variables de entorno.
El resto de módulos importan desde aquí — nunca leen os.environ directamente.
"""

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

load_dotenv()  # Lee .env si existe; las vars ya definidas en el entorno tienen prioridad


# Helpers de parseo
def _get(key: str, default: str | None = None) -> str:
    val = os.getenv(key, default)
    if val is None:
        raise EnvironmentError(f"Variable de entorno requerida no encontrada: '{key}'")
    return val


def _get_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _get_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _parse_date(val: str) -> date:
    return date.fromisoformat(val)


def _parse_channels(raw: str) -> Dict[str, Dict]:
    """
    Parsea: "Nombre|price_mult|demand_share,..."
    Valida que los demand_share sumen ~1.0
    """
    channels = {}
    for entry in raw.split(","):
        parts = entry.strip().split("|")
        if len(parts) != 3:
            raise ValueError(
                f"Canal mal formateado: '{entry}'. Esperado: nombre|price_mult|demand_share"
            )
        name, pm, ds = parts
        channels[name.strip()] = {
            "price_mult": float(pm),
            "demand_share": float(ds),
        }
    total_share = sum(c["demand_share"] for c in channels.values())
    if not (0.99 <= total_share <= 1.01):
        raise ValueError(
            f"Los demand_share deben sumar 1.0 (suma actual: {total_share:.4f})"
        )
    return channels


# Dataclass de configuración
@dataclass(frozen=True)
class Config:
    # Fechas
    start_date: date
    end_date: date

    # Catálogo
    n_skus: int
    random_seed: int

    # Parámetros de simulación
    trend_min: float
    trend_max: float
    noise_sigma: float
    price_variation: float

    # Canales
    channels: Dict[str, Dict] = field(compare=False)

    # Salida
    output_dir: Path
    sales_filename: str
    catalog_filename: str

    # Rutas completas (propiedades derivadas)
    @property
    def sales_path(self) -> Path:
        return self.output_dir / self.sales_filename

    @property
    def catalog_path(self) -> Path:
        return self.output_dir / self.catalog_filename

    def __post_init__(self):
        if self.start_date >= self.end_date:
            raise ValueError("START_DATE debe ser anterior a END_DATE")
        if self.trend_min >= self.trend_max:
            raise ValueError("TREND_MIN debe ser menor que TREND_MAX")
        if self.n_skus < 1:
            raise ValueError("N_SKUS debe ser >= 1")


def load_config() -> Config:
    """Punto de entrada único para obtener la configuración."""
    output_dir = Path(_get("OUTPUT_DIR", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        start_date=_parse_date(_get("START_DATE", "2021-01-01")),
        end_date=_parse_date(_get("END_DATE", "2024-12-31")),
        n_skus=_get_int("N_SKUS", 50),
        random_seed=_get_int("RANDOM_SEED", 42),
        trend_min=_get_float("TREND_MIN", -0.05),
        trend_max=_get_float("TREND_MAX", 0.20),
        noise_sigma=_get_float("NOISE_SIGMA", 0.15),
        price_variation=_get_float("PRICE_VARIATION", 0.02),
        channels=_parse_channels(
            _get(
                "CHANNELS",
                "Tienda Física|1.00|0.35,E-commerce|0.92|0.30,Mayorista|0.75|0.25,Marketplace|0.97|0.10",
            )
        ),
        output_dir=output_dir,
        sales_filename=_get("SALES_FILENAME", "sales_history.csv"),
        catalog_filename=_get("CATALOG_FILENAME", "sku_catalog.csv"),
    )
