"""Calendario del retail mexicano: quincenas (días de pago) y festivos.

Fuente única de verdad compartida por el generador (que inyecta el efecto en la
demanda) y por el notebook de feature engineering (que reconstruye los features y
mide su señal). Así el dataset y las variables del modelo siempre coinciden.
"""

from dataclasses import dataclass

import numpy as np

# En México se cobra el día 15 y el último día del mes. El gasto sube en la
# ventana inmediatamente alrededor de cada pago.
QUINCENA_STRENGTH = 0.30

# Días previos (además del propio día) con "compras de anticipación" a un festivo.
HOLIDAY_PRE_WINDOW = 2


@dataclass(frozen=True)
class Holiday:
    name: str
    month: int
    day: int
    mult: float


# Festivos de fecha fija relevantes para el retail mexicano. Se evitan solapes con
# los eventos comerciales de events.py (Reyes, Buen Fin, Navidad, etc.).
MX_HOLIDAYS: list[Holiday] = [
    Holiday("Año Nuevo", 1, 1, 1.20),
    Holiday("Día de la Constitución", 2, 5, 1.15),
    Holiday("Natalicio de Juárez", 3, 21, 1.12),
    Holiday("Día del Trabajo", 5, 1, 1.18),
    Holiday("Independencia", 9, 16, 1.30),
    Holiday("Día de Muertos", 11, 2, 1.25),
    Holiday("Revolución", 11, 20, 1.15),
    Holiday("Virgen de Guadalupe", 12, 12, 1.28),
]


def quincena_mask(dates) -> np.ndarray:
    """Bool por fecha: ¿cae en la ventana de gasto de quincena?

    `dates` es un DatetimeIndex (o similar con .day / .days_in_month).
    Ventanas: mediados (15–17) y fin/inicio de mes (últimos 2 días + días 1–2).
    """
    day = np.asarray(dates.day)
    dim = np.asarray(dates.days_in_month)
    mid = (day >= 15) & (day <= 17)
    end = (day >= dim - 1) | (day <= 2)
    return mid | end


def quincena_factor_series(dates) -> np.ndarray:
    """Multiplicador de demanda por quincena para cada fecha."""
    return np.where(quincena_mask(dates), 1.0 + QUINCENA_STRENGTH, 1.0)


def holiday_factor_series(dates, pre_window: int = HOLIDAY_PRE_WINDOW):
    """Multiplicador y nombre de festivo por fecha (incluye ventana pre-festivo)."""
    n = len(dates)
    mult = np.ones(n)
    names = np.array([""] * n, dtype=object)
    month = np.asarray(dates.month)
    day = np.asarray(dates.day)
    for h in MX_HOLIDAYS:
        idxs = np.where((month == h.month) & (day == h.day))[0]
        for i in idxs:
            for off in range(pre_window + 1):
                j = i - off
                if j >= 0:
                    mult[j] *= h.mult
                    if off == 0:
                        names[j] = h.name
    return mult, names
