import numpy as np

# Índice mensual (promedio = 1). Índice 0 = enero.
MONTHLY_INDEX = np.array(
    [
        0.80,
        0.75,
        0.85,
        0.90,
        1.10,
        0.95,
        0.90,
        0.88,
        0.92,
        1.00,
        1.25,
        1.45,
    ]
)

# Factor por día de la semana (lun=0 … dom=6)
WEEKDAY_FACTOR = np.array([0.85, 0.88, 0.92, 0.95, 1.10, 1.25, 1.05])


def monthly_factor(month: int, seas_strength: float) -> float:
    """month: 1–12"""
    return 1 + seas_strength * (MONTHLY_INDEX[month - 1] - 1)


def weekday_factor(dayofweek: int) -> float:
    """dayofweek: 0 (lunes) – 6 (domingo)"""
    return WEEKDAY_FACTOR[dayofweek]
