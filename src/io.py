"""Utilidades de E/S compartidas por los notebooks y la app.

Centraliza la localización de los CSV de `output/` para no repetir el helper en
cada notebook (fuente única, igual que el resto de `src/`).
"""

from pathlib import Path


def find_csv(filename: str = "sales_history.csv", max_levels: int = 4) -> Path:
    """Localiza `output/<filename>` subiendo desde el directorio actual.

    Busca la carpeta `output/` en el cwd y hasta `max_levels` niveles hacia
    arriba, para que funcione tanto desde la raíz del repo como desde
    `notebooks/`. Lanza `FileNotFoundError` con una pista si no existe.
    """
    base = Path().resolve()
    for _ in range(max_levels):
        candidate = base / "output" / filename
        if candidate.exists():
            return candidate
        base = base.parent
    raise FileNotFoundError(f"No se encontró '{filename}'. Corre main.py primero.")
