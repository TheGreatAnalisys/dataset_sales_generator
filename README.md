# Sales History Dataset Generator

Generador de datos sintéticos de ventas históricas con **tendencia**, **estacionalidad**, **eventos especiales** y **múltiples canales de venta**. Ideal para practicar análisis de datos, forecasting y visualización.

---

## Video explicativo

> *[Cómo crear un Dataset de Ventas con Python Forecasting Realista](https://youtu.be/FCKstgzxLwU)*

---

## Estructura del proyecto

```
dataset_sales_generator/
├── .env.example            # Plantilla de configuración (copia y renombra a .env)
├── requirements.txt        # Dependencias del generador (runtime)
├── requirements-dev.txt    # Dependencias adicionales para los notebooks
├── environment.yml         # Entorno conda/mamba equivalente
├── main.py                 # Punto de entrada del generador
├── app.py                  # Dashboard Streamlit del framework (streamlit run app.py)
├── src/
│   ├── config.py           # Carga y validación de variables de entorno
│   ├── catalog.py          # Generación del catálogo de SKUs
│   ├── events.py           # Definición de eventos comerciales
│   ├── calendar_mx.py      # Calendario mexicano: quincenas y festivos
│   ├── seasonality.py      # Factores de estacionalidad mensual y semanal
│   ├── generator.py        # Lógica principal de generación de ventas
│   ├── features.py         # Matriz supervisada para forecasting con ML (lags, ventanas)
│   ├── metrics.py          # Métricas de forecasting (WAPE, MASE, sMAPE, FVA…)
│   ├── intermittent.py     # Demanda intermitente (Croston, SBA, TSB, ADIDA)
│   ├── hierarchy.py        # Reconciliación jerárquica (bottom-up, top-down, MinT)
│   ├── backtest.py         # Validación cruzada temporal (walk-forward)
│   └── pipeline.py         # Framework end-to-end: dato → pronóstico → decisión
├── tests/                  # Suite de pytest (una por módulo de src/)
├── notebooks/              # Serie de análisis y forecasting (EDA, estacionariedad,
│                           #   descomposición, autocorrelación, outliers, modelado,
│                           #   métricas, validación y framework completo)
└── legacy/                 # Versiones antiguas de notebooks (referencia histórica)
```

---

## Instalación

### 1. Clona el repositorio

```bash
git clone https://github.com/thegreatanalisys/dataset_sales_generator.git
cd dataset_sales_generator
```

### 2. Crea y activa un entorno virtual

```bash
# Con mamba / conda
mamba create -n sales_env python=3.11
mamba activate sales_env

# O con venv
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
```

### 3. Instala las dependencias

```bash
# Solo para ejecutar el generador (main.py)
pip install -r requirements.txt

# Para además ejecutar los notebooks de análisis
pip install -r requirements-dev.txt
```

> Alternativa con conda/mamba: `mamba env create -f environment.yml`

### 4. Configura tus variables de entorno

```bash
cp .env.example .env
# Edita .env con los valores que quieras
```

---

## Uso

```bash
python main.py
```

Los archivos generados se guardan en la carpeta `output/` (se crea automáticamente):

| Archivo | Descripción |
|---|---|
| `sales_history.csv` | Dataset principal de ventas |
| `sku_catalog.csv` | Catálogo de los 50 SKUs con sus atributos |
| `sku_tiers.csv` | Segmentación de SKUs en tiers de forecasting (generado en el notebook 07) |

---

## Variables de entorno

Todas las variables tienen valores por defecto, por lo que el script funciona sin `.env`. Puedes sobreescribir cualquiera según tus necesidades.

| Variable | Default | Descripción |
|---|---|---|
| `START_DATE` | `2021-01-01` | Inicio del histórico (formato `YYYY-MM-DD`) |
| `END_DATE` | `2024-12-31` | Fin del histórico |
| `N_SKUS` | `50` | Número de SKUs a generar |
| `RANDOM_SEED` | `42` | Semilla para reproducibilidad |
| `TREND_MIN` | `-0.05` | Tendencia anual mínima por SKU (-5%) |
| `TREND_MAX` | `0.20` | Tendencia anual máxima por SKU (+20%) |
| `NOISE_SIGMA` | `0.15` | Desviación del ruido log-normal en demanda |
| `PRICE_VARIATION` | `0.02` | Variación diaria de precio (±2%) |
| `CHANNELS` | ver `.env.example` | Canales con su multiplicador de precio y share de demanda |
| `OUTPUT_DIR` | `output` | Carpeta de salida |
| `SALES_FILENAME` | `sales_history.csv` | Nombre del archivo de ventas |
| `CATALOG_FILENAME` | `sku_catalog.csv` | Nombre del catálogo de SKUs |

### Formato de `CHANNELS`

Cada canal se define como `Nombre|price_mult|demand_share`, separados por coma. Los `demand_share` deben sumar exactamente `1.0`.

```
CHANNELS=Tienda Física|1.00|0.35,E-commerce|0.92|0.30,Mayorista|0.75|0.25,Marketplace|0.97|0.10
```

---

## Características del dataset

- **4 años** de datos diarios (configurable)
- **50 SKUs** distribuidos en 5 categorías: Electrónica, Ropa, Alimentos, Hogar y Deportes
- **Perfiles de demanda mixtos:** ~70% SKUs regulares + ~30% *slow-movers* de **demanda intermitente** (con días en cero), para practicar segmentación y métodos como Croston
- **Tendencia** anual propia por SKU (crecimiento o decrecimiento)
- **Estacionalidad mensual y semanal** con intensidad variable por SKU
- **4 canales de venta** con precios diferenciados
- **Calendario mexicano:** efecto de **quincena** (pago el 15 y fin de mes) y **festivos nacionales** (16 de septiembre, Día de Muertos, Guadalupe, etc.)
- **Precio y promociones:** demanda con **elasticidad-precio** por categoría (los básicos son inelásticos; los discrecionales, elásticos) y **campañas de promoción** con descuentos reales
- **Eventos especiales** con picos de demanda:

| Evento | Mes | Multiplicador |
|---|---|---|
| Buen Fin | Noviembre | ×3.5 |
| Black Friday | Noviembre | ×3.0 |
| Hot Sale | Mayo | ×2.8 |
| Cyber Monday | Diciembre | ×2.4 |
| Navidad | Diciembre | ×2.2 |
| Día de las Madres | Mayo | ×2.0 |
| Reyes | Enero | ×1.6 |
| San Valentín | Febrero | ×1.5 |

---

## Columnas del dataset

| Columna | Tipo | Descripción |
|---|---|---|
| `date` | date | Fecha de la venta |
| `year` | int | Año |
| `month` | int | Mes |
| `week` | int | Semana del año |
| `weekday` | str | Nombre del día |
| `sku_id` | str | Identificador del producto |
| `category` | str | Categoría del producto |
| `channel` | str | Canal de venta |
| `units_sold` | int | Unidades vendidas |
| `unit_price` | float | Precio unitario (con variación diaria) |
| `revenue` | float | Ingreso total (`units_sold × unit_price`) |
| `event` | str | Evento especial activo (`Regular` si ninguno) |
| `on_promo` | int | 1 si el SKU está en promoción ese día, 0 si no |
| `discount` | float | Profundidad del descuento promocional (0 si no hay promo) |
| `trend_mult` | float | Multiplicador de tendencia acumulado |

---

## Desarrollo

```bash
# Instala dependencias de desarrollo (incluye el runtime)
pip install -r requirements-dev.txt

# Activa los hooks de pre-commit (nbstripout limpia outputs, black formatea)
pre-commit install

# Corre los tests
pytest -q
```

Los notebooks se versionan **sin outputs** — `nbstripout` los limpia automáticamente en cada commit, manteniendo el repositorio ligero. El CI (GitHub Actions) corre `black --check` y `pytest` en cada push y pull request.

---

## Contribuciones

¡Las contribuciones son bienvenidas! Si quieres agregar nuevas categorías, eventos o canales, abre un *Pull Request* o un *Issue*.

---

## Licencia

MIT — libre para uso personal y comercial.
