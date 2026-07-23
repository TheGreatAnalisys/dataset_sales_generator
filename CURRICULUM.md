# Currículum — *Forecasting de Ventas de la A a la Z*

Mapa pedagógico de la serie: el **orden** de los videos/notebooks, **por qué** cada
uno va donde va, y las **dependencias** entre ellos. Este documento es la fuente de
verdad del orden; los notebooks deben respetarlo en sus referencias cruzadas.

> **Video 1** es el generador de datos (`main.py` + este repo). Los notebooks
> arrancan en el Video 2. La serie tiene **22 videos** en **4 fases**.

---

## Las 4 fases

| Fase | Videos | Meta | Pregunta que responde |
|---|---|---|---|
| **1 · Fundamentos** | 2–6 | Entender la serie | ¿Qué hay dentro de estos datos? |
| **2 · Negocio y Features** | 7–10 | Preparar la señal | ¿Qué SKUs importan y con qué variables? |
| **3 · Modelado** | 11–18 | El zoológico de modelos | ¿Qué modelo predice mejor, y cómo lo mido? |
| **4 · Rigor y Producción** | 19–22 | Evaluar, tunear, entregar | ¿En qué confío y cómo lo llevo a una decisión? |

---

## Orden detallado y dependencias

### Fase 1 · Fundamentos *(ya publicada — no reordenar)*

| Video | Notebook | Depende de | Por qué va aquí |
|---|---|---|---|
| **2** | EDA | — | Da la **intuición** de tendencia y estacionalidad viéndolas. |
| **3** | Estacionariedad (ADF/KPSS, transformaciones) | V2 | Formaliza lo que el EDA insinuó e introduce las **herramientas de estabilización**: log (varianza) y diferenciación (tendencia). |
| **4** | Descomposición (Clásica vs STL) | V3 | **Consume el log** del V3 para separar componentes multiplicativos. Aquí no se *eliminan*: se *separan para entender*. |
| **5** | Autocorrelación (ACF/PACF) | V3, V4 | ACF/PACF solo son interpretables sobre serie **estacionaria/desestacionalizada** → por eso van después de V3 y V4. Identifican `p,q` para ARIMA. |
| **6** | Detección de Outliers vs. Eventos | V2–V5 | Cierra fundamentos: distingue anomalía real de evento comercial conocido. |

> **Nota conceptual — Estacionariedad (V3) vs. Descomposición (V4): dos lentes, no dos pasos.**
> Es la confusión más común de esta fase. Aclararla explícitamente en el V4.
>
> | | Estacionariedad (V3) | Descomposición (V4) |
> |---|---|---|
> | Objetivo | **Preparar para modelar** (ARIMA exige estacionariedad) | **Entender e interpretar** los componentes |
> | Tendencia | La **elimina** por diferenciación `(1−B)` | La **estima y separa** (no la borra) |
> | Estacionalidad | La elimina por diferenciación estacional `(1−Bˢ)` | La **aísla** (para desestacionalizar / re-estacionalizar) |
> | Logaritmo | Estabiliza **varianza** (multiplicativo→aditivo) | Igual: linealiza para poder separar |
>
> **El logaritmo NO elimina tendencia ni estacionalidad** — solo estabiliza varianza.
> La estacionalidad, en el mundo de la estacionariedad, se quita con diferenciación
> **estacional** `(1−Bˢ)`, nunca con log ni con diferenciación simple.

### Fase 2 · Negocio y Features

| Video | Notebook | Depende de | Por qué va aquí |
|---|---|---|---|
| **7** | Diagnóstico de Negocio / Tiers | V2–V6 | Bisagra: segmenta SKUs (A/B regulares vs C intermitentes). Decide **qué merece un modelo dedicado** — motiva todo lo que sigue. |
| **8** | Features de Calendario | V7 | Reconstruye quincenas/festivos (fuente única: `src/calendar_mx.py`). Se usan pronto (SARIMAX exógenas V13, holidays de Prophet V14). |
| **9** | Lags y Ventanas Móviles | V7 | Memoria de la serie sin fuga temporal. **Su consumidor principal es GBM (V16)** → recordar la conexión ahí. |
| **10** | Precio y Promoción | V7 | Elasticidad-precio por categoría; cierra la preparación de señal. |

### Fase 3 · Modelado

> **Cambio clave respecto a la versión anterior:** el **Video 11 (Evaluación 101)**
> es nuevo y va **antes** del primer modelo. Sin él, los videos 12–18 comparaban
> modelos con métricas (WAPE/MASE) que solo se explicaban al final de la serie —
> una referencia hacia adelante que dejaba sin fundamento cada comparación.

| Video | Notebook | Depende de | Por qué va aquí |
|---|---|---|---|
| **11** | **Evaluación 101** *(NUEVO)* | V7 | La **brújula** antes de modelar: split train/test **temporal** + WAPE, MASE y bias. Todo modelo posterior se juzga con esto. El split es simple y honesto (un solo holdout); el V20 lo hace robusto. |
| **12** | Baselines Inteligentes | V11 | El piso que todo modelo debe superar (naive estacional, drift). |
| **13** | ARIMA / SARIMA / SARIMAX | V3, V5, V11 | Modelo clásico; usa estacionariedad (V3) y órdenes de ACF/PACF (V5). |
| **14** | Prophet | V11 | Alternativa aditiva con holidays (V8). |
| **15** | Demanda Intermitente (Croston/SBA/TSB) | V7, V11 | El tier C del V7; métricas basadas en tasa, no en acierto de fecha. |
| **16** | XGBoost / LightGBM | V8, V9, V10, V11 | **Cierra el préstamo** de features: consume lags (V9), calendario (V8), precio (V10). |
| **17** | Forecasting Jerárquico | V11 | Reconciliación SKU→categoría→total (`src/hierarchy.py`). |
| **18** | Foundation Models (Chronos) | V11 | Zero-shot; cierra el zoológico de modelos. |

### Fase 4 · Rigor y Producción

| Video | Notebook | Depende de | Por qué va aquí |
|---|---|---|---|
| **19** | **Métricas avanzadas** | V11 | Profundiza lo que el primer (V11) no cubrió: **sMAPE, FVA, las trampas del MAPE** y descomposición del sesgo. (Antes era el V18 "Métricas".) |
| **20** | Validación Cruzada Temporal | V11, V13–V18 | Walk-forward: convierte el split único del V11 en una **distribución** de error. Aquí es donde se corrige el "no confíes en un solo split". |
| **21** | Optimización con Optuna | V20 | Tuneo de hiperparámetros — **requiere** un esquema de validación (V20). |
| **22** | Framework Completo | Todo | Capstone: dato → pronóstico por tier → reconciliación → métricas → decisión de inventario (`src/pipeline.py`, `app.py`). |

---

## Resumen del reordenamiento (vs. serie original de 21 videos)

- **+1 video:** se inserta **V11 · Evaluación 101** antes de Baselines.
- **Renumeración:** todo lo que era V11–V21 sube **+1** (Baselines V11→V12, …, Framework V21→V22).
- **V18 "Métricas" → V19 "Métricas avanzadas"** (refocalizado: sMAPE/FVA/MAPE, no lo básico ya visto en V11).
- **Fase 1 (V2–V6) intacta** — ya publicada y con orden correcto.

### Pendientes de implementación

1. **Crear** el notebook `11_Evaluacion_101.ipynb` (WAPE/MASE/bias + split temporal simple).
2. **Renumerar** notebooks 11–21 → 12–22 y actualizar sus referencias cruzadas ("Video N").
3. **Actualizar comentarios en `src/`** que citan números de video/notebook
   (`metrics.py` → "Video 19", `backtest.py` → "notebook 20", `hierarchy.py` → "notebook 17", etc.).
4. **Bug en NB04 (Descomposición), sección 5:** dice *"ACF/PACF viene en el Video 4"* — debe ser **Video 5**.
5. **Añadir en NB04** la celda "dos lentes" (tabla de la Nota conceptual de arriba).
