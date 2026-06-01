# MECTESIS — Código reproducible (anexo de tesis)

Código y datos para reproducir los experimentos de la tesis sobre **comparación de
modelos clásicos de series de tiempo frente a modelos fundacionales** (Amazon
**Chronos-2**), tanto en simulaciones Monte Carlo como en datos macroeconómicos
argentinos reales.

Este repositorio es una versión **limpia y autocontenida** pensada para acompañar el
anexo de la tesis: contiene únicamente el paquete `mectesis`, los 4 notebooks
principales, el dato de origen y los resultados de simulación ya generados.

## Estructura

```
.
├── mectesis/                     # Paquete Python (DGPs, modelos, engines, métricas, empírico)
│   ├── dgp/                      # Procesos generadores de datos (RW, ARIMA, GARCH, VAR, VECM, ARIMAX, ...)
│   ├── models/                   # Modelos: ARIMA/SARIMA/ETS/Theta, VAR/VECM, GARCH, Chronos, ...
│   ├── simulation/               # Motores Monte Carlo (uni / multi / covariables)
│   ├── metrics/                  # Descomposición sesgo-varianza, Trace-MSFE, CRPS
│   └── empirical/               # Loaders, transforms, backtest rolling-origin, auto-selección
├── data/
│   └── raw/
│       └── data.csv              # Dato de origen (series macro BCRA/INDEC)
├── notebooks/
│   ├── experimentos_univariados.ipynb
│   ├── experimentos_multivariados.ipynb
│   ├── experimentos_covariados.ipynb
│   ├── validacion_empirica.ipynb   # Figuras + tabla resumen sobre datos reales (autocontenido)
│   └── results/                  # Resultados de simulación YA generados (versionados)
│       ├── univariados/
│       ├── multivariados/
│       └── covariados/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Instalación

Requiere Python ≥ 3.10.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -e .          # instala el paquete `mectesis` y sus dependencias
```

> La primera vez que se usa un modelo Chronos se descargan los pesos desde Hugging Face.
> Todo corre en **CPU** (no se requiere GPU).

## Cómo correr los notebooks

Abrir Jupyter desde la raíz del repo y ejecutar los notebooks **desde la carpeta
`notebooks/`** (los notebooks asumen ese directorio de trabajo):

```bash
jupyter notebook   # o jupyter lab
```

### Notebooks de simulación (Monte Carlo)
- `experimentos_univariados.ipynb` → `notebooks/results/univariados/`
- `experimentos_multivariados.ipynb` → `notebooks/results/multivariados/`
- `experimentos_covariados.ipynb` → `notebooks/results/covariados/`

Cada uno escribe/lee sus resultados en `notebooks/results/<experimento>/exp_*.csv`.
**Los CSV ya vienen versionados en el repo**: los notebooks cachean por archivo, así que al
re-ejecutarlos detectan los resultados existentes y **saltean la simulación pesada**,
reconstruyendo solo las tablas y figuras de resumen. Para regenerar todo desde cero, borrar
el contenido del directorio de resultados correspondiente.

### Notebook de validación empírica
- `validacion_empirica.ipynb` — **autocontenido**: su único insumo de datos es
  `data/raw/data.csv` (vía `build_panel`). Genera en `outputs/`:
  - `panel_series.pdf`, `forecast_univariado.pdf`, `forecast_multivariado.pdf`,
    `forecast_covariables.pdf`
  - `tabla_resumen_pi.csv` / `tabla_resumen_long.csv`

  La tabla resumen **recomputa** el backtest rolling-origin completo (no lee CSV
  pre-computados). Ese paso refitea cada modelo en cada origen, incluido Chronos en CPU, y
  puede tardar **del orden de minutos a unas horas** según el hardware.

## Dato de origen

`data/raw/data.csv` (export estilo BCRA/INDEC: separador `;`, encoding `latin-1`, decimal
`,`). Contiene 6 series macroeconómicas mensuales (≈ dic-2016 a 2026):

| columna | serie |
|---------|-------|
| `ipc`    | inflación mensual (%) — objetivo principal (π) |
| `tpm`    | tasa de política monetaria (%) |
| `badlar` | tasa BADLAR bancos privados (%) |
| `tcm`    | tipo de cambio mayorista (ARS/USD) |
| `m2`     | variación interanual de M2 privado (%) |
| `rem`    | expectativa de inflación a 12m, mediana REM (%) |
