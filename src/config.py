"""
Configuración central del MVP.

Todas las decisiones metodológicas (semilla, umbrales, metales del índice,
variables excluidas por fuga de datos, fondo geoquímico) viven aquí para que
sean auditables y reproducibles desde un solo lugar.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
CONFIG_DIR = ROOT / "config"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
LOGS = REPORTS / "logs"
MODELS_DIR = ROOT / "models"

for _d in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, CONFIG_DIR, FIGURES, TABLES, LOGS, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

#: Ruta del CSV crudo del OEFA. Se puede sobreescribir con la variable de
#: entorno ``OEFA_CSV`` (útil si el archivo pesa y no se quiere copiar).
RAW_CSV = Path(os.environ.get("OEFA_CSV", DATA_RAW / "Monitoreos_SEDIMENTO.csv"))

# --------------------------------------------------------------------------
# Reproducibilidad
# --------------------------------------------------------------------------
SEED = 42
SEEDS = [42, 7, 2026, 1234, 99]
N_FOLDS = 5
N_BOOTSTRAP = 1000

# --------------------------------------------------------------------------
# Filtros de dominio
# --------------------------------------------------------------------------
DEPARTAMENTO_OBJETIVO = "MOQUEGUA"
MATRIZ_OBJETIVO = "Sedimento"

#: Caja envolvente aproximada de Moquegua en UTM 19S. Se usa para validar
#: coordenadas; los registros fuera de rango se marcan, no se eliminan.
UTM_BBOX = {"este": (200_000, 450_000), "norte": (8_000_000, 8_300_000)}

# --------------------------------------------------------------------------
# Calidad de datos
# --------------------------------------------------------------------------
#: Un parámetro entra al análisis descriptivo si tiene al menos esta cobertura.
COBERTURA_MIN_ANALISIS = 0.10
#: Un parámetro entra al modelo supervisado si tiene al menos esta cobertura.
COBERTURA_MIN_MODELO = 0.60
#: Una muestra se conserva si tiene al menos esta fracción de los parámetros
#: retenidos para modelado.
COBERTURA_MIN_MUESTRA = 0.50
#: Parámetros con más de este % de valores censurados se tratan como binarios.
UMBRAL_CENSURA_BINARIA = 0.80

# --------------------------------------------------------------------------
# Tratamiento de valores bajo el límite de detección (LD)
# --------------------------------------------------------------------------
#: Método principal del MVP (guía, paso 3).
METODO_CENSURA = "ld_medio"  # ld_medio | ld_raiz2 | ld_superior
FACTORES_CENSURA = {"ld_medio": 0.5, "ld_raiz2": 0.5 ** 0.5, "ld_superior": 1.0}

# --------------------------------------------------------------------------
# Diseño analítico: índice de riesgo y fuga de datos
# --------------------------------------------------------------------------
#: Metales que COMPONEN el índice de riesgo (PLI / Igeo / RI).
#: Se eligen porque son los que tienen guía CCME ISQG/PEL disponible.
METALES_INDICE = ["Arsénico", "Cadmio", "Cromo", "Cobre", "Mercurio", "Níquel", "Plomo", "Zinc"]

#: Factores de respuesta tóxica de Hakanson (1980) para el riesgo ecológico.
FACTORES_TOXICOS = {
    "Arsénico": 10, "Cadmio": 30, "Cromo": 2, "Cobre": 5,
    "Mercurio": 40, "Níquel": 5, "Plomo": 5, "Zinc": 1,
}

#: Abundancia media en la corteza continental (Turekian & Wedepohl, 1961),
#: en mg/kg. Se usa solo como análisis de sensibilidad del fondo.
FONDO_CORTEZA = {
    "Arsénico": 1.8, "Cadmio": 0.2, "Cromo": 90.0, "Cobre": 45.0,
    "Mercurio": 0.08, "Níquel": 68.0, "Plomo": 20.0, "Zinc": 95.0,
}

#: Percentil local usado como fondo geoquímico (Bn) en el escenario principal.
PERCENTIL_FONDO = 25
PERCENTILES_SENSIBILIDAD = [10, 25, 50]

#: Columnas que NUNCA pueden ser features porque definen el target.
#: Es la barrera anti-fuga de datos; se verifica en tests/test_no_leakage.py.
COLUMNAS_PROHIBIDAS_FEATURES = {
    "PLI", "PLI_log", "RI", "RI_log", "riesgo_alto", "riesgo_alto_pel",
    "n_metales_sobre_isqg", "n_metales_sobre_pel", "clase_riesgo",
    "igeo_max", "igeo_medio", "cf_max", "cf_medio", "score_metalico",
    "excede_pel", "excede_isqg",
}

#: Grupos espaciales para la validación cruzada (evita fuga espacial).
COLUMNA_GRUPO = "PUNTO_NORMALIZADO"
#: Grupos alternativos que se reportan como análisis de sensibilidad.
COLUMNAS_GRUPO_ALTERNATIVAS = ["PUNTO_NORMALIZADO", "DISTRITO", "CUENCA"]

# --------------------------------------------------------------------------
# Modelado
# --------------------------------------------------------------------------
TAREA_PRINCIPAL = "regresion_pli"
TAREAS = {
    "regresion_pli": {
        "tipo": "regresion",
        "target": "PLI_log",
        "descripcion": "log1p(PLI) de 8 metales con guía CCME, predicho con variables que no componen el índice",
    },
    "clasificacion_riesgo": {
        "tipo": "clasificacion",
        "target": "excede_pel",
        "descripcion": (
            "La muestra supera el PEL (nivel de efecto probable, CCME) de al menos "
            "un metal del índice; mismas entradas sin fuga"
        ),
    },
}

#: Hold-out temporal: se entrena con años <= corte y se prueba con los recientes.
ANIO_CORTE_TEMPORAL = 2023

# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
DPI = 150
FIG_FORMAT = "png"
PALETA_RIESGO = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
