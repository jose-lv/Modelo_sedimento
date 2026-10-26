"""Análisis exploratorio y figuras (RF-11)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from . import plots
from .utils import LOG, guardar_csv


def estadisticos_descriptivos(ancho: pd.DataFrame, params: list[str]) -> pd.DataFrame:
    filas = []
    for p in params:
        if p not in ancho.columns:
            continue
        v = ancho[p].dropna()
        if v.empty:
            continue
        filas.append({
            "parametro": p,
            "n": int(len(v)),
            "media": float(v.mean()),
            "de": float(v.std()),
            "min": float(v.min()),
            "p25": float(v.quantile(0.25)),
            "mediana": float(v.median()),
            "p75": float(v.quantile(0.75)),
            "p90": float(v.quantile(0.90)),
            "max": float(v.max()),
        })
    return pd.DataFrame(filas).sort_values("parametro").reset_index(drop=True)


def ejecutar(riesgo: pd.DataFrame, cobertura: pd.DataFrame, guias: pd.DataFrame) -> dict:
    """Genera tablas y figuras del EDA."""
    params_modelo = cobertura.loc[cobertura["cobertura"] >= cfg.COBERTURA_MIN_MODELO, "parametro"].tolist()
    params_analisis = cobertura.loc[cobertura["cobertura"] >= cfg.COBERTURA_MIN_ANALISIS, "parametro"].tolist()

    estad = estadisticos_descriptivos(riesgo, params_analisis)
    guardar_csv(estad, cfg.TABLES / "estadisticos_descriptivos.csv")

    plots.cobertura_barras(cobertura)
    if "pct_censurado" in cobertura.columns:
        plots.censura_barras(
            cobertura.rename(columns={"parametro": "parametro_estandar", "n_no_nulos": "n"})
        )
    plots.distribuciones(riesgo, params_modelo)
    plots.correlaciones(riesgo, [p for p in params_modelo if p in riesgo.columns][:24])
    plots.boxplot_por_grupo(riesgo, "PLI", "DISTRITO")
    plots.serie_temporal(riesgo, "PLI")
    plots.mapa_puntos(riesgo, "PLI")
    plots.mapa_nivel(riesgo)
    plots.barras_nivel_riesgo(riesgo)
    plots.excedencias_por_parametro(riesgo, guias)

    # Resumen ejecutivo del dataset
    resumen = {
        "n_muestras": int(len(riesgo)),
        "n_puntos": int(riesgo["PUNTO_NORMALIZADO"].nunique()),
        "n_parametros_totales": int(len(cobertura)),
        "n_parametros_modelo": int(len(params_modelo)),
        "n_parametros_analisis": int(len(params_analisis)),
        "anio_min": int(pd.to_numeric(riesgo["ANIO"], errors="coerce").min()),
        "anio_max": int(pd.to_numeric(riesgo["ANIO"], errors="coerce").max()),
        "pct_riesgo_alto_pli": float(100 * riesgo["PLI"].gt(1).mean()),
        "pct_excede_pel": float(100 * riesgo["excede_pel"].mean()),
        "mediana_PLI": float(riesgo["PLI"].median()),
        "mediana_RI": float(riesgo["RI"].median()),
        "distritos": sorted(riesgo["DISTRITO"].dropna().unique().tolist()),
        "nivel_riesgo_conteo": riesgo["nivel_riesgo"].value_counts().to_dict(),
    }
    LOG.info("EDA: %s muestras, %s puntos, %s-%s",
             resumen["n_muestras"], resumen["n_puntos"], resumen["anio_min"], resumen["anio_max"])
    return {"estadisticos": estad, "resumen": resumen}
