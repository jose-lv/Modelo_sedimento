"""
Interpretación con SHAP (RF-15) e importancia por permutación como control.

SHAP explica el modelo *ya transformado* (imputación + escalado), por eso se
aplica sobre la salida de todos los pasos del pipeline excepto el estimador.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from . import config as cfg
from . import plots
from .utils import LOG, guardar_csv, guardar_json


def _preprocesador(pipe):
    """Todos los pasos del pipeline menos el estimador final."""
    return pipe[:-1]


def _valores_shap(modelo, X_trans: pd.DataFrame):
    """Devuelve una matriz (n × features) de valores SHAP."""
    estimador = modelo[-1]
    nombre = type(estimador).__name__
    if nombre in {"RandomForestRegressor", "RandomForestClassifier",
                  "XGBRegressor", "XGBClassifier", "ExtraTreesRegressor"}:
        explainer = shap.TreeExplainer(estimador)
    elif nombre in {"RidgeCV", "LogisticRegression", "LinearRegression"}:
        explainer = shap.LinearExplainer(estimador, X_trans)
    else:
        explainer = shap.Explainer(estimador.predict, X_trans)

    valores = explainer.shap_values(X_trans)
    if isinstance(valores, list):          # clasificadores de dos clases
        valores = valores[-1]
    valores = np.asarray(valores)
    if valores.ndim == 3:                  # (n, features, clases)
        valores = valores[:, :, -1]
    base = getattr(explainer, "expected_value", 0.0)
    if isinstance(base, (list, np.ndarray)):
        base = float(np.ravel(base)[-1])
    return valores, float(base), explainer


def ejecutar(modelo, matriz: dict, tarea: str, top: int = 15) -> dict:
    """Calcula SHAP, figuras y tabla de contribuciones por muestra."""
    X = matriz["X"]
    pre = _preprocesador(modelo)
    X_trans = pd.DataFrame(pre.transform(X), columns=X.columns, index=X.index)

    valores, base, explainer = _valores_shap(modelo, X_trans)
    LOG.info("SHAP: %d muestras × %d variables (modelo %s)",
             *valores.shape, type(modelo[-1]).__name__)

    # --- Importancia global ----------------------------------------------
    imp = pd.DataFrame({
        "variable": X.columns,
        "shap_abs_medio": np.abs(valores).mean(axis=0),
        "shap_medio": valores.mean(axis=0),
    }).sort_values("shap_abs_medio", ascending=False).reset_index(drop=True)
    imp["ranking"] = np.arange(1, len(imp) + 1)
    guardar_csv(imp, cfg.TABLES / f"shap_importancia_{tarea}.csv")

    top_vars = imp["variable"].head(top).tolist()
    plots.shap_summary(valores, X_trans, nombre=f"16_shap_summary_{tarea}", max_display=top)
    plots.shap_bar(valores, X_trans, nombre=f"17_shap_bar_{tarea}", max_display=top)
    plots.shap_dependence(valores, X_trans, top_vars[:5], nombre=f"18_shap_dependence_{tarea}")

    # --- Contribución por muestra (para el ranking del tablero) -----------
    idx_top = np.argmax(np.abs(valores), axis=1)
    contrib = pd.DataFrame({
        "variable_mas_influyente": X.columns[idx_top],
        "shap_variable_top": valores[np.arange(len(valores)), idx_top],
        "shap_abs_max": np.abs(valores).max(axis=1),
    }, index=X.index)
    contrib = contrib.join(
        matriz["meta"][["id_muestra", cfg.COLUMNA_GRUPO, "DISTRITO"]], how="left"
    )

    # --- Waterfall para 3 muestras críticas ------------------------------
    orden = np.argsort(-np.abs(valores).sum(axis=1))[:3]
    criticas = []
    for i, pos in enumerate(orden, start=1):
        fila = X_trans.iloc[pos]
        ruta = plots.shap_waterfall(
            valores[pos], base, fila,
            titulo=f"Muestra crítica {i}",
            nombre=f"19_shap_waterfall_{tarea}_{i}",
        )
        criticas.append({
            "id_muestra": matriz["meta"]["id_muestra"].iloc[pos],
            "shap_total_abs": float(np.abs(valores[pos]).sum()),
            "figura": ruta.name if ruta else None,
        })

    # --- Importancia por permutación (control independiente) -------------
    try:
        pi = permutation_importance(
            modelo, X, matriz["y"], n_repeats=10, random_state=cfg.SEED,
            scoring="r2" if matriz["tipo"] == "regresion" else "roc_auc",
            n_jobs=1,
        )
        tabla_pi = pd.DataFrame({
            "variable": X.columns,
            "importancia_media": pi.importances_mean,
            "importancia_sd": pi.importances_std,
        }).sort_values("importancia_media", ascending=False).reset_index(drop=True)
        guardar_csv(tabla_pi, cfg.TABLES / f"importancia_permutacion_{tarea}.csv")
        plots.importancia_permutacion(tabla_pi)
    except Exception as exc:  # pragma: no cover
        LOG.warning("No se pudo calcular la importancia por permutación: %s", exc)
        tabla_pi = pd.DataFrame()

    guardar_json({
        "tarea": tarea,
        "modelo": type(modelo[-1]).__name__,
        "top_5": imp.head(5).to_dict(orient="records"),
        "muestras_criticas": criticas,
        "concordancia_top10_shap_vs_permutacion": (
            len(set(imp["variable"].head(10)) & set(tabla_pi["variable"].head(10)))
            if len(tabla_pi) else None
        ),
    }, cfg.TABLES / f"resumen_shap_{tarea}.json")

    return {
        "importancia": imp,
        "contribucion": contrib,
        "valores": valores,
        "X_transformado": X_trans,
        "base": base,
        "explainer": explainer,
        "importancia_permutacion": tabla_pi,
    }
