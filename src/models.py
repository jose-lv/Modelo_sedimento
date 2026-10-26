"""
Entrenamiento y evaluación comparativa de modelos (RF-13 y RF-14).

Decisiones de validación
------------------------
* **Agrupada**: los folds se construyen por punto de muestreo normalizado, de
  modo que un mismo punto nunca cae en entrenamiento y prueba a la vez.
* **Repetida**: se repite con 5 semillas y se reporta media ± desviación,
  porque con ~400 muestras una sola partición es muy ruidosa.
* **Hold-out temporal**: se entrena hasta 2023 y se prueba con 2024-2026,
  que es el escenario real de uso (predecir campañas futuras).
* La imputación ocurre **dentro** del pipeline, nunca antes de dividir.
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from . import config as cfg
from . import plots
from .utils import LOG, guardar_csv, guardar_json

warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------
# Transformadores auxiliares
# --------------------------------------------------------------------------
class Winsorizer(BaseEstimator, TransformerMixin):
    """Recorta los extremos al percentil ``limite`` y ``1 - limite`` (RF-09)."""

    def __init__(self, limite: float = 0.01):
        self.limite = limite

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        with np.errstate(all="ignore"):
            self.inferior_ = np.nanquantile(X, self.limite, axis=0)
            self.superior_ = np.nanquantile(X, 1 - self.limite, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.inferior_, self.superior_)


# --------------------------------------------------------------------------
# Particiones agrupadas
# --------------------------------------------------------------------------
def _folds_grupos_permutados(grupos: pd.Series, n_splits: int, seed: int):
    """GroupKFold con asignación aleatoria de grupos a folds (repetible)."""
    unicos = np.array(sorted(grupos.astype(str).unique()))
    rng = np.random.default_rng(seed)
    orden = rng.permutation(len(unicos))
    asignacion = {g: int(orden[i] % n_splits) for i, g in enumerate(unicos)}
    fold_de_fila = grupos.astype(str).map(asignacion).to_numpy()
    for k in range(n_splits):
        yield np.where(fold_de_fila != k)[0], np.where(fold_de_fila == k)[0]


def generar_folds(y: pd.Series, grupos: pd.Series, tipo: str, n_splits: int, seed: int):
    """Devuelve índices (train, test) respetando los grupos espaciales."""
    if tipo == "clasificacion":
        y_int = y.astype(int)
        if y_int.value_counts().min() >= n_splits:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return list(cv.split(np.zeros(len(y_int)), y_int, groups=grupos))
    return list(_folds_grupos_permutados(grupos, n_splits, seed))


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
def construir_modelos(tipo: str, seed: int = cfg.SEED) -> dict:
    """Baseline + modelo lineal + dos ensamblados de árboles (guía, sección 8)."""
    imputador = lambda: SimpleImputer(strategy="median")

    if tipo == "regresion":
        return {
            "Media (baseline)": Pipeline([
                ("imp", imputador()),
                ("m", DummyRegressor(strategy="mean")),
            ]),
            "Ridge": Pipeline([
                ("imp", imputador()),
                ("win", Winsorizer(0.01)),
                ("sc", StandardScaler()),
                ("m", RidgeCV(alphas=np.logspace(-3, 3, 25))),
            ]),
            "RandomForest": Pipeline([
                ("imp", imputador()),
                ("m", RandomForestRegressor(
                    n_estimators=500, min_samples_leaf=2, max_features=0.5,
                    random_state=seed, n_jobs=-1,
                )),
            ]),
            "XGBoost": Pipeline([
                ("imp", imputador()),
                ("m", XGBRegressor(
                    n_estimators=400, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                    random_state=seed, n_jobs=-1, verbosity=0,
                )),
            ]),
        }

    return {
        "Prior (baseline)": Pipeline([
            ("imp", imputador()),
            ("m", DummyClassifier(strategy="prior")),
        ]),
        "Regresión logística": Pipeline([
            ("imp", imputador()),
            ("win", Winsorizer(0.01)),
            ("sc", StandardScaler()),
            ("m", __import__("sklearn.linear_model", fromlist=["LogisticRegression"])
             .LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "RandomForest": Pipeline([
            ("imp", imputador()),
            ("m", RandomForestClassifier(
                n_estimators=500, min_samples_leaf=2, max_features=0.5,
                class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("imp", imputador()),
            ("m", XGBClassifier(
                n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss",
                random_state=seed, n_jobs=-1, verbosity=0,
            )),
        ]),
    }


#: Rejillas ligeras para RandomizedSearchCV (con validación agrupada interna).
ESPACIOS_BUSQUEDA = {
    "RandomForest": {
        "m__n_estimators": [300, 500, 800],
        "m__max_depth": [None, 6, 10, 16],
        "m__min_samples_leaf": [1, 2, 4],
        "m__max_features": ["sqrt", 0.3, 0.5, 0.8],
    },
    "XGBoost": {
        "m__n_estimators": [300, 600],
        "m__max_depth": [3, 4, 6],
        "m__learning_rate": [0.03, 0.05, 0.1],
        "m__subsample": [0.7, 0.9],
        "m__colsample_bytree": [0.6, 0.9],
        "m__reg_lambda": [1.0, 3.0],
    },
}


def ajustar_hiperparametros(pipe, nombre, X, y, grupos, tipo, n_iter=20, seed=cfg.SEED):
    """Búsqueda aleatoria con validación agrupada interna (guía, sección 8)."""
    espacio = ESPACIOS_BUSQUEDA.get(nombre)
    if espacio is None:
        return pipe
    cv_inner = generar_folds(y, grupos, tipo, n_splits=3, seed=seed)
    scoring = "r2" if tipo == "regresion" else "f1"
    busca = RandomizedSearchCV(
        clone(pipe), espacio, n_iter=n_iter, cv=cv_inner, scoring=scoring,
        random_state=seed, n_jobs=1, refit=True, error_score=np.nan,
    )
    busca.fit(X, y, groups=grupos)
    LOG.info("    %s: mejores hiperparámetros %s (%.3f)", nombre, busca.best_params_, busca.best_score_)
    return busca.best_estimator_


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------
def _metricas_regresion(y_true_log, y_pred_log) -> dict:
    """
    Métricas en la escala del modelo (log1p) y en la escala interpretable (PLI).
    """
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    return {
        "r2_log": r2_score(y_true_log, y_pred_log),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "r2_pli": r2_score(np.expm1(y_true_log), np.expm1(y_pred_log)),
        "rmse_pli": float(
            np.sqrt(mean_squared_error(np.expm1(y_true_log), np.expm1(y_pred_log)))
        ),
    }


def _metricas_clasificacion(y_true, y_pred, y_proba) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    m = {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        m["roc_auc"] = roc_auc_score(y_true, y_proba)
        m["pr_auc"] = average_precision_score(y_true, y_proba)
    else:
        m["roc_auc"] = np.nan
        m["pr_auc"] = np.nan
    return m


def predecir(pipe, X, tipo: str):
    pred = pipe.predict(X)
    proba = None
    if tipo == "clasificacion":
        try:
            proba = pipe.predict_proba(X)[:, 1]
        except (AttributeError, IndexError):
            proba = None
    return pred, proba


# --------------------------------------------------------------------------
# Validación cruzada repetida
# --------------------------------------------------------------------------
def validacion_cruzada(matriz: dict, modelos: dict, n_splits=cfg.N_FOLDS,
                       seeds=cfg.SEEDS) -> pd.DataFrame:
    """Evalúa cada modelo con validación agrupada repetida."""
    X, y, grupos, tipo = matriz["X"], matriz["y"], matriz["grupos"], matriz["tipo"]
    filas = []
    for nombre, pipe in modelos.items():
        LOG.info("  CV %-22s", nombre)
        for seed in seeds:
            for k, (tr, te) in enumerate(generar_folds(y, grupos, tipo, n_splits, seed)):
                modelo = clone(pipe)
                modelo.fit(X.iloc[tr], y.iloc[tr])
                pred, proba = predecir(modelo, X.iloc[te], tipo)

                if tipo == "regresion":
                    met = _metricas_regresion(y.iloc[te], pred)
                else:
                    met = _metricas_clasificacion(y.iloc[te], pred, proba)

                met.update({"modelo": nombre, "semilla": seed, "fold": k,
                            "n_train": len(tr), "n_test": len(te)})
                filas.append(met)
    return pd.DataFrame(filas)


def resumir_cv(detalle: pd.DataFrame, tipo: str) -> pd.DataFrame:
    """Media ± desviación por modelo, ordenada por la métrica principal."""
    metricas = (["r2_log", "rmse_log", "mae_log", "r2_pli", "rmse_pli"]
                if tipo == "regresion" else ["f1", "roc_auc", "pr_auc", "balanced_accuracy"])
    filas = []
    for nombre, g in detalle.groupby("modelo"):
        fila = {"modelo": nombre, "n_evaluaciones": len(g)}
        for m in metricas:
            if m in g.columns:
                fila[f"{m}_media"] = g[m].mean()
                fila[f"{m}_sd"] = g[m].std()
        filas.append(fila)
    res = pd.DataFrame(filas)
    principal = "r2_log_media" if tipo == "regresion" else "roc_auc_media"
    return res.sort_values(principal, ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Hold-out temporal y bootstrap
# --------------------------------------------------------------------------
def holdout_temporal(matriz: dict, modelos: dict, anio_corte: int = cfg.ANIO_CORTE_TEMPORAL) -> pd.DataFrame:
    """Entrena con años <= corte y evalúa con los posteriores."""
    X, y, meta, tipo = matriz["X"], matriz["y"], matriz["meta"], matriz["tipo"]
    anio = meta["ANIO"].to_numpy()
    tr = np.where(anio <= anio_corte)[0]
    te = np.where(anio > anio_corte)[0]
    if len(te) < 5 or len(tr) < 20:
        LOG.warning("Hold-out temporal no factible (train=%d, test=%d)", len(tr), len(te))
        return pd.DataFrame()

    filas = []
    for nombre, pipe in modelos.items():
        modelo = clone(pipe)
        modelo.fit(X.iloc[tr], y.iloc[tr])
        pred, proba = predecir(modelo, X.iloc[te], tipo)
        if tipo == "regresion":
            met = _metricas_regresion(y.iloc[te], pred)
        else:
            met = _metricas_clasificacion(y.iloc[te], pred, proba)
        met.update({"modelo": nombre, "n_train": len(tr), "n_test": len(te),
                    "anio_corte": anio_corte})
        filas.append(met)
    return pd.DataFrame(filas)


def bootstrap_holdout(matriz: dict, pipe, tipo: str, metrica: str,
                      anio_corte: int = cfg.ANIO_CORTE_TEMPORAL,
                      n_boot: int = cfg.N_BOOTSTRAP, seed: int = cfg.SEED) -> dict:
    """Intervalo de confianza al 95 % de la métrica en el hold-out temporal."""
    X, y, meta = matriz["X"], matriz["y"], matriz["meta"]
    anio = meta["ANIO"].to_numpy()
    tr, te = np.where(anio <= anio_corte)[0], np.where(anio > anio_corte)[0]
    if len(te) < 10:
        return {}
    modelo = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
    pred, proba = predecir(modelo, X.iloc[te], tipo)

    rng = np.random.default_rng(seed)
    valores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(te), len(te))
        yt = y.iloc[te].to_numpy()[idx]
        if tipo == "regresion":
            yp = np.asarray(pred)[idx]
            valores.append(r2_score(yt, yp) if np.var(yt) > 0 else np.nan)
        else:
            yt = yt.astype(int)
            if len(np.unique(yt)) < 2:
                valores.append(np.nan)
                continue
            pp = np.asarray(proba)[idx] if proba is not None else None
            valores.append(
                roc_auc_score(yt, pp) if metrica == "roc_auc" else f1_score(yt, (pp > 0.5).astype(int), zero_division=0)
            )
    valores = np.array(valores, dtype=float)
    return {
        "metrica": metrica,
        "estimacion": float(np.nanmean(valores)),
        "ic95_inferior": float(np.nanpercentile(valores, 2.5)),
        "ic95_superior": float(np.nanpercentile(valores, 97.5)),
        "n_bootstrap": n_boot,
    }


# --------------------------------------------------------------------------
# Predicciones fuera de fold (diagnóstico)
# --------------------------------------------------------------------------
def predicciones_out_of_fold(matriz: dict, pipe, n_splits: int = cfg.N_FOLDS,
                             seed: int = cfg.SEED) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Predicciones honestas para las figuras de diagnóstico: cada muestra se
    predice con un modelo que nunca la vio (misma partición agrupada que la CV).
    """
    X, y, grupos, tipo = matriz["X"], matriz["y"], matriz["grupos"], matriz["tipo"]
    oof = np.full(len(y), np.nan, dtype=float)
    proba = np.full(len(y), np.nan, dtype=float) if tipo == "clasificacion" else None

    for tr, te in generar_folds(y, grupos, tipo, n_splits, seed):
        modelo = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        pred, pr = predecir(modelo, X.iloc[te], tipo)
        oof[te] = np.asarray(pred, dtype=float)
        if proba is not None and pr is not None:
            proba[te] = np.asarray(pr, dtype=float)

    if proba is not None and np.isnan(proba).all():
        proba = None
    return oof, proba


# --------------------------------------------------------------------------
# Orquestación de una tarea
# --------------------------------------------------------------------------
def entrenar_tarea(matriz: dict, tarea: str, tunear: bool = True) -> dict:
    """CV repetida + tuning + hold-out temporal + modelo final serializado."""
    tipo = matriz["tipo"]
    modelos = construir_modelos(tipo)

    detalle = validacion_cruzada(matriz, modelos)
    resumen = resumir_cv(detalle, tipo)
    guardar_csv(detalle, cfg.TABLES / f"cv_detalle_{tarea}.csv")
    guardar_csv(resumen, cfg.TABLES / f"metricas_cv_{tarea}.csv")
    LOG.info("  Resumen CV:\n%s", resumen.round(3).to_string(index=False))

    # --- Tuning ligero de los modelos de árboles --------------------------
    if tunear:
        for nombre in ("RandomForest", "XGBoost"):
            if nombre in modelos:
                modelos[nombre] = ajustar_hiperparametros(
                    modelos[nombre], nombre, matriz["X"], matriz["y"],
                    matriz["grupos"], tipo,
                )

    # --- Hold-out temporal ------------------------------------------------
    holdout = holdout_temporal(matriz, modelos)
    if len(holdout):
        guardar_csv(holdout, cfg.TABLES / f"holdout_{tarea}.csv")
        LOG.info("  Hold-out temporal:\n%s", holdout.round(3).to_string(index=False))

    # --- Mejor modelo -----------------------------------------------------
    principal = "r2_log_media" if tipo == "regresion" else "roc_auc_media"
    mejor_nombre = resumen.iloc[0]["modelo"]
    LOG.info("  Mejor modelo: %s", mejor_nombre)

    final = clone(modelos[mejor_nombre]).fit(matriz["X"], matriz["y"])
    joblib.dump(final, cfg.MODELS_DIR / f"{tarea}__{mejor_nombre.replace(' ', '_')}.joblib")

    # --- Figuras de diagnóstico con predicciones fuera de fold -------------
    oof, oof_proba = predicciones_out_of_fold(matriz, modelos[mejor_nombre])
    plots.comparacion_modelos(resumen, tipo)
    if tipo == "regresion":
        plots.predicho_vs_real(
            matriz["y"], oof, etiqueta="log1p(PLI)",
            titulo="Fuera de fold: predicho vs. observado (log1p PLI)",
        )
    elif oof_proba is not None:
        auc_oof = roc_auc_score(matriz["y"].astype(int), oof_proba)
        plots.curva_roc(
            matriz["y"].astype(int), oof_proba, auc_oof,
            titulo="Curva ROC fuera de fold",
        )
    guardar_csv(
        pd.DataFrame({
            "y_observado": matriz["y"].to_numpy(),
            "y_predicho_oof": oof,
            **({"proba_oof": oof_proba} if oof_proba is not None else {}),
        }),
        cfg.TABLES / f"predicciones_oof_{tarea}.csv",
    )

    ic = bootstrap_holdout(
        matriz, modelos[mejor_nombre], tipo,
        "r2" if tipo == "regresion" else "roc_auc",
    )
    if tipo == "regresion":
        diagnostico_oof = {"r2_oof": float(r2_score(matriz["y"], oof))}
    else:
        diagnostico_oof = {
            "roc_auc_oof": (
                float(roc_auc_score(matriz["y"].astype(int), oof_proba))
                if oof_proba is not None else None
            )
        }
    guardar_json(
        {
            "tarea": tarea,
            "descripcion": matriz["descripcion"],
            "target": matriz["target"],
            "mejor_modelo": mejor_nombre,
            "metrica_principal": principal,
            "n_muestras": int(len(matriz["y"])),
            "n_features": int(matriz["X"].shape[1]),
            "n_grupos": int(matriz["grupos"].nunique()),
            "resumen_cv": resumen.to_dict(orient="records"),
            "holdout_temporal": holdout.to_dict(orient="records"),
            "bootstrap_holdout": ic,
            "diagnostico_oof": diagnostico_oof,
            "semillas": cfg.SEEDS,
            "n_folds": cfg.N_FOLDS,
        },
        cfg.TABLES / f"resumen_modelo_{tarea}.json",
    )

    return {
        "modelos": modelos,
        "detalle": detalle,
        "resumen": resumen,
        "holdout": holdout,
        "mejor_nombre": mejor_nombre,
        "modelo_final": final,
        "bootstrap": ic,
    }


def predecir_dataset(modelo, matriz: dict, tarea: str) -> pd.DataFrame:
    """Predicciones sobre todas las muestras (para el tablero)."""
    pred, proba = predecir(modelo, matriz["X"], matriz["tipo"])
    out = matriz["meta"].copy()
    if matriz["tipo"] == "regresion":
        out["pred_log"] = pred
        out["PLI_predicho"] = np.expm1(pred)
    else:
        out["prob_riesgo_alto"] = proba if proba is not None else np.nan
        out["clase_predicha"] = pred
    return out
