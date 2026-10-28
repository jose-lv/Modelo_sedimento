"""
Identificación de patrones: clustering y detección de anomalías (RF-12).

Se trabaja sobre los parámetros con cobertura suficiente (por defecto los
metales), transformados con log1p y estandarizados. Se comparan K-Means y
mezcla gaussiana (GMM) eligiendo k por silueta y BIC, y se marcan focos
anómalos con Isolation Forest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from . import config as cfg
from .utils import LOG, guardar_csv, guardar_figura, guardar_json
from . import plots

#: Variables de contexto que NO deben entrar al clustering: son constantes o
#: índices de posición/tiempo y dominarían la distancia sin aportar química.
EXCLUIR_PATRONES = {
    "ANIO", "MES_NUM", "COORD_ESTE", "COORD_NORTE", "TXZONA",
    "n_censurados",
}


def _matriz_patrones(riesgo: pd.DataFrame, cobertura: pd.DataFrame,
                     umbral: float = cfg.COBERTURA_MIN_MODELO) -> tuple[pd.DataFrame, list[str]]:
    """Selecciona y transforma las variables para el análisis no supervisado."""
    params = cobertura.loc[cobertura["cobertura"] >= umbral, "parametro"].tolist()
    params = [p for p in params if p in riesgo.columns]
    # Solo concentraciones (excluye coordenadas, año, índices)
    params = [p for p in params if not p.startswith(("Igeo", "CF_", "ratio_", "Bn_"))]
    params = [p for p in params if p not in {"PLI", "RI"} | EXCLUIR_PATRONES]

    X = riesgo[params].apply(pd.to_numeric, errors="coerce")
    X = np.log1p(X.clip(lower=0))
    X = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X), columns=params, index=X.index
    )
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=params, index=X.index)
    return X, params


def _elegir_kmeans(X: np.ndarray, k_max: int = 8) -> tuple[int, pd.DataFrame]:
    filas = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=25, random_state=cfg.SEED).fit(X)
        filas.append({
            "k": k,
            "inercia": float(km.inertia_),
            "silueta": float(silhouette_score(X, km.labels_)),
        })
    tabla = pd.DataFrame(filas)
    mejor_k = int(tabla.loc[tabla["silueta"].idxmax(), "k"])
    return mejor_k, tabla


def _elegir_gmm(X: np.ndarray, k_max: int = 8) -> tuple[int, pd.DataFrame]:
    filas = []
    for k in range(2, k_max + 1):
        gmm = GaussianMixture(
            n_components=k, covariance_type="full", random_state=cfg.SEED, n_init=5
        ).fit(X)
        filas.append({
            "k": k,
            "bic": float(gmm.bic(X)),
            "aic": float(gmm.aic(X)),
            "silueta": float(silhouette_score(X, gmm.predict(X))),
        })
    tabla = pd.DataFrame(filas)
    mejor_k = int(tabla.loc[tabla["bic"].idxmin(), "k"])
    return mejor_k, tabla


def _perfil_clusters(riesgo: pd.DataFrame, params: list[str],
                     etiquetas: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perfil por cluster: z-score de cada parámetro (matriz densa) y una tabla
    resumen con los 4 elementos que más se elevan respecto de la media global.
    """
    numericos = [
        p for p in params
        if p in riesgo.columns and pd.api.types.is_numeric_dtype(riesgo[p])
    ]
    base = np.log1p(riesgo[numericos].astype(float).clip(lower=0))
    z = (base - base.mean()) / base.std(ddof=0).replace(0, np.nan)
    z = z.dropna(axis=1, how="all")
    numericos = list(z.columns)
    z["cluster"] = etiquetas
    perfil_z = z.groupby("cluster").mean().astype(float)

    perfil_crudo = riesgo[numericos].groupby(etiquetas).median()

    filas = []
    for c in perfil_z.index:
        fila = perfil_z.loc[c].dropna()
        top = fila.sort_values(ascending=False).head(4)
        bajo = fila.sort_values(ascending=True).head(3)
        filas.append({
            "cluster": int(c),
            "n_muestras": int((etiquetas == c).sum()),
            "elementos_dominantes": ", ".join(top.index),
            "z_max": round(float(top.iloc[0]), 2) if len(top) else np.nan,
            "elementos_depletados": ", ".join(bajo.index),
            "z_min": round(float(bajo.iloc[0]), 2) if len(bajo) else np.nan,
        })
    res = pd.DataFrame(filas)
    res = res.merge(
        perfil_crudo.median(axis=1).rename("mediana_global_mg_kg").reset_index(),
        left_on="cluster", right_on="index", how="left",
    ).drop(columns="index")
    return res.sort_values("cluster").reset_index(drop=True), perfil_z.reset_index()


def ejecutar(riesgo: pd.DataFrame, cobertura: pd.DataFrame) -> pd.DataFrame:
    """Corre PCA + K-Means + GMM + Isolation Forest y guarda resultados."""
    X, params = _matriz_patrones(riesgo, cobertura)
    LOG.info("Patrones: %d muestras × %d variables", *X.shape)

    # --- PCA --------------------------------------------------------------
    pca = PCA(n_components=min(10, X.shape[1]), random_state=cfg.SEED).fit(X)
    comp = pca.transform(X)
    var_exp = pca.explained_variance_ratio_

    # --- K-Means ----------------------------------------------------------
    k_km, tabla_km = _elegir_kmeans(X.to_numpy())
    km = KMeans(n_clusters=k_km, n_init=50, random_state=cfg.SEED).fit(X)
    guardar_csv(tabla_km, cfg.TABLES / "seleccion_k_kmeans.csv")

    # --- GMM --------------------------------------------------------------
    k_gmm, tabla_gmm = _elegir_gmm(X.to_numpy())
    gmm = GaussianMixture(
        n_components=k_gmm, covariance_type="full", random_state=cfg.SEED, n_init=10
    ).fit(X)
    guardar_csv(tabla_gmm, cfg.TABLES / "seleccion_k_gmm.csv")

    # --- Isolation Forest ------------------------------------------------
    iso = IsolationForest(
        n_estimators=400, contamination="auto", random_state=cfg.SEED, n_jobs=-1
    ).fit(X)
    score = -iso.score_samples(X)          # mayor = más anómalo
    umbral = np.quantile(score, 0.95)

    res = pd.DataFrame({
        "PC1": comp[:, 0],
        "PC2": comp[:, 1],
        "cluster_kmeans": km.labels_,
        "cluster_gmm": gmm.predict(X),
        "anomalia_score": score,
        "es_anomalia": score >= umbral,
    }, index=X.index)
    if "id_muestra" in riesgo.columns:
        res.insert(0, "id_muestra", riesgo.loc[X.index, "id_muestra"].to_numpy())

    perfil, perfil_z = _perfil_clusters(riesgo, params, km.labels_)
    guardar_csv(perfil, cfg.TABLES / "perfil_clusters.csv")
    guardar_csv(perfil_z, cfg.TABLES / "perfil_clusters_z.csv")
    guardar_json({
        "k_kmeans": k_km,
        "k_gmm": k_gmm,
        "varianza_PC1": float(var_exp[0]),
        "varianza_PC2": float(var_exp[1]),
        "varianza_acumulada_2PC": float(var_exp[:2].sum()),
        "variables": params,
        "umbral_anomalia_p95": float(umbral),
    }, cfg.TABLES / "resumen_patrones.json")

    # --- Figuras ----------------------------------------------------------
    plots.pca_clusters(res, perfil)
    plots.seleccion_k(tabla_km, tabla_gmm)
    plots.mapa_anomalias(riesgo, res)
    plots.calor_perfil(riesgo, params, km.labels_)

    LOG.info("Clusters K-Means: k=%d | GMM: k=%d | anomalías: %d (5 %%)",
             k_km, k_gmm, int(res["es_anomalia"].sum()))
    return res
