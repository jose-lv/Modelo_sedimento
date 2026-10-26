"""
Figuras del MVP. Todas se guardan en ``reports/figures`` con la misma paleta
y estilo para que el capítulo de resultados sea consistente.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.lines import Line2D

from . import config as cfg
from .utils import guardar_figura

plt.rcParams.update({
    "figure.autolayout": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 9,
})

CMAP_RIESGO = LinearSegmentedColormap.from_list("riesgo", cfg.PALETA_RIESGO)
COLORES_CLUSTER = plt.get_cmap("tab10").colors


# --------------------------------------------------------------------------
# EDA
# --------------------------------------------------------------------------
def cobertura_barras(rep: pd.DataFrame, top: int = 30):
    d = rep.sort_values("cobertura", ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.22 * len(d))))
    colores = ["#2c7bb6" if c >= cfg.COBERTURA_MIN_MODELO else "#fdae61" if c >= cfg.COBERTURA_MIN_ANALISIS else "#bdbdbd"
               for c in d["cobertura"]]
    ax.barh(d["parametro"], 100 * d["cobertura"], color=colores)
    ax.axvline(100 * cfg.COBERTURA_MIN_MODELO, color="#d7191c", ls="--", lw=1,
               label=f"umbral modelo ({int(100*cfg.COBERTURA_MIN_MODELO)} %)")
    ax.axvline(100 * cfg.COBERTURA_MIN_ANALISIS, color="#fdae61", ls=":", lw=1,
               label=f"umbral análisis ({int(100*cfg.COBERTURA_MIN_ANALISIS)} %)")
    ax.set_xlabel("Cobertura (% de muestras con dato)")
    ax.set_title("Cobertura por parámetro")
    ax.legend(loc="lower right", fontsize=7)
    return guardar_figura(fig, "01_cobertura_parametros")


def censura_barras(censura: pd.DataFrame, top: int = 25):
    d = censura[censura["n"] >= 30].sort_values("pct_censurado", ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.25 * len(d))))
    ax.barh(d["parametro_estandar"], d["pct_censurado"], color="#7b3294")
    ax.axvline(100 * cfg.UMBRAL_CENSURA_BINARIA, color="#d7191c", ls="--", lw=1,
               label="80 % (pasa a binaria)")
    ax.set_xlabel("% de valores bajo el límite de detección")
    ax.set_title("Censura por parámetro (n ≥ 30)")
    ax.legend(fontsize=7)
    return guardar_figura(fig, "02_censura_parametros")


def distribuciones(ancho: pd.DataFrame, params: list[str], ncols: int = 4):
    params = [p for p in params if p in ancho.columns][:16]
    if not params:
        return None
    nrows = int(np.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes, params):
        v = ancho[p].dropna()
        v = v[v > 0]
        if len(v) > 2:
            ax.hist(np.log10(v), bins=25, color="#2c7bb6", alpha=0.85)
        ax.set_title(p, fontsize=8)
        ax.set_xlabel("log10(mg/kg)")
        ax.tick_params(labelsize=7)
    for ax in axes[len(params):]:
        ax.axis("off")
    fig.suptitle("Distribución de concentraciones (escala log)", y=1.01)
    return guardar_figura(fig, "03_distribuciones")


def correlaciones(ancho: pd.DataFrame, params: list[str], nombre: str = "04_correlaciones"):
    params = [p for p in params if p in ancho.columns]
    if len(params) < 3:
        return None
    corr = ancho[params].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(0.42 * len(params) + 2, 0.42 * len(params) + 1.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(params, rotation=90, fontsize=6)
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params, fontsize=6)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman ρ")
    ax.set_title("Correlación de Spearman entre contaminantes")
    return guardar_figura(fig, nombre)


def boxplot_por_grupo(riesgo: pd.DataFrame, variable: str, grupo: str = "DISTRITO",
                      nombre: str = "05_pli_por_distrito"):
    if variable not in riesgo.columns:
        return None
    d = riesgo.dropna(subset=[variable, grupo])
    orden = d.groupby(grupo)[variable].median().sort_values().index
    fig, ax = plt.subplots(figsize=(8, 4))
    datos = [d.loc[d[grupo] == g, variable].values for g in orden]
    ax.boxplot(datos, labels=[str(g)[:18] for g in orden], patch_artist=True,
               boxprops=dict(facecolor="#abd9e9"), medianprops=dict(color="#d7191c"))
    ax.set_ylabel(variable)
    ax.set_title(f"{variable} por {grupo.lower()}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return guardar_figura(fig, nombre)


def serie_temporal(riesgo: pd.DataFrame, variable: str = "PLI", nombre: str = "06_serie_temporal"):
    if variable not in riesgo.columns or "ANIO" not in riesgo.columns:
        return None
    d = riesgo.dropna(subset=[variable, "ANIO"])
    g = d.groupby("ANIO")[variable].agg(["median", "count", lambda s: s.quantile(0.9)])
    g.columns = ["mediana", "n", "p90"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(g.index.astype(int), g["mediana"], "-o", color="#2c7bb6", label="mediana")
    ax.plot(g.index.astype(int), g["p90"], "--", color="#d7191c", label="percentil 90")
    ax.axhline(1, color="gray", lw=1, ls=":", label="PLI = 1")
    ax.set_xlabel("Año")
    ax.set_ylabel(variable)
    ax.set_title(f"Evolución temporal de {variable}")
    ax2 = ax.twinx()
    ax2.bar(g.index.astype(int), g["n"], alpha=0.15, color="gray")
    ax2.set_ylabel("n muestras", color="gray")
    ax2.grid(False)
    ax.legend(loc="upper left", fontsize=7)
    return guardar_figura(fig, nombre)


# --------------------------------------------------------------------------
# Mapas
# --------------------------------------------------------------------------
def mapa_puntos(riesgo: pd.DataFrame, variable: str = "PLI", nombre: str = "07_mapa_pli",
                titulo: str | None = None):
    """Mapa de puntos en coordenadas UTM coloreados por una variable."""
    d = riesgo.dropna(subset=["COORD_ESTE", "COORD_NORTE", variable])
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 6))
    este, norte = d["COORD_ESTE"] / 1000.0, d["COORD_NORTE"] / 1000.0
    if variable == "PLI":
        tope = max(5.0, float(d[variable].max()) * 1.001 + 1e-6)
        norm = BoundaryNorm([0, 1, 2, 3, 5, tope], CMAP_RIESGO.N, clip=True)
        sc = ax.scatter(este, norte, c=d[variable],
                        cmap=CMAP_RIESGO, norm=norm, s=42, edgecolor="k", linewidth=0.3)
        cb = fig.colorbar(sc, ax=ax, shrink=0.75, extend="max",
                          ticks=[0, 1, 2, 3, 5, tope])
        cb.set_label("PLI")
        cb.ax.set_yticklabels(["0", "1", "2", "3", "5", f"{tope:.0f}"], fontsize=7)
    else:
        sc = ax.scatter(este, norte, c=d[variable],
                        cmap="viridis", s=42, edgecolor="k", linewidth=0.3)
        fig.colorbar(sc, ax=ax, shrink=0.75, label=variable)
    ax.set_aspect("equal")
    ax.set_xlabel("Este (km, UTM 19S)")
    ax.set_ylabel("Norte (km, UTM 19S)")
    ax.set_title(titulo or f"Distribución espacial de {variable} ({len(d)} muestras)")
    return guardar_figura(fig, nombre)


def mapa_anomalias(riesgo: pd.DataFrame, patrones: pd.DataFrame):
    d = riesgo.join(patrones[["anomalia_score", "es_anomalia"]], how="inner")
    d = d.dropna(subset=["COORD_ESTE", "COORD_NORTE"])
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 6))
    normal = d[~d["es_anomalia"]]
    anom = d[d["es_anomalia"]]
    ax.scatter(normal["COORD_ESTE"] / 1000, normal["COORD_NORTE"] / 1000, s=26, c="#bdbdbd",
               label=f"Normal (n={len(normal)})", edgecolor="k", linewidth=0.2)
    ax.scatter(anom["COORD_ESTE"] / 1000, anom["COORD_NORTE"] / 1000, s=70, c="#d7191c", marker="^",
               label=f"Anomalía (n={len(anom)})", edgecolor="k", linewidth=0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("Este (km, UTM 19S)")
    ax.set_ylabel("Norte (km, UTM 19S)")
    ax.set_title("Focos anómalos (Isolation Forest, percentil 95)")
    ax.legend(fontsize=8)
    return guardar_figura(fig, "08_mapa_anomalias")


def mapa_nivel(riesgo: pd.DataFrame, nombre: str = "07_mapa_nivel_riesgo"):
    """Mapa categórico por nivel de riesgo normativo (CCME)."""
    d = riesgo.dropna(subset=["COORD_ESTE", "COORD_NORTE"]).copy()
    if d.empty:
        return None
    colores = {"Bajo": "#2c7bb6", "Medio": "#ffffbf", "Alto": "#fdae61",
               "Muy alto": "#d7191c", "Sin dato": "#bdbdbd"}
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for nivel in ["Bajo", "Medio", "Alto", "Muy alto", "Sin dato"]:
        sub = d[d["nivel_riesgo"] == nivel]
        if sub.empty:
            continue
        ax.scatter(sub["COORD_ESTE"] / 1000, sub["COORD_NORTE"] / 1000, s=42, c=colores[nivel],
                   edgecolor="k", linewidth=0.3, label=f"{nivel} (n={len(sub)})")
    ax.set_aspect("equal")
    ax.set_xlabel("Este (km, UTM 19S)")
    ax.set_ylabel("Norte (km, UTM 19S)")
    ax.set_title("Nivel de riesgo por excedencia de guías CCME")
    ax.legend(fontsize=8, loc="best")
    return guardar_figura(fig, nombre)


def barras_nivel_riesgo(riesgo: pd.DataFrame, nombre: str = "20_nivel_riesgo"):
    orden = ["Bajo", "Medio", "Alto", "Muy alto", "Sin dato"]
    conteo = riesgo["nivel_riesgo"].value_counts().reindex(orden).fillna(0)
    colores = ["#2c7bb6", "#ffffbf", "#fdae61", "#d7191c", "#bdbdbd"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    barras = ax.bar(conteo.index, conteo.values, color=colores, edgecolor="k", linewidth=0.4)
    for b, v in zip(barras, conteo.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{int(v)}\n({100*v/len(riesgo):.0f} %)", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Muestras")
    ax.set_title("Distribución del nivel de riesgo (guías CCME)")
    ax.set_ylim(0, conteo.max() * 1.25)
    return guardar_figura(fig, nombre)


# --------------------------------------------------------------------------
# Patrones
# --------------------------------------------------------------------------
def pca_clusters(patrones: pd.DataFrame, perfil: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, col, titulo in zip(axes, ["cluster_kmeans", "cluster_gmm"],
                               ["K-Means", "Mezcla gaussiana"]):
        for c in sorted(patrones[col].unique()):
            d = patrones[patrones[col] == c]
            ax.scatter(d["PC1"], d["PC2"], s=26, color=COLORES_CLUSTER[int(c) % 10],
                       label=f"C{c} (n={len(d)})", alpha=0.85, edgecolor="k", linewidth=0.2)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Clusters {titulo}")
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Patrones de contaminación en el espacio PCA")
    return guardar_figura(fig, "09_pca_clusters")


def seleccion_k(tabla_km: pd.DataFrame, tabla_gmm: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].plot(tabla_km["k"], tabla_km["silueta"], "-o", color="#2c7bb6")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Silueta")
    axes[0].set_title("K-Means: elección de k")
    axes[1].plot(tabla_gmm["k"], tabla_gmm["bic"], "-o", color="#d7191c")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("BIC")
    axes[1].set_title("GMM: elección de k")
    return guardar_figura(fig, "10_seleccion_k")


def calor_perfil(riesgo: pd.DataFrame, params: list[str], etiquetas: np.ndarray):
    # Solo variables numéricas: "Fizz Rating" y similares son categóricas.
    numericos = [
        p for p in params
        if p in riesgo.columns and pd.api.types.is_numeric_dtype(riesgo[p])
        and not pd.api.types.is_bool_dtype(riesgo[p])
    ]
    if not numericos:
        return None
    base = np.log1p(riesgo[numericos].astype(float).clip(lower=0))
    z = (base - base.mean()) / base.std(ddof=0).replace(0, np.nan)
    z = z.dropna(axis=1, how="all")
    params = list(z.columns)
    z["cluster"] = etiquetas
    perfil = z.groupby("cluster").mean().astype(float)
    fig, ax = plt.subplots(figsize=(max(8, 0.34 * len(params)), 0.55 * len(perfil) + 1.6))
    im = ax.imshow(perfil.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(params, rotation=90, fontsize=6)
    ax.set_yticks(range(len(perfil)))
    ax.set_yticklabels([f"Cluster {c}" for c in perfil.index], fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="z-score (log)")
    ax.set_title("Perfil químico medio por cluster")
    return guardar_figura(fig, "11_perfil_clusters")


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
def comparacion_modelos(resumen: pd.DataFrame, tipo: str):
    metrica = "r2_log_media" if tipo == "regresion" else "roc_auc_media"
    sd = metrica.replace("_media", "_sd")
    d = resumen.sort_values(metrica)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(d) + 2))
    ax.barh(d["modelo"], d[metrica], xerr=d.get(sd, 0), color="#2c7bb6",
            error_kw=dict(ecolor="#333", capsize=3))
    ax.set_xlabel(metrica.replace("_media", "") + " (media ± DE en CV agrupada)")
    ax.set_title("Comparación de modelos")
    if tipo == "regresion":
        ax.axvline(0, color="gray", lw=1, ls=":")
    return guardar_figura(fig, f"12_comparacion_modelos_{tipo}")


def predicho_vs_real(y_true: pd.Series, y_pred: np.ndarray, etiqueta: str = "PLI",
                     titulo: str | None = None, nombre: str = "13_predicho_vs_real"):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(y_true, y_pred, s=24, alpha=0.75, color="#2c7bb6", edgecolor="k", linewidth=0.3)
    lim = [min(np.nanmin(y_true), np.nanmin(y_pred)), max(np.nanmax(y_true), np.nanmax(y_pred))]
    ax.plot(lim, lim, "--", color="#d7191c", lw=1)
    ax.set_xlabel(f"{etiqueta} observado")
    ax.set_ylabel(f"{etiqueta} predicho")
    ax.set_title(titulo or "Predicho vs. observado")
    return guardar_figura(fig, nombre)


def curva_roc(y_true, y_proba, auc: float, titulo: str | None = None,
              nombre: str = "14_curva_roc"):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot(fpr, tpr, color="#2c7bb6", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(titulo or "Curva ROC")
    ax.legend()
    return guardar_figura(fig, nombre)


def importancia_permutacion(imp: pd.DataFrame, top: int = 15):
    d = imp.sort_values("importancia_media", ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(d) + 1.8))
    ax.barh(d["variable"], d["importancia_media"], xerr=d["importancia_sd"],
            color="#7b3294", error_kw=dict(ecolor="#333", capsize=3))
    ax.set_xlabel("Caída de la métrica al permutar (media ± DE)")
    ax.set_title("Importancia por permutación")
    return guardar_figura(fig, "15_importancia_permutacion")


def shap_summary(shap_values, X: pd.DataFrame, nombre: str = "16_shap_summary", max_display: int = 15):
    import shap
    fig = plt.figure(figsize=(8, 0.35 * min(max_display, X.shape[1]) + 2))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    ruta = cfg.FIGURES / f"{nombre}.{cfg.FIG_FORMAT}"
    plt.savefig(ruta, dpi=cfg.DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def shap_bar(shap_values, X: pd.DataFrame, nombre: str = "17_shap_bar", max_display: int = 15):
    import shap
    fig = plt.figure(figsize=(8, 0.35 * min(max_display, X.shape[1]) + 2))
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display, show=False)
    ruta = cfg.FIGURES / f"{nombre}.{cfg.FIG_FORMAT}"
    plt.savefig(ruta, dpi=cfg.DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def shap_dependence(shap_values, X: pd.DataFrame, variables: list[str], nombre: str = "18_shap_dependence"):
    import shap
    # explain.ejecutar entrega un ndarray; se acepta también shap.Explanation.
    valores = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)
    variables = [v for v in variables if v in X.columns][:5]
    if not variables:
        return None
    fig, axes = plt.subplots(len(variables), 1, figsize=(7, 3 * len(variables)))
    axes = np.atleast_1d(axes)
    for ax, v in zip(axes, variables):
        shap.dependence_plot(v, valores, X, ax=ax, show=False, interaction_index=None)
        ax.set_title(f"Dependencia SHAP: {v}", fontsize=9)
    fig.tight_layout()
    return guardar_figura(fig, nombre)


def shap_waterfall(valores_fila, base_value, X_fila: pd.Series, titulo: str, nombre: str):
    import shap
    fig = plt.figure(figsize=(8, 4.6))
    exp = shap.Explanation(
        values=valores_fila, base_values=base_value,
        data=X_fila.values, feature_names=list(X_fila.index),
    )
    shap.plots.waterfall(exp, max_display=12, show=False)
    ruta = cfg.FIGURES / f"{nombre}.{cfg.FIG_FORMAT}"
    plt.savefig(ruta, dpi=cfg.DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def excedencias_por_parametro(riesgo: pd.DataFrame, guias: pd.DataFrame, nombre: str = "19_excedencias"):
    filas = []
    for _, g in guias.iterrows():
        p = g["parametro"]
        if p not in riesgo.columns:
            continue
        v = riesgo[p].dropna()
        if v.empty:
            continue
        filas.append({
            "parametro": p,
            "pct_sobre_isqg": 100 * (v > g["isqg_mg_kg"]).mean() if pd.notna(g["isqg_mg_kg"]) else np.nan,
            "pct_sobre_pel": 100 * (v > g["pel_mg_kg"]).mean() if pd.notna(g["pel_mg_kg"]) else np.nan,
            "n": len(v),
        })
    d = pd.DataFrame(filas).dropna(subset=["pct_sobre_isqg"]).sort_values("pct_sobre_isqg")
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(d) + 1.8))
    y = np.arange(len(d))
    ax.barh(y - 0.2, d["pct_sobre_isqg"], height=0.4, color="#fdae61", label="sobre ISQG")
    ax.barh(y + 0.2, d["pct_sobre_pel"], height=0.4, color="#d7191c", label="sobre PEL")
    ax.set_yticks(y)
    ax.set_yticklabels(d["parametro"])
    ax.set_xlabel("% de muestras que exceden la guía")
    ax.set_title("Excedencia de guías de calidad de sedimento (CCME)")
    ax.legend(fontsize=8)
    return guardar_figura(fig, nombre)
