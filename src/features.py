"""
Selección de variables, cobertura, faltantes y construcción de la matriz de
modelado (RF-08 y RF-09).

Regla anti-fuga
---------------
La variable objetivo (PLI / Igeo / RI) se calcula con 8 metales. Esos metales
—y cualquier columna derivada de ellos— quedan **prohibidos** como entradas.
El modelo aprende de litogénicos no incluidos en el índice, parámetros
físico-químicos, granulometría, hidrocarburos, ubicación y tiempo.
``verificar_no_fuga`` falla ruidosamente si esta regla se rompe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .utils import LOG, guardar_csv

#: Metales y metaloides que NO componen el índice y sirven de contexto
#: geoquímico (litogénicos, mayores y trazas no normadas).
FEATURES_LITOGENICAS = [
    "Aluminio", "Hierro", "Manganeso", "Calcio", "Magnesio", "Potasio", "Sodio",
    "Titanio", "Estroncio", "Litio", "Bario", "Vanadio", "Cobalto", "Boro",
    "Berilio", "Estaño", "Bismuto", "Selenio", "Molibdeno", "Antimonio", "Talio",
    "Fósforo", "Silicio", "Uranio", "Cerio", "Torio", "Wolframio",
]

FEATURES_FISICO = [
    "pH pasta", "Materia orgánica", "Carbono Orgánico Total (COT)",
    "Sulfato", "Sulfuro", "Cianuro Total", "Conductividad", "Aceites y Grasas",
]

FEATURES_ABA = [
    "Potencial de Neutralización Neto (NNP)", "Potencial de Acidez Máximo (MPA)",
    "Ratio Potencial de Neutralización (RNP)", "Fizz Rating",
    "Potencial de Neutralización Sobek",
]

FEATURES_GRANULOMETRIA = ["% Arcilla", "% Limo", "% Arena", "% Grava"]

FEATURES_ORGANICOS_DERIVADOS = ["suma_pah16", "suma_btex", "tph_total"]

FEATURES_ESPACIALES = ["COORD_ESTE", "COORD_NORTE", "ANIO", "MES_NUM"]

#: Variables estrictamente positivas y muy asimétricas: se transforman con
#: log1p antes del modelado (RF-09).
FEATURES_LOG = (
    FEATURES_LITOGENICAS + FEATURES_ORGANICOS_DERIVADOS
    + ["ratio_Fe_Al", "ratio_Mn_Fe", "ratio_Ca_Mg",
       "Sulfato", "Sulfuro", "Cianuro Total", "Materia orgánica",
       "Carbono Orgánico Total (COT)", "Conductividad", "Aceites y Grasas"]
)

FEATURES_CATEGORICAS = ["TIPO_PUNT", "DISTRITO", "TXORIGEN"]

PAH16 = [
    "Acenafteno", "Acenaftileno", "Antraceno", "Benzo(a)antraceno", "Benzo(a)pireno",
    "Benzo(b)fluoranteno", "Benzo(Ghi)perileno", "Benzo(k)fluoranteno", "Criseno",
    "Dibenzo(a,h)antraceno", "Fenantreno", "Fluoranteno", "Fluoreno",
    "Indeno(1,2,3-cd)pireno", "Naftaleno", "Pireno",
]
BTEX = ["Benceno", "Tolueno", "Etilbenceno", "Xilenos"]
TPH = ["TPH (C6-C40)", "TPH F1 (C6-C10)", "TPH F2 (C10-C28)", "TPH F3 (C28-C40)"]


# --------------------------------------------------------------------------
# Cobertura (RF-08)
# --------------------------------------------------------------------------
def reporte_cobertura(ancho: pd.DataFrame, censura: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cobertura, censura y rango de cada parámetro del dataset ancho."""
    familias = (
        censura.set_index("parametro_estandar")["familia"].to_dict() if censura is not None else {}
    )
    filas = []
    n = len(ancho)
    for col in ancho.columns:
        if col == "id_muestra":
            continue
        s = ancho[col]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        filas.append({
            "parametro": col,
            "familia": familias.get(col, "meta"),
            "n_no_nulos": int(s.notna().sum()),
            "cobertura": round(float(s.notna().mean()), 4),
            "min": float(np.nanmin(s)) if s.notna().any() else np.nan,
            "mediana": float(np.nanmedian(s)) if s.notna().any() else np.nan,
            "max": float(np.nanmax(s)) if s.notna().any() else np.nan,
        })
    rep = pd.DataFrame(filas).sort_values("cobertura", ascending=False)
    rep["retenido_analisis"] = rep["cobertura"] >= cfg.COBERTURA_MIN_ANALISIS
    rep["retenido_modelo"] = rep["cobertura"] >= cfg.COBERTURA_MIN_MODELO
    if censura is not None:
        rep = rep.merge(
            censura[["parametro_estandar", "pct_censurado"]],
            left_on="parametro", right_on="parametro_estandar", how="left",
        ).drop(columns="parametro_estandar")
    return rep.reset_index(drop=True)


def parametros_retenidos(cobertura: pd.DataFrame, umbral: float) -> list[str]:
    return cobertura.loc[cobertura["cobertura"] >= umbral, "parametro"].tolist()


# --------------------------------------------------------------------------
# Variables derivadas
# --------------------------------------------------------------------------
def _suma_si_suficiente(df: pd.DataFrame, columnas: list[str], minimo: int) -> pd.Series:
    presentes = [c for c in columnas if c in df.columns]
    if not presentes:
        return pd.Series(np.nan, index=df.index)
    sub = df[presentes]
    suma = sub.sum(axis=1, min_count=1)
    return suma.where(sub.notna().sum(axis=1) >= minimo)


def agregar_variables_derivadas(ancho: pd.DataFrame) -> pd.DataFrame:
    """Suma de PAH16, suma de BTEX, TPH total y proxies geoquímicos."""
    df = ancho.copy()

    df["suma_pah16"] = _suma_si_suficiente(df, PAH16, minimo=8)
    df["suma_btex"] = _suma_si_suficiente(df, BTEX, minimo=2)

    tph_cols = [c for c in TPH if c in df.columns]
    if tph_cols:
        df["tph_total"] = df[tph_cols].max(axis=1, skipna=True)
    else:
        df["tph_total"] = np.nan

    # Proxies litogénicos: relaciones entre elementos mayores
    if {"Hierro", "Aluminio"} <= set(df.columns):
        df["ratio_Fe_Al"] = df["Hierro"] / df["Aluminio"].replace(0, np.nan)
    if {"Manganeso", "Hierro"} <= set(df.columns):
        df["ratio_Mn_Fe"] = df["Manganeso"] / df["Hierro"].replace(0, np.nan)
    if {"Calcio", "Magnesio"} <= set(df.columns):
        df["ratio_Ca_Mg"] = df["Calcio"] / df["Magnesio"].replace(0, np.nan)

    return df


def aplicar_log(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """log1p sobre concentraciones (asimetría típica de datos ambientales)."""
    out = df.copy()
    for c in columnas:
        if c in out.columns:
            out[c] = np.log1p(out[c].clip(lower=0))
    return out


# --------------------------------------------------------------------------
# Matriz de modelado (RF-08 / RF-09)
# --------------------------------------------------------------------------
def columnas_features(riesgo: pd.DataFrame, metales_indice: list[str] | None = None) -> list[str]:
    """
    Lista blanca de entradas: sólo variables que NO componen el target.
    """
    metales_indice = metales_indice or cfg.METALES_INDICE
    candidatas: list[str] = []
    for grupo in (
        FEATURES_LITOGENICAS, FEATURES_FISICO, FEATURES_ABA, FEATURES_GRANULOMETRIA,
        FEATURES_ORGANICOS_DERIVADOS, FEATURES_ESPACIALES,
        ["ratio_Fe_Al", "ratio_Mn_Fe", "ratio_Ca_Mg"],
    ):
        candidatas += [c for c in grupo if c in riesgo.columns]

    # Excluir cualquier columna que contenga el nombre de un metal del índice
    # (p. ej. "CF_Cobre", "Igeo_Cobre", "ratio_pel_Cobre").
    def _toca_indice(col: str) -> bool:
        c = col.lower()
        return any(m.lower() in c for m in metales_indice)

    candidatas = [c for c in candidatas if not _toca_indice(c)]
    candidatas = [c for c in candidatas if c not in cfg.COLUMNAS_PROHIBIDAS_FEATURES]
    return list(dict.fromkeys(candidatas))


def verificar_no_fuga(features: list[str], metales_indice: list[str] | None = None) -> None:
    """Barrera dura: aborta si alguna entrada compone el índice de riesgo."""
    metales_indice = metales_indice or cfg.METALES_INDICE
    problemas = []
    for f in features:
        if f in cfg.COLUMNAS_PROHIBIDAS_FEATURES:
            problemas.append(f"{f} es una variable objetivo")
        elif f in metales_indice:
            problemas.append(f"{f} compone el índice de riesgo")
        elif any(m.lower() == f.lower() for m in metales_indice):
            problemas.append(f"{f} compone el índice de riesgo")
    if problemas:
        raise ValueError("FUGA DE DATOS detectada: " + "; ".join(problemas))
    LOG.info("Anti-fuga OK: %d features, ninguno de los %d metales del índice",
             len(features), len(metales_indice))


def construir_matriz(
    riesgo: pd.DataFrame,
    tarea: str,
    cobertura: pd.DataFrame | None = None,
) -> dict:
    """
    Construye ``X``, ``y``, grupos y metadatos para una tarea de modelado.

    Los faltantes NO se imputan aquí: la imputación ocurre dentro del pipeline
    de validación para no filtrar información entre folds.
    """
    spec = cfg.TAREAS[tarea]
    target = spec["target"]

    features = columnas_features(riesgo)
    verificar_no_fuga(features)

    base = riesgo.dropna(subset=[target]).copy()

    # Muestras con cobertura mínima de las variables de entrada
    if cobertura is not None:
        retenidos = set(parametros_retenidos(cobertura, cfg.COBERTURA_MIN_MODELO))
        exigidas = [f for f in features if f in retenidos]
    else:
        exigidas = features
    if exigidas:
        frac = base[exigidas].notna().mean(axis=1)
        base = base[frac >= cfg.COBERTURA_MIN_MUESTRA]

    X = base[features].copy()
    y = base[target].astype(float)

    # Transformación log de concentraciones (RF-09)
    for c in [c for c in FEATURES_LOG if c in X.columns]:
        X[c] = np.log1p(X[c].clip(lower=0))

    # Categóricas → dummies (categorías globales, sin usar el target)
    cats = [c for c in FEATURES_CATEGORICAS if c in X.columns]
    if cats:
        dummies = pd.get_dummies(
            X[cats].astype(str), prefix=cats, dummy_na=True, drop_first=False
        ).astype(float)
        X = pd.concat([X.drop(columns=cats), dummies], axis=1)

    X = X.astype(float)
    columnas_constantes = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if columnas_constantes:
        X = X.drop(columns=columnas_constantes)
        LOG.info("Se descartan %d columnas constantes", len(columnas_constantes))

    # Grupos espaciales
    grupos = base[cfg.COLUMNA_GRUPO].astype(str).fillna("SIN_PUNTO")

    LOG.info("Tarea %-22s | n=%d | features=%d | target=%s",
             tarea, len(X), X.shape[1], target)

    return {
        "X": X,
        "y": y,
        "grupos": grupos,
        "meta": base[["id_muestra", cfg.COLUMNA_GRUPO, "DISTRITO", "ANIO",
                      "COORD_ESTE", "COORD_NORTE", "PLI", "nivel_pli",
                      "metal_dominante", "metal_dominante_normativo"]].copy(),
        "features": list(X.columns),
        "target": target,
        "tipo": spec["tipo"],
        "descripcion": spec["descripcion"],
    }


def resumen_faltantes(X: pd.DataFrame) -> pd.DataFrame:
    """Porcentaje de faltantes por variable de entrada (RF-08)."""
    tab = pd.DataFrame({
        "variable": X.columns,
        "pct_faltante": (100 * X.isna().mean()).round(2).values,
        "n_unicos": [X[c].nunique(dropna=True) for c in X.columns],
    }).sort_values("pct_faltante", ascending=False)
    return tab.reset_index(drop=True)
