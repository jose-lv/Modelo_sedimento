"""
Índices de riesgo y comparación con guías de calidad (RF-10).

Índices calculados
------------------
* Factor de contaminación  CF_i = C_i / Bn_i
* Índice geoacumulación   Igeo = log2( C_i / (1.5 · Bn_i) )
* Índice de carga         PLI = (∏ CF_i)^(1/n)
* Riesgo ecológico        RI  = Σ Tr_i · CF_i   (Hakanson, 1980)
* Excedencia normativa    C_i / ISQG y C_i / PEL (CCME, agua dulce)

Fondo geoquímico (Bn)
---------------------
Se usa el percentil 25 de las propias muestras de Moquegua (fondo local),
que es la opción recomendada en la guía frente a la corteza promedio. El
análisis de sensibilidad compara P10, P25, P50 y la corteza de
Turekian & Wedepohl (1961).
"""
from __future__ import annotations

import glob
import re

import numpy as np
import pandas as pd

from . import config as cfg
from .utils import LOG, guardar_csv

#: Nombre en la tabla CCME → nombre canónico del proyecto.
CCME_A_CANONICO = {
    "Arsenic": "Arsénico",
    "Cadmium": "Cadmio",
    "Chromium (total)": "Cromo",
    "Copper": "Cobre",
    "Lead": "Plomo",
    "Mercury": "Mercurio",
    "Zinc": "Zinc",
    "Acenaphthene": "Acenafteno",
    "Acenaphthylene": "Acenaftileno",
    "Anthracene": "Antraceno",
    "Benz(a)anthracene": "Benzo(a)antraceno",
    "Benzo(a)pyrene": "Benzo(a)pireno",
    "Chrysene": "Criseno",
    "Dibenz(a,h)anthracene": "Dibenzo(a,h)antraceno",
    "Fluoranthene": "Fluoranteno",
    "Fluorene": "Fluoreno",
    "Naphthalene": "Naftaleno",
    "Phenanthrene": "Fenantreno",
    "Pyrene": "Pireno",
}

#: Valores que la tabla CCME no trae y se completan con otra fuente citada.
COMPLEMENTO_GUIAS = {
    "Níquel": {
        "isqg": 20.9, "pel": 51.6,
        "fuente": "NOAA ERL/ERM (Long et al., 1995); CCME no fija ISQG/PEL para Ni",
    }
}


def _numero_celda(x) -> float:
    """Convierte '37 300', '35,700' o 'No data' a número (µg/kg)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).replace("\xa0", "").replace(" ", "").replace(",", "").strip()
    if not s or s.lower().startswith("no data"):
        return np.nan
    m = re.match(r"^-?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else np.nan


def cargar_guias_ccme() -> pd.DataFrame:
    """
    Lee el CSV de guías CCME descargado junto al dataset y devuelve una tabla
    en mg/kg con ISQG y PEL de agua dulce (RF-10).
    """
    patron = str(cfg.ROOT.parent / "Canadian Council*.csv")
    archivos = glob.glob(patron)
    if not archivos:
        LOG.warning("No se encontró el CSV de guías CCME en %s; se usará solo el complemento", patron)
        crudo = pd.DataFrame(columns=["Chemical", "Groups", "FW_ISQG", "FW_PEL"])
    else:
        crudo = pd.read_csv(archivos[0], encoding="utf-8", skiprows=[1])
        crudo = crudo.rename(columns={
            crudo.columns[0]: "Chemical",
            crudo.columns[1]: "Groups",
            crudo.columns[3]: "FW_ISQG",
            crudo.columns[4]: "FW_PEL",
        })

    filas = []
    for _, r in crudo.iterrows():
        canon = CCME_A_CANONICO.get(str(r["Chemical"]).strip())
        if canon is None:
            continue
        isqg = _numero_celda(r["FW_ISQG"]) / 1000.0  # µg/kg → mg/kg
        pel = _numero_celda(r["FW_PEL"]) / 1000.0
        if np.isnan(isqg) and np.isnan(pel):
            continue
        filas.append({
            "parametro": canon,
            "isqg_mg_kg": isqg,
            "pel_mg_kg": pel,
            "fuente": "CCME (1999-2002), agua dulce, µg/kg → mg/kg",
        })

    for param, vals in COMPLEMENTO_GUIAS.items():
        filas.append({
            "parametro": param,
            "isqg_mg_kg": vals["isqg"],
            "pel_mg_kg": vals["pel"],
            "fuente": vals["fuente"],
        })

    guias = (
        pd.DataFrame(filas)
        .drop_duplicates("parametro")
        .sort_values("parametro")
        .reset_index(drop=True)
    )
    guardar_csv(guias, cfg.CONFIG_DIR / "guias_calidad.csv")
    LOG.info("Guías de calidad: %d parámetros con ISQG/PEL", len(guias))
    return guias


# --------------------------------------------------------------------------
# Fondo geoquímico
# --------------------------------------------------------------------------
def fondo_local(ancho: pd.DataFrame, metales: list[str], percentil: int = None) -> pd.Series:
    """Bn por metal como percentil de las muestras (fondo local)."""
    percentil = cfg.PERCENTIL_FONDO if percentil is None else percentil
    presentes = [m for m in metales if m in ancho.columns]
    fondo = ancho[presentes].quantile(percentil / 100.0, numeric_only=True)
    fondo = fondo[fondo > 0]
    if len(fondo) < len(presentes):
        faltan = sorted(set(presentes) - set(fondo.index))
        LOG.warning("Sin fondo positivo para: %s", faltan)
    return fondo


# --------------------------------------------------------------------------
# Índices
# --------------------------------------------------------------------------
def _clase_igeo(x: float) -> float:
    if pd.isna(x):
        return np.nan
    if x <= 0:
        return 0
    return min(int(np.ceil(x)), 6)


def _nivel_pli(x: float) -> str:
    if pd.isna(x):
        return "Sin dato"
    if x <= 1:
        return "Bajo"
    if x <= 2:
        return "Medio"
    if x <= 3:
        return "Alto"
    return "Muy alto"


def _nivel_ri(x: float) -> str:
    if pd.isna(x):
        return "Sin dato"
    if x < 150:
        return "Bajo"
    if x < 300:
        return "Moderado"
    if x < 600:
        return "Considerable"
    return "Muy alto"


def calcular_indices(
    ancho: pd.DataFrame,
    metales: list[str] | None = None,
    fondo: pd.Series | None = None,
    guias: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Calcula índices por muestra. Devuelve un DataFrame indexado como ``ancho``.

    Se exige un mínimo de 4 de los 8 metales del índice para calcular PLI/RI;
    con menos, el índice no es representativo y queda como faltante.
    """
    metales = metales or cfg.METALES_INDICE
    presentes = [m for m in metales if m in ancho.columns]
    if fondo is None:
        fondo = fondo_local(ancho, presentes)
    guias = cargar_guias_ccme() if guias is None else guias

    res = pd.DataFrame(index=ancho.index)

    conc = ancho[presentes].apply(pd.to_numeric, errors="coerce")
    conc = conc.where(conc > 0)

    # --- CF e Igeo por metal ---------------------------------------------
    cf = conc.div(fondo, axis=1)
    igeo = np.log2(cf / 1.5)

    for m in presentes:
        res[f"CF_{m}"] = cf[m]
        res[f"Igeo_{m}"] = igeo[m]
        res[f"clase_igeo_{m}"] = igeo[m].map(_clase_igeo)

    # --- PLI / RI ---------------------------------------------------------
    n_validos = cf.notna().sum(axis=1)
    res["n_metales_indice"] = n_validos
    log_cf = np.log(cf.where(cf > 0))
    pli = np.exp(log_cf.mean(axis=1, skipna=True))
    res["PLI"] = pli.where(n_validos >= 4)
    res["PLI_log"] = np.log1p(res["PLI"])

    tr = pd.Series({m: cfg.FACTORES_TOXICOS.get(m, 1.0) for m in presentes})
    res["RI"] = (cf.mul(tr, axis=1)).sum(axis=1, min_count=4).where(n_validos >= 4)
    res["RI_log"] = np.log1p(res["RI"])

    res["CF_max"] = cf.max(axis=1, skipna=True)
    res["CF_medio"] = cf.mean(axis=1, skipna=True)
    res["Igeo_max"] = igeo.max(axis=1, skipna=True)
    res["Igeo_medio"] = igeo.mean(axis=1, skipna=True)
    # idxmax por fila evita el FutureWarning de pandas con filas 100 % NA.
    res["metal_dominante"] = cf.apply(
        lambda fila: fila.idxmax() if fila.notna().any() else pd.NA, axis=1
    ).astype("object")
    res.loc[res["CF_max"].isna(), "metal_dominante"] = pd.NA

    res["nivel_pli"] = res["PLI"].map(_nivel_pli)
    res["nivel_ri"] = res["RI"].map(_nivel_ri)
    res["clase_igeo_max"] = res["Igeo_max"].map(_clase_igeo)

    # --- Excedencia frente a guías ---------------------------------------
    guias_idx = guias.set_index("parametro")
    sobre_isqg, sobre_pel = [], []
    for m in presentes:
        if m not in guias_idx.index:
            continue
        isqg = guias_idx.loc[m, "isqg_mg_kg"]
        pel = guias_idx.loc[m, "pel_mg_kg"]
        c = conc[m]
        if pd.notna(isqg):
            res[f"ratio_isqg_{m}"] = c / isqg
            sobre_isqg.append(c > isqg)
        if pd.notna(pel):
            res[f"ratio_pel_{m}"] = c / pel
            sobre_pel.append(c > pel)

    res["n_metales_sobre_isqg"] = (
        pd.concat(sobre_isqg, axis=1).sum(axis=1) if sobre_isqg else 0
    )
    res["n_metales_sobre_pel"] = (
        pd.concat(sobre_pel, axis=1).sum(axis=1) if sobre_pel else 0
    )
    res["excede_isqg"] = res["n_metales_sobre_isqg"] > 0
    res["excede_pel"] = res["n_metales_sobre_pel"] > 0

    # Metal con mayor presión normativa: el que más se acerca/supera su guía.
    # Es más informativo para riesgo que el CF, que premia metales con fondo
    # muy bajo (p. ej. Hg) aunque su concentración absoluta sea pequeña.
    ratios_pel = [c for c in res.columns if c.startswith("ratio_pel_")]
    ratios_isqg = [c for c in res.columns if c.startswith("ratio_isqg_")]
    ratios_norm = ratios_pel or ratios_isqg
    if ratios_norm:
        res["metal_dominante_normativo"] = res[ratios_norm].apply(
            lambda fila: (
                fila.idxmax().replace("ratio_pel_", "").replace("ratio_isqg_", "")
                if fila.notna().any() else pd.NA
            ),
            axis=1,
        ).astype("object")
    else:
        res["metal_dominante_normativo"] = pd.NA

    # --- Variable objetivo binaria ---------------------------------------
    res["riesgo_alto"] = (res["PLI"] > 1).astype("Int64")
    res.loc[res["PLI"].isna(), "riesgo_alto"] = pd.NA
    res["riesgo_alto_pel"] = res["excede_pel"].astype("Int64")
    res.loc[res["PLI"].isna(), "riesgo_alto_pel"] = pd.NA

    # --- Nivel de riesgo normativo (0-3) ----------------------------------
    # Se ancla en las guías CCME, no en un umbral arbitrario de PLI.
    res["nivel_riesgo"] = np.select(
        [
            res["n_metales_sobre_pel"] >= 3,
            res["n_metales_sobre_pel"] >= 1,
            res["n_metales_sobre_isqg"] >= 1,
        ],
        ["Muy alto", "Alto", "Medio"],
        default="Bajo",
    )
    sin_datos = res["n_metales_indice"].fillna(0).eq(0)
    res.loc[sin_datos, "nivel_riesgo"] = "Sin dato"

    orden = {"Bajo": 0, "Medio": 1, "Alto": 2, "Muy alto": 3, "Sin dato": np.nan}
    res["nivel_riesgo_num"] = res["nivel_riesgo"].map(orden)

    # --- Fondo usado (trazabilidad) --------------------------------------
    for m in presentes:
        res[f"Bn_{m}"] = fondo.get(m, np.nan)

    return res


# --------------------------------------------------------------------------
# Sensibilidad
# --------------------------------------------------------------------------
def sensibilidad_fondo(ancho: pd.DataFrame, metales: list[str] | None = None) -> pd.DataFrame:
    """PLI bajo distintos fondos geoquímicos y su correlación de rangos."""
    metales = metales or cfg.METALES_INDICE
    escenarios = {}
    for p in cfg.PERCENTILES_SENSIBILIDAD:
        escenarios[f"P{p}"] = calcular_indices(
            ancho, metales, fondo=fondo_local(ancho, metales, p)
        )["PLI"]
    corteza = pd.Series(cfg.FONDO_CORTEZA).reindex(metales).dropna()
    escenarios["Corteza"] = calcular_indices(ancho, metales, fondo=corteza)["PLI"]

    tabla = pd.DataFrame(escenarios)
    resumen = pd.DataFrame({
        "escenario": tabla.columns,
        "mediana_PLI": tabla.median().round(3).values,
        "pct_PLI_mayor_1": (100 * (tabla > 1).mean()).round(1).values,
        "spearman_vs_P25": [
            tabla.corr(method="spearman").loc[c, "P25"] if "P25" in tabla else np.nan
            for c in tabla.columns
        ],
    })
    return resumen


def sensibilidad_censura(largo: pd.DataFrame, ancho_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Compara el PLI con tres sustituciones del límite de detección.
    Requiere el dataset largo (con ``valor`` y ``censurado``).
    """
    filas = []
    util = largo[largo["parametro_estandar"].isin(cfg.METALES_INDICE)]
    for metodo, factor in cfg.FACTORES_CENSURA.items():
        vals = np.where(util["censurado"], util["valor"] * factor, util["valor"])
        tmp = util.assign(valor_proc=vals)
        ancho = tmp.pivot_table(
            index="id_muestra", columns="parametro_estandar", values="valor_proc", aggfunc="median"
        )
        fondo = fondo_local(ancho, cfg.METALES_INDICE)
        pli = calcular_indices(ancho, fondo=fondo)["PLI"]
        filas.append({
            "metodo": metodo,
            "factor_LD": factor,
            "mediana_PLI": round(float(np.nanmedian(pli)), 4),
            "pct_PLI_mayor_1": round(float(100 * np.nanmean(pli > 1)), 1),
            "_pli": pli,
        })
    tabla = pd.DataFrame(filas)
    referencia = tabla.loc[tabla["metodo"] == cfg.METODO_CENSURA, "_pli"].iloc[0]
    tabla["spearman_vs_principal"] = [
        round(float(pd.Series(p).corr(pd.Series(referencia), method="spearman")), 4)
        for p in tabla["_pli"]
    ]
    return tabla.drop(columns="_pli")


def construir_dataset_riesgo(ancho: pd.DataFrame) -> pd.DataFrame:
    """Dataset final de muestras + índices (RF-18)."""
    guias = cargar_guias_ccme()
    indices = calcular_indices(ancho, guias=guias)
    return pd.concat([ancho, indices], axis=1)
