"""
Limpieza del dataset largo y construcción del dataset ancho (RF-03 a RF-07).

Reglas aplicadas, en orden, con su bitácora:

R1  Consolidar las columnas de valor en ``valor``.
R2  Normalizar parámetros y unidades a mg/kg (RF-04).
R3  Marcar resultados bajo el límite de detección (RF-05).
R4  Descartar filas sin valor numérico.
R5  Descartar "placeholders" (valor 0 sin unidad y sin signo): son filas
    duplicadas que el OEFA publica en blanco y que, si no se eliminan,
    introducen ceros falsos y "duplicados con conflicto".
R6  Separar ensayos de lixiviación (mg/L): no son comparables con mg/kg.
R7  Descartar concentraciones negativas o con unidad no convertible.
R8  Resolver duplicados (muestra, parámetro) y registrar los conflictos.
R9  Validar coordenadas UTM (se marcan, no se eliminan).
R10 Pivotar a formato ancho: una fila por muestra.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as cfg
from .load import consolidar_valor
from .parametros import aplicar_normalizacion
from .utils import LOG, Bitacora, guardar_csv


# --------------------------------------------------------------------------
# Claves y metadatos
# --------------------------------------------------------------------------
def construir_id_muestra(df: pd.DataFrame) -> pd.DataFrame:
    """Clave única de muestra: CUC | PUNTO_MUESTREO | FECHA_PTO."""
    df = df.copy()
    df["id_muestra"] = (
        df["CUC"].astype(str).str.strip()
        + "|"
        + df["PUNTO_MUESTREO"].astype(str).str.strip()
        + "|"
        + df["FECHA_PTO"].astype(str).str.strip()
    )
    return df


_NUM_FINAL = re.compile(r"(?<=-)(\d+)$")


def normalizar_punto(nombre: str) -> str:
    """
    Unifica el nombre del punto de muestreo.

    ``ESP-SED-01`` y ``ESP-SED-1`` son el mismo punto escrito por campañas
    distintas. Si no se unifican, la validación agrupada deja el mismo punto
    en entrenamiento y prueba (fuga espacial).
    """
    s = " ".join(str(nombre).upper().split())
    s = s.replace("_", "-")
    return _NUM_FINAL.sub(lambda m: str(int(m.group(1))), s)


def extraer_ubicacion(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae departamento, provincia y distrito del primer ubigeo."""
    df = df.copy()
    primero = df["TXUBIGEO"].astype(str).str.split("/").str[0].str.strip()
    partes = primero.str.split(",")
    df["DEPARTAMENTO"] = partes.str[0].str.strip().str.upper()
    df["PROVINCIA"] = partes.str[1].str.strip().str.upper()
    df["DISTRITO"] = partes.str[2].str.strip().str.upper()
    return df


def _parsear_fecha(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["FECHA_PTO"], errors="coerce", format="mixed")
    df["ANIO"] = df["fecha"].dt.year.fillna(pd.to_numeric(df["ANHO"], errors="coerce"))
    df["MES_NUM"] = df["fecha"].dt.month
    df["ANIO"] = df["ANIO"].astype("Int64")
    df["MES_NUM"] = df["MES_NUM"].astype("Int64")
    return df


# --------------------------------------------------------------------------
# Limpieza del formato largo
# --------------------------------------------------------------------------
def limpiar_largo(df: pd.DataFrame) -> tuple[pd.DataFrame, Bitacora]:
    bit = Bitacora()
    n = len(df)

    # --- R1/R2: consolidar y normalizar ----------------------------------
    df = consolidar_valor(df)
    bit.registrar("R1 Consolidar columnas de valor", n, len(df),
                  "Una sola columna 'valor' por registro")

    df = aplicar_normalizacion(df)
    df = construir_id_muestra(df)
    df = extraer_ubicacion(df)
    df = _parsear_fecha(df)
    df["PUNTO_NORMALIZADO"] = df["PUNTO_MUESTREO"].map(normalizar_punto)
    df["GRID_5KM"] = (
        (pd.to_numeric(df["COORD_ESTE"], errors="coerce") // 5000).astype("Int64").astype(str)
        + "_"
        + (pd.to_numeric(df["COORD_NORTE"], errors="coerce") // 5000).astype("Int64").astype(str)
    )

    # --- R3: censura ------------------------------------------------------
    signo = df["SIGNO_PARAMETRO"].astype(str).str.strip()
    df["censurado"] = signo.eq("<")
    df["mayor_que"] = signo.eq(">")
    factor = cfg.FACTORES_CENSURA[cfg.METODO_CENSURA]
    # El valor reportado con "<" es el límite de detección: se sustituye por
    # una fracción del LD (método principal: LD/2).
    df["valor_proc"] = np.where(df["censurado"], df["valor"] * factor, df["valor"])

    # --- R4: sin valor numérico ------------------------------------------
    antes = len(df)
    df = df[df["valor"].notna()].copy()
    bit.registrar("R4 Descartar valor no numérico", antes, len(df),
                  "Texto, celdas vacías o '-' en todas las columnas de valor")

    # --- R5: placeholders ------------------------------------------------
    antes = len(df)
    placeholder = (
        df["valor"].eq(0)
        & df["UNIDAD_MEDIDA_PARAMETRO"].isna()
        & df["SIGNO_PARAMETRO"].isna()
    )
    df = df[~placeholder].copy()
    bit.registrar("R5 Descartar placeholders (0 sin unidad ni signo)", antes, len(df),
                  "Filas espejo del OEFA: 0.0 sin unidad ni signo que duplican la medición real")

    # --- R6: lixiviados ---------------------------------------------------
    antes = len(df)
    df = df[~df["es_lixiviado"]].copy()
    bit.registrar("R6 Excluir ensayos de lixiviación (mg/L)", antes, len(df),
                  "Otra matriz de exposición: no comparable con mg/kg de sedimento")

    # --- R7: unidades no convertibles y negativos ------------------------
    antes = len(df)
    df = df[df["factor_a_mg_kg"].notna()].copy()
    bit.registrar("R7a Excluir unidad no convertible", antes, len(df),
                  "Unidades como 'g', 'sin unidad' o de campo en concentraciones")

    antes = len(df)
    concentracion = df["familia"].isin(
        {"metal", "metaloide", "mayor", "elemento", "pah", "btex", "tph", "anion"}
    )
    negativos = concentracion & df["valor"].lt(0)
    df = df[~negativos].copy()
    bit.registrar("R7b Excluir concentraciones negativas", antes, len(df),
                  "Físicamente imposibles; los negativos de ABA (NNP/RNP) se conservan")

    # --- R8: duplicados (muestra, parámetro) ------------------------------
    antes = len(df)
    clave = ["id_muestra", "parametro_estandar"]
    agg = (
        df.groupby(clave, as_index=False)
        .agg(
            valor=("valor", "median"),
            valor_proc=("valor_proc", "median"),
            censurado=("censurado", "any"),
            mayor_que=("mayor_que", "any"),
            n_mediciones=("valor", "size"),
            n_valores_distintos=("valor", "nunique"),
            familia=("familia", "first"),
        )
    )
    conflictos = agg[agg["n_valores_distintos"] > 1]
    bit.registrar("R8 Consolidar duplicados (muestra, parámetro)", antes, len(agg),
                  f"Se conserva la mediana; {len(conflictos)} pares tenían valores distintos")
    if len(conflictos):
        guardar_csv(conflictos, cfg.TABLES / "duplicados_con_conflicto.csv")

    # Contexto: una fila por muestra (first ignora NaN)
    contexto = (
        df.drop(columns=["parametro_estandar", "familia", "valor", "valor_proc",
                         "censurado", "mayor_que"])
        .groupby("id_muestra", as_index=False)
        .first()
    )
    agg = agg.merge(contexto, on="id_muestra", how="left", validate="many_to_one")

    # --- R9: coordenadas --------------------------------------------------
    este = pd.to_numeric(agg["COORD_ESTE"], errors="coerce")
    norte = pd.to_numeric(agg["COORD_NORTE"], errors="coerce")
    dentro = (
        este.between(*cfg.UTM_BBOX["este"]) & norte.between(*cfg.UTM_BBOX["norte"])
    )
    agg["coord_valida"] = dentro.fillna(False)
    n_invalidas = int((~agg["coord_valida"]).sum())
    if n_invalidas:
        LOG.warning("R9 %d registros con coordenadas fuera de la caja UTM de Moquegua", n_invalidas)

    agg = agg.sort_values(["id_muestra", "parametro_estandar"]).reset_index(drop=True)
    return agg, bit


# --------------------------------------------------------------------------
# Dataset ancho (RF-07)
# --------------------------------------------------------------------------
META_COLUMNAS = [
    "CUC", "PUNTO_MUESTREO", "PUNTO_NORMALIZADO", "fecha", "ANIO", "MES_NUM",
    "COORD_ESTE", "COORD_NORTE", "TXZONA", "DEPARTAMENTO", "PROVINCIA", "DISTRITO",
    "GRID_5KM", "TIPO_PUNT", "TXORIGEN", "LABORATORIO_ANONIMIZADO", "TAREA_TDR",
    "EXPEDIENTE", "RS", "coord_valida",
]


def construir_ancho(largo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pivota el formato largo a una fila por muestra (RF-07).

    Devuelve ``(ancho, meta)``. Las columnas del ancho son los nombres
    estándar de parámetro, en mg/kg cuando corresponde.
    """
    util = largo[largo["familia"].isin(
        {"metal", "metaloide", "mayor", "elemento", "pah", "btex", "tph", "anion",
         "fisico", "aba", "granulometria"}
    )].copy()

    ancho = (
        util.pivot_table(
            index="id_muestra",
            columns="parametro_estandar",
            values="valor_proc",
            aggfunc="median",
        )
        .sort_index()
    )
    ancho.columns.name = None

    # Metadatos: primera aparición por muestra
    meta = (
        largo.sort_values("id_muestra")
        .groupby("id_muestra", as_index=False)
        .agg({c: "first" for c in META_COLUMNAS if c in largo.columns})
        .set_index("id_muestra")
    )

    # Flags de censura por muestra
    cens = (
        largo.groupby("id_muestra")["censurado"].sum().rename("n_censurados")
    )
    meta = meta.join(cens)

    ancho = meta.join(ancho)
    return ancho.reset_index(), meta.reset_index()


def resumen_censura(largo: pd.DataFrame) -> pd.DataFrame:
    """Porcentaje de valores censurados por parámetro (RF-05)."""
    tab = (
        largo.groupby("parametro_estandar")
        .agg(
            n=("valor", "size"),
            n_censurados=("censurado", "sum"),
            pct_censurado=("censurado", "mean"),
            mediana=("valor_proc", "median"),
            maximo=("valor", "max"),
            familia=("familia", "first"),
        )
        .sort_values("n", ascending=False)
    )
    tab["pct_censurado"] = (100 * tab["pct_censurado"]).round(2)
    return tab.reset_index()


def ejecutar(df_crudo: pd.DataFrame) -> dict:
    """Corre la limpieza completa y guarda los artefactos intermedios."""
    largo, bitacora = limpiar_largo(df_crudo)
    ancho, meta = construir_ancho(largo)

    bitacora.guardar(cfg.TABLES / "log_limpieza.csv")
    guardar_csv(largo, cfg.DATA_INTERIM / "largo_limpio.csv")
    guardar_csv(ancho, cfg.DATA_INTERIM / "ancho_bruto.csv")

    censura = resumen_censura(largo)
    guardar_csv(censura, cfg.TABLES / "censura_por_parametro.csv")

    LOG.info("Dataset ancho: %d muestras × %d columnas", len(ancho), ancho.shape[1])
    return {
        "largo": largo,
        "ancho": ancho,
        "meta": meta,
        "bitacora": bitacora.a_dataframe(),
        "censura": censura,
    }
