"""
Normalización de nombres de parámetros y unidades (RF-04).

El CSV del OEFA trae el mismo analito escrito de varias formas
("Arsénico", "Arsenico", "Arsénico Total"). Aquí se construye una tabla de
equivalencias versionada en ``config/parametros.csv`` y se aplica al dataset.

Decisiones:
* Los nombres canónicos conservan tildes (para las tablas de la tesis).
* La comparación se hace sin tildes, sin signos y en minúsculas.
* Las unidades de lixiviación (mg/L) NO se convierten a mg/kg: pertenecen a
  otra familia y se excluyen del análisis de fase sólida.
"""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from . import config as cfg
from .utils import LOG, clave_parametro, guardar_csv

# --------------------------------------------------------------------------
# Diccionario de parámetros
# --------------------------------------------------------------------------
# (nombre_canonico, familia, [alias explícitos], usar_en_modelo, notas)
_ESPECIFICACION = [
    # ---- Metales y metaloides -------------------------------------------
    ("Aluminio", "mayor", [], True, "Litogénico; no compone el índice"),
    ("Antimonio", "metal", [], True, ""),
    ("Arsénico", "metaloide", ["Arsenico Total", "Arsénico Total"], True, "Índice"),
    ("Bario", "metal", ["Bario Total Real"], True, ""),
    ("Berilio", "metal", [], True, ""),
    ("Bismuto", "metal", [], True, ""),
    ("Boro", "metaloide", [], True, ""),
    ("Cadmio", "metal", ["Cadmio Total"], True, "Índice"),
    ("Calcio", "mayor", [], True, ""),
    ("Cerio", "metal", [], True, "Tierra rara"),
    ("Cobalto", "metal", [], True, ""),
    ("Cobre", "metal", [], True, "Índice"),
    ("Cromo", "metal", [], True, "Índice"),
    ("Cromo Hexavalente", "metal", ["Cromo Hexavalente Total"], False, "Cobertura mínima"),
    ("Estaño", "metal", [], True, ""),
    ("Estroncio", "metal", [], True, ""),
    ("Fósforo", "mayor", ["Fosforo"], True, ""),
    ("Hierro", "mayor", [], True, "Litogénico; no compone el índice"),
    ("Litio", "metal", [], True, ""),
    ("Magnesio", "mayor", [], True, ""),
    ("Manganeso", "metal", [], True, "Litogénico; no compone el índice"),
    ("Mercurio", "metal", ["Mercurio Total"], True, "Índice"),
    ("Molibdeno", "metal", [], True, ""),
    ("Níquel", "metal", ["Niquel"], True, "Índice"),
    ("Plata", "metal", [], True, ""),
    ("Plomo", "metal", ["Plomo Total"], True, "Índice"),
    ("Potasio", "mayor", [], True, ""),
    ("Selenio", "metaloide", [], True, ""),
    ("Silicio", "mayor", [], True, ""),
    ("Sodio", "mayor", [], True, ""),
    ("Talio", "metal", [], True, ""),
    ("Titanio", "mayor", [], True, ""),
    ("Torio", "metal", [], True, "Radionucleido natural"),
    ("Uranio", "metal", [], True, "Radionucleido natural"),
    ("Vanadio", "metal", [], True, ""),
    ("Wolframio", "metal", [], True, ""),
    ("Zinc", "metal", [], True, "Índice"),
    ("Azufre Total", "elemento", [], False, "Indicador de sulfuros"),
    # ---- Hidrocarburos aromáticos policíclicos (PAH) ---------------------
    ("Acenafteno", "pah", [], False, ""),
    ("Acenaftileno", "pah", [], False, ""),
    ("Antraceno", "pah", [], False, ""),
    ("Benzo(a)antraceno", "pah", [], False, ""),
    ("Benzo(a)pireno", "pah", [], False, "Cancerígeno; guía CCME"),
    ("Benzo(b)fluoranteno", "pah", [], False, ""),
    ("Benzo(Ghi)perileno", "pah", ["Benzo(ghi)perileno"], False, ""),
    ("Benzo(k)fluoranteno", "pah", [], False, ""),
    ("Criseno", "pah", [], False, ""),
    ("Dibenzo(a,h)antraceno", "pah", [], False, ""),
    ("Fenantreno", "pah", [], False, ""),
    ("Fluoranteno", "pah", [], False, ""),
    ("Fluoreno", "pah", [], False, ""),
    ("Indeno(1,2,3-cd)pireno", "pah", [], False, ""),
    ("Naftaleno", "pah", [], False, ""),
    ("Pireno", "pah", [], False, ""),
    # ---- BTEX ------------------------------------------------------------
    ("Benceno", "btex", [], False, ""),
    ("Tolueno", "btex", [], False, ""),
    ("Etilbenceno", "btex", [], False, ""),
    ("Xilenos", "btex", [], False, "Suma de xilenos"),
    ("o-Xileno", "btex", [], False, ""),
    ("m-Xileno", "btex", [], False, ""),
    ("p-Xileno", "btex", [], False, ""),
    ("m,p-Xileno", "btex", [], False, ""),
    ("BTEX total", "btex", ["BTEX (Benceno, Tolueno, Etilbenceno, Xilenos)"], False, ""),
    # ---- Hidrocarburos totales de petróleo -------------------------------
    ("TPH (C6-C40)", "tph", ["Hidrocarburos Totales de Petroleo-TPH",
                             "Hidrocarburos Totales de Petróleo -TPH"], False, ""),
    ("TPH F1 (C6-C10)", "tph", ["Fracciones de Hidrocarburos F1 (C6-C10)",
                                "Hidrocarburos Totales de Petroleo F1",
                                "Hidrocarburos Totales de Petroleo F1 (C6 - C10)"], False, ""),
    ("TPH F2 (C10-C28)", "tph", ["Fracciones de Hidrocarburos F2 (C10-C28)",
                                 "Hidrocarburos Totales de Petroleo F2",
                                 "Hidrocarburos Totales de Petroleo F2 (C10 - C28)"], False, ""),
    ("TPH F3 (C28-C40)", "tph", ["Fracciones de Hidrocarburos F3 (&gt;C28-C40)",
                                 "Fracciones de Hidrocarburos F3 (C28-C40)",
                                 "Hidrocarburos Totales de Petroleo F3",
                                 "Hidrocarburos Totales de Petroleo F3 (C28 - C40)"], False, ""),
    ("TPH > C40", "tph", ["TPH mayor a C40"], False, ""),
    # ---- Aniones y nutrientes -------------------------------------------
    ("Sulfato", "anion", ["Sulfato Total", "Sulfatos", "Sulfato(s)", "Sulfatos, SO4-2"], False, ""),
    ("Sulfuro", "anion", ["Sulfuro Total", "Sulfuros"], False, ""),
    ("Cianuro Total", "anion", [], False, ""),
    ("Cianuro Libre", "anion", [], False, ""),
    ("Cianuro WAD", "anion", [], False, ""),
    ("Nitrato", "anion", ["Nitratos", "Nitratos, NO3-"], False, ""),
    ("Nitrato (como N)", "anion", ["Nitratos (como N)"], False, ""),
    ("Nitrito", "anion", ["Nitritos", "Nitritos, NO2-"], False, ""),
    ("Nitrito (como N)", "anion", ["Nitritos (como N)"], False, ""),
    ("Cloruro", "anion", ["Cloruros", "Cloruros, Cl-"], False, ""),
    ("Fluoruro", "anion", ["Fluoruros", "Fluoruros, F-"], False, ""),
    ("Bromuro", "anion", ["Bromuro, Br-"], False, ""),
    ("Fosfato", "anion", ["Fosfatos", "Fosfatos, PO4-3"], False, ""),
    ("Fosfato (como P)", "anion", ["Fosfatos (como P)"], False, ""),
    ("Amonio", "anion", ["Amonio (NH4+)"], False, ""),
    # ---- Físico-químicos -------------------------------------------------
    ("pH pasta", "fisico", ["pH Pasta", "pH en pasta"], False, ""),
    ("pH", "fisico", [], False, ""),
    ("Materia orgánica", "fisico", [], False, ""),
    ("Carbono Orgánico Total (COT)", "fisico", [], False, ""),
    ("Conductividad", "fisico", [], False, ""),
    ("Aceites y Grasas", "fisico", ["Aceites y Grasas ó MEH"], False, ""),
    ("Organoléptico", "fisico", [], False, ""),
    # ---- Pruebas ABA (ácido-base) ---------------------------------------
    ("Potencial de Neutralización Neto (NNP)", "aba", [], False, "Puede ser negativo"),
    ("Potencial de Acidez Máximo (MPA)", "aba", [], False, ""),
    ("Potencial de Acidez Maximo (PAM Basado en Azufre total)", "aba", [], False, ""),
    ("Potencial de Neutralización Sobek", "aba", [], False, ""),
    ("Potencial de Neutralizacion (PN)", "aba", [], False, ""),
    ("Ratio Potencial de Neutralización (RNP)", "aba", [], False, "Puede ser negativo"),
    ("Relacion PN/PA", "aba", [], False, ""),
    ("Fizz Rating", "aba", [], False, ""),
    # ---- Granulometría ---------------------------------------------------
    ("% Arcilla", "granulometria", [], False, ""),
    ("% Limo", "granulometria", [], False, ""),
    ("% Arena", "granulometria", [], False, ""),
    ("% Grava", "granulometria", [], False, ""),
    # ---- Genéricos / no analíticos --------------------------------------
    ("Metales Totales", "otro", [], False, "Registro genérico, sin analito"),
    ("Análisis granulométrico", "otro", [], False, "Registro genérico"),
    ("Shake Flask en aniones", "lixiviado", [], False, "Ensayo de lixiviación (mg/L)"),
    ("Compuestos Orgánicos Volátiles-COV", "otro", [], False, ""),
    ("Macrobentos", "otro", [], False, ""),
    ("PAHs", "pah", [], False, "Suma de PAH"),
    ("BTEX", "btex", [], False, "Suma de BTEX"),
    ("ABA", "aba", [], False, ""),
]

#: Familias cuyos valores son concentraciones en fase sólida y se convierten
#: a mg/kg peso seco.
FAMILIAS_CONCENTRACION = {"metal", "metaloide", "mayor", "elemento", "pah", "btex", "tph", "anion"}

#: Factores de conversión de unidad a mg/kg según la unidad de origen.
FACTORES_UNIDAD = {
    "mg/kg": 1.0,
    "mg/kg ms": 1.0,
    "mg/kg peso seco": 1.0,
    "kg/tm": 1_000.0,       # 1 kg/t = 1 g/kg = 1000 mg/kg
    "g/kg": 1_000.0,
    "%": 10_000.0,          # 1 % = 10 g/kg = 10 000 mg/kg
    "ug/kg": 0.001,
    "µg/kg": 0.001,
}

#: Unidades que identifican un ensayo de lixiviación (otra familia).
UNIDADES_LIXIVIADO = {"mg/l", "µg/l", "ug/l", "mg/s"}


def _construir_lookup() -> dict[str, tuple[str, str, bool, str]]:
    """clave normalizada → (canónico, familia, usar_en_modelo, notas)."""
    lookup: dict[str, tuple[str, str, bool, str]] = {}
    for canon, familia, alias, usa, notas in _ESPECIFICACION:
        lookup[clave_parametro(canon)] = (canon, familia, usa, notas)
        for a in alias:
            lookup[clave_parametro(a)] = (canon, familia, usa, notas)
    return lookup


LOOKUP = _construir_lookup()

#: Alias adicionales que aparecen solo como columnas de valor del CSV.
_COLUMNAS_VALOR = {
    "OTROS": "OTROS",
    "ABA": "ABA",
    "Aniones": "Aniones",
    "Análisis.granulométrico": "Análisis granulométrico",
    "BTEX": "BTEX",
    "BTEX..Benceno..Tolueno..Etilbenceno..Xilenos.": "BTEX total",
    "Hidrocarburos.Aromáticos.Policíclicos.PAH": "PAHs",
    "Metales.Totales": "Metales Totales",
    "Metales.Totales.y.Mercurio": "Metales Totales",
    "PAHs": "PAHs",
    "Shake.Flash.en.aniones..mg.L.": "Shake Flask en aniones",
    "Shake.Flask.en.metales..mg.L..o.Shake.Flask.Test..Solubity.Testing..SFT.": "Shake Flask en metales",
}


def canonicalizar(nombre: str) -> tuple[str, str, bool, str]:
    """Devuelve (nombre_canónico, familia, usar_en_modelo, notas)."""
    k = clave_parametro(html.unescape(str(nombre)))
    if k in LOOKUP:
        return LOOKUP[k]
    limpio = " ".join(html.unescape(str(nombre)).split())
    return (limpio, "otro", False, "Sin equivalencia en el diccionario")


def factor_unidad(familia: str, unidad: str) -> float | None:
    """Factor para llevar el valor a mg/kg. ``None`` = no convertible."""
    if familia not in FAMILIAS_CONCENTRACION:
        return 1.0
    u = str(unidad).strip().lower()
    if u in UNIDADES_LIXIVIADO:
        return None
    return FACTORES_UNIDAD.get(u)


def es_lixiviado(unidad: str) -> bool:
    return str(unidad).strip().lower() in UNIDADES_LIXIVIADO


def tabla_equivalencias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye ``config/parametros.csv`` con una fila por cada combinación
    (parámetro original, unidad) realmente presente en los datos.
    """
    base = df[["PARAMETRO", "UNIDAD_MEDIDA_PARAMETRO"]].drop_duplicates().copy()
    filas = []
    for _, r in base.iterrows():
        canon, familia, usa, notas = canonicalizar(r["PARAMETRO"])
        fac = factor_unidad(familia, r["UNIDAD_MEDIDA_PARAMETRO"])
        filas.append(
            {
                "parametro_original": r["PARAMETRO"],
                "unidad_original": r["UNIDAD_MEDIDA_PARAMETRO"],
                "parametro_estandar": canon,
                "familia": familia,
                "es_concentracion": familia in FAMILIAS_CONCENTRACION,
                "factor_a_mg_kg": fac,
                "unidad_estandar": "mg/kg MS" if fac not in (None, 1.0) else r["UNIDAD_MEDIDA_PARAMETRO"],
                "conversion_valida": fac is not None,
                "usar_en_modelo": usa,
                "notas": notas,
            }
        )
    tabla = pd.DataFrame(filas).sort_values(["familia", "parametro_estandar", "unidad_original"])
    tabla["n_registros"] = tabla.apply(
        lambda r: int(
            (
                (df["PARAMETRO"] == r["parametro_original"])
                & (df["UNIDAD_MEDIDA_PARAMETRO"] == r["unidad_original"])
            ).sum()
        ),
        axis=1,
    )
    return tabla.reset_index(drop=True)


def aplicar_normalizacion(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas ``parametro_estandar``, ``familia`` y ``factor``."""
    cache = {}

    def _map(nombre):
        if nombre not in cache:
            cache[nombre] = canonicalizar(nombre)
        return cache[nombre]

    info = df["PARAMETRO"].map(_map)
    df["parametro_estandar"] = [x[0] for x in info]
    df["familia"] = [x[1] for x in info]
    df["_usar_modelo"] = [x[2] for x in info]
    df["factor_a_mg_kg"] = [
        factor_unidad(f, u) for f, u in zip(df["familia"], df["UNIDAD_MEDIDA_PARAMETRO"])
    ]
    df["es_lixiviado"] = df["UNIDAD_MEDIDA_PARAMETRO"].map(es_lixiviado)
    return df


def generar_tabla_parametros(df: pd.DataFrame) -> pd.DataFrame:
    """Genera y guarda la tabla de equivalencias (RF-04)."""
    tabla = tabla_equivalencias(df)
    guardar_csv(tabla, cfg.CONFIG_DIR / "parametros.csv")
    sin_map = tabla[tabla["familia"] == "otro"]
    if len(sin_map):
        LOG.warning("Parámetros sin equivalencia explícita: %d combinaciones", len(sin_map))
    LOG.info("Tabla de equivalencias: %d filas, %d parámetros estándar",
             len(tabla), tabla["parametro_estandar"].nunique())
    return tabla
