"""
Carga y filtrado del CSV del OEFA (RF-01 y RF-02).

El archivo crudo nunca se modifica. Para acelerar las corridas sucesivas se
guarda una copia intermedia en ``data/interim/`` (pickle) identificada por el
hash del CSV original: si el crudo cambia, la caché se invalida sola.
"""
from __future__ import annotations

import pickle

import pandas as pd

from . import config as cfg
from .utils import LOG, Bitacora, sha256_archivo, sin_acentos

#: Columnas mínimas que el pipeline necesita para funcionar.
COLUMNAS_REQUERIDAS = [
    "TXORIGEN", "ANHO", "MES", "CUC", "FECHAINI", "FECHAFIN", "TXUBIGEO", "TXZONA",
    "COORD_ESTE", "COORD_NORTE", "PUNTO_MUESTREO", "TIPO_PUNT", "MATRIZ",
    "FECHA_PTO", "HORA_PTO", "PARAMETRO", "SIGNO_PARAMETRO",
    "UNIDAD_MEDIDA_PARAMETRO", "RS",
]

#: Columnas donde el OEFA deposita el resultado numérico según el tipo de ensayo.
COLUMNAS_VALOR = [
    "OTROS", "ABA", "Aniones", "Análisis.granulométrico", "BTEX",
    "BTEX..Benceno..Tolueno..Etilbenceno..Xilenos.",
    "Hidrocarburos.Aromáticos.Policíclicos.PAH", "Metales.Totales",
    "Metales.Totales.y.Mercurio", "PAHs", "Shake.Flash.en.aniones..mg.L.",
    "Shake.Flask.en.metales..mg.L..o.Shake.Flask.Test..Solubity.Testing..SFT.",
]

CACHE_RAW = cfg.DATA_INTERIM / "raw_moquegua.pkl"
CACHE_HASH = cfg.DATA_INTERIM / "raw_moquegua.sha256"


def validar_columnas(df: pd.DataFrame) -> None:
    """RF-01: falla con un mensaje claro si falta una columna esperada."""
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise ValueError(
            "El CSV no tiene la estructura esperada del OEFA. "
            f"Columnas faltantes: {faltantes}. "
            f"Columnas presentes: {list(df.columns)}"
        )
    valores_faltantes = [c for c in COLUMNAS_VALOR if c not in df.columns]
    if valores_faltantes:
        raise ValueError(
            "Faltan columnas de valor (donde se almacenan los resultados): "
            f"{valores_faltantes}"
        )
    LOG.info("RF-01 OK: %d columnas, %d de valor", len(df.columns), len(COLUMNAS_VALOR))


def _leer_csv(ruta) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(ruta, encoding=enc, low_memory=False)
            LOG.info("CSV leído con codificación %s: %s filas × %s columnas",
                     enc, f"{len(df):,}", df.shape[1])
            df.columns = [c.strip() for c in df.columns]
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo decodificar {ruta} con utf-8 ni latin-1")


def cargar_crudo(usar_cache: bool = True) -> pd.DataFrame:
    """Carga el CSV crudo aplicando filtro de matriz y de departamento."""
    if not cfg.RAW_CSV.exists():
        raise FileNotFoundError(
            f"No se encontró el CSV del OEFA en {cfg.RAW_CSV}. "
            "Copie el archivo a data/raw/ o defina la variable de entorno OEFA_CSV."
        )

    huella = sha256_archivo(cfg.RAW_CSV)
    if usar_cache and CACHE_RAW.exists() and CACHE_HASH.exists():
        if CACHE_HASH.read_text(encoding="utf-8").strip() == huella:
            LOG.info("Usando caché intermedia %s", CACHE_RAW.name)
            with open(CACHE_RAW, "rb") as fh:
                return pickle.load(fh)

    bitacora = Bitacora()
    df = _leer_csv(cfg.RAW_CSV)
    validar_columnas(df)

    n0 = len(df)
    # --- RF-02a: solo la matriz sedimento ---------------------------------
    if "MATRIZ" in df.columns:
        df = df[df["MATRIZ"].astype(str).str.strip().str.lower() == cfg.MATRIZ_OBJETIVO.lower()]
    bitacora.registrar("Filtro matriz = Sedimento", n0, len(df),
                       "El estudio es de sedimentos; se descartan agua y suelo")

    # --- RF-02b: solo Moquegua --------------------------------------------
    df = filtrar_moquegua(df, bitacora)
    df = df.reset_index(drop=True)

    LOG.info("Registros retenidos: %s de %s (%.2f %%)",
             f"{len(df):,}", f"{n0:,}", 100 * len(df) / n0)
    bitacora.guardar(cfg.TABLES / "log_filtros_carga.csv")

    CACHE_RAW.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_RAW, "wb") as fh:
        pickle.dump(df, fh, protocol=4)
    CACHE_HASH.write_text(huella, encoding="utf-8")
    return df


def filtrar_moquegua(df: pd.DataFrame, bitacora: Bitacora | None = None) -> pd.DataFrame:
    """
    Filtra los registros cuyo ubigeo corresponde a Moquegua.

    ``TXUBIGEO`` tiene el formato ``DEPARTAMENTO, PROVINCIA, DISTRITO`` y puede
    contener varios ubigeos separados por ``/`` cuando una supervisión abarcó
    varias zonas. Se toma el **primer** segmento porque es el que identifica al
    departamento del registro. Hay 10 372 filas de sedimento cuyo texto menciona
    Moquegua pero cuyo primer departamento es otro (una supervisión nacional que
    abarca Amazonas, Cajamarca, Lambayeque, Loreto y Piura); sus coordenadas
    caen fuera de la caja UTM de Moquegua (0 de 10 372 dentro).
    """
    ubigeo = df["TXUBIGEO"].astype(str).str.upper()
    primer_departamento = (
        ubigeo.str.split("/").str[0].str.strip().str.split(",").str[0].str.strip()
    )
    objetivo = sin_acentos(cfg.DEPARTAMENTO_OBJETIVO).upper()
    mascara = primer_departamento.map(sin_acentos).str.upper() == objetivo

    if bitacora is not None:
        bitacora.registrar(
            "Filtro departamento = Moquegua", len(df), int(mascara.sum()),
            "TXUBIGEO (primer segmento); los ubigeos agrupados no se pueden georreferenciar",
        )
    return df[mascara].copy()


def consolidar_valor(df: pd.DataFrame) -> pd.DataFrame:
    """
    RF-03: consolida las 12 columnas de valor en una sola columna ``valor``.

    Sólo una columna tiene dato por fila (verificado: 99.95 % de las filas).
    Los valores vienen como texto con coma decimal en algunos registros.
    """
    df = df.copy()
    valores = df[COLUMNAS_VALOR].apply(
        lambda s: pd.to_numeric(
            s.astype(str).str.replace(",", ".", regex=False).str.strip(),
            errors="coerce",
        )
    )
    df["valor"] = valores.bfill(axis=1).iloc[:, 0]
    df["tipo_analisis"] = valores.notna().idxmax(axis=1)
    df.loc[valores.notna().sum(axis=1) == 0, "tipo_analisis"] = pd.NA
    df["n_columnas_con_valor"] = valores.notna().sum(axis=1)
    return df
