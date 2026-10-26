"""Utilidades transversales: logging, trazabilidad, E/S y formato."""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def get_logger(name: str = "pmv") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(cfg.LOGS / "pipeline.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


LOG = get_logger()


@contextmanager
def paso(nombre: str):
    """Cronometra y registra un paso del pipeline."""
    t0 = time.perf_counter()
    LOG.info("▶ %s", nombre)
    try:
        yield
    finally:
        LOG.info("  ✓ %s (%.1f s)", nombre, time.perf_counter() - t0)


# --------------------------------------------------------------------------
# Texto
# --------------------------------------------------------------------------
def sin_acentos(texto: str) -> str:
    """Devuelve el texto sin tildes y en minúsculas (clave de comparación)."""
    if texto is None or (isinstance(texto, float) and np.isnan(texto)):
        return ""
    nfkd = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def clave_parametro(texto: str) -> str:
    """Normaliza un nombre de parámetro para emparejarlo con el diccionario."""
    base = sin_acentos(texto)
    for ch in ("(", ")", ".", ",", ";", ":", "-", "_", "/", "'", '"', "+"):
        base = base.replace(ch, " ")
    return " ".join(base.split())


def slug(texto: str, max_len: int = 60) -> str:
    base = sin_acentos(texto)
    keep = [c if (c.isalnum() or c in " _-") else " " for c in base]
    out = " ".join("".join(keep).split()).replace(" ", "_")
    return out[:max_len].strip("_")


# --------------------------------------------------------------------------
# E/S
# --------------------------------------------------------------------------
def guardar_csv(df: pd.DataFrame, ruta: Path, index: bool = False) -> Path:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=index, encoding="utf-8-sig")
    return ruta


def leer_csv(ruta: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(ruta, encoding="utf-8-sig", **kwargs)


def guardar_json(obj, ruta: Path) -> Path:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)
    return ruta


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (Path,)):
        return str(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


def sha256_archivo(ruta: Path, bloque: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        while chunk := fh.read(bloque):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
def nueva_figura(filas: int = 1, cols: int = 1, **kwargs):
    fig, axes = plt.subplots(filas, cols, **kwargs)
    return fig, axes


def guardar_figura(fig, nombre: str) -> Path:
    ruta = cfg.FIGURES / f"{nombre}.{cfg.FIG_FORMAT}"
    fig.savefig(ruta, dpi=cfg.DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    LOG.info("  figura → %s", ruta.name)
    return ruta


# --------------------------------------------------------------------------
# Trazabilidad de la limpieza (RF-06)
# --------------------------------------------------------------------------
class Bitacora:
    """Registra cuántas filas entran y salen de cada regla de limpieza."""

    def __init__(self):
        self.registros: list[dict] = []

    def registrar(self, regla: str, antes: int, despues: int, motivo: str) -> None:
        self.registros.append(
            {
                "regla": regla,
                "filas_antes": int(antes),
                "filas_despues": int(despues),
                "filas_eliminadas": int(antes - despues),
                "pct_eliminado": round(100 * (antes - despues) / antes, 3) if antes else 0.0,
                "motivo": motivo,
            }
        )
        LOG.info("  · %-42s %7d → %7d (−%d)", regla, antes, despues, antes - despues)

    def a_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.registros)

    def guardar(self, ruta: Path) -> Path:
        return guardar_csv(self.a_dataframe(), ruta)
