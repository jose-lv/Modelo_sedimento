"""
Barrera anti-fuga de datos.

El PLI/Igeo/RI se calculan con 8 metales. Si alguno de esos metales (o una
columna derivada de ellos) entra como variable de entrada, el modelo aprende
la respuesta y las métricas dejan de ser válidas.

Ejecutar:  pytest -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import features


@pytest.fixture(scope="module")
def riesgo_sintetico() -> pd.DataFrame:
    """Dataset mínimo con los 8 metales del índice y variables de contexto."""
    rng = np.random.default_rng(cfg.SEED)
    n = 40
    df = pd.DataFrame({
        m: rng.uniform(1, 50, n) for m in cfg.METALES_INDICE
    })
    for c in features.FEATURES_LITOGENICAS + features.FEATURES_FISICO:
        df[c] = rng.uniform(0.1, 10, n)
    for c in features.FEATURES_ESPACIALES:
        df[c] = rng.uniform(0, 1000, n)
    df["TIPO_PUNT"] = rng.choice(["SEDIMENTO", "AGUA"], n)
    df["DISTRITO"] = rng.choice(["CARUMAS", "TORATA"], n)
    df["TXORIGEN"] = rng.choice(["MINERO", "OTRO"], n)

    # Columnas derivadas del índice: deben quedar prohibidas.
    df["PLI"] = rng.uniform(0.5, 5, n)
    df["PLI_log"] = np.log1p(df["PLI"])
    df["RI"] = rng.uniform(10, 600, n)
    df["CF_Cobre"] = rng.uniform(0.1, 9, n)
    df["Igeo_Plomo"] = rng.uniform(-1, 4, n)
    df["n_metales_sobre_pel"] = rng.integers(0, 4, n)
    df["excede_pel"] = df["n_metales_sobre_pel"] > 0
    return df


def test_metales_del_indice_nunca_son_features(riesgo_sintetico):
    feats = features.columnas_features(riesgo_sintetico)
    for metal in cfg.METALES_INDICE:
        assert metal not in feats, f"{metal} compone el target y entró como feature"


def test_columnas_derivadas_del_indice_estan_prohibidas(riesgo_sintetico):
    feats = features.columnas_features(riesgo_sintetico)
    prohibidas = [c for c in feats if c in cfg.COLUMNAS_PROHIBIDAS_FEATURES]
    assert not prohibidas, f"Columnas objetivo filtradas como feature: {prohibidas}"

    derivadas = [c for c in feats if c.startswith(("CF_", "Igeo_", "Bn_", "ratio_pel_"))]
    assert not derivadas, f"Derivadas del índice filtradas como feature: {derivadas}"


def test_verificar_no_fuga_aborta_con_metal_del_indice():
    with pytest.raises(ValueError, match="FUGA DE DATOS"):
        features.verificar_no_fuga(["Plomo", "pH pasta"])


def test_verificar_no_fuga_aborta_con_target():
    with pytest.raises(ValueError, match="FUGA DE DATOS"):
        features.verificar_no_fuga(["PLI_log"])


def test_verificar_no_fuga_acepta_contexto_geoquimico():
    # No debe lanzar: son litogénicos/fisicoquímicos fuera del índice.
    features.verificar_no_fuga(["Hierro", "Aluminio", "pH pasta", "COORD_ESTE"])


def test_el_target_no_es_reconstruible_desde_las_features(riesgo_sintetico):
    """
    Comprobación empírica: ninguna feature individual explica el PLI.
    Si alguien reintrodujera un metal del índice, |r| sería ~1.
    """
    feats = features.columnas_features(riesgo_sintetico)
    numericas = [
        c for c in feats
        if pd.api.types.is_numeric_dtype(riesgo_sintetico[c])
    ]
    correlaciones = (
        riesgo_sintetico[numericas]
        .corrwith(riesgo_sintetico["PLI"])
        .abs()
        .dropna()
    )
    assert correlaciones.max() < 0.95, (
        "Alguna feature casi reproduce el PLI: "
        f"{correlaciones.idxmax()} (r={correlaciones.max():.3f})"
    )
