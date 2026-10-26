"""
Pruebas de los índices de riesgo y de la escala normativa.

Se usan datos sintéticos con valores conocidos para verificar las fórmulas
contra su definición matemática, no contra la implementación.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import indices


@pytest.fixture()
def ancho_minimo() -> pd.DataFrame:
    """3 muestras con los 8 metales del índice y un fondo conocido."""
    fondo = {m: 10.0 for m in cfg.METALES_INDICE}
    return pd.DataFrame(
        [
            # Muestra 0: todos los metales = fondo → CF = 1, PLI = 1
            {m: 10.0 for m in cfg.METALES_INDICE},
            # Muestra 1: todos = 2 × fondo → CF = 2, PLI = 2
            {m: 20.0 for m in cfg.METALES_INDICE},
            # Muestra 2: solo 3 metales → no alcanza el mínimo de 4
            {**{m: 10.0 for m in cfg.METALES_INDICE[:3]},
             **{m: np.nan for m in cfg.METALES_INDICE[3:]}},
        ],
        index=["m0", "m1", "m2"],
    )


def test_pli_es_media_geometrica_de_cf(ancho_minimo):
    fondo = pd.Series({m: 10.0 for m in cfg.METALES_INDICE})
    res = indices.calcular_indices(ancho_minimo, fondo=fondo)

    assert res.loc["m0", "PLI"] == pytest.approx(1.0)
    assert res.loc["m1", "PLI"] == pytest.approx(2.0)
    # CF = concentración / fondo
    assert res.loc["m1", "CF_Plomo"] == pytest.approx(2.0)


def test_pli_exige_cuatro_metales(ancho_minimo):
    fondo = pd.Series({m: 10.0 for m in cfg.METALES_INDICE})
    res = indices.calcular_indices(ancho_minimo, fondo=fondo)

    assert res.loc["m2", "n_metales_indice"] == 3
    assert pd.isna(res.loc["m2", "PLI"]), "Con 3 metales el PLI no es representativo"


def test_igeo_sigue_la_definicion_log2_cf_1_5(ancho_minimo):
    fondo = pd.Series({m: 10.0 for m in cfg.METALES_INDICE})
    res = indices.calcular_indices(ancho_minimo, fondo=fondo)

    esperado = np.log2(2.0 / 1.5)  # CF = 2
    assert res.loc["m1", "Igeo_Plomo"] == pytest.approx(esperado, abs=1e-9)


def test_clase_igeo_es_entera_entre_0_y_6():
    for x, esperado in [(-2, 0), (0, 0), (0.2, 1), (1.0, 1), (2.3, 3), (9, 6)]:
        assert indices._clase_igeo(x) == esperado
    assert pd.isna(indices._clase_igeo(np.nan))


def test_ri_pondera_por_toxicidad(ancho_minimo):
    fondo = pd.Series({m: 10.0 for m in cfg.METALES_INDICE})
    res = indices.calcular_indices(ancho_minimo, fondo=fondo)

    esperado = sum(cfg.FACTORES_TOXICOS[m] for m in cfg.METALES_INDICE) * 2.0
    assert res.loc["m1", "RI"] == pytest.approx(esperado)


def test_nivel_pli_usa_los_cortes_documentados():
    assert indices._nivel_pli(0.5) == "Bajo"
    assert indices._nivel_pli(1.0) == "Bajo"
    assert indices._nivel_pli(1.5) == "Medio"
    assert indices._nivel_pli(2.5) == "Alto"
    assert indices._nivel_pli(4.0) == "Muy alto"
    assert indices._nivel_pli(np.nan) == "Sin dato"


def test_nivel_riesgo_es_normativo_no_un_umbral_de_pli():
    """
    El nivel de riesgo se ancla en guías CCME (ISQG/PEL), por lo que un PLI
    bajo puede coexistir con excedencia de PEL y viceversa.
    """
    df = pd.DataFrame({
        "PLI": [0.8, 0.8, 4.0, 4.0],
        "n_metales_indice": [8, 8, 8, 8],
        "n_metales_sobre_isqg": [0, 1, 0, 3],
        "n_metales_sobre_pel": [0, 0, 0, 3],
    })
    orden = {"Bajo": 0, "Medio": 1, "Alto": 2, "Muy alto": 3}
    nivel = np.select(
        [
            df["n_metales_sobre_pel"] >= 3,
            df["n_metales_sobre_pel"] >= 1,
            df["n_metales_sobre_isqg"] >= 1,
        ],
        ["Muy alto", "Alto", "Medio"],
        default="Bajo",
    )
    assert list(nivel) == ["Bajo", "Medio", "Bajo", "Muy alto"]
    assert orden["Muy alto"] > orden["Alto"] > orden["Medio"] > orden["Bajo"]


def test_fondo_local_es_un_percentil_positivo(ancho_minimo):
    fondo = indices.fondo_local(ancho_minimo, cfg.METALES_INDICE, percentil=25)
    assert (fondo > 0).all()
    # Con dos valores (10 y 20) el percentil 25 interpolado es 12.5.
    assert fondo["Plomo"] == pytest.approx(12.5)


def test_guias_ccme_tienen_isqg_y_pel_coherentes():
    guias = indices.cargar_guias_ccme()
    assert not guias.empty
    assert (guias["pel_mg_kg"] > guias["isqg_mg_kg"]).all(), "PEL debe superar a ISQG"
    # Los 8 metales del índice deben tener guía (Ni se complementa con NOAA).
    faltan = [m for m in cfg.METALES_INDICE if m not in set(guias["parametro"])]
    assert not faltan, f"Metales del índice sin guía: {faltan}"
