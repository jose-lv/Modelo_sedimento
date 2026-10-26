"""Pruebas de las agregaciones espaciales y del ranking de puntos."""
import pandas as pd

from src import report


def _riesgo_sintetico() -> pd.DataFrame:
    return pd.DataFrame({
        "GRID_5KM": ["A", "A", "B", "B", "C"],
        "PUNTO_NORMALIZADO": ["p1", "p2", "p3", "p4", "p5"],
        "DISTRITO": ["Ilo"] * 5,
        "PLI": [2.0, 3.0, 5.0, 6.0, 1.0],
        "RI": [1.0, 2.0, 3.0, 4.0, 0.5],
        "nivel_pli": ["Medio", "Alto", "Muy alto", "Muy alto", "Bajo"],
        "nivel_riesgo": ["Medio", "Alto", "Muy alto", "Muy alto", "Bajo"],
        "nivel_riesgo_num": [1, 2, 3, 3, 0],
        "n_metales_sobre_pel": [0, 1, 2, 3, 0],
        "excede_pel": [False, True, True, True, False],
        "metal_dominante": ["Hg", "Hg", "As", "Hg", "Zn"],
        "metal_dominante_normativo": ["As", "Pb", "Cu", "Pb", "Zn"],
        "COORD_ESTE": [200000.0] * 5,
        "COORD_NORTE": [8000000.0] * 5,
        "ANIO": [2020] * 5,
    })


def test_zonas_criticas_ordena_por_severidad_y_agrega():
    zonas = report.zonas_criticas(_riesgo_sintetico(), top=2)

    assert list(zonas["GRID_5KM"]) == ["B", "A"]
    assert zonas.loc[0, "n_muestras"] == 2
    assert zonas.loc[0, "n_puntos"] == 2
    assert zonas.loc[0, "nivel_riesgo"] == "Muy alto"
    assert zonas.loc[0, "metal_dominante_normativo"] == "Cu"


def test_zonas_criticas_desempata_por_numero_de_puntos():
    df = _riesgo_sintetico()
    # Celda D: mismo nivel y % alto que B, pero con más puntos y menor PLI.
    extra = df.iloc[[2, 2, 2]].copy()
    extra["GRID_5KM"] = "D"
    extra["PUNTO_NORMALIZADO"] = ["p6", "p7", "p8"]
    df = pd.concat([df, extra], ignore_index=True)

    zonas = report.zonas_criticas(df, top=2)
    assert zonas.loc[0, "GRID_5KM"] == "D"
    assert zonas.loc[0, "n_puntos"] == 3
    assert zonas.loc[0, "PLI_mediano"] < zonas.loc[1, "PLI_mediano"]
    # El IC95 inferior premia la celda con más muestras aunque ambas estén al 100 %.
    assert zonas.loc[0, "pct_alto_ic95_inf"] > zonas.loc[1, "pct_alto_ic95_inf"]


def test_zonas_criticas_sin_columna_espacial_devuelve_vacio():
    zonas = report.zonas_criticas(_riesgo_sintetico().drop(columns="GRID_5KM"))
    assert zonas.empty


def test_ranking_usa_metal_normativo_y_conserva_el_cf():
    ranking = report.ranking_puntos(_riesgo_sintetico(), top=10)

    assert "metal_dominante_normativo" in ranking.columns
    assert "metal_enriquecido_cf" in ranking.columns
    # Ordenado por severidad normativa y luego por PLI mediano.
    assert ranking.loc[0, "nivel_riesgo_num"] == 3
    assert ranking.loc[0, "PLI_mediano"] == 6.0
    # El metal normativo no debe ser el Hg que premia el CF.
    assert ranking.loc[0, "metal_dominante_normativo"] == "Pb"
    assert ranking.loc[0, "metal_enriquecido_cf"] == "Hg"
