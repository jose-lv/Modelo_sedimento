"""Pruebas de las agregaciones espaciales, el ranking y el mapa interactivo."""
from pathlib import Path

import pandas as pd
import pytest

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


def test_zonas_criticas_top_none_devuelve_todas_las_celdas():
    zonas = report.zonas_criticas(_riesgo_sintetico(), top=None)
    assert len(zonas) == 3
    assert set(zonas["GRID_5KM"]) == {"A", "B", "C"}


def test_geometria_celda_5km_usa_cuatro_esquinas_utm():
    geom = report.geometria_celda_5km("74_1633", zona=19)
    assert geom is not None
    assert len(geom) == 4

    lat_so, lon_so = report.utm_a_latlon(370000, 8165000, zona=19)
    lat_no, lon_no = report.utm_a_latlon(370000, 8170000, zona=19)
    assert abs(geom[0][0] - float(lat_so)) < 1e-6
    assert abs(geom[0][1] - float(lon_so)) < 1e-6
    assert abs(geom[3][0] - float(lat_no)) < 1e-6
    assert abs(geom[3][1] - float(lon_no)) < 1e-6


def _riesgo_mapa() -> pd.DataFrame:
    df = _riesgo_sintetico()
    df["id_muestra"] = [f"id-{i}" for i in range(len(df))]
    df["GRID_5KM"] = ["74_1633", "74_1633", "75_1633", "75_1633", "74_1634"]
    df["TXZONA"] = 19
    df["fecha"] = "2022-01-01"
    df["n_metales_sobre_isqg"] = [0, 1, 2, 3, 0]
    return df


def test_mapa_folium_incluye_controles_y_capas(tmp_path):
    pytest.importorskip("folium")
    df = _riesgo_mapa()
    patrones = pd.DataFrame({
        "cluster_kmeans": [0, 1, 2, 0, 1],
        "cluster_gmm": [0, 0, 1, 1, 2],
        "es_anomalia": [True, False, False, True, False],
    })
    predicciones = {
        "regresion_pli": pd.DataFrame({
            "id_muestra": df["id_muestra"],
            "PLI_predicho": [2.1, 2.8, 5.2, 4.9, 1.2],
        }),
        "clasificacion_riesgo": pd.DataFrame({
            "id_muestra": df["id_muestra"],
            "prob_riesgo_alto": [0.1, 0.4, 0.9, 0.8, 0.2],
            "clase_predicha": [0, 0, 1, 1, 0],
        }),
    }
    ruta = report.mapa_folium(
        df, patrones=patrones, predicciones=predicciones,
        ruta=tmp_path / "mapa.html",
    )
    assert ruta is not None
    texto = Path(ruta).read_text(encoding="utf-8")

    assert "id='modo-color'" in texto
    assert "id='celdas'" in texto
    assert "id='ver-anomalias'" in texto
    assert "window.addEventListener('load'" in texto
    assert "cluster_kmeans" in texto
    assert "cluster_gmm" in texto
    assert "PLI_predicho_mapa" in texto
    assert "prob_riesgo_alto_mapa" in texto
    assert "clase_predicha_mapa" in texto
    assert "Celdas de 5 × 5 km" in texto
    assert 'var CELDAS = {"top": feature_group_' in texto
    assert '"todas": feature_group_' in texto
    assert "Anomalías (Isolation Forest)" in texto
    assert "no equivalen a niveles de riesgo" in texto


def test_exportar_dataset_final_une_predicciones_por_id(tmp_path, monkeypatch):
    riesgo = _riesgo_sintetico().iloc[[0, 2, 4]].reset_index(drop=True)
    riesgo["id_muestra"] = ["a", "b", "c"]
    patrones = pd.DataFrame({"cluster_kmeans": [0, 1, 2]})
    predicciones = pd.DataFrame({
        "id_muestra": ["c", "a", "b"],
        "PLI_predicho": [3.0, 1.0, 2.0],
    })
    monkeypatch.setattr(report.cfg, "DATA_PROCESSED", tmp_path)

    final = report.exportar_dataset_final(riesgo, patrones, predicciones)
    por_id = final.set_index("id_muestra")

    # La unión debe seguir el id, no el orden de las filas.
    assert por_id.loc["a", "PLI_predicho"] == 1.0
    assert por_id.loc["b", "PLI_predicho"] == 2.0
    assert por_id.loc["c", "PLI_predicho"] == 3.0
    assert por_id.loc["a", "PLI_predicho_mapa"] == 1.0
    assert "in-sample" in por_id.loc["a", "fuente_pli"]
    assert (tmp_path / "dataset_final.csv").exists()


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
