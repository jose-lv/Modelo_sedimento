"""
Mapa de focos críticos, ranking de puntos, exportación de resultados
(RF-16 y RF-18) y verificación del checklist de aceptación del MVP.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as cfg
from .utils import LOG, guardar_csv, guardar_json


# --------------------------------------------------------------------------
# Coordenadas
# --------------------------------------------------------------------------
def utm_a_latlon(este, norte, zona: int = 19):
    """Convierte UTM (sur) a latitud/longitud usando pyproj."""
    from pyproj import Transformer
    tr = Transformer.from_crs(f"EPSG:327{zona:02d}", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(np.asarray(este, dtype=float), np.asarray(norte, dtype=float))
    return lat, lon


# --------------------------------------------------------------------------
# Ranking de puntos críticos
# --------------------------------------------------------------------------
def ranking_puntos(riesgo: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Agrega por punto de muestreo y ordena por severidad normativa y PLI."""
    cols = ["PUNTO_NORMALIZADO", "DISTRITO", "PLI", "RI", "nivel_pli",
            "nivel_riesgo", "nivel_riesgo_num", "n_metales_sobre_pel",
            "metal_dominante_normativo", "metal_dominante", "excede_pel",
            "COORD_ESTE", "COORD_NORTE", "ANIO"]
    cols = [c for c in cols if c in riesgo.columns]
    d = riesgo[cols].dropna(subset=["PLI"])

    g = d.groupby("PUNTO_NORMALIZADO").agg(
        n_muestras=("PLI", "size"),
        PLI_mediano=("PLI", "median"),
        PLI_max=("PLI", "max"),
        RI_mediano=("RI", "median"),
        distrito=("DISTRITO", "first"),
        metal_dominante_normativo=("metal_dominante_normativo",
                                   lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan),
        metal_enriquecido_cf=("metal_dominante",
                              lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan),
        pct_excede_pel=("excede_pel", "mean"),
        max_metales_sobre_pel=("n_metales_sobre_pel", "max"),
        nivel_riesgo_num=("nivel_riesgo_num", "max"),
        este=("COORD_ESTE", "first"),
        norte=("COORD_NORTE", "first"),
        anio_min=("ANIO", "min"),
        anio_max=("ANIO", "max"),
    ).reset_index()
    g["pct_excede_pel"] = (100 * g["pct_excede_pel"]).round(1)
    g["nivel"] = g["nivel_riesgo_num"].map({0: "Bajo", 1: "Medio", 2: "Alto", 3: "Muy alto"})
    g["nivel_pli"] = pd.cut(
        g["PLI_mediano"], bins=[-np.inf, 1, 2, 3, np.inf],
        labels=["Bajo", "Medio", "Alto", "Muy alto"],
    )
    g = g.sort_values(["nivel_riesgo_num", "PLI_mediano"], ascending=[False, False])
    return g.head(top).reset_index(drop=True)


def ranking_con_shap(ranking: pd.DataFrame, contribucion: pd.DataFrame) -> pd.DataFrame:
    """Añade la variable que más aporta según SHAP en las muestras del punto."""
    if contribucion is None or contribucion.empty or cfg.COLUMNA_GRUPO not in contribucion:
        return ranking
    por_punto = (
        contribucion.groupby(cfg.COLUMNA_GRUPO)["variable_mas_influyente"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
        .rename("variable_shap_dominante")
    )
    return ranking.merge(
        por_punto, left_on="PUNTO_NORMALIZADO", right_index=True, how="left"
    )


def zonas_criticas(riesgo: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """
    Agrega las muestras por celda de 5 × 5 km y ordena las zonas por severidad
    normativa. Responde al objetivo del plan de localizar zonas críticas, no
    solo puntos aislados.
    """
    if "GRID_5KM" not in riesgo.columns:
        return pd.DataFrame()
    cols = ["GRID_5KM", "PUNTO_NORMALIZADO", "DISTRITO", "PLI", "RI",
            "nivel_riesgo", "nivel_riesgo_num", "n_metales_sobre_pel",
            "metal_dominante", "metal_dominante_normativo",
            "COORD_ESTE", "COORD_NORTE", "ANIO"]
    d = riesgo[[c for c in cols if c in riesgo.columns]].dropna(subset=["GRID_5KM"])
    if d.empty:
        return pd.DataFrame()

    g = d.groupby("GRID_5KM").agg(
        n_muestras=("PLI", "size"),
        n_puntos=("PUNTO_NORMALIZADO", "nunique"),
        PLI_mediano=("PLI", "median"),
        PLI_max=("PLI", "max"),
        max_metales_sobre_pel=("n_metales_sobre_pel", "max"),
        nivel_riesgo_num=("nivel_riesgo_num", "max"),
        pct_alto=("nivel_riesgo", lambda s: 100 * s.isin(["Alto", "Muy alto"]).mean()),
        distrito=("DISTRITO", lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan),
        metal_dominante_normativo=("metal_dominante_normativo",
                                   lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan),
        este=("COORD_ESTE", "median"),
        norte=("COORD_NORTE", "median"),
        anio_min=("ANIO", "min"),
        anio_max=("ANIO", "max"),
    ).reset_index()

    g["nivel_riesgo"] = g["nivel_riesgo_num"].map(
        {0: "Bajo", 1: "Medio", 2: "Alto", 3: "Muy alto"}
    )
    g["pct_alto"] = g["pct_alto"].round(1)

    # Límite inferior del IC 95 % de Wilson para la proporción de muestras en
    # Alto/Muy alto. Evita que una celda con una sola muestra al 100 % desplace
    # a una zona con decenas de puntos y 95 % de excedencia.
    z = 1.96
    p = g["pct_alto"] / 100.0
    n = g["n_muestras"].astype(float)
    denom = 1.0 + z * z / n
    centro = (p + z * z / (2.0 * n)) / denom
    margen = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    g["pct_alto_ic95_inf"] = (100.0 * (centro - margen)).clip(lower=0).round(1)

    g = g.sort_values(
        ["nivel_riesgo_num", "pct_alto_ic95_inf", "n_puntos", "PLI_mediano"],
        ascending=False,
    ).head(top)
    return g.reset_index(drop=True)


# --------------------------------------------------------------------------
# Mapa interactivo
# --------------------------------------------------------------------------
def mapa_folium(riesgo: pd.DataFrame, patrones: pd.DataFrame | None = None,
                ruta=None) -> str | None:
    """Mapa HTML con puntos coloreados por nivel de riesgo."""
    import folium

    d = riesgo.dropna(subset=["COORD_ESTE", "COORD_NORTE"]).copy()
    if d.empty:
        return None
    lat, lon = utm_a_latlon(d["COORD_ESTE"], d["COORD_NORTE"],
                            zona=int(d["TXZONA"].mode().iloc[0]) if "TXZONA" in d else 19)
    d["lat"], d["lon"] = lat, lon
    if patrones is not None:
        d = d.join(patrones[["cluster_kmeans", "es_anomalia"]], how="left")

    # Colores alineados con plots.mapa_nivel (escala CCME ISQG/PEL).
    colores = {"Bajo": "#2c7bb6", "Medio": "#ffffbf", "Alto": "#fdae61",
               "Muy alto": "#d7191c", "Sin dato": "#bdbdbd"}
    centro = [float(d["lat"].mean()), float(d["lon"].mean())]
    mapa = folium.Map(location=centro, zoom_start=10, tiles="OpenStreetMap")

    for _, r in d.iterrows():
        nivel = str(r.get("nivel_riesgo", "Sin dato"))
        color = colores.get(nivel, "#777777")
        pli = r.get("PLI", float("nan"))
        pli_txt = f"{pli:.2f}" if pd.notna(pli) else "s/d"
        ri = r.get("RI", float("nan"))
        ri_txt = f"{ri:.0f}" if pd.notna(ri) else "s/d"
        popup = (
            f"<b>{r['PUNTO_NORMALIZADO']}</b><br>"
            f"Distrito: {r.get('DISTRITO', '')}<br>"
            f"Fecha: {str(r.get('fecha', ''))[:10]}<br>"
            f"Nivel de riesgo: <b>{nivel}</b><br>"
            f"Metales &gt; PEL: {r.get('n_metales_sobre_pel', 's/d')} / 8<br>"
            f"Metales &gt; ISQG: {r.get('n_metales_sobre_isqg', 's/d')} / 8<br>"
            f"PLI: {pli_txt}<br>"
            f"RI: {ri_txt}<br>"
            f"Metal dominante (norma): {r.get('metal_dominante_normativo', '')}<br>"
            f"Metal más enriquecido (CF): {r.get('metal_dominante', '')}<br>"
            f"Cluster: {r.get('cluster_kmeans', '')}"
        )
        folium.CircleMarker(
            location=[r["lat"], r["lon"]], radius=5.5, color=color, weight=1,
            fill=True, fill_color=color, fill_opacity=0.85, popup=folium.Popup(popup, max_width=300),
        ).add_to(mapa)

    leyenda = (
        "<div style='position:fixed;bottom:30px;left:30px;z-index:9999;background:white;"
        "padding:10px;border:1px solid #ccc;font-size:12px;'>"
        "<b>Nivel de riesgo (guías CCME ISQG/PEL)</b><br>"
        "<span style='color:#2c7bb6'>●</span> Bajo: ningún metal &gt; ISQG<br>"
        "<span style='color:#ffffbf'>●</span> Medio: ≥1 metal &gt; ISQG<br>"
        "<span style='color:#fdae61'>●</span> Alto: ≥1 metal &gt; PEL<br>"
        "<span style='color:#d7191c'>●</span> Muy alto: ≥3 metales &gt; PEL<br>"
        "<span style='color:#bdbdbd'>●</span> Sin dato</div>"
    )
    mapa.get_root().html.add_child(folium.Element(leyenda))

    ruta = ruta or (cfg.REPORTS / "mapa_focos.html")
    mapa.save(str(ruta))
    LOG.info("Mapa interactivo → %s", ruta)
    return str(ruta)


# --------------------------------------------------------------------------
# Exportación (RF-18)
# --------------------------------------------------------------------------
def exportar_dataset_final(riesgo: pd.DataFrame, patrones: pd.DataFrame | None,
                           predicciones: pd.DataFrame | None) -> pd.DataFrame:
    final = riesgo.copy()
    if patrones is not None:
        final = final.join(patrones, how="left")
    if predicciones is not None:
        cols = [c for c in predicciones.columns if c.startswith(("PLI_predicho", "prob_", "clase_"))]
        if cols:
            final = final.join(predicciones[cols], how="left")
    guardar_csv(final, cfg.DATA_PROCESSED / "dataset_final.csv")
    return final


# --------------------------------------------------------------------------
# Checklist de aceptación (guía, sección 12)
# --------------------------------------------------------------------------
def checklist(estado: dict) -> pd.DataFrame:
    items = [
        ("El CSV crudo se transforma en un dataset ancho con un solo comando",
         estado.get("pipeline_un_comando", False), "python run_pipeline.py"),
        ("Existe un log con el número de filas eliminadas por cada regla",
         estado.get("log_limpieza", False), "reports/tables/log_limpieza.csv"),
        ("El porcentaje de datos censurados por parámetro está documentado",
         estado.get("censura", False), "reports/tables/censura_por_parametro.csv"),
        ("No hay fuga de datos: el target no depende de las entradas",
         estado.get("sin_fuga", False), "tests/test_no_leakage.py + verificar_no_fuga()"),
        ("Validación agrupada por punto o cuenca",
         estado.get("validacion_agrupada", False), f"GroupKFold por {cfg.COLUMNA_GRUPO}"),
        ("Al menos 3 modelos comparados con R², RMSE y MAE",
         estado.get("tres_modelos", False), "reports/tables/metricas_cv_regresion_pli.csv"),
        ("SHAP generado y los 5 factores principales interpretados",
         estado.get("shap", False), "reports/figures/16_shap_summary_*.png"),
        ("Mapa con zonas y focos críticos",
         estado.get("mapa", False), "reports/mapa_focos.html"),
        ("Tablero ejecutable localmente",
         estado.get("tablero", False), "streamlit run app/streamlit_app.py"),
        ("README con instrucciones para reproducir todo",
         estado.get("readme", False), "README.md"),
    ]
    tabla = pd.DataFrame(items, columns=["criterio", "cumplido", "evidencia"])
    tabla["cumplido"] = tabla["cumplido"].map({True: "✔", False: "✘"})
    guardar_csv(tabla, cfg.TABLES / "checklist_aceptacion.csv")
    return tabla


# --------------------------------------------------------------------------
# Resumen en Markdown
# --------------------------------------------------------------------------
def escribir_resumen(contexto: dict) -> str:
    r = contexto["resumen_eda"]
    modelos = contexto.get("modelos", {})
    lineas = [
        "# MVP – Modelo de ML para contaminación de sedimentos en Moquegua",
        "",
        "Resumen generado automáticamente por `run_pipeline.py`. "
        "Todas las cifras provienen de `reports/tables/`.",
        "",
        "## 1. Dataset",
        "",
        f"- **Muestras**: {r['n_muestras']} (una fila por muestra, formato ancho)",
        f"- **Puntos de muestreo**: {r['n_puntos']}",
        f"- **Periodo**: {r['anio_min']}–{r['anio_max']}",
        f"- **Parámetros con cobertura ≥ {int(100*cfg.COBERTURA_MIN_MODELO)} %**: {r['n_parametros_modelo']}",
        f"- **Parámetros con cobertura ≥ {int(100*cfg.COBERTURA_MIN_ANALISIS)} %**: {r['n_parametros_analisis']}",
        f"- **Distritos**: {', '.join(map(str, r['distritos']))}",
        "",
        "## 2. Riesgo",
        "",
        f"- **PLI mediano**: {r['mediana_PLI']:.2f}",
        f"- **Muestras con PLI > 1** (sedimento deteriorado): {r['pct_riesgo_alto_pli']:.1f} %",
        f"- **Muestras que exceden el PEL** de al menos un metal: {r['pct_excede_pel']:.1f} %",
        "",
    ]
    nivel = r.get("nivel_riesgo_conteo") or {}
    if nivel:
        lineas += [
            "**Nivel de riesgo normativo** (anclado en guías CCME ISQG/PEL, "
            "no en un umbral arbitrario de PLI):",
            "",
            "| Nivel | Criterio | Muestras | % |",
            "|---|---|---:|---:|",
        ]
        criterios = {
            "Bajo": "ningún metal > ISQG",
            "Medio": "≥1 metal > ISQG",
            "Alto": "≥1 metal > PEL",
            "Muy alto": "≥3 metales > PEL",
            "Sin dato": "sin metales del índice",
        }
        for k in ["Bajo", "Medio", "Alto", "Muy alto", "Sin dato"]:
            n = int(nivel.get(k, 0))
            lineas.append(
                f"| {k} | {criterios[k]} | {n} | {100*n/max(r['n_muestras'],1):.1f} % |"
            )
        lineas.append("")
    lineas += [
        "## 3. Modelos supervisados",
        "",
    ]
    for tarea, info in modelos.items():
        lineas += [
            f"### {tarea}",
            "",
            f"- Target: `{info['target']}` — {info['descripcion']}",
            f"- Mejor modelo: **{info['mejor_nombre']}**",
            f"- n = {info['n_muestras']}, {info['n_features']} variables, "
            f"{info['n_grupos']} grupos espaciales",
            "",
            info["tabla"].to_markdown(index=False),
            "",
        ]
    lineas += [
        "## 4. Patrones",
        "",
        contexto.get("texto_patrones", ""),
        "",
        "## 5. Zonas críticas (celdas de 5 × 5 km)",
        "",
        "Zonas agregadas por celda UTM de 5 km, ordenadas por severidad normativa. "
        "Responden al objetivo de localizar zonas críticas y no solo puntos aislados.",
        "",
        contexto.get("texto_zonas", "") or "_(sin datos de grilla)_",
        "",
        "## 6. Checklist de aceptación",
        "",
        contexto["checklist"].to_markdown(index=False),
        "",
        "## 7. Notas metodológicas",
        "",
        "- **Tamaño muestral**: el plan de tesis estima ~270 muestras y 29 metales; "
        f"el CSV vigente de OEFA entrega {r['n_muestras']} muestras útiles y "
        f"{r['n_parametros_analisis']} parámetros con cobertura suficiente. "
        "La diferencia se documenta en el README y no altera el objetivo del plan.",
        "- **Anti-fuga**: los 8 metales que componen el PLI/Igeo/RI "
        "(As, Cd, Cr, Cu, Hg, Ni, Pb, Zn) están prohibidos como entradas; "
        "el modelo usa litogénicos no normados, fisicoquímicos, granulometría, "
        "hidrocarburos, ubicación y tiempo.",
        "- **Validación**: GroupKFold repetido (5 semillas) por punto normalizado, "
        "más hold-out temporal (entrena ≤2023, prueba 2024-2026) e IC bootstrap 95 %.",
        "- **Fondo geoquímico**: P25 local como escenario principal, con "
        "sensibilidad a P10/P25/P50 y a la corteza de Turekian & Wedepohl.",
        "- **Guías**: CCME ISQG/PEL para 8 metales; Ni se complementa con "
        "NOAA ERL/ERM porque CCME no publica valor para ese metal.",
        "",
    ]
    texto = "\n".join(lineas)
    ruta = cfg.REPORTS / "RESUMEN_MVP.md"
    ruta.write_text(texto, encoding="utf-8")
    LOG.info("Resumen → %s", ruta)
    return texto
