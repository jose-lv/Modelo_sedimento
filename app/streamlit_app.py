"""
Tablero del MVP – Contaminación de sedimentos en Moquegua (OEFA).

Ejecutar desde la raíz del proyecto:

    streamlit run app/streamlit_app.py

El tablero es de solo lectura: consume los artefactos que produce
``python run_pipeline.py`` (data/processed, reports/tables, reports/figures).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"

COLORES_NIVEL = {
    "Bajo": "#2c7bb6",
    "Medio": "#e6c229",
    "Alto": "#fdae61",
    "Muy alto": "#d7191c",
    "Sin dato": "#bdbdbd",
}
ORDEN_NIVEL = ["Bajo", "Medio", "Alto", "Muy alto", "Sin dato"]

st.set_page_config(
    page_title="Sedimentos Moquegua – Riesgo y ML",
    page_icon="🌊",
    layout="wide",
)


# --------------------------------------------------------------------------
# Carga de artefactos
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_datos() -> pd.DataFrame:
    ruta = PROCESSED / "dataset_final.csv"
    if not ruta.exists():
        ruta = PROCESSED / "dataset_riesgo.csv"
    if not ruta.exists():
        return pd.DataFrame()
    df = pd.read_csv(ruta, encoding="utf-8-sig")
    if "ANIO" in df.columns:
        df["ANIO"] = pd.to_numeric(df["ANIO"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def cargar_csv(nombre: str) -> pd.DataFrame:
    ruta = TABLES / nombre
    if not ruta.exists():
        return pd.DataFrame()
    return pd.read_csv(ruta, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def cargar_json(nombre: str) -> dict:
    ruta = TABLES / nombre
    if not ruta.exists():
        return {}
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def figura(nombre: str, caption: str | None = None):
    ruta = FIGURES / nombre
    if ruta.exists():
        st.image(str(ruta), caption=caption, width="stretch")
    else:
        st.info(f"Falta la figura `{nombre}`. Ejecuta `python run_pipeline.py`.")


def aviso_sin_datos():
    st.error(
        "No se encontró `data/processed/dataset_final.csv`.\n\n"
        "Ejecuta primero:\n\n```\npython run_pipeline.py\n```"
    )
    st.stop()


# --------------------------------------------------------------------------
# Filtros laterales
# --------------------------------------------------------------------------
def barra_lateral(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("🌊 Sedimentos Moquegua")
    st.sidebar.caption("OEFA · ML para patrones de contaminación y riesgo")

    if df.empty:
        return df

    distritos = sorted(df["DISTRITO"].dropna().unique()) if "DISTRITO" in df else []
    sel_dist = st.sidebar.multiselect("Distrito", distritos, default=distritos)

    anios = pd.to_numeric(df.get("ANIO"), errors="coerce")
    if anios is not None and anios.notna().any():
        lo, hi = int(anios.min()), int(anios.max())
        rango = st.sidebar.slider("Periodo", lo, hi, (lo, hi))
    else:
        rango = None

    niveles = [n for n in ORDEN_NIVEL if "nivel_riesgo" in df and n in set(df["nivel_riesgo"])]
    sel_nivel = st.sidebar.multiselect("Nivel de riesgo", niveles, default=niveles)

    st.sidebar.divider()
    st.sidebar.caption(
        "Nivel de riesgo anclado en guías CCME ISQG/PEL "
        "(Bajo: ningún metal > ISQG; Medio: ≥1 > ISQG; "
        "Alto: ≥1 > PEL; Muy alto: ≥3 > PEL)."
    )

    d = df.copy()
    if sel_dist and "DISTRITO" in d:
        d = d[d["DISTRITO"].isin(sel_dist)]
    if rango is not None:
        d = d[anios.loc[d.index].between(*rango)]
    if sel_nivel and "nivel_riesgo" in d:
        d = d[d["nivel_riesgo"].isin(sel_nivel)]
    return d


# --------------------------------------------------------------------------
# 1. Resumen
# --------------------------------------------------------------------------
def pantalla_resumen(df: pd.DataFrame):
    st.title("Resumen ejecutivo")
    if df.empty:
        aviso_sin_datos()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Muestras", f"{len(df):,}".replace(",", " "))
    c2.metric("Puntos de muestreo", f"{df['PUNTO_NORMALIZADO'].nunique():,}".replace(",", " "))
    pli = pd.to_numeric(df.get("PLI"), errors="coerce")
    c3.metric("PLI mediano", f"{pli.median():.2f}" if pli.notna().any() else "s/d")
    pel = pd.to_numeric(df.get("n_metales_sobre_pel"), errors="coerce")
    pct_pel = 100 * (pel.fillna(0) > 0).mean() if pel.notna().any() else np.nan
    c4.metric("Excede ≥1 PEL", f"{pct_pel:.0f} %" if np.isfinite(pct_pel) else "s/d")
    if "nivel_riesgo" in df:
        pct_alto = 100 * df["nivel_riesgo"].isin(["Alto", "Muy alto"]).mean()
        c5.metric("Riesgo alto o muy alto", f"{pct_alto:.0f} %")
    else:
        c5.metric("Riesgo alto o muy alto", "s/d")

    st.divider()

    izq, der = st.columns([1.05, 1])
    with izq:
        st.subheader("Distribución del nivel de riesgo")
        if "nivel_riesgo" in df:
            conteo = (
                df["nivel_riesgo"].value_counts().reindex(ORDEN_NIVEL).dropna()
            )
            tabla = pd.DataFrame({
                "nivel": conteo.index,
                "muestras": conteo.astype(int).values,
                "porcentaje": (100 * conteo / len(df)).round(1).values,
            })
            st.bar_chart(
                tabla.set_index("nivel")["muestras"],
                color="#2c7bb6",
                height=280,
            )
            st.dataframe(tabla, hide_index=True, width="stretch")
        else:
            st.info("El dataset no contiene `nivel_riesgo`.")

        st.subheader("Metales por encima del PEL")
        cols_pel = [f"n_metales_sobre_pel", "n_metales_sobre_isqg"]
        if all(c in df.columns for c in cols_pel):
            dist = (
                pd.to_numeric(df["n_metales_sobre_pel"], errors="coerce")
                .value_counts().sort_index()
            )
            st.bar_chart(dist, color="#fdae61", height=240)
    with der:
        st.subheader("Indicadores clave")
        filas = [
            ("Periodo", f"{int(df['ANIO'].min())}–{int(df['ANIO'].max())}"
             if df.get("ANIO") is not None and df["ANIO"].notna().any() else "s/d"),
            ("PLI máximo", f"{pli.max():.2f}" if pli.notna().any() else "s/d"),
            ("RI mediano", f"{pd.to_numeric(df.get('RI'), errors='coerce').median():.0f}"
             if df.get("RI") is not None else "s/d"),
            ("Metal con mayor presión normativa",
             df["metal_dominante_normativo"].mode().iloc[0]
             if "metal_dominante_normativo" in df
             and not df["metal_dominante_normativo"].mode().empty else "s/d"),
            ("Anomalías (Isolation Forest)",
             f"{int(df['es_anomalia'].sum())}" if "es_anomalia" in df else "s/d"),
            ("Clusters K-Means",
             f"{int(df['cluster_kmeans'].nunique())}" if "cluster_kmeans" in df else "s/d"),
        ]
        st.dataframe(
            pd.DataFrame(filas, columns=["indicador", "valor"]),
            hide_index=True, width="stretch",
        )
        st.subheader("Resumen del pipeline")
        ruta = REPORTS / "RESUMEN_MVP.md"
        if ruta.exists():
            with st.expander("Ver RESUMEN_MVP.md"):
                st.markdown(ruta.read_text(encoding="utf-8"))
        else:
            st.caption("Aún no existe `reports/RESUMEN_MVP.md`.")


# --------------------------------------------------------------------------
# 2. Mapa
# --------------------------------------------------------------------------
def pantalla_mapa(df: pd.DataFrame):
    st.title("Mapa de focos críticos")
    if df.empty:
        aviso_sin_datos()

    st.caption(
        "Color = nivel de riesgo normativo (CCME ISQG/PEL). "
        "Pasa el cursor por un punto para ver metales sobre PEL, PLI y cluster."
    )

    try:
        import folium
        from src.report import utm_a_latlon
    except Exception as exc:  # pragma: no cover
        st.warning(f"No se pudo cargar folium/pyproj: {exc}")
        return

    d = df.dropna(subset=["COORD_ESTE", "COORD_NORTE"]).copy()
    if d.empty:
        st.info("No hay muestras con coordenadas para los filtros elegidos.")
        return

    zona = int(d["TXZONA"].mode().iloc[0]) if "TXZONA" in d else 19
    lat, lon = utm_a_latlon(d["COORD_ESTE"], d["COORD_NORTE"], zona=zona)
    d["lat"], d["lon"] = lat, lon

    mapa = folium.Map(
        location=[float(d["lat"].mean()), float(d["lon"].mean())],
        zoom_start=10, tiles="OpenStreetMap",
    )
    for _, r in d.iterrows():
        nivel = str(r.get("nivel_riesgo", "Sin dato"))
        color = COLORES_NIVEL.get(nivel, "#777777")
        pli_v = pd.to_numeric(pd.Series([r.get("PLI")]), errors="coerce").iloc[0]
        popup = (
            f"<b>{r.get('PUNTO_NORMALIZADO', '')}</b><br>"
            f"{r.get('DISTRITO', '')} · {str(r.get('fecha', ''))[:10]}<br>"
            f"Nivel: <b>{nivel}</b><br>"
            f"Metales &gt; PEL: {r.get('n_metales_sobre_pel', 's/d')} / 8<br>"
            f"PLI: {f'{pli_v:.2f}' if pd.notna(pli_v) else 's/d'}<br>"
            f"Metal dominante (norma): {r.get('metal_dominante_normativo', 's/d')}<br>"
            f"Cluster: {r.get('cluster_kmeans', 's/d')}"
        )
        folium.CircleMarker(
            location=[r["lat"], r["lon"]], radius=5,
            color=color, weight=1, fill=True, fill_color=color, fill_opacity=0.85,
            popup=folium.Popup(popup, max_width=280),
        ).add_to(mapa)

    leyenda = "".join(
        f"<div><span style='color:{COLORES_NIVEL[n]}'>●</span> {n}</div>"
        for n in ORDEN_NIVEL
    )
    html = (
        "<div style='position:fixed;bottom:24px;left:24px;z-index:9999;"
        "background:white;padding:8px 10px;border:1px solid #ccc;font-size:12px;'>"
        f"<b>Nivel de riesgo</b>{leyenda}</div>"
    )
    mapa.get_root().html.add_child(folium.Element(html))

    import streamlit.components.v1 as components
    components.html(mapa._repr_html_(), height=620, scrolling=True)

    estatico = REPORTS / "mapa_focos.html"
    if estatico.exists():
        st.download_button(
            "Descargar mapa interactivo completo (HTML)",
            data=estatico.read_bytes(),
            file_name="mapa_focos_moquegua.html",
            mime="text/html",
        )

    st.subheader("Mapas estáticos del pipeline")
    c1, c2 = st.columns(2)
    with c1:
        figura("07_mapa_nivel_riesgo.png", "Nivel de riesgo normativo")
    with c2:
        figura("08_mapa_anomalias.png", "Focos anómalos (Isolation Forest)")


# --------------------------------------------------------------------------
# 3. Ranking
# --------------------------------------------------------------------------
def pantalla_ranking(df: pd.DataFrame):
    st.title("Ranking de puntos críticos")
    if df.empty:
        aviso_sin_datos()

    ranking = cargar_csv("ranking_puntos_criticos.csv")
    if ranking.empty:
        st.info("Falta `reports/tables/ranking_puntos_criticos.csv`.")
    else:
        st.caption(
            "Puntos ordenados por severidad normativa (máximo nivel de riesgo) "
            "y luego por PLI mediano. La columna `variable_shap_dominante` indica "
            "qué variable empuja más la predicción en ese punto."
        )
        st.dataframe(ranking, hide_index=True, width="stretch")

        if {"PLI_mediano", "distrito"}.issubset(ranking.columns):
            top = ranking.head(15).set_index("PUNTO_NORMALIZADO")
            st.subheader("PLI mediano de los puntos más críticos")
            st.bar_chart(top["PLI_mediano"], color="#d7191c", height=320)

    st.divider()
    st.subheader("Zonas críticas (celdas de 5 × 5 km)")
    zonas = cargar_csv("zonas_criticas.csv")
    if zonas.empty:
        st.info("Falta `reports/tables/zonas_criticas.csv`.")
    else:
        st.caption(
            "Agregación espacial por celda UTM de 5 km: resume varios puntos en una "
            "zona operativa y evita sobreinterpretar muestras aisladas. Ordenadas por "
            "severidad normativa, % de muestras en Alto/Muy alto y número de puntos."
        )
        cols_z = [
            "GRID_5KM", "distrito", "n_muestras", "n_puntos", "nivel_riesgo",
            "pct_alto", "pct_alto_ic95_inf", "PLI_mediano",
            "max_metales_sobre_pel", "metal_dominante_normativo",
        ]
        vista = zonas[[c for c in cols_z if c in zonas.columns]].rename(columns={
            "GRID_5KM": "celda_5km",
            "pct_alto": "%_alto_muy_alto",
            "pct_alto_ic95_inf": "%_alto_IC95_inf",
            "max_metales_sobre_pel": "max_metales_PEL",
            "metal_dominante_normativo": "metal_dominante",
        })
        st.dataframe(vista, hide_index=True, width="stretch")
        st.download_button(
            "Descargar zonas críticas (CSV)",
            data=zonas.to_csv(index=False).encode("utf-8-sig"),
            file_name="zonas_criticas_moquegua.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("Explorar una muestra")
    if "PUNTO_NORMALIZADO" in df:
        punto = st.selectbox(
            "Punto de muestreo", sorted(df["PUNTO_NORMALIZADO"].dropna().unique())
        )
        sub = df[df["PUNTO_NORMALIZADO"] == punto]
        cols = [
            c for c in [
                "id_muestra", "fecha", "DISTRITO", "PLI", "RI", "nivel_riesgo",
                "n_metales_sobre_isqg", "n_metales_sobre_pel", "metal_dominante",
                "cluster_kmeans", "PLI_predicho",
            ] if c in sub.columns
        ]
        st.dataframe(sub[cols], hide_index=True, width="stretch")

        metales = [c for c in df.columns if c in {
            "Arsénico", "Cadmio", "Cromo", "Cobre", "Mercurio",
            "Níquel", "Plomo", "Zinc",
        }]
        if metales:
            st.markdown("**Concentraciones de los metales del índice (mg/kg)**")
            etiqueta_muestra = (
                sub["id_muestra"].astype(str) + " · " + sub["fecha"].astype(str)
            )
            tabla_m = sub[metales].copy()
            tabla_m.index = etiqueta_muestra.values
            st.dataframe(tabla_m.T, width="stretch")


# --------------------------------------------------------------------------
# 4. Modelo
# --------------------------------------------------------------------------
def pantalla_modelo():
    st.title("Modelos y explicabilidad")

    tareas = {
        "Regresión del PLI": ("regresion_pli", "regresion"),
        "Clasificación de excedencia PEL": ("clasificacion_riesgo", "clasificacion"),
    }
    etiqueta = st.radio(
        "Tarea", list(tareas), horizontal=True, label_visibility="collapsed"
    )
    tarea, tipo = tareas[etiqueta]

    resumen = cargar_json(f"resumen_modelo_{tarea}.json")
    if not resumen:
        st.info(
            f"Falta `reports/tables/resumen_modelo_{tarea}.json`. "
            "Ejecuta el pipeline sin `--solo-datos`."
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mejor modelo", resumen.get("mejor_modelo", "s/d"))
    c2.metric("Muestras", resumen.get("n_muestras", "s/d"))
    c3.metric("Variables", resumen.get("n_features", "s/d"))
    c4.metric("Grupos espaciales", resumen.get("n_grupos", "s/d"))

    st.caption(resumen.get("descripcion", ""))

    st.subheader("Validación cruzada agrupada repetida")
    met = cargar_csv(f"metricas_cv_{tarea}.csv")
    if not met.empty:
        st.dataframe(met.round(3), hide_index=True, width="stretch")

    holdout = cargar_csv(f"holdout_{tarea}.csv")
    if not holdout.empty:
        st.subheader("Hold-out temporal (entrena ≤2023, prueba 2024–2026)")
        st.dataframe(holdout.round(3), hide_index=True, width="stretch")

    ic = resumen.get("bootstrap_holdout") or {}
    if ic:
        st.info(
            f"IC 95 % bootstrap para `{ic.get('metrica')}` en el hold-out: "
            f"{ic.get('ic95_inferior', float('nan')):.3f} – "
            f"{ic.get('ic95_superior', float('nan')):.3f}"
        )
    oof = resumen.get("diagnostico_oof") or {}
    if oof:
        etiqueta_met = "R²" if "r2_oof" in oof else "ROC-AUC"
        valor = oof.get("r2_oof", oof.get("roc_auc_oof"))
        if valor is not None:
            st.caption(f"{etiqueta_met} fuera de fold (5 folds agrupados): {valor:.3f}")

    st.subheader("Comparación de modelos")
    figura(f"12_comparacion_modelos_{tipo}.png")

    st.divider()
    st.subheader("Importancia de variables (SHAP)")
    imp = cargar_csv(f"shap_importancia_{tarea}.csv")
    if not imp.empty:
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.dataframe(imp.head(15).round(4), hide_index=True, width="stretch")
        with c2:
            figura(f"17_shap_bar_{tarea}.png", "Importancia media |SHAP|")
        figura(f"16_shap_summary_{tarea}.png", "Distribución de valores SHAP")

        pi = cargar_csv(f"importancia_permutacion_{tarea}.csv")
        if not pi.empty:
            st.subheader("Control: importancia por permutación")
            st.dataframe(pi.head(15).round(4), hide_index=True, width="stretch")
            concordancia = cargar_json(f"resumen_shap_{tarea}.json").get(
                "concordancia_top10_shap_vs_permutacion"
            )
            if concordancia is not None:
                st.caption(
                    f"Coincidencia top-10 SHAP vs. permutación: "
                    f"{concordancia}/10 variables."
                )

    st.divider()
    st.subheader("Diagnóstico del modelo")
    c1, c2 = st.columns(2)
    with c1:
        figura("13_predicho_vs_real.png", "Predicho vs. real (PLI)")
    with c2:
        if tipo == "clasificacion":
            figura("14_curva_roc.png", "Curva ROC en validación")
        else:
            figura("06_serie_temporal.png", "Evolución temporal del PLI")


# --------------------------------------------------------------------------
# 5. Datos
# --------------------------------------------------------------------------
def pantalla_datos(df: pd.DataFrame):
    st.title("Explorador de datos")
    if df.empty:
        aviso_sin_datos()

    st.caption(
        f"{len(df)} muestras × {df.shape[1]} columnas. "
        "Usa los filtros de la barra lateral para acotar."
    )

    grupos = {
        "Identificación y riesgo": [
            "id_muestra", "fecha", "PUNTO_NORMALIZADO", "DISTRITO", "ANIO",
            "PLI", "PLI_log", "RI", "nivel_riesgo", "n_metales_sobre_isqg",
            "n_metales_sobre_pel", "metal_dominante", "metal_dominante_normativo",
            "excede_pel",
        ],
        "Metales del índice": [
            "Arsénico", "Cadmio", "Cromo", "Cobre", "Mercurio",
            "Níquel", "Plomo", "Zinc",
        ],
        "Fisicoquímica": [
            "pH pasta", "Materia orgánica", "Carbono Orgánico Total (COT)",
            "Sulfato", "Sulfuro", "Conductividad",
        ],
        "Patrones y predicción": [
            "PC1", "PC2", "cluster_kmeans", "cluster_gmm",
            "anomalia_score", "es_anomalia", "PLI_predicho",
            "prob_riesgo_alto", "clase_predicha",
        ],
    }
    opciones = {
        k: [c for c in v if c in df.columns] for k, v in grupos.items()
    }
    opciones = {k: v for k, v in opciones.items() if v}
    opciones["Todas las columnas"] = list(df.columns)

    vista = st.selectbox("Vista de columnas", list(opciones))
    cols = opciones[vista]
    st.dataframe(df[cols], hide_index=True, width="stretch", height=460)

    st.download_button(
        "Descargar dataset filtrado (CSV)",
        data=df[cols].to_csv(index=False).encode("utf-8-sig"),
        file_name="sedimentos_moquegua_filtrado.csv",
        mime="text/csv",
    )

    with st.expander("Calidad de datos: cobertura y censura"):
        cob = cargar_csv("cobertura_parametros.csv")
        if not cob.empty:
            st.dataframe(cob, hide_index=True, width="stretch")
        cens = cargar_csv("censura_por_parametro.csv")
        if not cens.empty:
            st.markdown("**Censura por parámetro**")
            st.dataframe(cens, hide_index=True, width="stretch")

    with st.expander("Trazabilidad de la limpieza"):
        log = cargar_csv("log_limpieza.csv")
        if not log.empty:
            st.dataframe(log, hide_index=True, width="stretch")


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------
def main():
    df = cargar_datos()
    filtrado = barra_lateral(df)

    tabs = st.tabs(["📊 Resumen", "🗺️ Mapa", "🏆 Ranking", "🤖 Modelo", "🗃️ Datos"])
    with tabs[0]:
        pantalla_resumen(filtrado)
    with tabs[1]:
        pantalla_mapa(filtrado)
    with tabs[2]:
        pantalla_ranking(filtrado)
    with tabs[3]:
        pantalla_modelo()
    with tabs[4]:
        pantalla_datos(filtrado)


if __name__ == "__main__":
    main()
