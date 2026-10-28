"""
Mapa de focos críticos, ranking de puntos, exportación de resultados
(RF-16 y RF-18) y verificación del checklist de aceptación del MVP.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .utils import LOG, guardar_csv, guardar_json


# --------------------------------------------------------------------------
# Paletas y escalas del mapa
# --------------------------------------------------------------------------
#: Colores del nivel de riesgo normativo (coherentes con plots.mapa_nivel).
COLORES_NIVEL = {
    "Bajo": "#2c7bb6",
    "Medio": "#ffffbf",
    "Alto": "#fdae61",
    "Muy alto": "#d7191c",
    "Sin dato": "#bdbdbd",
}

#: Paleta de clusters deliberadamente distinta de la escala de riesgo: un
#: cluster describe una firma química, no un nivel de peligro.
PALETA_CLUSTER = [
    "#1b9e77", "#7570b3", "#e7298a", "#66a61e",
    "#a6761d", "#666666", "#8c510a", "#01665e",
]

#: Escala secuencial para el PLI (observado y predicho). Los cortes coinciden
#: con los usados en el informe para que ambas vistas sean comparables.
BINS_PLI = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, np.inf]
COLORES_PLI = ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#e31a1c"]
ETIQUETAS_PLI = ["< 1", "1 – 2", "2 – 3", "3 – 5", "5 – 10", "≥ 10"]

#: Probabilidad del clasificador de exceder al menos un PEL.
BINS_PROB = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
COLORES_PROB = ["#f2f0f7", "#cbc9e2", "#9e9ac8", "#6a51a3", "#3f007d"]
ETIQUETAS_PROB = ["0 – 0,2", "0,2 – 0,4", "0,4 – 0,6", "0,6 – 0,8", "0,8 – 1,0"]

COLORES_CLASE = {0.0: "#4d9221", 1.0: "#b2182b"}
ETIQUETAS_CLASE = {0.0: "No excede ≥1 PEL", 1.0: "Excede ≥1 PEL"}

#: Nombres en español del perfil de clusters → símbolo químico.
SIMBOLOS_METALES = {
    "Aluminio": "Al", "Antimonio": "Sb", "Arsénico": "As", "Azufre Total": "S",
    "Bario": "Ba", "Berilio": "Be", "Bismuto": "Bi", "Boro": "B",
    "Cadmio": "Cd", "Calcio": "Ca", "Cerio": "Ce", "Cobalto": "Co",
    "Cobre": "Cu", "Cromo": "Cr", "Cromo Hexavalente": "Cr(VI)",
    "Estaño": "Sn", "Estroncio": "Sr", "Fósforo": "P", "Hierro": "Fe",
    "Litio": "Li", "Magnesio": "Mg", "Manganeso": "Mn", "Mercurio": "Hg",
    "Molibdeno": "Mo", "Níquel": "Ni", "Plata": "Ag", "Plomo": "Pb",
    "Potasio": "K", "Selenio": "Se", "Silicio": "Si", "Sodio": "Na",
    "Talio": "Tl", "Titanio": "Ti", "Torio": "Th", "Uranio": "U",
    "Vanadio": "V", "Wolframio": "W", "Zinc": "Zn",
}

#: Modos de color del mapa. El identificador es también el nombre de la capa.
MODOS_COLOR = {
    "nivel_riesgo": "Riesgo observado (CCME)",
    "PLI": "PLI observado",
    "cluster_kmeans": "Cluster K-Means",
    "cluster_gmm": "Cluster GMM",
    "PLI_predicho_mapa": "PLI predicho (OOF)",
    "prob_riesgo_alto_mapa": "Prob. exceder PEL (predicha)",
    "clase_predicha_mapa": "Clase predicha (excede PEL)",
}


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
    )
    if top is not None:
        g = g.head(top)
    return g.reset_index(drop=True)


# --------------------------------------------------------------------------
# Utilidades del mapa
# --------------------------------------------------------------------------
def _leer_csv(ruta) -> pd.DataFrame | None:
    ruta = Path(ruta)
    if not ruta.exists():
        return None
    try:
        return pd.read_csv(ruta, encoding="utf-8-sig")
    except (OSError, pd.errors.ParserError) as exc:
        LOG.warning("No se pudo leer %s: %s", ruta, exc)
        return None


def _es_verdadero(valor) -> bool:
    if pd.isna(valor):
        return False
    if isinstance(valor, str):
        return valor.strip().lower() in {"true", "1", "sí", "si", "yes"}
    return bool(valor)


def _fmt(valor, dec: int = 2, sufijo: str = "") -> str:
    try:
        if pd.isna(valor):
            return "s/d"
        return f"{float(valor):.{dec}f}{sufijo}"
    except (TypeError, ValueError):
        return "s/d"


def _simbolos_elementos(texto) -> str:
    """Convierte 'Antimonio, Plomo, Arsénico' en 'Sb–Pb–As'."""
    if not isinstance(texto, str) or not texto.strip():
        return ""
    partes = [p.strip() for p in texto.split(",") if p.strip()]
    return "–".join(SIMBOLOS_METALES.get(p, p) for p in partes)


def _etiquetas_clusters(perfil: pd.DataFrame | None) -> dict[int, str]:
    if perfil is None or perfil.empty or "cluster" not in perfil.columns:
        return {}
    etiquetas = {}
    for _, r in perfil.iterrows():
        try:
            c = int(r["cluster"])
        except (TypeError, ValueError):
            continue
        simbolos = _simbolos_elementos(r.get("elementos_dominantes"))
        etiquetas[c] = f"C{c} · {simbolos}" if simbolos else f"C{c}"
    return etiquetas


def _cargar_perfil_clusters() -> pd.DataFrame | None:
    return _leer_csv(cfg.TABLES / "perfil_clusters.csv")


def _unir_por_id(base: pd.DataFrame, extra: pd.DataFrame | None,
                 columnas: list[str]) -> pd.DataFrame:
    """Une por ``id_muestra`` y, si no existe, por posición (compatibilidad)."""
    if extra is None or getattr(extra, "empty", True):
        return base
    cols = [c for c in columnas if c in extra.columns and c not in base.columns]
    if not cols:
        return base
    extra = extra.copy()
    if "id_muestra" in base.columns and "id_muestra" in extra.columns:
        extra = extra.drop_duplicates("id_muestra").set_index("id_muestra")
        return base.join(extra[cols], on="id_muestra")
    extra = extra.reset_index(drop=True)
    if len(extra) != len(base):
        LOG.warning("Sin id_muestra y longitud distinta: no se unen %s", cols)
        return base
    extra.index = base.index
    return base.join(extra[cols], how="left")


def _predicciones_finales(predicciones=None) -> pd.DataFrame:
    """Reúne las predicciones del modelo final (in-sample) por ``id_muestra``."""
    if isinstance(predicciones, dict):
        marcos = [v for v in predicciones.values() if isinstance(v, pd.DataFrame)]
    elif isinstance(predicciones, pd.DataFrame):
        marcos = [predicciones]
    else:
        marcos = []
        for nombre in ("regresion_pli", "clasificacion_riesgo"):
            f = _leer_csv(cfg.DATA_PROCESSED / f"predicciones_{nombre}.csv")
            if f is not None:
                marcos.append(f)
    if not marcos:
        return pd.DataFrame()

    out = None
    for f in marcos:
        if "id_muestra" not in f.columns:
            continue
        cols = ["id_muestra"] + [
            c for c in ("PLI_predicho", "prob_riesgo_alto", "clase_predicha")
            if c in f.columns
        ]
        f = f[cols].drop_duplicates("id_muestra")
        out = f if out is None else out.merge(f, on="id_muestra", how="outer")
    return out if out is not None else pd.DataFrame()


def _predicciones_mapa(predicciones=None) -> pd.DataFrame:
    """
    Combina predicciones OOF (honestas) con las del modelo final.

    Para las muestras con OOF se usa la predicción fuera de fold; si el
    archivo OOF no existe o no se puede emparejar, se usa la del modelo final
    y se marca explícitamente en ``fuente_pli`` / ``fuente_prob``.
    """
    final = _predicciones_finales(predicciones)
    out = final.copy()
    oof_reg = _leer_csv(cfg.TABLES / "predicciones_oof_regresion_pli.csv")
    oof_cla = _leer_csv(cfg.TABLES / "predicciones_oof_clasificacion_riesgo.csv")

    if out.empty and oof_reg is not None and "id_muestra" in oof_reg.columns:
        out = oof_reg[["id_muestra"]].drop_duplicates().copy()
    if out.empty:
        return out

    if oof_reg is not None and not oof_reg.empty:
        reg = oof_reg.copy()
        if "id_muestra" not in reg.columns and len(reg) == len(final):
            reg["id_muestra"] = final["id_muestra"].to_numpy()
        if "id_muestra" in reg.columns and "y_predicho_oof" in reg.columns:
            reg["PLI_predicho_oof"] = np.expm1(
                pd.to_numeric(reg["y_predicho_oof"], errors="coerce")
            ).clip(lower=0)
            out = out.merge(
                reg[["id_muestra", "PLI_predicho_oof"]].drop_duplicates("id_muestra"),
                on="id_muestra", how="left",
            )

    if oof_cla is not None and not oof_cla.empty:
        cla = oof_cla.copy()
        if "id_muestra" not in cla.columns and len(cla) == len(final):
            cla["id_muestra"] = final["id_muestra"].to_numpy()
        if "id_muestra" in cla.columns:
            if "proba_oof" in cla.columns:
                proba = pd.to_numeric(cla["proba_oof"], errors="coerce")
                cla["prob_riesgo_alto_oof"] = proba
                cla["clase_predicha_oof"] = (
                    proba.ge(0.5).astype(float).where(proba.notna())
                )
            elif "y_predicho_oof" in cla.columns:
                cla["clase_predicha_oof"] = pd.to_numeric(
                    cla["y_predicho_oof"], errors="coerce"
                )
            cols = ["id_muestra"] + [
                c for c in ("prob_riesgo_alto_oof", "clase_predicha_oof")
                if c in cla.columns
            ]
            out = out.merge(
                cla[cols].drop_duplicates("id_muestra"), on="id_muestra", how="left"
            )

    for c in ("PLI_predicho", "prob_riesgo_alto", "clase_predicha",
              "PLI_predicho_oof", "prob_riesgo_alto_oof", "clase_predicha_oof"):
        if c not in out.columns:
            out[c] = np.nan

    out["PLI_predicho_mapa"] = out["PLI_predicho_oof"]
    out["fuente_pli"] = np.select(
        [out["PLI_predicho_mapa"].notna(), out["PLI_predicho"].notna()],
        ["OOF (fuera de fold)", "modelo final (in-sample)"],
        default="sin predicción",
    )
    out["PLI_predicho_mapa"] = out["PLI_predicho_mapa"].fillna(out["PLI_predicho"])

    out["prob_riesgo_alto_mapa"] = out["prob_riesgo_alto_oof"]
    out["fuente_prob"] = np.select(
        [out["prob_riesgo_alto_mapa"].notna(), out["prob_riesgo_alto"].notna()],
        ["OOF (fuera de fold)", "modelo final (in-sample)"],
        default="sin predicción",
    )
    out["prob_riesgo_alto_mapa"] = out["prob_riesgo_alto_mapa"].fillna(
        out["prob_riesgo_alto"]
    )
    out["clase_predicha_mapa"] = out["clase_predicha_oof"].fillna(
        out["clase_predicha"]
    )
    return out


def geometria_celda_5km(codigo, zona: int = 19):
    """
    Reconstruye el polígono de una celda ``GRID_5KM`` (p. ej. ``74_1633``).

    El código identifica la esquina suroeste en unidades de 5 km; se usan las
    cuatro esquinas UTM y se reproyectan, no un rectángulo lat/lon.
    """
    if codigo is None:
        return None
    try:
        if pd.isna(codigo):
            return None
    except (TypeError, ValueError):
        pass
    texto = str(codigo).strip()
    if "_" not in texto:
        return None
    try:
        este5, norte5 = texto.split("_", 1)
        e0 = int(float(este5)) * 5000
        n0 = int(float(norte5)) * 5000
    except (TypeError, ValueError):
        return None
    esquinas = [(e0, n0), (e0 + 5000, n0), (e0 + 5000, n0 + 5000), (e0, n0 + 5000)]
    lats, lons = utm_a_latlon(
        [c[0] for c in esquinas], [c[1] for c in esquinas], zona=zona
    )
    return [[float(la), float(lo)] for la, lo in zip(lats, lons)]


def _color_bins(valor, bins, colores) -> str:
    try:
        if pd.isna(valor):
            return "#bdbdbd"
        x = float(valor)
    except (TypeError, ValueError):
        return "#bdbdbd"
    idx = int(np.digitize([x], bins[1:-1], right=False)[0])
    return colores[min(max(idx, 0), len(colores) - 1)]


def _color_modo(r: pd.Series, modo: str) -> str:
    if modo == "nivel_riesgo":
        return COLORES_NIVEL.get(str(r.get("nivel_riesgo", "Sin dato")), "#777777")
    if modo == "PLI":
        return _color_bins(r.get("PLI"), BINS_PLI, COLORES_PLI)
    if modo == "PLI_predicho_mapa":
        return _color_bins(r.get("PLI_predicho_mapa"), BINS_PLI, COLORES_PLI)
    if modo == "prob_riesgo_alto_mapa":
        return _color_bins(r.get("prob_riesgo_alto_mapa"), BINS_PROB, COLORES_PROB)
    if modo == "clase_predicha_mapa":
        valor = r.get("clase_predicha_mapa")
        if pd.isna(valor):
            return "#bdbdbd"
        return COLORES_CLASE.get(float(valor), "#777777")
    if modo in {"cluster_kmeans", "cluster_gmm"}:
        valor = r.get(modo)
        if pd.isna(valor):
            return "#bdbdbd"
        return PALETA_CLUSTER[int(valor) % len(PALETA_CLUSTER)]
    return "#777777"


def _texto_cluster(valor, etiquetas: dict[int, str]) -> str:
    if pd.isna(valor):
        return "s/d"
    try:
        c = int(valor)
    except (TypeError, ValueError):
        return "s/d"
    return etiquetas.get(c, f"C{c}")


def _popup_muestra(r: pd.Series, etiquetas_km: dict[int, str]) -> str:
    nivel = html.escape(str(r.get("nivel_riesgo", "Sin dato")))
    punto = html.escape(str(r.get("PUNTO_NORMALIZADO", "s/d")))
    distrito = html.escape(str(r.get("DISTRITO", "s/d")))
    fecha = html.escape(str(r.get("fecha", ""))[:10])
    id_m = html.escape(str(r.get("id_muestra", "")))
    metal_norma = html.escape(str(r.get("metal_dominante_normativo", "s/d")))
    metal_cf = html.escape(str(r.get("metal_dominante", "s/d")))
    cluster_km = html.escape(_texto_cluster(r.get("cluster_kmeans"), etiquetas_km))
    cluster_gmm = html.escape(_texto_cluster(r.get("cluster_gmm"), {}))
    anomalia = "Sí" if _es_verdadero(r.get("es_anomalia")) else "No"

    pli_p = r.get("PLI_predicho_mapa")
    if pd.notna(pli_p):
        fuente = html.escape(str(r.get("fuente_pli", "")))
        pli_p_txt = (
            f"{float(pli_p):.2f} "
            f"<span style='color:#666;font-size:10px;'>({fuente})</span>"
        )
    else:
        pli_p_txt = "s/d"

    prob = r.get("prob_riesgo_alto_mapa")
    if pd.notna(prob):
        fuente = html.escape(str(r.get("fuente_prob", "")))
        prob_txt = (
            f"{float(prob):.3f} "
            f"<span style='color:#666;font-size:10px;'>({fuente})</span>"
        )
    else:
        prob_txt = "s/d"

    clase = r.get("clase_predicha_mapa")
    if pd.isna(clase):
        clase_txt = "s/d"
    else:
        clase_txt = ETIQUETAS_CLASE.get(float(clase), "s/d")

    return (
        "<div style='font-size:12px;line-height:1.45;'>"
        f"<b>{punto}</b><br>"
        f"{distrito} · {fecha}<br>"
        f"Nivel de riesgo: <b>{nivel}</b><br>"
        f"Metales &gt; PEL: {_fmt(r.get('n_metales_sobre_pel'), 0)} / 8 · "
        f"&gt; ISQG: {_fmt(r.get('n_metales_sobre_isqg'), 0)} / 8<br>"
        f"PLI: <b>{_fmt(r.get('PLI'))}</b> · RI: {_fmt(r.get('RI'), 0)}<br>"
        f"Metal dominante (norma): {metal_norma}<br>"
        f"Metal más enriquecido (CF): {metal_cf}<br>"
        f"Cluster K-Means: {cluster_km}<br>"
        f"Cluster GMM: {cluster_gmm}<br>"
        f"Anomalía (Isolation Forest): <b>{anomalia}</b>"
        "<hr style='margin:5px 0;border:none;border-top:1px solid #ddd;'>"
        f"PLI predicho: <b>{pli_p_txt}</b><br>"
        f"Prob. exceder PEL: <b>{prob_txt}</b><br>"
        f"Clase predicha: <b>{clase_txt}</b><br>"
        "<span style='color:#666;font-size:10px;'>"
        "OOF = predicción fuera de fold. La clase predicha no es un nivel normativo."
        "</span>"
        f"<br><span style='color:#888;font-size:9px;'>{id_m}</span>"
        "</div>"
    )


def _popup_anomalia(r: pd.Series) -> str:
    punto = html.escape(str(r.get("PUNTO_NORMALIZADO", "s/d")))
    distrito = html.escape(str(r.get("DISTRITO", "s/d")))
    fecha = html.escape(str(r.get("fecha", ""))[:10])
    return (
        f"<b>{punto}</b><br>"
        f"{distrito} · {fecha}<br>"
        "Anomalía Isolation Forest (percentil 95)<br>"
        f"Score: {_fmt(r.get('anomalia_score'), 3)}"
    )


def _popup_celda(r: pd.Series) -> str:
    celda = html.escape(str(r.get("GRID_5KM", "s/d")))
    distrito = html.escape(str(r.get("distrito", "s/d")))
    nivel = html.escape(str(r.get("nivel_riesgo", "s/d")))
    metal = html.escape(str(r.get("metal_dominante_normativo", "s/d")))
    return (
        f"<b>Celda {celda}</b><br>"
        f"Distrito: {distrito}<br>"
        f"Muestras: {_fmt(r.get('n_muestras'), 0)} · "
        f"Puntos: {_fmt(r.get('n_puntos'), 0)}<br>"
        f"Nivel de riesgo máximo: <b>{nivel}</b><br>"
        f"% Alto/Muy alto: {_fmt(r.get('pct_alto'), 1, ' %')} "
        f"(IC95 inf. {_fmt(r.get('pct_alto_ic95_inf'), 1, ' %')})<br>"
        f"PLI mediano: {_fmt(r.get('PLI_mediano'))}<br>"
        f"Máx. metales &gt; PEL: {_fmt(r.get('max_metales_sobre_pel'), 0)} / 8<br>"
        f"Metal dominante (norma): {metal}"
    )


def _capa_celdas(zonas: pd.DataFrame, nombre: str, zona: int):
    import folium

    fg = folium.FeatureGroup(name=nombre)
    for _, r in zonas.iterrows():
        geom = geometria_celda_5km(r.get("GRID_5KM"), zona=zona)
        if geom is None:
            continue
        nivel = str(r.get("nivel_riesgo", "Sin dato"))
        color = COLORES_NIVEL.get(nivel, "#777777")
        folium.Polygon(
            locations=geom, color=color, weight=1.5, fill=True,
            fill_color=color, fill_opacity=0.18,
            popup=folium.Popup(_popup_celda(r), max_width=300),
            tooltip=f"Celda {r.get('GRID_5KM')} · {nivel}",
        ).add_to(fg)
    return fg


def _leyenda_html(titulo: str, filas: list[tuple[str, str]], nota: str = "") -> str:
    cuerpo = "".join(
        "<div style='margin:1px 0;'>"
        f"<span style='color:{color};font-size:15px;text-shadow:0 0 1px #333;'>"
        f"●</span> {texto}</div>"
        for color, texto in filas
    )
    nota_html = (
        f"<div style='margin-top:4px;color:#555;font-size:10.5px;'>{nota}</div>"
        if nota else ""
    )
    return f"<b>{titulo}</b>{cuerpo}{nota_html}"


def _conteo_bins(serie, bins) -> list[int]:
    s = pd.to_numeric(serie, errors="coerce").dropna()
    if s.empty:
        return [0] * (len(bins) - 1)
    idx = np.digitize(s.to_numpy(dtype=float), bins[1:-1], right=False)
    return np.bincount(idx, minlength=len(bins) - 1)[:len(bins) - 1].tolist()


def _leyenda_riesgo(d: pd.DataFrame) -> str:
    conteo = d["nivel_riesgo"].value_counts()
    criterios = {
        "Bajo": "ningún metal &gt; ISQG",
        "Medio": "≥1 metal &gt; ISQG",
        "Alto": "≥1 metal &gt; PEL",
        "Muy alto": "≥3 metales &gt; PEL",
        "Sin dato": "sin metales del índice",
    }
    filas = [
        (COLORES_NIVEL[n], f"<b>{n}</b> ({int(conteo.get(n, 0))}) – {criterios[n]}")
        for n in ["Bajo", "Medio", "Alto", "Muy alto", "Sin dato"]
    ]
    return _leyenda_html(
        "Nivel de riesgo (CCME ISQG/PEL)", filas,
        "Anclado en guías CCME; no es un umbral arbitrario de PLI.",
    )


def _leyenda_pli(titulo: str, serie, nota: str) -> str:
    conteo = _conteo_bins(serie, BINS_PLI)
    filas = [
        (color, f"{etiqueta} ({n})")
        for color, etiqueta, n in zip(COLORES_PLI, ETIQUETAS_PLI, conteo)
    ]
    return _leyenda_html(titulo, filas, nota)


def _leyenda_bins(titulo: str, serie, bins, colores, etiquetas, nota: str) -> str:
    conteo = _conteo_bins(serie, bins)
    filas = [
        (color, f"{etiqueta} ({n})")
        for color, etiqueta, n in zip(colores, etiquetas, conteo)
    ]
    return _leyenda_html(titulo, filas, nota)


def _leyenda_clusters(titulo: str, serie, etiquetas: dict[int, str]) -> str:
    conteo = pd.to_numeric(serie, errors="coerce").dropna().astype(int).value_counts()
    filas = [
        (PALETA_CLUSTER[int(c) % len(PALETA_CLUSTER)],
         f"{html.escape(etiquetas.get(int(c), f'C{int(c)}'))} ({int(n)})")
        for c, n in conteo.sort_index().items()
    ]
    if not filas:
        filas = [("#bdbdbd", "Sin datos")]
    return _leyenda_html(
        titulo, filas,
        "Los clusters describen firmas químicas; no equivalen a niveles de riesgo.",
    )


def _leyenda_clase(serie) -> str:
    s = pd.to_numeric(serie, errors="coerce")
    filas = [
        (COLORES_CLASE[0.0], f"{ETIQUETAS_CLASE[0.0]} ({int((s == 0).sum())})"),
        (COLORES_CLASE[1.0], f"{ETIQUETAS_CLASE[1.0]} ({int((s == 1).sum())})"),
        ("#bdbdbd", f"Sin predicción ({int(s.isna().sum())})"),
    ]
    return _leyenda_html(
        "Clase predicha", filas,
        "Umbral 0,5 sobre la probabilidad OOF. No es un nivel normativo.",
    )


def _envolver_leyenda(modo: str, contenido: str, visible: bool = False) -> str:
    estilo = "display:block;" if visible else "display:none;"
    return (
        f"<div id='leyenda-{modo}' class='leyenda-mapa' "
        "style='position:fixed;bottom:64px;left:24px;z-index:9999;"
        "background:rgba(255,255,255,.96);padding:9px 11px;border:1px solid #bbb;"
        "border-radius:5px;font-size:11.5px;line-height:1.45;max-width:340px;"
        f"box-shadow:0 1px 5px rgba(0,0,0,.25);{estilo}'>{contenido}</div>"
    )


def _panel_controles() -> str:
    def opcion(valor: str) -> str:
        return f"<option value='{valor}'>{html.escape(MODOS_COLOR[valor])}</option>"

    opciones = (
        "<optgroup label='Observado'>"
        + opcion("nivel_riesgo") + opcion("PLI")
        + "</optgroup><optgroup label='Patrones (no son riesgo)'>"
        + opcion("cluster_kmeans") + opcion("cluster_gmm")
        + "</optgroup><optgroup label='Predicho'>"
        + opcion("PLI_predicho_mapa") + opcion("prob_riesgo_alto_mapa")
        + opcion("clase_predicha_mapa")
        + "</optgroup>"
    )
    return (
        "<div id='panel-mapa' style='position:fixed;top:10px;right:12px;z-index:9999;"
        "background:rgba(255,255,255,.97);padding:10px 12px;border:1px solid #bbb;"
        "border-radius:5px;font-size:12px;width:250px;box-shadow:0 1px 5px rgba(0,0,0,.25);'>"
        "<div style='font-weight:700;font-size:13px;margin-bottom:6px;'>"
        "Focos críticos · Moquegua</div>"
        "<label for='modo-color' style='display:block;font-weight:600;'>"
        "Color de los puntos</label>"
        "<select id='modo-color' style='width:100%;margin:2px 0 7px 0;padding:3px;'>"
        f"{opciones}</select>"
        "<label for='celdas' style='display:block;font-weight:600;'>"
        "Celdas de 5 × 5 km</label>"
        "<select id='celdas' style='width:100%;margin:2px 0 7px 0;padding:3px;'>"
        "<option value='ninguna'>No mostrar</option>"
        "<option value='top'>Top 15 críticas</option>"
        "<option value='todas'>Todas</option></select>"
        "<label style='display:block;margin:5px 0;'>"
        "<input type='checkbox' id='ver-anomalias' checked> "
        "Anomalías (Isolation Forest)</label>"
        "<div style='font-size:10.5px;color:#555;border-top:1px solid #ddd;"
        "padding-top:5px;margin-top:5px;'>"
        "Los clusters describen patrones químicos; no son niveles de riesgo.</div>"
        "</div>"
    )


def _script_controles(mapa, capas: dict, capas_celdas: dict, fg_anom) -> str:
    def objeto_capas(dic: dict) -> str:
        return "{" + ",".join(
            f"{json.dumps(k)}: {v.get_name()}" for k, v in dic.items()
        ) + "}"

    capas_js = objeto_capas(capas)
    celdas_js = objeto_capas(capas_celdas)
    leyendas_js = "{" + ",".join(
        f"{json.dumps(k)}: {json.dumps('leyenda-' + k)}" for k in capas
    ) + "}"
    return f"""
<script>
window.addEventListener('load', function () {{
  var MAPA = {mapa.get_name()};
  var CAPAS = {capas_js};
  var CELDAS = {celdas_js};
  var ANOMALIAS = {fg_anom.get_name()};
  var LEYENDAS = {leyendas_js};
  function alternarModo(modo) {{
    Object.keys(CAPAS).forEach(function (k) {{
      if (k === modo) {{ MAPA.addLayer(CAPAS[k]); }} else {{ MAPA.removeLayer(CAPAS[k]); }}
      var el = document.getElementById(LEYENDAS[k]);
      if (el) {{ el.style.display = (k === modo) ? 'block' : 'none'; }}
    }});
  }}
  function alternarCeldas(valor) {{
    Object.keys(CELDAS).forEach(function (k) {{
      if (k === valor) {{ MAPA.addLayer(CELDAS[k]); }} else {{ MAPA.removeLayer(CELDAS[k]); }}
    }});
  }}
  function alternarAnomalias(ver) {{
    if (ver) {{ MAPA.addLayer(ANOMALIAS); }} else {{ MAPA.removeLayer(ANOMALIAS); }}
  }}
  var selModo = document.getElementById('modo-color');
  var selCeldas = document.getElementById('celdas');
  var chkAnom = document.getElementById('ver-anomalias');
  if (selModo) {{ selModo.addEventListener('change', function (e) {{ alternarModo(e.target.value); }}); }}
  if (selCeldas) {{ selCeldas.addEventListener('change', function (e) {{ alternarCeldas(e.target.value); }}); }}
  if (chkAnom) {{ chkAnom.addEventListener('change', function (e) {{ alternarAnomalias(e.target.checked); }}); }}
  if (selModo) {{ alternarModo(selModo.value); }}
  if (selCeldas) {{ alternarCeldas(selCeldas.value); }}
  if (chkAnom) {{ alternarAnomalias(chkAnom.checked); }}
}});
</script>
"""


# --------------------------------------------------------------------------
# Mapa interactivo
# --------------------------------------------------------------------------
def mapa_folium(riesgo: pd.DataFrame, patrones: pd.DataFrame | None = None,
                predicciones=None, ruta=None) -> str | None:
    """
    Mapa HTML interactivo de focos críticos.

    Capas: riesgo observado, PLI observado, clusters K-Means/GMM, PLI y
    probabilidad predichos (OOF cuando existen), clase predicha, anomalías y
    celdas de 5 × 5 km. El panel superior derecho permite alternar las capas.
    """
    import folium

    d = riesgo.dropna(subset=["COORD_ESTE", "COORD_NORTE"]).copy()
    if d.empty:
        return None
    zona = 19
    if "TXZONA" in d.columns and d["TXZONA"].notna().any():
        zona = int(d["TXZONA"].mode().iloc[0])
    lat, lon = utm_a_latlon(d["COORD_ESTE"], d["COORD_NORTE"], zona=zona)
    d["lat"], d["lon"] = lat, lon

    d = _unir_por_id(
        d, patrones, ["cluster_kmeans", "cluster_gmm", "es_anomalia", "anomalia_score"]
    )
    pred = _predicciones_mapa(predicciones)
    d = _unir_por_id(d, pred, [
        "PLI_predicho", "PLI_predicho_oof", "PLI_predicho_mapa", "fuente_pli",
        "prob_riesgo_alto", "prob_riesgo_alto_oof", "prob_riesgo_alto_mapa",
        "fuente_prob", "clase_predicha", "clase_predicha_mapa",
    ])

    if "nivel_riesgo" not in d.columns:
        d["nivel_riesgo"] = "Sin dato"
    d["nivel_riesgo"] = d["nivel_riesgo"].fillna("Sin dato").astype(str)
    for c in ("PLI", "cluster_kmeans", "cluster_gmm", "PLI_predicho_mapa",
              "prob_riesgo_alto_mapa", "clase_predicha_mapa"):
        if c not in d.columns:
            d[c] = np.nan
    if "es_anomalia" not in d.columns:
        d["es_anomalia"] = False
    for c in ("fuente_pli", "fuente_prob"):
        if c not in d.columns:
            d[c] = "sin predicción"
        else:
            d[c] = d[c].fillna("sin predicción")

    etiquetas_km = _etiquetas_clusters(_cargar_perfil_clusters())

    centro = [float(d["lat"].mean()), float(d["lon"].mean())]
    mapa = folium.Map(
        location=centro, zoom_start=10, tiles="OpenStreetMap", control_scale=True
    )

    # Celdas 5 × 5 km: se dibujan debajo de los puntos.
    zonas = zonas_criticas(riesgo, top=None)
    capas_celdas = {}
    if not zonas.empty:
        capas_celdas["top"] = _capa_celdas(
            zonas.head(15), "Celdas 5×5 km (top 15)", zona
        )
        capas_celdas["todas"] = _capa_celdas(zonas, "Celdas 5×5 km (todas)", zona)
        for capa in capas_celdas.values():
            capa.add_to(mapa)

    capas = {}
    for modo, etiqueta in MODOS_COLOR.items():
        fg = folium.FeatureGroup(name=etiqueta)
        for _, r in d.iterrows():
            color = _color_modo(r, modo)
            folium.CircleMarker(
                location=[r["lat"], r["lon"]], radius=5.5, color="#333333",
                weight=0.6, fill=True, fill_color=color, fill_opacity=0.88,
                popup=folium.Popup(_popup_muestra(r, etiquetas_km), max_width=330),
            ).add_to(fg)
        fg.add_to(mapa)
        capas[modo] = fg

    fg_anom = folium.FeatureGroup(name="Anomalías (Isolation Forest)")
    for _, r in d[d["es_anomalia"].map(_es_verdadero)].iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]], radius=8.5, color="#111111",
            weight=1.6, dash_array="3", fill=False,
            popup=folium.Popup(_popup_anomalia(r), max_width=300),
        ).add_to(fg_anom)
    fg_anom.add_to(mapa)

    leyendas = {
        "nivel_riesgo": _leyenda_riesgo(d),
        "PLI": _leyenda_pli(
            "PLI observado", d["PLI"],
            "PLI observado; no es un nivel normativo.",
        ),
        "cluster_kmeans": _leyenda_clusters(
            "Cluster K-Means", d["cluster_kmeans"], etiquetas_km
        ),
        "cluster_gmm": _leyenda_clusters(
            "Cluster GMM", d["cluster_gmm"], {}
        ),
        "PLI_predicho_mapa": _leyenda_pli(
            "PLI predicho", d["PLI_predicho_mapa"],
            "Predicción fuera de fold (OOF) cuando existe; si no, modelo final "
            "(in-sample). No es un nivel normativo.",
        ),
        "prob_riesgo_alto_mapa": _leyenda_bins(
            "Prob. de exceder ≥1 PEL", d["prob_riesgo_alto_mapa"],
            BINS_PROB, COLORES_PROB, ETIQUETAS_PROB,
            "Probabilidad del clasificador; OOF cuando existe.",
        ),
        "clase_predicha_mapa": _leyenda_clase(d["clase_predicha_mapa"]),
    }
    html_leyendas = "".join(
        _envolver_leyenda(modo, contenido, visible=(modo == "nivel_riesgo"))
        for modo, contenido in leyendas.items()
    )
    mapa.get_root().html.add_child(folium.Element(html_leyendas))
    mapa.get_root().html.add_child(folium.Element(_panel_controles()))
    mapa.get_root().html.add_child(
        folium.Element(_script_controles(mapa, capas, capas_celdas, fg_anom))
    )

    ruta = Path(ruta) if ruta else (cfg.REPORTS / "mapa_focos.html")
    mapa.save(str(ruta))
    LOG.info("Mapa interactivo → %s", ruta)
    return str(ruta)


# --------------------------------------------------------------------------
# Exportación (RF-18)
# --------------------------------------------------------------------------
def exportar_dataset_final(riesgo: pd.DataFrame, patrones: pd.DataFrame | None,
                           predicciones=None) -> pd.DataFrame:
    """
    Une índices, patrones y predicciones por ``id_muestra``.

    Las predicciones incluyen las del modelo final y, cuando existen, las
    predicciones fuera de fold (OOF) con su fuente explícita.
    """
    final = riesgo.copy()
    if patrones is not None:
        extra = patrones.drop(
            columns=[c for c in patrones.columns if c in final.columns], errors="ignore"
        )
        final = final.join(extra, how="left")
    pred = _predicciones_mapa(predicciones)
    if pred is not None and not pred.empty:
        cols = [c for c in pred.columns
                if c != "id_muestra" and c not in final.columns]
        if cols:
            if "id_muestra" in final.columns and "id_muestra" in pred.columns:
                final = final.merge(
                    pred[["id_muestra"] + cols], on="id_muestra", how="left"
                )
            else:
                final = final.join(pred[cols], how="left")
    for c in ("fuente_pli", "fuente_prob"):
        if c in final.columns:
            final[c] = final[c].fillna("sin predicción")
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
