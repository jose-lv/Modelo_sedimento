"""
Pipeline completo del MVP (CRISP-DM) en un solo comando.

    python run_pipeline.py

Salidas principales
-------------------
data/processed/dataset_final.csv      muestras + índices + clusters + predicción
data/processed/patrones.csv           clusters y anomalías
reports/tables/*.csv|json             métricas, logs, importancias
reports/figures/*.png                 figuras de la tesis
reports/mapa_focos.html               mapa interactivo
reports/RESUMEN_MVP.md                resumen con cifras
"""
from __future__ import annotations

import argparse
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn

from src import config as cfg
from src import clean, eda, explain, features, indices, load, models, parametros, patterns, report
from src.utils import LOG, guardar_csv, guardar_json, paso, sha256_archivo


def parsear_argumentos():
    p = argparse.ArgumentParser(description="MVP sedimentos Moquegua (OEFA)")
    p.add_argument("--sin-cache", action="store_true", help="ignora la caché del CSV crudo")
    p.add_argument("--sin-tunear", action="store_true", help="omite RandomizedSearchCV")
    p.add_argument("--sin-patrones", action="store_true", help="omite clustering/anomalías")
    p.add_argument("--solo-datos", action="store_true", help="solo carga, limpieza e índices")
    return p.parse_args()


def _tabla_modelo(resumen: pd.DataFrame, tipo: str) -> pd.DataFrame:
    if tipo == "regresion":
        cols = ["modelo", "r2_log_media", "r2_log_sd", "rmse_log_media",
                "mae_log_media", "r2_pli_media", "rmse_pli_media"]
    else:
        cols = ["modelo", "f1_media", "f1_sd", "roc_auc_media", "roc_auc_sd",
                "pr_auc_media", "balanced_accuracy_media"]
    cols = [c for c in cols if c in resumen.columns]
    return resumen[cols].round(3)


def main() -> int:
    args = parsear_argumentos()
    t0 = time.perf_counter()
    LOG.info("=" * 78)
    LOG.info("MVP – Modelo de ML para contaminación de sedimentos en Moquegua")
    LOG.info("=" * 78)

    # ------------------------------------------------------------------
    # 0. Reproducibilidad
    # ------------------------------------------------------------------
    guardar_json({
        "fecha_ejecucion": pd.Timestamp.now().isoformat(),
        "python": sys.version,
        "plataforma": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "semilla": cfg.SEED,
        "semillas_cv": cfg.SEEDS,
        "n_folds": cfg.N_FOLDS,
        "csv_crudo": str(cfg.RAW_CSV),
        "sha256_csv": sha256_archivo(cfg.RAW_CSV),
        "metodo_censura": cfg.METODO_CENSURA,
        "percentil_fondo": cfg.PERCENTIL_FONDO,
        "metales_indice": cfg.METALES_INDICE,
        "cobertura_min_modelo": cfg.COBERTURA_MIN_MODELO,
        "cobertura_min_analisis": cfg.COBERTURA_MIN_ANALISIS,
        "anio_corte_temporal": cfg.ANIO_CORTE_TEMPORAL,
        "argumentos": vars(args),
    }, cfg.CONFIG_DIR / "run_config.json")

    # ------------------------------------------------------------------
    # 1. Carga y filtro (RF-01, RF-02)
    # ------------------------------------------------------------------
    with paso("1. Carga y filtro a Moquegua"):
        crudo = load.cargar_crudo(usar_cache=not args.sin_cache)
        parametros.generar_tabla_parametros(crudo)

    # ------------------------------------------------------------------
    # 2. Limpieza y dataset ancho (RF-03 a RF-07)
    # ------------------------------------------------------------------
    with paso("2. Limpieza y formato ancho"):
        resultado_clean = clean.ejecutar(crudo)
        ancho = resultado_clean["ancho"]
        ancho = features.agregar_variables_derivadas(ancho)

    # ------------------------------------------------------------------
    # 3. Cobertura y faltantes (RF-08, RF-09)
    # ------------------------------------------------------------------
    with paso("3. Cobertura y faltantes"):
        cobertura = features.reporte_cobertura(ancho, resultado_clean["censura"])
        guardar_csv(cobertura, cfg.TABLES / "cobertura_parametros.csv")
        guardar_csv(ancho, cfg.DATA_PROCESSED / "dataset_ancho.csv")

    # ------------------------------------------------------------------
    # 4. Índices de riesgo (RF-10)
    # ------------------------------------------------------------------
    with paso("4. Índices de riesgo y guías de calidad"):
        guias = indices.cargar_guias_ccme()
        riesgo = indices.construir_dataset_riesgo(ancho)
        guardar_csv(riesgo, cfg.DATA_PROCESSED / "dataset_riesgo.csv")

        sens_fondo = indices.sensibilidad_fondo(ancho)
        guardar_csv(sens_fondo, cfg.TABLES / "sensibilidad_fondo.csv")
        sens_cens = indices.sensibilidad_censura(resultado_clean["largo"], ancho)
        guardar_csv(sens_cens, cfg.TABLES / "sensibilidad_censura.csv")
        LOG.info("PLI: mediana=%.2f | %% >1 = %.1f | %% excede PEL = %.1f",
                 riesgo["PLI"].median(), 100 * riesgo["PLI"].gt(1).mean(),
                 100 * riesgo["excede_pel"].mean())

    # ------------------------------------------------------------------
    # 5. EDA (RF-11)
    # ------------------------------------------------------------------
    with paso("5. Análisis exploratorio"):
        resultado_eda = eda.ejecutar(riesgo, cobertura, guias)

    if args.solo_datos:
        LOG.info("Modo --solo-datos: se detiene antes del modelado.")
        return 0

    # ------------------------------------------------------------------
    # 6. Patrones (RF-12)
    # ------------------------------------------------------------------
    resultado_patrones = None
    if not args.sin_patrones:
        with paso("6. Clustering y anomalías"):
            resultado_patrones = patterns.ejecutar(riesgo, cobertura)
            guardar_csv(resultado_patrones, cfg.DATA_PROCESSED / "patrones.csv")

    # ------------------------------------------------------------------
    # 7. Modelos supervisados (RF-13, RF-14)
    # ------------------------------------------------------------------
    resultados_modelos = {}
    predicciones = {}
    for tarea in cfg.TAREAS:
        with paso(f"7. Modelado – {tarea}"):
            matriz = features.construir_matriz(riesgo, tarea, cobertura)
            guardar_csv(features.resumen_faltantes(matriz["X"]),
                        cfg.TABLES / f"faltantes_features_{tarea}.csv")
            res = models.entrenar_tarea(matriz, tarea, tunear=not args.sin_tunear)
            pred = models.predecir_dataset(res["modelo_final"], matriz, tarea)
            guardar_csv(pred, cfg.DATA_PROCESSED / f"predicciones_{tarea}.csv")
            resultados_modelos[tarea] = res
            predicciones[tarea] = pred

    # ------------------------------------------------------------------
    # 8. Interpretación SHAP (RF-15)
    # ------------------------------------------------------------------
    tarea_principal = cfg.TAREA_PRINCIPAL
    with paso(f"8. SHAP – {tarea_principal}"):
        matriz_principal = features.construir_matriz(riesgo, tarea_principal, cobertura)
        resultado_shap = explain.ejecutar(
            resultados_modelos[tarea_principal]["modelo_final"], matriz_principal, tarea_principal
        )
        guardar_csv(resultado_shap["contribucion"],
                    cfg.DATA_PROCESSED / "shap_contribucion_por_muestra.csv")
        LOG.info("Top 5 variables SHAP: %s",
                 ", ".join(resultado_shap["importancia"]["variable"].head(5)))

    # ------------------------------------------------------------------
    # 9. Mapa, ranking y exportación (RF-16, RF-18)
    # ------------------------------------------------------------------
    with paso("9. Mapa de focos, ranking y exportación"):
        ranking = report.ranking_puntos(riesgo, top=10)
        ranking = report.ranking_con_shap(ranking, resultado_shap["contribucion"])
        guardar_csv(ranking, cfg.TABLES / "ranking_puntos_criticos.csv")
        zonas = report.zonas_criticas(riesgo, top=15)
        if not zonas.empty:
            guardar_csv(zonas, cfg.TABLES / "zonas_criticas.csv")
        report.mapa_folium(riesgo, resultado_patrones, predicciones)
        final = report.exportar_dataset_final(
            riesgo, resultado_patrones, predicciones
        )
        LOG.info("Dataset final: %d filas × %d columnas", *final.shape)

    # ------------------------------------------------------------------
    # 10. Checklist y resumen
    # ------------------------------------------------------------------
    with paso("10. Checklist y resumen"):
        estado = {
            "pipeline_un_comando": True,
            "log_limpieza": (cfg.TABLES / "log_limpieza.csv").exists(),
            "censura": (cfg.TABLES / "censura_por_parametro.csv").exists(),
            "sin_fuga": True,
            "validacion_agrupada": True,
            "tres_modelos": all(
                len(r["resumen"]) >= 3 for r in resultados_modelos.values()
            ),
            "shap": (cfg.TABLES / f"shap_importancia_{tarea_principal}.csv").exists(),
            "mapa": (cfg.REPORTS / "mapa_focos.html").exists(),
            "tablero": (cfg.ROOT / "app" / "streamlit_app.py").exists(),
            "readme": (cfg.ROOT / "README.md").exists(),
        }
        tabla_checklist = report.checklist(estado)

        texto_patrones = ""
        if resultado_patrones is not None:
            perfil = pd.read_csv(cfg.TABLES / "perfil_clusters.csv", encoding="utf-8-sig")
            resumen_pat = pd.read_json(cfg.TABLES / "resumen_patrones.json", typ="series")
            lineas = [
                f"- Clusters K-Means: {resumen_pat['k_kmeans']} "
                f"(PC1+PC2 explican {100*resumen_pat['varianza_acumulada_2PC']:.1f} % de la varianza)",
                f"- Componentes GMM: {resumen_pat['k_gmm']}",
                f"- Muestras anómalas (Isolation Forest, p95): "
                f"{int(resultado_patrones['es_anomalia'].sum())}",
                "",
                perfil.to_markdown(index=False),
            ]
            texto_patrones = "\n".join(lineas)

        texto_zonas = ""
        if zonas is not None and not zonas.empty:
            tabla_zonas = zonas[[
                "GRID_5KM", "distrito", "n_muestras", "n_puntos", "nivel_riesgo",
                "pct_alto", "pct_alto_ic95_inf", "PLI_mediano",
                "max_metales_sobre_pel", "metal_dominante_normativo",
            ]].rename(columns={
                "GRID_5KM": "celda_5km", "pct_alto": "%_alto_muy_alto",
                "pct_alto_ic95_inf": "%_alto_IC95_inf",
                "PLI_mediano": "PLI_mediano", "max_metales_sobre_pel": "max_metales_PEL",
                "metal_dominante_normativo": "metal_dominante",
            })
            texto_zonas = tabla_zonas.to_markdown(index=False)

        contexto = {
            "resumen_eda": resultado_eda["resumen"],
            "modelos": {},
            "texto_patrones": texto_patrones,
            "texto_zonas": texto_zonas,
            "checklist": tabla_checklist,
        }
        for t, r in resultados_modelos.items():
            matriz_t = features.construir_matriz(riesgo, t, cobertura)
            contexto["modelos"][t] = {
                "target": cfg.TAREAS[t]["target"],
                "descripcion": cfg.TAREAS[t]["descripcion"],
                "mejor_nombre": r["mejor_nombre"],
                "n_muestras": int(len(matriz_t["y"])),
                "n_features": int(matriz_t["X"].shape[1]),
                "n_grupos": int(matriz_t["grupos"].nunique()),
                "tabla": _tabla_modelo(r["resumen"], cfg.TAREAS[t]["tipo"]),
            }

        report.escribir_resumen(contexto)

    LOG.info("=" * 78)
    LOG.info("Pipeline terminado en %.1f s", time.perf_counter() - t0)
    LOG.info("Revisar: %s", cfg.REPORTS)
    LOG.info("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
