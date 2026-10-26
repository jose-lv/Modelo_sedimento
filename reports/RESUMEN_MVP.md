# MVP – Modelo de ML para contaminación de sedimentos en Moquegua

Resumen generado automáticamente por `run_pipeline.py`. Todas las cifras provienen de `reports/tables/`.

## 1. Dataset

- **Muestras**: 394 (una fila por muestra, formato ancho)
- **Puntos de muestreo**: 193
- **Periodo**: 2013–2026
- **Parámetros con cobertura ≥ 60 %**: 39
- **Parámetros con cobertura ≥ 10 %**: 47
- **Distritos**: CARUMAS, CHOJATA, EL ALGARROBAL, ICHUÑA, ILO, PACOCHA, TORATA

## 2. Riesgo

- **PLI mediano**: 2.44
- **Muestras con PLI > 1** (sedimento deteriorado): 79.7 %
- **Muestras que exceden el PEL** de al menos un metal: 66.5 %

**Nivel de riesgo normativo** (anclado en guías CCME ISQG/PEL, no en un umbral arbitrario de PLI):

| Nivel | Criterio | Muestras | % |
|---|---|---:|---:|
| Bajo | ningún metal > ISQG | 19 | 4.8 % |
| Medio | ≥1 metal > ISQG | 89 | 22.6 % |
| Alto | ≥1 metal > PEL | 199 | 50.5 % |
| Muy alto | ≥3 metales > PEL | 63 | 16.0 % |
| Sin dato | sin metales del índice | 24 | 6.1 % |

## 3. Modelos supervisados

### regresion_pli

- Target: `PLI_log` — log1p(PLI) de 8 metales con guía CCME, predicho con variables que no componen el índice
- Mejor modelo: **XGBoost**
- n = 357, 45 variables, 170 grupos espaciales

| modelo           |   r2_log_media |   r2_log_sd |   rmse_log_media |   mae_log_media |   r2_pli_media |   rmse_pli_media |
|:-----------------|---------------:|------------:|-----------------:|----------------:|---------------:|-----------------:|
| XGBoost          |          0.823 |       0.043 |            0.262 |           0.197 |          0.774 |            1.702 |
| RandomForest     |          0.781 |       0.046 |            0.292 |           0.222 |          0.689 |            2.02  |
| Ridge            |          0.675 |       0.087 |            0.354 |           0.279 |          0.606 |            2.259 |
| Media (baseline) |         -0.031 |       0.05  |            0.64  |           0.535 |         -0.085 |            3.821 |

### clasificacion_riesgo

- Target: `excede_pel` — La muestra supera el PEL (nivel de efecto probable, CCME) de al menos un metal del índice; mismas entradas sin fuga
- Mejor modelo: **RandomForest**
- n = 357, 45 variables, 170 grupos espaciales

| modelo              |   f1_media |   f1_sd |   roc_auc_media |   roc_auc_sd |   pr_auc_media |   balanced_accuracy_media |
|:--------------------|-----------:|--------:|----------------:|-------------:|---------------:|--------------------------:|
| RandomForest        |      0.955 |   0.02  |           0.971 |        0.024 |          0.984 |                     0.912 |
| XGBoost             |      0.961 |   0.021 |           0.964 |        0.028 |          0.98  |                     0.929 |
| Regresión logística |      0.919 |   0.035 |           0.94  |        0.033 |          0.973 |                     0.898 |
| Prior (baseline)    |      0.845 |   0.03  |           0.5   |        0     |          0.732 |                     0.5   |

## 4. Patrones

- Clusters K-Means: 6 (PC1+PC2 explican 34.9 % de la varianza)
- Componentes GMM: 8
- Muestras anómalas (Isolation Forest, p95): 20

|   cluster |   n_muestras | elementos_dominantes                  |   z_max | elementos_depletados        |   z_min |   mediana_global_mg_kg |
|----------:|-------------:|:--------------------------------------|--------:|:----------------------------|--------:|-----------------------:|
|         0 |           22 | Antimonio, Plomo, Arsénico, Molibdeno |    0.62 | Aluminio, Hierro, Potasio   |   -3.58 |                6.76825 |
|         1 |          178 | Manganeso, Titanio, Magnesio, Calcio  |    0.55 | Talio, Antimonio, Arsénico  |   -0.76 |               10.6875  |
|         2 |          111 | Plomo, Arsénico, Hierro, Bario        |    1.02 | Manganeso, Titanio, Cobalto |   -0.93 |               40.725   |
|         3 |           24 | Zinc, Cobre, Cadmio, Níquel           |    2.46 | Mercurio, Bario, Arsénico   |   -0.24 |               42.45    |
|         4 |           36 | Estaño, Talio, Titanio, Antimonio     |    1.47 | Arsénico, Bario, Plomo      |   -0.98 |                9.45    |
|         5 |           23 | Boro, Mercurio, Sodio, Cobre          |    2.76 | Titanio, Plomo, Antimonio   |   -0.72 |               58.25    |

## 5. Zonas críticas (celdas de 5 × 5 km)

Zonas agregadas por celda UTM de 5 km, ordenadas por severidad normativa. Responden al objetivo de localizar zonas críticas y no solo puntos aislados.

|   celda_5km | distrito   |   n_muestras |   n_puntos | nivel_riesgo   |   %_alto_muy_alto |   %_alto_IC95_inf |   PLI_mediano |   max_metales_PEL | metal_dominante   |
|------------:|:-----------|-------------:|-----------:|:---------------|------------------:|------------------:|--------------:|------------------:|:------------------|
|     74_1633 | CARUMAS    |           79 |         68 | Muy alto       |              94.9 |              87.6 |       3.85918 |                 5 | Arsénico          |
|     73_1633 | CARUMAS    |           37 |         33 | Muy alto       |              97.3 |              86.2 |       3.78613 |                 4 | Arsénico          |
|     70_1634 | CARUMAS    |           24 |         14 | Muy alto       |             100   |              86.2 |       4.11447 |                 3 | Arsénico          |
|     72_1632 | CARUMAS    |           19 |         10 | Muy alto       |             100   |              83.2 |       3.68976 |                 3 | Arsénico          |
|     70_1633 | CARUMAS    |            9 |          6 | Muy alto       |             100   |              70.1 |       4.25426 |                 3 | Arsénico          |
|     71_1634 | CARUMAS    |            8 |          4 | Muy alto       |             100   |              67.6 |       1.19429 |                 3 | Arsénico          |
|     61_1636 | CARUMAS    |            6 |          6 | Muy alto       |             100   |              61   |       4.70089 |                 3 | Arsénico          |
|     64_1633 | CARUMAS    |            6 |          4 | Muy alto       |             100   |              61   |       5.20764 |                 3 | Arsénico          |
|     63_1634 | CARUMAS    |            3 |          3 | Muy alto       |             100   |              43.8 |       3.6356  |                 3 | Arsénico          |
|     69_1633 | CARUMAS    |            2 |          2 | Muy alto       |             100   |              34.2 |       2.49434 |                 3 | Arsénico          |
|     61_1633 | CARUMAS    |            2 |          1 | Muy alto       |             100   |              34.2 |      11.7476  |                 5 | Arsénico          |
|     63_1622 | TORATA     |            6 |          6 | Muy alto       |              66.7 |              30   |       1.7002  |                 5 | Plomo             |
|     62_1622 | TORATA     |           11 |          9 | Muy alto       |              54.5 |              28   |       3.24161 |                 4 | Cobre             |
|     64_1621 | TORATA     |           17 |         14 | Muy alto       |              41.2 |              21.6 |       1.54359 |                 4 | Cobre             |
|     62_1636 | CARUMAS    |            1 |          1 | Muy alto       |             100   |              20.7 |      14.2942  |                 4 | Arsénico          |

## 6. Checklist de aceptación

| criterio                                                           | cumplido   | evidencia                                      |
|:-------------------------------------------------------------------|:-----------|:-----------------------------------------------|
| El CSV crudo se transforma en un dataset ancho con un solo comando | ✔          | python run_pipeline.py                         |
| Existe un log con el número de filas eliminadas por cada regla     | ✔          | reports/tables/log_limpieza.csv                |
| El porcentaje de datos censurados por parámetro está documentado   | ✔          | reports/tables/censura_por_parametro.csv       |
| No hay fuga de datos: el target no depende de las entradas         | ✔          | tests/test_no_leakage.py + verificar_no_fuga() |
| Validación agrupada por punto o cuenca                             | ✔          | GroupKFold por PUNTO_NORMALIZADO               |
| Al menos 3 modelos comparados con R², RMSE y MAE                   | ✔          | reports/tables/metricas_cv_regresion_pli.csv   |
| SHAP generado y los 5 factores principales interpretados           | ✔          | reports/figures/16_shap_summary_*.png          |
| Mapa con zonas y focos críticos                                    | ✔          | reports/mapa_focos.html                        |
| Tablero ejecutable localmente                                      | ✔          | streamlit run app/streamlit_app.py             |
| README con instrucciones para reproducir todo                      | ✔          | README.md                                      |

## 7. Notas metodológicas

- **Tamaño muestral**: el plan de tesis estima ~270 muestras y 29 metales; el CSV vigente de OEFA entrega 394 muestras útiles y 47 parámetros con cobertura suficiente. La diferencia se documenta en el README y no altera el objetivo del plan.
- **Anti-fuga**: los 8 metales que componen el PLI/Igeo/RI (As, Cd, Cr, Cu, Hg, Ni, Pb, Zn) están prohibidos como entradas; el modelo usa litogénicos no normados, fisicoquímicos, granulometría, hidrocarburos, ubicación y tiempo.
- **Validación**: GroupKFold repetido (5 semillas) por punto normalizado, más hold-out temporal (entrena ≤2023, prueba 2024-2026) e IC bootstrap 95 %.
- **Fondo geoquímico**: P25 local como escenario principal, con sensibilidad a P10/P25/P50 y a la corteza de Turekian & Wedepohl.
- **Guías**: CCME ISQG/PEL para 8 metales; Ni se complementa con NOAA ERL/ERM porque CCME no publica valor para ese metal.
