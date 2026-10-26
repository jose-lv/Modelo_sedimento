# MVP · Detección de patrones de contaminación y estimación de riesgo en sedimentos de Moquegua

Implementación ejecutable del plan de tesis **«Modelos de aprendizaje automático
para identificar patrones de contaminación y estimar el riesgo en sedimentos de
cuerpos de agua de Moquegua»**, sobre el registro público de monitoreos
ambientales del OEFA.

El objetivo del plan se mantiene intacto: **identificar patrones de
contaminación y estimar riesgo** con aprendizaje automático, evaluado con
R²/RMSE/MAE, validación K-fold, interpretación SHAP y localización de zonas
críticas. Lo que cambia respecto del anteproyecto son las **decisiones técnicas**
(documentadas en la sección [Decisiones metodológicas](#decisiones-metodológicas)),
todas trazables a evidencia del propio dataset.

---

## 1. Resultados en una mirada

| Indicador | Valor |
|---|---|
| Muestras de sedimento en Moquegua | **394** (2013–2026) |
| Puntos de muestreo normalizados | **193** |
| Distritos | Carumas (288), Torata (78), Ichuña (12), Pacocha (7), El Algarrobal (4), Chojata (3), Ilo (2) |
| Metales del índice (con guía CCME) | As, Cd, Cr, Cu, Hg, Ni, Pb, Zn |
| PLI mediano | **2.44** |
| Muestras con PLI > 1 | **79.7 %** |
| Muestras que exceden ≥ 1 PEL | **66.5 %** |
| Nivel de riesgo *Alto* o *Muy alto* | **66.5 %** (262/394) |
| Zonas críticas (celdas de 5 × 5 km) | **15**, todas *Muy alto*: 12 en Carumas y 3 en Torata |
| Mejor modelo (regresión de `log1p(PLI)`) | **XGBoost**, R² = 0.82 ± 0.04 (CV agrupada repetida) |

> Las cifras exactas y actualizadas viven en `reports/tables/` y en
> `reports/RESUMEN_MVP.md`, que se regeneran con cada ejecución.

---

## 2. Datos

**Fuente.** `data/raw/Monitoreos_SEDIMENTO.csv` — Registro de monitoreos
ambientales del OEFA (matriz sedimento). 257 092 filas a nivel nacional.

**Filtro a Moquegua.** Se aplica sobre el **primer segmento** de `TXUBIGEO`
(código de departamento), no sobre el ubigeo completo:

| Etapa | Filas |
|---|---:|
| Registro nacional | 257 092 |
| Moquegua (primer segmento de `TXUBIGEO`) | 14 889 |
| Tras reglas de limpieza R1–R10 | 12 254 |
| Dataset ancho (una fila por muestra) | **394 muestras × 101 columnas** |

**Verificación del filtro.** Se comprobó que ninguna de las 10 372 filas cuyo
ubigeo agrupado caía en Moquegua tiene coordenadas dentro de la caja envolvente
UTM 19S del departamento; filtrar por ubigeo completo habría introducido
muestras de otras regiones.

### 2.1 Hallazgo de calidad que cambia el dataset

El CSV codifica celdas vacías como `valor = 0` **sin unidad y sin signo**. Son
*placeholders*, no mediciones. La regla **R5** los elimina:

- 14 889 → 12 834 filas (−2 055).
- Al quitarlos **desaparecen los 208 duplicados conflictivos** (dos valores
  distintos para la misma muestra y parámetro), sin necesidad de promediar ni
  descartar muestras.
- Los valores negativos solo aparecen en pruebas ABA (NNP/RNP) y son
  físicamente válidos; se conservan. Solo se descartan negativos en las
  familias de concentración.

### 2.2 Censura (valores bajo el límite de detección)

El porcentaje de censura por parámetro se documenta en
`reports/tables/censura_por_parametro.csv`. Imputación principal: **LD/2**,
con análisis de sensibilidad a LD/√2 y LD. Los parámetros con censura > 80 %
se tratan como binarios (detectado / no detectado).

---

## 3. Decisiones metodológicas

Cada decisión responde a un problema concreto encontrado en los datos.

### 3.1 Variable objetivo y anti-fuga de datos

El **PLI** se calcula con 8 metales:

```
PLI = exp( (1/n) · Σ ln( Cᵢ / Bnᵢ ) )      n ≥ 4 metales
```

Si esos metales entran como variables de entrada, el modelo aprende la
respuesta y las métricas son ficticias. Por eso:

- **Prohibidos como features**: As, Cd, Cr, Cu, Hg, Ni, Pb, Zn y cualquier
  columna derivada (`CF_*`, `Igeo_*`, `Bn_*`, `n_metales_sobre_*`).
- **Permitidos**: litogénicos no normados (Al, Fe, Mn, Ti, …), fisicoquímicos,
  granulometría, hidrocarburos, coordenadas y tiempo.
- La regla se verifica en dos lugares: `features.verificar_no_fuga()` (aborta el
  pipeline) y `tests/test_no_leakage.py` (falla la suite).

### 3.2 Fondo geoquímico

`Bn` = **percentil 25 local** de cada metal (fondo geogénico de la cuenca).
Se reporta sensibilidad a P10, P50 y a la corteza de Turekian & Wedepohl (1961).

### 3.3 Nivel de riesgo normativo (mejora sobre la guía)

La guía propone clasificar el riesgo por umbrales de PLI. Con estos datos eso es
inútil: el **79.7 %** de las muestras tiene PLI > 1, así que «riesgo alto» deja
de discriminar. Se reemplaza por una escala anclada en las guías **CCME
ISQG/PEL** (valores de efecto umbral y probable):

| Nivel | Criterio | Muestras |
|---|---|---:|
| Bajo | ningún metal > ISQG | 19 |
| Medio | ≥ 1 metal > ISQG | 89 |
| Alto | ≥ 1 metal > PEL | 199 |
| Muy alto | ≥ 3 metales > PEL | 63 |
| Sin dato | sin metales del índice | 24 |

El PLI se conserva como variable objetivo de regresión y como indicador
continuo; no como definición de riesgo.

### 3.4 Variable de clasificación

Se descartó `PLI > 1` (79.7 % positivos, modelo trivialmente optimista) y se usa
**`excede_pel`**: la muestra supera el PEL de al menos un metal del índice
(64 % / 36 %, balanceado).

### 3.5 Validación

| Esquema | Para qué |
|---|---|
| **GroupKFold repetido (5 semillas)** por `PUNTO_NORMALIZADO` | métricas honestas; un punto nunca cae en train y test a la vez |
| **Hold-out temporal** (entrena ≤ 2023, prueba 2024–2026) | escenario real: predecir campañas futuras |
| **Bootstrap 95 %** sobre el hold-out | intervalo de confianza de la métrica |
| **Permutación** como control de SHAP | verifica que la explicación no dependa de la técnica |

`PUNTO_NORMALIZADO` normaliza variantes como `ESP-SED-01` → `ESP-SED-1` para
evitar fuga espacial entre nombres equivalentes.

### 3.6 Guías de calidad

`config/guias_calidad.csv` contiene ISQG/PEL de CCME para 20 parámetros. CCME no
publica valor para **níquel**, así que se complementa con **NOAA ERL/ERM**
(20.9 / 51.6 mg/kg) y se deja explícito en la columna `fuente`.

### 3.7 Dos definiciones de «metal dominante»

El dataset conserva dos lecturas complementarias, y conviene no confundirlas:

| Columna | Definición | Para qué sirve |
|---|---|---|
| `metal_dominante` | argmax del **factor de contaminación** `CF = C / Bn` | Detectar anomalías respecto al fondo geoquímico local |
| `metal_dominante_normativo` | argmax de `C / PEL` (o `C / ISQG` si el metal no tiene PEL) | Responder a la pregunta regulatoria: qué metal presiona más la norma |

El CF sobrepondera metales con fondo muy bajo (p. ej. Hg): una concentración
absoluta pequeña puede producir un CF alto. Por eso el **ranking de puntos, las
zonas críticas y el mapa** usan `metal_dominante_normativo`; el CF se mantiene
como indicador de enriquecimiento relativo.

---

## 4. Estructura del proyecto

```
pmv/
├── run_pipeline.py            # orquestador único (CRISP-DM)
├── requirements.txt
├── config/
│   ├── parametros.csv         # equivalencias de parámetros OEFA → nombre estándar
│   ├── guias_calidad.csv      # ISQG/PEL (CCME + complemento NOAA)
│   └── run_config.json        # semillas, versiones, sha256 del CSV
├── data/
│   ├── raw/                   # CSV original del OEFA (no se modifica)
│   ├── interim/               # caché de la carga filtrada
│   └── processed/             # dataset_ancho, dataset_riesgo, patrones, predicciones
├── src/
│   ├── config.py              # decisiones centrales y rutas
│   ├── load.py                # carga + filtro Moquegua
│   ├── parametros.py          # homologación de nombres
│   ├── clean.py               # reglas R1–R10 con bitácora
│   ├── features.py            # cobertura, derivadas, matriz anti-fuga
│   ├── indices.py             # Igeo, CF, PLI, RI, nivel_riesgo, sensibilidad
│   ├── models.py              # pipelines, CV agrupada, hold-out, bootstrap
│   ├── patterns.py            # K-Means, GMM, PCA, Isolation Forest
│   ├── explain.py             # SHAP + permutación
│   ├── plots.py               # todas las figuras
│   ├── eda.py                 # tablas y figuras exploratorias
│   ├── report.py              # mapa folium, ranking, resumen, checklist
│   └── utils.py               # logging, guardado, hashing
├── app/streamlit_app.py       # tablero (5 pantallas)
├── notebooks/                 # 01–05, narrativa de la tesis
├── tests/                     # anti-fuga e índices
└── reports/
    ├── figures/               # PNG para el documento
    ├── tables/                # CSV/JSON con todas las métricas
    ├── logs/                  # bitácoras de ejecución
    ├── mapa_focos.html        # mapa interactivo
    └── RESUMEN_MVP.md         # resumen autogenerado con cifras
```

---

## 5. Instalación y ejecución

```bash
# 1. Entorno (Python 3.11–3.12)
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Pipeline completo (datos → EDA → patrones → modelos → SHAP → mapa)
python run_pipeline.py

# 3. Solo la etapa de datos (rápido, sin modelos)
python run_pipeline.py --solo-datos

# 4. Tablero
streamlit run app/streamlit_app.py

# 5. Pruebas
pytest -q
```

**Opciones del pipeline**

| Flag | Efecto |
|---|---|
| `--sin-cache` | ignora la caché intermedia y relee el CSV crudo |
| `--sin-tunear` | omite `RandomizedSearchCV` (más rápido, métricas iguales) |
| `--sin-patrones` | omite clustering y anomalías |
| `--solo-datos` | se detiene antes del modelado |

**Ruta del CSV.** Por defecto `data/raw/Monitoreos_SEDIMENTO.csv`; se puede
sobreescribir con la variable de entorno `OEFA_CSV`.

**Nota para clones desde GitHub.** El CSV crudo (~98 MB) y los archivos
intermedios **no se versionan** por tamaño (están en `.gitignore`). Descarga
`Monitoreos_SEDIMENTO.csv` del portal de datos abiertos del OEFA y guárdalo en
`data/raw/`. Sin ese archivo el pipeline se detiene en el paso 1. Los artefactos
de `data/processed/`, `models/` y `reports/` sí se incluyen, de modo que el
tablero funciona sin reentrenar.

---

## 6. Modelos

Se comparan cuatro enfoques por tarea: *baseline* (media/prior), modelo lineal
regularizado, Random Forest y XGBoost. La imputación y el escalado viven
**dentro** del `Pipeline`, de modo que nunca se filtran estadísticos del test al
train.

### 6.1 Regresión de `log1p(PLI)`

| Modelo | R² (log) | RMSE (log) | MAE (log) | R² (escala PLI) |
|---|---:|---:|---:|---:|
| **XGBoost** | **0.823 ± 0.043** | 0.262 | 0.197 | 0.774 |
| Random Forest | 0.781 ± 0.046 | 0.292 | 0.222 | 0.689 |
| Ridge | 0.675 ± 0.087 | 0.354 | 0.279 | 0.606 |
| Media (baseline) | −0.031 ± 0.050 | 0.640 | 0.535 | −0.085 |

n = 357 muestras, 45 variables, 25 evaluaciones (5 folds × 5 semillas).

**Diagnóstico fuera de fold** (una predicción por muestra, sin que el modelo la
haya visto): **R² = 0.841**. En el **hold-out temporal** (2024–2026) el R² cae a
**0.536** con IC 95 % bootstrap **[0.300, 0.721]**, que es la cifra honesta para
predecir campañas futuras. Ambas se guardan en
`reports/tables/resumen_modelo_regresion_pli.json` y alimentan
`13_predicho_vs_real.png`.

### 6.2 Clasificación de `excede_pel`

| Modelo | F1 | ROC-AUC | PR-AUC | Balanced accuracy |
|---|---:|---:|---:|---:|
| **XGBoost** | **0.961 ± 0.021** | 0.964 ± 0.028 | 0.980 | 0.929 |
| Random Forest | 0.955 ± 0.020 | **0.971 ± 0.024** | 0.984 | 0.912 |
| Regresión logística | 0.919 ± 0.035 | 0.940 ± 0.033 | 0.973 | 0.898 |
| Prior (baseline) | 0.845 ± 0.030 | 0.500 | 0.732 | 0.500 |

n = 357, 45 variables, 25 evaluaciones. El rendimiento es alto porque los
litogénicos de contexto (Fe, Al, Mn) covarían con la carga metálica de origen
minero; **no hay fuga** —ninguno de los 8 metales del índice entra como
entrada—, pero conviene leerlo como *capacidad de discriminar con proxies* y no
como prueba de causalidad. El baseline *prior* alcanza F1 = 0.845 porque predice
siempre la clase mayoritaria (73 % de las muestras modeladas); por eso el F1 se
reporta junto al balanced accuracy.

**Fuera de fold**: ROC-AUC = **0.967**. En el hold-out temporal, ROC-AUC =
**0.946** con IC 95 % **[0.872, 0.993]**.

### 6.3 Interpretación

Las cinco variables que más empujan la predicción del PLI son **Fe, Mo, Co, Tl y
Sb**, seguidas de `ratio_Ca_Mg`, Al, la coordenada Norte, B y K. La concordancia
con la importancia por permutación es **9 de 10**, de modo que la explicación no
depende de la técnica.

La lectura geoquímica es coherente con un sistema mineralizado de tipo
pórfido/epitermal: Fe como fase portadora (óxidos/sulfuros) y Mo-Co-Tl-Sb como
elementos trazadores de alteración hidrotermal. Ninguno de los 8 metales que
componen el PLI entra al modelo, así que la señal proviene del contexto
litogénico y no de una fuga.

`reports/tables/shap_importancia_regresion_pli.csv` y
`reports/tables/importancia_permutacion_regresion_pli.csv` contienen el ranking
completo. `reports/tables/resumen_shap_regresion_pli.json` guarda las muestras
críticas y la concordancia top-10 entre ambas técnicas.

### 6.4 Zonas críticas

Las 394 muestras se agregaron en celdas UTM de **5 × 5 km** y se ordenaron por
severidad normativa, **límite inferior del IC 95 % de Wilson** del porcentaje de
muestras en *Alto/Muy alto*, número de puntos y PLI mediano. El IC de Wilson
evita que una celda con una sola muestra al 100 % desplace a una zona con
decenas de puntos y 95 % de excedencia. Las 15 celdas del ranking son **todas
*Muy alto*** y cubren **230 muestras (58 %)**: **12 en Carumas** y **3 en
Torata**. El metal de mayor presión normativa es el **arsénico en 12 celdas**,
con plomo en `63_1622` y cobre en `62_1622` y `64_1621` (las tres en Torata).

Las zonas más persistentes son:

| Celda | Distrito | Muestras | Puntos | PLI mediano | % alto/muy alto | IC95 inferior |
|---|---|---:|---:|---:|---:|---:|
| `74_1633` | Carumas | 79 | 68 | 3.86 | 94.9 | 87.6 |
| `73_1633` | Carumas | 37 | 33 | 3.79 | 97.3 | 86.2 |
| `70_1634` | Carumas | 24 | 14 | 4.11 | 100 | 86.2 |
| `72_1632` | Carumas | 19 | 10 | 3.69 | 100 | 83.2 |

La celda con PLI mediano más alto (`62_1636`, 14.3) tiene una sola muestra: es
un indicio, no prevalencia de zona. Por eso el orden usa el IC95 inferior y
desempata por número de puntos y PLI mediano, no por el máximo aislado.

El contraste entre definiciones justifica la decisión de la sección 3.7: por
factor de contaminación el metal dominante sería **mercurio en 113 muestras**
(arsénico en 91), pero por presión normativa es **arsénico en 285** y cobre en
56. El ranking, las zonas y el mapa usan la segunda lectura.

---

## 7. Tablero

`streamlit run app/streamlit_app.py` abre cinco pantallas:

1. **Resumen** — KPIs, distribución del nivel de riesgo, metales sobre PEL.
2. **Mapa** — puntos coloreados por nivel normativo, con filtros laterales.
3. **Ranking** — puntos críticos ordenados por severidad y PLI, con el metal de
   mayor presión normativa (`metal_dominante_normativo`), el metal más
   enriquecido respecto al fondo (`metal_enriquecido_cf`), la variable SHAP
   dominante, las **zonas críticas** por celda de 5 × 5 km y la exploración de
   una muestra.
4. **Modelo** — métricas de CV, hold-out temporal, IC bootstrap, SHAP y control
   por permutación.
5. **Datos** — explorador con vistas por grupo de columnas, calidad de datos y
   descarga del subconjunto filtrado.

El tablero es **de solo lectura**: consume los artefactos del pipeline.

---

## 8. Pruebas

```bash
pytest -q     # 19 pruebas
```

- `tests/test_no_leakage.py` — los 8 metales del índice y sus derivadas nunca
  entran como features; `verificar_no_fuga` aborta; ninguna feature reproduce el
  PLI (|r| < 0.95).
- `tests/test_indices.py` — PLI = media geométrica de CF, Igeo = log2(CF/1.5),
  RI ponderado por toxicidad, mínimo de 4 metales, cortes de nivel, coherencia
  PEL > ISQG en las guías.
- `tests/test_report.py` — las zonas se agregan y ordenan por severidad; el
  ranking usa `metal_dominante_normativo` y conserva el CF como columna aparte.

---

## 9. Limitaciones

1. **Muestreo no probabilístico.** Las campañas del OEFA priorizan zonas de
   influencia minera; los porcentajes no son prevalence poblacional.
2. **Datos censurados.** La imputación LD/2 es estándar pero introduce sesgo;
   por eso se reporta el análisis de sensibilidad.
3. **Fondo geoquímico.** El P25 local es una aproximación; no sustituye un
   estudio de línea base con testigos de referencia.
4. **Deriva temporal.** El hold-out 2024–2026 tiene pocas muestras; el IC
   bootstrap es obligatorio para leerlo.
5. **Causalidad.** SHAP explica el modelo, no el mecanismo ambiental. Las
   hipótesis de causa requieren verificación de campo.
6. **Resolución espacial.** 193 puntos para todo un departamento; el mapa
   interpola visualmente, no genera superficies de riesgo continuas.

---

## 10. Diferencias con el plan de tesis

El plan estima **270 muestras y 29 metales**. El CSV vigente entrega **394
muestras útiles y 83 parámetros estándar** (47 con cobertura ≥ 10 % y 39 con
cobertura ≥ 60 %, suficiente para modelar). No se recortó el dataset para forzar
la cifra del plan: se reporta el tamaño real y se documenta la diferencia. El
objetivo, las métricas, la validación K-fold, SHAP y la localización de zonas
críticas se mantienen.

Mejoras incorporadas sobre la guía de implementación:

| Guía original | Implementación |
|---|---|
| Riesgo por umbral de PLI | Nivel normativo CCME ISQG/PEL (0–3) |
| K-fold simple | GroupKFold repetido por punto + hold-out temporal + bootstrap |
| Sin control anti-fuga explícito | Barrera dura + prueba automatizada |
| SHAP sin contraste | SHAP + importancia por permutación |
| Un solo fondo geoquímico | P25 principal + sensibilidad P10/P50/corteza |
| Sin tratamiento de placeholders | Regla R5 que elimina 2 055 filas espurias |
| Sin censura documentada | % de censura por parámetro y sensibilidad |
| Focos críticos solo como puntos | Ranking de puntos **y** zonas críticas por celda de 5 × 5 km |

---

## 11. Reproducibilidad

`config/run_config.json` registra en cada ejecución: fecha, versiones de Python,
pandas, numpy y scikit-learn, semillas (`42, 7, 2026, 1234, 99`), número de
folds, sha256 del CSV crudo y todas las decisiones numéricas.

```bash
python run_pipeline.py --sin-cache   # reconstrucción total desde el CSV
```

---

## 12. Referencias metodológicas

- Hakanson, L. (1980). An ecological risk index for aquatic pollution control.
  *Water Research*, 14(8), 975–1001.
- Müller, G. (1969). Index of geoaccumulation in sediments of the Rhine River.
  *GeoJournal*, 2(3), 108–118.
- CCME (1999–2002). *Canadian Sediment Quality Guidelines for the Protection of
  Aquatic Life* — ISQG/PEL de agua dulce.
- Turekian, K. K. & Wedepohl, K. H. (1961). Distribution of the elements in
  some major units of the Earth's crust. *GSA Bulletin*, 72(2), 175–192.
- Lundberg, S. M. & Lee, S.-I. (2017). A unified approach to interpreting model
  predictions. *NeurIPS*.
- NOAA (1996). *Sediment Quality Guidelines* — ERL/ERM (complemento para Ni).
