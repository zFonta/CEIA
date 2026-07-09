

# Trabajo Final Aprendizaje de Máquina - 2B 2026
# Machine Learning - Predicción de paradas en boxes (Fórmula 1)

### Docentes:
 - Facundo Adrián Lucianna
 - Maria Carina Roldán

Pipeline de Machine Learning que predice, vuelta a vuelta, si un piloto de Fórmula 1 entrará a boxes en la **vuelta siguiente**.

## Objetivo

El proyecto pone a prueba un **pipeline de entrenamiento de modelos** con el fin de encontrar el **mejor clasificador capaz de anticipar cuándo un piloto de F1 entra al pit**. Concretamente, es un problema de **clasificación binaria** sobre la variable `PitNextLap` (¿el piloto para en boxes en la próxima vuelta?), construida a nivel de vuelta.

El evento "para en boxes" es **poco frecuente**, por lo que el target está fuertemente desbalanceado. Por este motivo, se buscaron métricas de optimización apropiadas para el problema y formas mitigarlo. Se compara un **baseline sin machine learning** (heurística de dominio) contra varios modelos de ML optimizados con Optuna, e incorporando una estrategia para el tratamiento del desbalanceo (ponderación de clases).

## Dataset

Datos a **nivel de vuelta**: cada fila es una vuelta completada por un piloto. Se construye en el notebook `00_Preprocesamiento.ipynb` a partir de dos fuentes que se cruzan y consolidan:

- **Kaggle** — [F1 Strategy Dataset | Pit Stop Prediction](https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction) (base principal, vía `kagglehub`).
- **OpenF1 API** — datos de vueltas y clima, descargados con manejo de *rate-limit*. Además, su campo `is_pit_out_lap` se usa como *ground truth* para corregir la etiqueta del target (`PitNextLap_fixed`).

Variables principales: `LapNumber`, `Position`, `LapTime (s)`, `Stint`, `TyreLife`, `Normalized_TyreLife`, `Compound_Encoded`, `LapTime_Delta`, `Cumulative_Degradation`, `Position_Change`, `RaceProgress`, además del target `PitNextLap`.

## Estructura del repositorio

| Archivo | Descripción |
|---|---|
| `00_Preprocesamiento.ipynb` | Ingesta y consolidación de datos: carga de Kaggle + OpenF1 API, *merge* maestro, controles de calidad (QA), corrección del target con la API como *ground truth*, limpieza (exclusión de carreras anómalas, eliminación de la última vuelta de cada piloto, descarte de columnas no usadas) y EDA. **Exporta** `train_data.parquet` y `test_data.parquet`. |
| `01_Entrenamiento.ipynb` | Pipeline de entrenamiento "clásico". Preprocesamiento (WOE para categóricas, imputación + estandarización para numéricas), baseline heurístico y entrenamiento de 5 modelos (Regresión Logística, Random Forest, LightGBM, XGBoost, CatBoost) con Optuna + validación cruzada. El desbalanceo se aborda con **ponderación de clases** (`class_weight` / `auto_class_weights`). Cierra con la validación en test. |
| `optimize.py` | Módulo con `train_with_optuna(...)`: orquesta la búsqueda de hiperparámetros con **Optuna (TPE)** y **StratifiedKFold**, reentrena el mejor modelo sobre todo el train y devuelve `(best_model, results_df)`. |
| `evaluation.py` | Utilidades de evaluación: `evaluate_model_pr` (PR-AUC + umbral óptimo por F1 + gráficos), `evaluate_model`, `compare_train_test`, `plot_confusion_matrices`, `plot_roc_pr_curves`, `plot_feature_importance`, entre otras. |
| `README.md` | Este archivo. |

Carpetas que se crean al ejecutar los notebooks (no están versionadas):

- `./data/` — `train_data.parquet` y `test_data.parquet` (generadas por `00`).
- `./modelos/` — modelos entrenados en formato `.joblib` (generados por `01` y `02`).

## Metodología

- **Baseline sin ML.** Una heurística de dominio (`HeuristicaPitStop`) que predice la parada cuando la antigüedad del neumático (`TyreLife`) supera un umbral, partiendo de que el neumático se degrada con las vueltas. Sirve como piso: cualquier modelo útil debería superarla.
- **Optimización.** `train_with_optuna` recibe un clasificador (o un pipeline completo), una grilla de hiperparámetros y la métrica objetivo; muestrea con Optuna, evalúa con validación cruzada estratificada y reentrena el mejor sobre todo el train.
- **Métrica.** Por el desbalanceo se optimiza **`f1_macro`** y la evaluación se centra en la **clase minoritaria** (F1/precision/recall de la clase positiva, macro-F1 y área bajo la curva PR). El *accuracy* no es representativo: un modelo que nunca prediga una parada ya obtiene *accuracy* alto sin detectar nada útil.
- **Tratamiento del desbalanceo.** Se recurre a la ponderación de clases dentro de cada modelo.
## Resultados (resumen)

Los modelos basados en **árboles/boosting (Random Forest, LightGBM, XGBoost, CatBoost)** muestran mejor performance predictiva que la regresión logística, al captar patrones más complejos. Aun así, a todos les cuesta la clase minoritaria: anticipar el evento raro sigue siendo difícil. Las conclusiones del trabajo apuntan a que la **ponderación de clases por sí sola es insuficiente** para el desbalanceo, por lo que se sugiere explorar técnicas adicionales que mitiguen esta caraterísticas del dataset. Concretamente, se puede incoporar SMOTE, tanto recurriendo al oversampling como al undersampling, explorando ambas posibilidades. Adicionalmente, también sería deseable **incorporar más datos o fuentes adicionales** para mejorar los resultados.

## Autores

- Juan Sebastián Bonals — jsbonals@gmail.com
- Federico Santiago Fontanari — federicofontanari@gmail.com
- Jose Andres Montes de Oca — amontesdeoca1982@gmail.com
