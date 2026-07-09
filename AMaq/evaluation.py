"""
f1_evaluation.py
================

Utilidades de evaluación para comparar la performance de los modelos en
train y test sobre las métricas pedidas: F1, precision y recall.

Funciones principales
----------------------
- `evaluate_model`        : métricas de un modelo sobre un (X, y).
- `compare_train_test`    : tabla comparativa train vs test (+ gap).
- `plot_metric_comparison`: gráfico de barras train vs test.
- `plot_confusion_matrices`: matrices de confusión train/test.
- `plot_roc_pr_curves`    : Precision-Recall.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)

__all__ = [
    "evaluate_model",
    "compare_train_test",
    "plot_metric_comparison",
    "plot_confusion_matrices",
    "plot_roc_pr_curves",
    "plot_feature_importance",
]



# Orden de métricas en las salidas (las 5 pedidas + PR-AUC como extra útil)
METRIC_ORDER = ["AUC", "Accuracy", "F1", "Precision", "Recall", "PR_AUC"]

def evaluate_model_pr(y_true, y_pred_proba, model_name="Modelo"):
    """
    Genera un análisis completo de Precision-Recall para un modelo de clasificación.
    Encuentra automáticamente el umbral que maximiza el F1-Score y genera 3 visualizaciones.
    """
    # 1. Calcular PR-AUC y F1-Scores para todos los umbrales
    pr_auc = average_precision_score(y_true, y_pred_proba)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)

    # Encontrar el índice del mejor F1-Score
    idx_optimo = np.argmax(f1_scores)
    umbral_optimo = thresholds[idx_optimo]
    max_f1 = f1_scores[idx_optimo]

    print(f"=========================================")
    print(f"--- RENDIMIENTO: {model_name.upper()} ---")
    print(f"=========================================")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Umbral Óptimo sugerido: {umbral_optimo:.3f} (Max F1-Score: {max_f1:.3f})\n")

    # ==========================================
    # GRÁFICO 1: Precision, Recall y F1 vs Umbral
    # ==========================================
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.plot(thresholds, f1_scores, "k-", label="F1 Score", linewidth=2)

    plt.axvline(x=umbral_optimo, color='r', linestyle=':', label=f'Umbral Óptimo (~{umbral_optimo:.2f})')
    plt.axhline(y=max_f1, color='darkred', linestyle=':', label=f'Max F1 (~{max_f1:.2f})')
    plt.plot(umbral_optimo, max_f1, 'ro')

    plt.title(f"{model_name}: Precision, Recall y F1 Score vs Umbral")
    plt.xlabel("Umbral (Threshold)")
    plt.ylabel("Valor de la Métrica")
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # ==========================================
    # GRÁFICO 2: Curva Precision-Recall (PR)
    # ==========================================
    # plt.figure(figsize=(8, 5))
    # plt.plot(recalls, precisions, 'k-', linewidth=2, label=f'{model_name} (AUC = {pr_auc:.2f})')

    # # Marcar el punto del umbral óptimo en la curva PR
    # plt.plot(recalls[idx_optimo], precisions[idx_optimo], 'ro', markersize=8, label=f'Umbral Elegido ({umbral_optimo:.2f})')

    # plt.title(f'Curva Precision-Recall (PR) - {model_name}')
    # plt.xlabel('Recall (Sensibilidad)')
    # plt.ylabel('Precision (Valor Predictivo Positivo)')
    # plt.legend(loc='best')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.show()

    # ==========================================
    # REPORTE DE CLASIFICACIÓN Y MATRIZ
    # ==========================================
    # Aplicar el nuevo umbral
    y_pred = (y_pred_proba >= umbral_optimo).astype(int)

    print(f"\n--- Reporte de Clasificación ({model_name} - Umbral Ajustado) ---")
    print(classification_report(y_true, y_pred))

    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {model_name} (Umbral: {umbral_optimo:.2f})')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.show()

    return umbral_optimo


def _positive_scores(model, X) -> np.ndarray:
    """
    Devuelve el score de la clase positiva.

    Usa `predict_proba` si está disponible; si no, `decision_function`.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return scores


def evaluate_model(model, X, y, threshold: float = 0.5) -> dict:
    """
    Calcula las métricas de clasificación de `model` sobre (X, y).

    - AUC y PR_AUC usan el score continuo (independientes del umbral).
    - Accuracy / F1 / Precision / Recall usan la predicción de clase con
      el umbral dado (por defecto 0.5).

    `zero_division=0` evita warnings cuando el modelo no predice positivos.
    """
    y = np.asarray(y).astype(int)
    scores = _positive_scores(model, X)

    # Predicción de clase. Si trabajamos con probabilidades aplicamos el umbral;
    # para decision_function usamos el signo (umbral 0) salvo umbral explícito 0.5.
    if hasattr(model, "predict_proba"):
        y_pred = (scores >= threshold).astype(int)
    else:
        y_pred = model.predict(X).astype(int)

    return {
        "AUC": roc_auc_score(y, scores),
        "Accuracy": accuracy_score(y, y_pred),
        "F1": f1_score(y, y_pred, zero_division=0),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "PR_AUC": average_precision_score(y, scores),
    }


def compare_train_test(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Tabla comparativa de métricas en train vs test.

    Columnas: Train, Test, Gap (= Train - Test). Un Gap alto y positivo
    es señal de overfitting.
    """
    train_metrics = evaluate_model(model, X_train, y_train, threshold)
    test_metrics = evaluate_model(model, X_test, y_test, threshold)

    df = pd.DataFrame(
        {
            "Train": [train_metrics[m] for m in METRIC_ORDER],
            "Test": [test_metrics[m] for m in METRIC_ORDER],
        },
        index=METRIC_ORDER,
    )
    df["Gap (Train-Test)"] = df["Train"] - df["Test"]
    return df.round(4)


def plot_metric_comparison(comparison_df: pd.DataFrame, title: str = "") -> None:
    """Gráfico de barras comparando Train vs Test por métrica."""
    metrics = comparison_df.index.tolist()
    x = np.arange(len(metrics))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width / 2, comparison_df["Train"], width, label="Train", color="#4C72B0")
    b2 = ax.bar(x + width / 2, comparison_df["Test"], width, label="Test", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title or "Comparación de métricas: Train vs Test")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    threshold: float = 0.5,
    labels=("No Pit", "Pit next lap"),
) -> None:
    """Matrices de confusión lado a lado para train y test."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (X, y, name) in zip(
        axes,
        [(X_train, y_train, "Train"), (X_test, y_test, "Test")],
    ):
        y = np.asarray(y).astype(int)
        if hasattr(model, "predict_proba"):
            y_pred = (model.predict_proba(X)[:, 1] >= threshold).astype(int)
        else:
            y_pred = model.predict(X).astype(int)
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(f"Matriz de confusión — {name}")
    plt.tight_layout()
    plt.show()


def plot_roc_pr_curves(model, X_test, y_test) -> None:
    """Curvas ROC y Precision-Recall sobre el set de test."""
    y_test = np.asarray(y_test).astype(int)
    scores = _positive_scores(model, X_test)

    fpr, tpr, _ = roc_curve(y_test, scores)
    auc = roc_auc_score(y_test, scores)
    prec, rec, _ = precision_recall_curve(y_test, scores)
    ap = average_precision_score(y_test, scores)
    baseline = y_test.mean()  # prevalencia de la clase positiva

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(fpr, tpr, color="#4C72B0", lw=2, label=f"ROC (AUC = {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Azar")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Curva ROC (test)")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(rec, prec, color="#DD8452", lw=2, label=f"PR (AP = {ap:.3f})")
    axes[1].axhline(baseline, ls="--", color="gray", alpha=0.7,
                    label=f"Baseline (prevalencia = {baseline:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Curva Precision-Recall (test)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_feature_importance(fitted_pipeline, top_n: int = 20, model_step: str = "model",
                            preprocess_step: str = "preprocess") -> pd.DataFrame:
    """
    Grafica la importancia de las variables de un Pipeline ya entrenado.

    Soporta modelos con `feature_importances_` (árboles/ensembles) o `coef_`
    (lineales). Recupera los nombres de feature post-preprocesamiento desde el
    ColumnTransformer. Devuelve el DataFrame de importancias (todas las features).
    """
    model = fitted_pipeline.named_steps[model_step]
    pre = fitted_pipeline.named_steps[preprocess_step]
    try:
        names = pre.get_feature_names_out()
    except Exception:
        names = np.array([f"f{i}" for i in range(_n_features(model))])

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        xlabel = "Importancia (feature_importances_)"
    elif hasattr(model, "coef_"):
        importances = np.abs(np.ravel(model.coef_))
        xlabel = "|Coeficiente| (modelo lineal)"
    else:
        print("El modelo no expone feature_importances_ ni coef_.")
        return pd.DataFrame()

    imp_df = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    top = imp_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(9, max(4, 0.35 * len(top))))
    plt.barh(top["feature"], top["importance"], color="#55A868")
    plt.xlabel(xlabel)
    plt.title(f"Top {min(top_n, len(imp_df))} variables más importantes")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()
    return imp_df


def _n_features(model) -> int:
    return getattr(model, "n_features_in_", 0)
