import time
import numpy as np
import pandas as pd
import optuna
from sklearn.base import clone, is_classifier
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_with_optuna(
    train: pd.DataFrame,
    target_col: str,
    classifier,
    param_grid: dict,
    scoring: str = "roc_auc",
    n_trials: int = 50,
    n_folds: int = 5,
    random_state: int = 42,
) -> tuple:
    """
    Entrena un clasificador usando Optuna (TPE sampler) con cross-validation.

    Parameters
    ----------
    train : pd.DataFrame
        DataFrame que incluye features y la columna target.
    target_col : str
        Nombre de la columna target dentro de `train`.
    classifier :
        Instancia de un clasificador compatible con scikit-learn
        (debe implementar fit/predict y set_params).
    param_grid : dict
        Grilla de hiperparámetros. Cada clave es el nombre del
        hiperparámetro y el valor puede ser:
          - list/tuple de valores categóricos  → suggest_categorical
          - dict con keys "low", "high" y opcionalmente "step"/"log"
            → suggest_int  si ambos son int
            → suggest_float si alguno es float
        Ejemplos:
            {"n_estimators": [100, 200, 500],
             "max_depth": {"low": 2, "high": 10},
             "learning_rate": {"low": 1e-4, "high": 0.3, "log": True}}
    scoring : str
        Métrica de evaluación compatible con sklearn cross_val_score.
        Default: "roc_auc".
    n_trials : int
        Número de trials de Optuna. Default: 50.
    n_folds : int
        Número de folds para cross-validation. Default: 5.
    random_state : int
        Semilla aleatoria. Default: 42.

    Returns
    -------
    best_model :
        Modelo re-entrenado con los mejores hiperparámetros sobre
        todo el conjunto de entrenamiento.
    results_df : pd.DataFrame
        DataFrame con los resultados de cada trial:
        trial_number, params, mean_score, std_score, duration_sec, state.
    """

    X = train.drop(columns=[target_col])
    y = train[target_col]

    # Elegir CV estratificado para clasificadores
    use_stratified = is_classifier(classifier)
    cv = (
        StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        if use_stratified
        else KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    )

    trial_records = []

    def objective(trial: optuna.Trial) -> float:
        # Muestrear hiperparámetros según el tipo definido en param_grid
        suggested_params = {}
        for param_name, param_spec in param_grid.items():
            if isinstance(param_spec, (list, tuple)):
                suggested_params[param_name] = trial.suggest_categorical(
                    param_name, param_spec
                )
            elif isinstance(param_spec, dict):
                low = param_spec["low"]
                high = param_spec["high"]
                step = param_spec.get("step", None)
                use_log = param_spec.get("log", False)

                if isinstance(low, int) and isinstance(high, int):
                    suggested_params[param_name] = trial.suggest_int(
                        param_name, low, high, step=step or 1, log=use_log
                    )
                else:
                    suggested_params[param_name] = trial.suggest_float(
                        param_name,
                        float(low),
                        float(high),
                        step=step,
                        log=use_log,
                    )
            else:
                raise ValueError(
                    f"Formato no soportado para '{param_name}'. "
                    "Usá una lista/tuple o un dict con keys 'low' y 'high'."
                )

        model = clone(classifier)
        model.set_params(**suggested_params)

        t0 = time.perf_counter()
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        duration = time.perf_counter() - t0

        mean_score = scores.mean()
        std_score = scores.std()

        trial_records.append(
            {
                "trial_number": trial.number,
                "params": suggested_params.copy(),
                "mean_score": mean_score,
                "std_score": std_score,
                "duration_sec": round(duration, 4),
                "state": "COMPLETE",
            }
        )

        return mean_score

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Re-entrenar el mejor modelo sobre todo el dataset
    best_params = study.best_params
    best_model = clone(classifier)
    best_model.set_params(**best_params)
    best_model.fit(X, y)

    # Construir DataFrame de resultados
    results_df = pd.DataFrame(trial_records).sort_values(
        "mean_score", ascending=False
    ).reset_index(drop=True)

    print(f"✅ Mejor trial: #{study.best_trial.number}")
    print(f"   Score ({scoring}): {study.best_value:.4f}")
    print(f"   Parámetros: {best_params}")

    return best_model, results_df


# ---------------------------------------------------------------------------
# Ejemplo de uso
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier

    # Dataset sintético
    X_raw, y_raw = make_classification(
        n_samples=1_000, n_features=20, random_state=42
    )
    df_train = pd.DataFrame(X_raw, columns=[f"f{i}" for i in range(20)])
    df_train["target"] = y_raw

    # Clasificador base
    clf = GradientBoostingClassifier(random_state=42)

    # Grilla de hiperparámetros
    param_grid = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": {"low": 2, "high": 6},
        "learning_rate": {"low": 0.01, "high": 0.3, "log": True},
        "subsample": {"low": 0.5, "high": 1.0},
        "min_samples_split": {"low": 2, "high": 20},
    }

    best_model, results_df = train_with_optuna(
        train=df_train,
        target_col="target",
        classifier=clf,
        param_grid=param_grid,
        scoring="roc_auc",
        n_trials=30,
        n_folds=5,
    )

    print("\nTop 5 trials:")
    print(results_df.head())
