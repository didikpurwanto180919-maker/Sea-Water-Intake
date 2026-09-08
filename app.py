import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

# Suppress Optuna logging logs di console (optional)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==========================================
# OPTION A: HYPERPARAMETER TUNING DENGAN OPTUNA (RECOMMENDED)
# ==========================================
def optimize_xgboost_optuna(df_train: pd.DataFrame, n_trials: int = 30):
    """Melakukan hyperparameter tuning menggunakan Optuna TPE Sampler

    dengan 5-Fold Stratified Cross-Validation.
    """
    X = df_train[FEATURE_COLUMNS]
    y = df_train["risk_level"]

    def objective(trial):
        # Ruang pencarian hyperparameter (Search Space)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "eval_metric": "mlogloss",
            "n_jobs": -1,
        }

        model = XGBClassifier(**params)

        # Evaluasi dengan Stratified K-Fold CV untuk imbalanced risk distribution
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1
        )

        return scores.mean()

    # Inisialisasi study Optuna
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f" Best F1-Macro Score: {study.best_value:.4f}")
    print(" Best Hyperparameters Found:")
    for k, v in study.best_params.items():
        print(f"   {k}: {v}")

    # Latikan model final dengan parameter terbaik
    best_model = XGBClassifier(**study.best_params, random_state=42)
    best_model.fit(X, y)

    return best_model, study.best_params


# ==========================================
# OPTION B: HYPERPARAMETER TUNING DENGAN GRIDSEARCHCV
# ==========================================
from sklearn.model_selection import GridSearchCV


def optimize_xgboost_gridsearch(df_train: pd.DataFrame):
    """Melakukan tuning Exhaustive Grid Search pada parameter umum XGBoost."""
    X = df_train[FEATURE_COLUMNS]
    y = df_train["risk_level"]

    param_grid = {
        "n_estimators": [150, 250, 350],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
    }

    base_model = XGBClassifier(
        random_state=42, eval_metric="mlogloss", n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        verbose=1,
        n_jobs=-1,
    )

    grid_search.fit(X, y)

    print(f" Best F1-Macro Score: {grid_search.best_score_:.4f}")
    print(" Best Parameters:", grid_search.best_params_)

    return grid_search.best_estimator_, grid_search.best_params_


# ==========================================
# INTEGRASI KE FUNGSI STREAMLIT @st.cache_resource
# ==========================================
# Di dalam aplikasi Streamlit, ganti fungsi train_high_precision_model() menjadi:


@st.cache_resource
def train_tuned_high_precision_model():
    # 1. Generate Dataset Sintetis
    # (Menggunakan logika dataset sintetik dari kode utama Anda)
    _, train_df = train_high_precision_model()  # ambil dataframe awal

    # 2. Jalankan Tuning dengan Optuna (30 Iterasi)
    tuned_model, best_params = optimize_xgboost_optuna(
        train_df, n_trials=30
    )

    return tuned_model, train_df
