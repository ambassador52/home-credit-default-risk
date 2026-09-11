
from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform, loguniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier


# ============================================================
# SABİTLER
# ============================================================

RANDOM_STATE = 42
EXPECTED_SHAPE = (307_511, 122)
MODEL_COMPARISON_THRESHOLD = 0.10
THRESHOLDS = [0.05, 0.10, 0.50]

RATIO_COLUMNS = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_GOODS_RATIO",
    "EMPLOYED_BIRTH_RATIO",
]


# ============================================================
# ÖZELLİK MÜHENDİSLİĞİ
# ============================================================

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Öğrenme gerektirmeyen satır bazlı dönüşümler.

    - DAYS_EMPLOYED = 365243 -> NaN
    - Dört oranı güvenli biçimde üretir.
    """

    def __init__(self, add_ratios: bool = True):
        self.add_ratios = add_ratios

    def fit(self, X, y=None):
        return self

    @staticmethod
    def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        num = pd.to_numeric(numerator, errors="coerce")
        den = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)

        result = num / den
        return result.replace([np.inf, -np.inf], np.nan)

    def transform(self, X):
        X = X.copy()

        # Home Credit verisindeki özel değer:
        # gerçek çalışma süresi gibi kullanılmaz.
        X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].replace(365243, np.nan)

        if self.add_ratios:
            X["CREDIT_INCOME_RATIO"] = self.safe_ratio(
                X["AMT_CREDIT"], X["AMT_INCOME_TOTAL"]
            )
            X["ANNUITY_INCOME_RATIO"] = self.safe_ratio(
                X["AMT_ANNUITY"], X["AMT_INCOME_TOTAL"]
            )
            X["CREDIT_GOODS_RATIO"] = self.safe_ratio(
                X["AMT_CREDIT"], X["AMT_GOODS_PRICE"]
            )

            # Bu oran toplam çalışma hayatı değildir.
            # Mevcut işte geçen süreyi yaşa oranlar.
            X["EMPLOYED_BIRTH_RATIO"] = self.safe_ratio(
                X["DAYS_EMPLOYED"], X["DAYS_BIRTH"]
            )

        return X


# ============================================================
# PIPELINE
# ============================================================

def make_preprocessor(model_family: str) -> ColumnTransformer:
    """
    Öğrenilen tüm ön işlemler Pipeline içinde tutulur.

    Böylece imputer / encoder / scaler değerleri yalnızca
    ilgili eğitim katında öğrenilir.
    """

    if model_family == "linear":
        numeric_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipe = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="Bilinmiyor"),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True,
                    ),
                ),
            ]
        )

        sparse_threshold = 1.0

    else:
        numeric_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        categorical_pipe = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="Bilinmiyor"),
                ),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )

        sparse_threshold = 0.0

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, selector(dtype_include=np.number)),
            (
                "cat",
                categorical_pipe,
                selector(dtype_include=["object", "category"]),
            ),
        ],
        remainder="drop",
        sparse_threshold=sparse_threshold,
        verbose_feature_names_out=True,
    )


def make_pipeline(
    model_name: str,
    add_ratios: bool = True,
    model_jobs: int = 4,
) -> Pipeline:

    if model_name == "Dummy":
        model = DummyClassifier(
            strategy="prior",
            random_state=RANDOM_STATE,
        )
        family = "tree"

    elif model_name == "Logistic Regression":
        model = LogisticRegression(
            max_iter=1200,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
        family = "linear"

    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=14,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=model_jobs,
            random_state=RANDOM_STATE,
        )
        family = "tree"

    elif model_name == "XGBoost":
        model = XGBClassifier(
            n_estimators=220,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=5,
            subsample=0.90,
            colsample_bytree=0.90,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=model_jobs,
            random_state=RANDOM_STATE,
        )
        family = "tree"

    elif model_name == "LightGBM":
        model = LGBMClassifier(
            n_estimators=220,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.90,
            subsample_freq=1,
            colsample_bytree=0.90,
            n_jobs=model_jobs,
            verbosity=-1,
            random_state=RANDOM_STATE,
        )
        family = "tree"

    else:
        raise ValueError(f"Bilinmeyen model: {model_name}")

    return Pipeline(
        [
            ("features", FeatureEngineer(add_ratios=add_ratios)),
            ("preprocessor", make_preprocessor(family)),
            ("model", model),
        ]
    )


# ============================================================
# VERİ
# ============================================================

def find_default_data_path() -> Path:
    """
    PyCharm Run / Debug Console / normal Python çalıştırmalarında
    application_train.csv dosyasını yaygın konumlarda arar.
    """

    cwd = Path.cwd()

    candidates = [
        cwd / "application_train.csv",
        cwd / "data" / "raw" / "application_train.csv",
    ]

    if "__file__" in globals():
        script_path = Path(__file__).resolve()
        script_dir = script_path.parent
        project_root = script_dir.parent

        candidates.extend(
            [
                script_dir / "application_train.csv",
                project_root / "application_train.csv",
                project_root / "data" / "raw" / "application_train.csv",
            ]
        )

    unique_candidates = []
    seen = set()

    for candidate in candidates:
        candidate = candidate.resolve()

        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f" - {p}" for p in unique_candidates)

    raise FileNotFoundError(
        "application_train.csv bulunamadı.\n"
        "Aranan konumlar:\n"
        f"{searched}\n"
        "Dosyayı proje klasörüne veya data/raw/ altına koyabilirsiniz."
    )


def load_and_split(data_path: Path):
    print("CSV okunuyor...")

    df = pd.read_csv(data_path)

    if df.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Beklenen veri boyutu {EXPECTED_SHAPE}, bulunan {df.shape}"
        )

    required = {
        "TARGET",
        "SK_ID_CURR",
        "DAYS_EMPLOYED",
        "DAYS_BIRTH",
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Zorunlu sütunlar eksik: {sorted(missing)}"
        )

    y = df["TARGET"].astype(int)
    ids = df["SK_ID_CURR"].copy()

    # TARGET ve kimlik kesinlikle model girdisinden çıkarılır.
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])

    indices = np.arange(len(df))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=y.iloc[temp_idx],
        random_state=RANDOM_STATE,
    )

    return df, X, y, ids, train_idx, val_idx, test_idx


# ============================================================
# METRİKLER
# ============================================================

def metric_row(y_true, probabilities, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "flagged_count": int(predictions.sum()),
    }


def choose_threshold(rows: list[dict]) -> dict:
    """
    Önceden tanımlı eşitlik kuralı:

    1. F1 daha yüksek
    2. eşitse recall daha yüksek
    3. yine eşitse precision daha yüksek
    4. yine eşitse daha yüksek eşik
    """

    return max(
        rows,
        key=lambda row: (
            round(row["f1"], 12),
            row["recall"],
            row["precision"],
            row["threshold"],
        ),
    )


# ============================================================
# DOSYA YARDIMCILARI
# ============================================================

def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def model_comparison_complete(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        df = pd.read_csv(path)
    except Exception:
        return False

    expected = {
        "Dummy",
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "LightGBM",
    }

    return set(df["model"]) == expected


# ============================================================
# 1) TEMEL KONTROLLER
# ============================================================

def save_data_checks(
    df,
    X,
    y,
    ids,
    train_idx,
    val_idx,
    test_idx,
    out_dir: Path,
):

    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)

    overlap = {
        "train_validation": len(train_set & val_set),
        "train_test": len(train_set & test_set),
        "validation_test": len(val_set & test_set),
    }

    if any(overlap.values()):
        raise RuntimeError(
            f"Veri bölümlerinde çakışma bulundu: {overlap}"
        )

    if "TARGET" in X.columns:
        raise RuntimeError(
            "TARGET model girdisinde bulundu."
        )

    if "SK_ID_CURR" in X.columns:
        raise RuntimeError(
            "SK_ID_CURR model girdisinde bulundu."
        )

    split_table = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": "train",
                    "row_index": train_idx,
                    "SK_ID_CURR": ids.iloc[train_idx].to_numpy(),
                    "TARGET": y.iloc[train_idx].to_numpy(),
                }
            ),
            pd.DataFrame(
                {
                    "split": "validation",
                    "row_index": val_idx,
                    "SK_ID_CURR": ids.iloc[val_idx].to_numpy(),
                    "TARGET": y.iloc[val_idx].to_numpy(),
                }
            ),
            pd.DataFrame(
                {
                    "split": "test",
                    "row_index": test_idx,
                    "SK_ID_CURR": ids.iloc[test_idx].to_numpy(),
                    "TARGET": y.iloc[test_idx].to_numpy(),
                }
            ),
        ],
        ignore_index=True,
    )

    split_table.to_csv(
        out_dir / "tables" / "split_ids.csv",
        index=False,
    )

    summary = {
        "shape": list(df.shape),
        "target_counts": {
            str(k): int(v)
            for k, v in
            df["TARGET"].value_counts().sort_index().items()
        },
        "target_rate": float(df["TARGET"].mean()),
        "days_employed_365243_count": int(
            (df["DAYS_EMPLOYED"] == 365243).sum()
        ),
        "split_sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "split_target_rates": {
            "train": float(y.iloc[train_idx].mean()),
            "validation": float(y.iloc[val_idx].mean()),
            "test": float(y.iloc[test_idx].mean()),
        },
        "overlap": overlap,
        "TARGET_in_model_input": False,
        "SK_ID_CURR_in_model_input": False,
        "random_state": RANDOM_STATE,
        "stratified": True,
    }

    save_json(
        summary,
        out_dir / "data_validation.json",
    )


# ============================================================
# 2) MODEL KARŞILAŞTIRMASI
# ============================================================

def run_model_comparison(
    X,
    y,
    train_idx,
    val_idx,
    out_dir: Path,
):

    result_path = (
        out_dir
        / "tables"
        / "model_comparison_validation.csv"
    )

    if model_comparison_complete(result_path):
        print(
            "Model karşılaştırması daha önce tamamlanmış. "
            "Tekrar eğitilmiyor."
        )
        return pd.read_csv(result_path)

    print("\n=== MODEL KARŞILAŞTIRMASI ===")

    rows = []

    model_names = [
        "Dummy",
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "LightGBM",
    ]

    for model_name in model_names:
        print(
            f"Eğitiliyor: {model_name}",
            flush=True,
        )

        pipe = make_pipeline(
            model_name,
            add_ratios=True,
            model_jobs=4,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(
                X.iloc[train_idx],
                y.iloc[train_idx],
            )

        probabilities = pipe.predict_proba(
            X.iloc[val_idx]
        )[:, 1]

        row = {
            "model": model_name,
            **metric_row(
                y.iloc[val_idx],
                probabilities,
                MODEL_COMPARISON_THRESHOLD,
            ),
        }

        rows.append(row)

        print(
            f"  ROC-AUC={row['roc_auc']:.4f} | "
            f"AP={row['average_precision']:.4f} | "
            f"F1@0.10={row['f1']:.4f}",
            flush=True,
        )

    result = (
        pd.DataFrame(rows)
        .sort_values(
            "roc_auc",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    result.to_csv(
        result_path,
        index=False,
    )

    return result


# ============================================================
# 3) 30 x 5 LIGHTGBM TUNING
# ============================================================

def run_or_load_final_tuning(
    X,
    y,
    train_idx,
    out_dir: Path,
    model_dir: Path,
    search_jobs: int,
):

    tuned_path = (
        model_dir
        / "tuned_lightgbm_train_only.joblib"
    )

    tuning_summary_path = (
        out_dir
        / "lightgbm_tuning_summary.json"
    )

    cv_results_path = (
        out_dir
        / "tables"
        / "lightgbm_random_search_cv.csv"
    )

    # Uzun arama daha önce başarıyla bittiyse tekrar çalıştırma.
    if (
        tuned_path.exists()
        and tuning_summary_path.exists()
        and cv_results_path.exists()
    ):
        print(
            "\n30x5 tuning daha önce tamamlanmış. "
            "Kayıtlı model yükleniyor; arama tekrarlanmıyor."
        )

        return joblib.load(tuned_path)

    print("\n=== NİHAİ LIGHTGBM TUNING ===")
    print(
        "30 aday x 5 fold = toplam 150 fit."
    )
    print(
        "Arama yalnızca TRAIN bölümünde çalışır."
    )
    print(
        f"RandomizedSearchCV n_jobs={search_jobs}; "
        "LightGBM iç n_jobs=1."
    )

    search_pipe = make_pipeline(
        "LightGBM",
        add_ratios=True,
        model_jobs=1,
    )

    param_distributions = {
        "model__n_estimators": randint(120, 501),
        "model__num_leaves": randint(15, 64),
        "model__max_depth": randint(4, 11),
        "model__learning_rate": loguniform(0.02, 0.10),
        "model__min_child_samples": randint(20, 121),
        "model__subsample": uniform(0.70, 0.30),

        # subsample gerçekten etkin olsun.
        "model__subsample_freq": [1],

        "model__colsample_bytree": uniform(0.70, 0.30),
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=search_pipe,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="roc_auc",
        cv=cv,
        n_jobs=search_jobs,
        random_state=RANDOM_STATE,
        refit=True,
        return_train_score=False,
        verbose=2,
    )

    search.fit(
        X.iloc[train_idx],
        y.iloc[train_idx],
    )

    tuned = search.best_estimator_

    # Arama tamamlandıktan sonra normal tahmin için
    # model birkaç çekirdek kullanabilir.
    tuned.set_params(
        model__n_jobs=4
    )

    # Önce uzun işlemin tüm sonuçlarını kaydet.
    cv_results = pd.DataFrame(
        search.cv_results_
    )

    cv_results[
        [
            "rank_test_score",
            "mean_test_score",
            "std_test_score",
            "params",
        ]
    ].sort_values(
        "rank_test_score"
    ).to_csv(
        cv_results_path,
        index=False,
    )

    tuning_summary = {
        "n_iter": 30,
        "cv_folds": 5,
        "total_fits": 150,
        "scoring": "roc_auc",
        "search_data": "training split only",
        "best_cv_roc_auc": float(
            search.best_score_
        ),
        "best_params": search.best_params_,
        "subsample_freq_check": int(
            tuned.named_steps[
                "model"
            ].subsample_freq
        ),
        "random_state": RANDOM_STATE,
    }

    save_json(
        tuning_summary,
        tuning_summary_path,
    )

    # Modeli hemen kaydet.
    joblib.dump(
        tuned,
        tuned_path,
        compress=3,
    )

    print(
        "30x5 tuning tamamlandı ve diske kaydedildi.",
        flush=True,
    )
    print(
        f"En iyi CV ROC-AUC: "
        f"{search.best_score_:.6f}",
        flush=True,
    )

    return tuned


# ============================================================
# 4) VALIDATION EŞİK KARŞILAŞTIRMASI
# ============================================================

def run_threshold_comparison(
    tuned,
    X,
    y,
    val_idx,
    out_dir: Path,
):

    print("\n=== EŞİK KARŞILAŞTIRMASI ===")

    validation_probabilities = (
        tuned.predict_proba(
            X.iloc[val_idx]
        )[:, 1]
    )

    rows = [
        metric_row(
            y.iloc[val_idx],
            validation_probabilities,
            threshold,
        )
        for threshold in THRESHOLDS
    ]

    table = pd.DataFrame(rows)

    table[
        [
            "threshold",
            "precision",
            "recall",
            "f1",
            "TP",
            "FP",
            "TN",
            "FN",
            "flagged_count",
        ]
    ].to_csv(
        out_dir
        / "tables"
        / "threshold_comparison_validation.csv",
        index=False,
    )

    selected = choose_threshold(rows)

    selection_summary = {
        "candidate_thresholds": THRESHOLDS,
        "selection_metric": "F1",
        "tie_rule": (
            "F1 eşitse recall, sonra precision, "
            "sonra daha yüksek eşik"
        ),
        "selected_threshold": float(
            selected["threshold"]
        ),
        "validation_metrics": selected,
        "economic_optimum_claimed": False,
        "roc_auc_changes_with_threshold": False,
    }

    save_json(
        selection_summary,
        out_dir / "threshold_selection.json",
    )

    print(
        f"Seçilen eşik: "
        f"{selected['threshold']:.2f}"
    )
    print(
        f"Validation F1: "
        f"{selected['f1']:.4f}"
    )

    return selected


# ============================================================
# 5) DÖRT ORANIN KATKISI
# ============================================================

def run_ratio_ablation(
    tuned,
    X,
    y,
    train_idx,
    val_idx,
    out_dir: Path,
    model_dir: Path,
):

    result_path = (
        out_dir
        / "tables"
        / "feature_ratio_ablation_validation.csv"
    )

    if result_path.exists():
        print(
            "\nOranlı/oransız karşılaştırma daha önce tamamlanmış. "
            "Tekrar eğitilmiyor."
        )
        return pd.read_csv(result_path)

    print(
        "\n=== DÖRT ORANIN KONTROLLÜ KARŞILAŞTIRMASI ==="
    )

    tuned_probabilities = (
        tuned.predict_proba(
            X.iloc[val_idx]
        )[:, 1]
    )

    with_ratios = {
        "ratios_added": True,
        "roc_auc": float(
            roc_auc_score(
                y.iloc[val_idx],
                tuned_probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y.iloc[val_idx],
                tuned_probabilities,
            )
        ),
    }

    # Aynı LightGBM ayarları, aynı train/validation bölünmesi.
    # Yalnızca dört oran çıkarılır.
    best_model_params = (
        tuned.named_steps[
            "model"
        ].get_params()
    )

    keep_params = [
        "n_estimators",
        "num_leaves",
        "max_depth",
        "learning_rate",
        "min_child_samples",
        "subsample",
        "subsample_freq",
        "colsample_bytree",
        "random_state",
        "verbosity",
        "n_jobs",
    ]

    without_ratios_pipe = make_pipeline(
        "LightGBM",
        add_ratios=False,
        model_jobs=4,
    )

    without_ratios_pipe.named_steps[
        "model"
    ].set_params(
        **{
            key: best_model_params[key]
            for key in keep_params
        }
    )

    without_ratios_pipe.fit(
        X.iloc[train_idx],
        y.iloc[train_idx],
    )

    without_probabilities = (
        without_ratios_pipe.predict_proba(
            X.iloc[val_idx]
        )[:, 1]
    )

    without_ratios = {
        "ratios_added": False,
        "roc_auc": float(
            roc_auc_score(
                y.iloc[val_idx],
                without_probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y.iloc[val_idx],
                without_probabilities,
            )
        ),
    }

    result = pd.DataFrame(
        [
            without_ratios,
            with_ratios,
        ]
    )

    result["roc_auc_difference_vs_without"] = (
        result["roc_auc"]
        - without_ratios["roc_auc"]
    )

    result[
        "average_precision_difference_vs_without"
    ] = (
        result["average_precision"]
        - without_ratios["average_precision"]
    )

    result.to_csv(
        result_path,
        index=False,
    )

    # Uzun tuning tekrar gerektirmese de bu küçük ablation modeli
    # ayrıca saklanır.
    joblib.dump(
        without_ratios_pipe,
        model_dir
        / "ablation_lightgbm_without_ratios.joblib",
        compress=3,
    )

    return result


# ============================================================
# 6) GAIN IMPORTANCE
# ============================================================

def save_gain_importance(
    tuned,
    X,
    train_idx,
    out_dir: Path,
):

    print("\n=== GAIN FEATURE IMPORTANCE ===")

    feature_engineered_sample = (
        tuned.named_steps[
            "features"
        ].transform(
            X.iloc[train_idx].head(10)
        )
    )

    ratio_presence = {
        ratio: bool(
            ratio
            in feature_engineered_sample.columns
        )
        for ratio in RATIO_COLUMNS
    }

    if not all(
        ratio_presence.values()
    ):
        raise RuntimeError(
            "Dört oran model girdisinde bulunamadı: "
            f"{ratio_presence}"
        )

    transformed_feature_names = (
        tuned.named_steps[
            "preprocessor"
        ].get_feature_names_out()
    )

    gains = (
        tuned.named_steps[
            "model"
        ]
        .booster_
        .feature_importance(
            importance_type="gain"
        )
    )

    if (
        len(transformed_feature_names)
        != len(gains)
    ):
        raise RuntimeError(
            "Dönüştürülmüş sütun adları ile "
            "LightGBM importance uzunluğu uyuşmuyor."
        )

    importance = pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "gain": gains,
        }
    ).sort_values(
        "gain",
        ascending=False,
    )

    importance.head(20).to_csv(
        out_dir
        / "tables"
        / "lightgbm_gain_importance_top20.csv",
        index=False,
    )

    top10 = (
        importance
        .head(10)
        .sort_values(
            "gain",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(11, 6.5)
    )

    plt.barh(
        top10["feature"],
        top10["gain"],
    )

    plt.xlabel(
        "Gain importance"
    )
    plt.ylabel(
        "Dönüştürülmüş özellik"
    )
    plt.title(
        "LightGBM - Gain Temelli İlk 10 Özellik"
    )
    plt.tight_layout()

    plt.savefig(
        out_dir
        / "figures"
        / "lightgbm_gain_top10.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close()

    save_json(
        {
            "ratio_features_present_before_preprocessing":
                ratio_presence,
            "importance_type": "gain",
            "interpretation_warning": (
                "Feature importance nedensellik veya "
                "riskin artış/azalış yönünü göstermez."
            ),
        },
        out_dir / "feature_checks.json",
    )

    return ratio_presence


# ============================================================
# 7) FINAL PIPELINE + TEK SEFERLİK TEST
# ============================================================

def save_final_pipeline_and_run_test_once(
    tuned,
    selected_threshold,
    ratio_presence,
    X,
    y,
    test_idx,
    out_dir: Path,
    model_dir: Path,
):

    final_pipeline_path = (
        model_dir
        / "home_credit_final_pipeline.joblib"
    )

    final_test_path = (
        out_dir
        / "final_test_results.json"
    )

    # Pipeline önce kaydedilir.
    if not final_pipeline_path.exists():
        joblib.dump(
            tuned,
            final_pipeline_path,
            compress=3,
        )

    # Kaydedilmiş pipeline yüklenir.
    loaded_pipeline = joblib.load(
        final_pipeline_path
    )

    # Test daha önce ölçüldüyse tekrar hesaplanmaz.
    if final_test_path.exists():
        print(
            "\nFINAL TEST sonucu zaten mevcut. "
            "Test tekrar değerlendirilmedi."
        )

        with open(
            final_test_path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    print("\n=== FINAL TEST - TEK SEFER ===")
    print(
        "Model, hiperparametre ve eşik artık sabit."
    )
    print(
        "Test sonucu seçimleri değiştirmek için kullanılmayacak."
    )

    test_probabilities = (
        loaded_pipeline.predict_proba(
            X.iloc[test_idx]
        )[:, 1]
    )

    test_metrics = metric_row(
        y.iloc[test_idx],
        test_probabilities,
        selected_threshold["threshold"],
    )

    final_result = {
        "selected_model": "Tuned LightGBM",
        "model_fit_data": "training split only",
        "selected_threshold": float(
            selected_threshold["threshold"]
        ),
        "threshold_selected_on": "validation",
        "test_used_for_model_or_threshold_selection": False,
        "ratio_features_present": ratio_presence,
        "test_metrics": test_metrics,
        "note": (
            "Bu test sonucu görüldükten sonra model, "
            "hiperparametre veya eşik yeniden seçilmemelidir."
        ),
    }

    # Test ölçümünden hemen sonra kaydet.
    save_json(
        final_result,
        final_test_path,
    )

    print(
        "Final test sonucu kaydedildi."
    )

    return final_result


# ============================================================
# 8) SON KONTROLLER
# ============================================================

def run_final_checks(
    X,
    train_idx,
    val_idx,
    test_idx,
    model_dir: Path,
    out_dir: Path,
):

    final_pipeline_path = (
        model_dir
        / "home_credit_final_pipeline.joblib"
    )

    loaded = joblib.load(
        final_pipeline_path
    )

    smoke_probabilities = (
        loaded.predict_proba(
            X.iloc[test_idx[:5]]
        )[:, 1]
    )

    checks = {
        "train_validation_overlap": int(
            len(
                set(train_idx)
                & set(val_idx)
            )
        ),
        "train_test_overlap": int(
            len(
                set(train_idx)
                & set(test_idx)
            )
        ),
        "validation_test_overlap": int(
            len(
                set(val_idx)
                & set(test_idx)
            )
        ),
        "TARGET_in_model_input": bool(
            "TARGET" in X.columns
        ),
        "SK_ID_CURR_in_model_input": bool(
            "SK_ID_CURR" in X.columns
        ),
        "saved_pipeline_loaded": True,
        "prediction_count": int(
            len(smoke_probabilities)
        ),
        "prediction_min": float(
            smoke_probabilities.min()
        ),
        "prediction_max": float(
            smoke_probabilities.max()
        ),
    }

    save_json(
        checks,
        out_dir / "final_checks.json",
    )

    failed = []

    if (
        checks[
            "train_validation_overlap"
        ]
        != 0
    ):
        failed.append(
            "train-validation overlap"
        )

    if (
        checks[
            "train_test_overlap"
        ]
        != 0
    ):
        failed.append(
            "train-test overlap"
        )

    if (
        checks[
            "validation_test_overlap"
        ]
        != 0
    ):
        failed.append(
            "validation-test overlap"
        )

    if checks["TARGET_in_model_input"]:
        failed.append(
            "TARGET inputta"
        )

    if checks["SK_ID_CURR_in_model_input"]:
        failed.append(
            "SK_ID_CURR inputta"
        )

    if failed:
        raise RuntimeError(
            "Final kontroller başarısız: "
            + ", ".join(failed)
        )

    print(
        "\nFinal kontroller başarılı."
    )


# ============================================================
# 9) OTOMATİK SONUÇ ÖZETİ
# ============================================================

def save_run_summary(
    model_comparison,
    ratio_ablation,
    selected_threshold,
    final_result,
    out_dir: Path,
):

    best_base_model = (
        model_comparison
        .sort_values(
            "roc_auc",
            ascending=False,
        )
        .iloc[0]
    )

    ratio_without = (
        ratio_ablation[
            ratio_ablation[
                "ratios_added"
            ]
            == False
        ]
        .iloc[0]
    )

    ratio_with = (
        ratio_ablation[
            ratio_ablation[
                "ratios_added"
            ]
            == True
        ]
        .iloc[0]
    )

    ratio_auc_delta = (
        ratio_with["roc_auc"]
        - ratio_without["roc_auc"]
    )

    text = f"""HOME CREDIT FINAL RUN ÖZETİ
==============================

Model karşılaştırması en yüksek validation ROC-AUC:
- Model: {best_base_model['model']}
- ROC-AUC: {best_base_model['roc_auc']:.6f}
- Average Precision: {best_base_model['average_precision']:.6f}

Seçilen optimize model:
- Tuned LightGBM
- Eşik: {selected_threshold['threshold']:.2f}
- Validation F1: {selected_threshold['f1']:.6f}
- Validation precision: {selected_threshold['precision']:.6f}
- Validation recall: {selected_threshold['recall']:.6f}

Dört oran kontrollü validation karşılaştırması:
- Oransız ROC-AUC: {ratio_without['roc_auc']:.6f}
- Oranlı ROC-AUC: {ratio_with['roc_auc']:.6f}
- Fark: {ratio_auc_delta:+.6f}

Final test:
- ROC-AUC: {final_result['test_metrics']['roc_auc']:.6f}
- Average Precision: {final_result['test_metrics']['average_precision']:.6f}
- Precision: {final_result['test_metrics']['precision']:.6f}
- Recall: {final_result['test_metrics']['recall']:.6f}
- F1: {final_result['test_metrics']['f1']:.6f}
- TP: {final_result['test_metrics']['TP']}
- FP: {final_result['test_metrics']['FP']}
- TN: {final_result['test_metrics']['TN']}
- FN: {final_result['test_metrics']['FN']}
- Riskli işaretlenen: {final_result['test_metrics']['flagged_count']}

Yorum sınırları:
- ROC-AUC doğruluk yüzdesi değildir.
- Eşik finansal maliyet verisi olmadan ekonomik optimum değildir.
- Gain importance nedensellik veya risk yönü değildir.
- Çalışma/yaş oranı toplam çalışma hayatını değil mevcut işte geçen süre / yaşı temsil eder.
- Önceki EDA tüm veri üzerinde yapıldığı için testin proje boyunca hiç görülmediği iddia edilmez.
"""

    path = (
        out_dir
        / "FINAL_RUN_OZETI.txt"
    )

    path.write_text(
        text,
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=(
            "application_train.csv yolu. "
            "Boş bırakılırsa otomatik aranır."
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("final_outputs"),
        help=(
            "Final tablo/grafik/JSON klasörü."
        ),
    )

    parser.add_argument(
        "--models",
        type=Path,
        default=Path("final_models"),
        help=(
            "Kaydedilmiş model klasörü."
        ),
    )

    parser.add_argument(
        "--search-jobs",
        type=int,
        default=2,
        help=(
            "30x5 RandomizedSearchCV paralel iş sayısı. "
            "Laptop için 2 güvenli varsayılandır."
        ),
    )

    args, unknown_args = (
        parser.parse_known_args()
    )

    if unknown_args:
        print(
            "IDE tarafından eklenen argümanlar "
            f"yok sayıldı: {unknown_args}"
        )

    if args.data is None:
        args.data = (
            find_default_data_path()
        )

    print(
        f"Okunan veri dosyası: "
        f"{args.data.resolve()}"
    )

    out_dir = args.out
    model_dir = args.models

    (
        out_dir
        / "tables"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        out_dir
        / "figures"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------
    # Veri ve split
    # --------------------------

    (
        df,
        X,
        y,
        ids,
        train_idx,
        val_idx,
        test_idx,
    ) = load_and_split(
        args.data
    )

    save_data_checks(
        df,
        X,
        y,
        ids,
        train_idx,
        val_idx,
        test_idx,
        out_dir,
    )

    print(
        f"Split: train={len(train_idx)}, "
        f"validation={len(val_idx)}, "
        f"test={len(test_idx)}"
    )

    # --------------------------
    # Model karşılaştırması
    # --------------------------

    comparison = (
        run_model_comparison(
            X,
            y,
            train_idx,
            val_idx,
            out_dir,
        )
    )

    # --------------------------
    # Nihai 30 x 5 tuning
    # --------------------------

    tuned = (
        run_or_load_final_tuning(
            X,
            y,
            train_idx,
            out_dir,
            model_dir,
            search_jobs=args.search_jobs,
        )
    )

    # --------------------------
    # Validation eşik seçimi
    # --------------------------

    selected_threshold = (
        run_threshold_comparison(
            tuned,
            X,
            y,
            val_idx,
            out_dir,
        )
    )

    # --------------------------
    # Oran ablation
    # --------------------------

    ratio_ablation = (
        run_ratio_ablation(
            tuned,
            X,
            y,
            train_idx,
            val_idx,
            out_dir,
            model_dir,
        )
    )

    # --------------------------
    # Feature importance
    # --------------------------

    ratio_presence = (
        save_gain_importance(
            tuned,
            X,
            train_idx,
            out_dir,
        )
    )

    # --------------------------
    # Pipeline kaydı + test once
    # --------------------------

    final_result = (
        save_final_pipeline_and_run_test_once(
            tuned,
            selected_threshold,
            ratio_presence,
            X,
            y,
            test_idx,
            out_dir,
            model_dir,
        )
    )

    # --------------------------
    # Final kontroller
    # --------------------------

    run_final_checks(
        X,
        train_idx,
        val_idx,
        test_idx,
        model_dir,
        out_dir,
    )

    # --------------------------
    # İnsan okunur özet
    # --------------------------

    save_run_summary(
        comparison,
        ratio_ablation,
        selected_threshold,
        final_result,
        out_dir,
    )

    print(
        "\n===================================="
    )
    print(
        "FINAL ÇALIŞTIRMA TAMAMLANDI"
    )
    print(
        "===================================="
    )
    print(
        f"Çıktılar: {out_dir.resolve()}"
    )
    print(
        f"Modeller: {model_dir.resolve()}"
    )
    print(
        f"Seçilen eşik: "
        f"{selected_threshold['threshold']:.2f}"
    )
    print(
        f"Final test ROC-AUC: "
        f"{final_result['test_metrics']['roc_auc']:.6f}"
    )
    print(
        f"Final test AP: "
        f"{final_result['test_metrics']['average_precision']:.6f}"
    )
    print(
        "Programı yanlışlıkla yeniden çalıştırırsanız "
        "tamamlanmış 30x5 tuning ve final test tekrar yapılmaz."
    )


if __name__ == "__main__":
    main()
