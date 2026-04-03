import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, classification_report, confusion_matrix
)

from src.config import settings
from src.logger import setup_logger
from src.preprocess import (
    MissingValueNormalizer,
    TermParser,
    EmploymentLengthParser,
    ZipCodeParser,
    NumericCoercer,
    DateParser,
    DomainFeatureEngineer,
    DomainZeroImputer,
    ReferenceTimeImputer,
    PostEventFeatureDropper,
    ExplicitColumnDropper,
    ConstantColumnDropper,
    HighMissingnessDropper,
    HighCardinalityDropper,
    DateColumnDropper
)

logger = setup_logger("training", settings.LOG_LEVEL, settings.full_log_path)

# post-approval features
POST_EVENT_COLS = [
    # repayment & recovery
    "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries",
    "collection_recovery_fee",
    "last_pymnt_amnt",

    # post-issue dates
    "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d",

    # updated credit scores
    "last_fico_range_high", "last_fico_range_low",

    # hardship
    "hardship_flag", "hardship_type", "hardship_reason",
    "hardship_status", "deferral_term", "hardship_amount",
    "hardship_start_date", "hardship_end_date",
    "payment_plan_start_date", "hardship_length",
    "hardship_dpd", "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount",
    "hardship_last_payment_amount",

    # settlement
    "debt_settlement_flag", "debt_settlement_flag_date",
    "settlement_status", "settlement_date",
    "settlement_amount", "settlement_percentage",
    "settlement_term"
]

# explicit governance + proxy removal
EXPLICIT_DROP_COLS = [
    "id", "member_id", "url",
    "desc", "title", "grade",
    "sub_grade", "int_rate"
]

# columns where missing means "never happened" (not missing data)
REFERENCE_TIME_COLS = [
    "mths_since_last_delinq",
    "mths_since_last_record",
    "mths_since_last_major_derog",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_revol_delinq"
]


def save_model_with_metadata(model, metrics: dict, best_params: dict, model_dir: Path):
    """Save model with versioning and metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_filename = f"credit_risk_model_{timestamp}.pkl"
    model_path = model_dir / model_filename
    
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved: {model_path}")
    
    metadata = {
        "model_file": model_filename,
        "trained_at": datetime.now().isoformat(),
        "metrics": metrics,
        "best_params": {k: str(v) for k, v in best_params.items()},
        "data_path": str(settings.full_data_path),
        "train_test_split_date": settings.TRAIN_TEST_SPLIT_DATE
    }
    
    metadata_path = model_dir / f"credit_risk_model_{timestamp}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved: {metadata_path}")
    
    latest_pointer = model_dir / "latest_model.txt"
    with open(latest_pointer, "w") as f:
        f.write(model_filename)
    logger.info(f"Latest model pointer updated: {model_filename}")

    default_path = model_dir / settings.MODEL_NAME
    joblib.dump(model, default_path)
    logger.info(f"Default model updated: {default_path}")
    
    return model_path


def main():
    logger.info("Starting model training...")
    
    logger.info(f"Loading data from: {settings.full_data_path}")
    df = pd.read_csv(settings.full_data_path)
    logger.info(f"Loaded {len(df)} records")
    
    X = df.drop(columns=[settings.TARGET_COL])
    y = (df[settings.TARGET_COL] == "Charged Off").astype(int)
    
    X[settings.DATE_COL] = pd.to_datetime(X[settings.DATE_COL])
    X = X.sort_values(by=settings.DATE_COL)
    y = y.loc[X.index]
    
    # time based split to prevent data leakage
    train_mask = X[settings.DATE_COL] < settings.TRAIN_TEST_SPLIT_DATE
    test_mask = X[settings.DATE_COL] >= settings.TRAIN_TEST_SPLIT_DATE
    
    X_train = X.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()
    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]
    
    logger.info(f"Train shape: {X_train.shape}")
    logger.info(f"Test shape: {X_test.shape}")
    
    # class weight as the data is imbalanced
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"Scale pos weight: {scale_weight:.2f}")
    
    data_cleaning = Pipeline(steps=[
        ("missing_normalizer", MissingValueNormalizer()),
        ("term_parser", TermParser()),
        ("emp_length_parser", EmploymentLengthParser()),
        ("zip_parser", ZipCodeParser()),
        ("numeric_coercer", NumericCoercer()),
        ("date_parser", DateParser([settings.DATE_COL, "earliest_cr_line"])),
        ("domain_features", DomainFeatureEngineer()),
        ("domain_imputer", DomainZeroImputer()),
        ("reference_time_imputer", ReferenceTimeImputer(REFERENCE_TIME_COLS, fill_value=999)),
        ("post_event_dropper", PostEventFeatureDropper(POST_EVENT_COLS)),
        ("explicit_dropper", ExplicitColumnDropper(EXPLICIT_DROP_COLS)),
        ("constant_dropper", ConstantColumnDropper()),
        ("missingness_dropper", HighMissingnessDropper(0.75)),
        ("cardinality_dropper", HighCardinalityDropper(100)),
        ("date_dropper", DateColumnDropper())
    ])
    
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    ml_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, make_column_selector(dtype_include=np.number)),
            ("cat", categorical_pipe, make_column_selector(dtype_include=object))
        ],
        remainder="drop"
    )
    
    pipeline = Pipeline(steps=[
        ("data_cleaning", data_cleaning),
        ("ml_preprocessing", ml_preprocessor),
        ("model", XGBClassifier(
            n_estimators=300,
            random_state=settings.RANDOM_STATE,
            n_jobs=-1,
            scale_pos_weight=scale_weight,
            eval_metric="auc"
        ))
    ])
    
    # maintain time based order during cross-validation
    tscv = TimeSeriesSplit(n_splits=3)

    param_distributions = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 6, 10],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0]
    }
    
    hpt = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="roc_auc",
        cv=tscv,
        n_jobs=-1,
        verbose=2,
        random_state=settings.RANDOM_STATE
    )
    
    logger.info("Running hyperparameter tuning...")
    hpt.fit(X_train, y_train)
    
    best_model = hpt.best_estimator_
    
    logger.info(f"Best Parameters: {hpt.best_params_}")
    
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred))
    }
    
    logger.info("Final Evaluation Results")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    save_model_with_metadata(
        model=best_model,
        metrics=metrics,
        best_params=hpt.best_params_,
        model_dir=settings.full_model_dir
    )
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
