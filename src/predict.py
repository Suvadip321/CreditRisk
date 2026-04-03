import pandas as pd
import joblib
import numpy as np

from src.config import settings
from src.logger import get_logger

logger = get_logger("predict")

_model = None
_model_loaded = False


def load_model():
    """Load the model, try latest version first, then default."""
    global _model, _model_loaded
    
    model_dir = settings.full_model_dir
    
    latest_pointer = model_dir / "latest_model.txt"
    if latest_pointer.exists():
        latest_model_name = latest_pointer.read_text().strip()
        latest_model_path = model_dir / latest_model_name
        if latest_model_path.exists():
            logger.info(f"Loading latest model: {latest_model_name}")
            _model = joblib.load(latest_model_path)
            _model_loaded = True
            return _model
    
    if settings.full_model_path.exists():
        logger.info(f"Loading default model: {settings.MODEL_NAME}")
        _model = joblib.load(settings.full_model_path)
        _model_loaded = True
        return _model
    
    raise FileNotFoundError(
        f"No model found. Train the model first.\n"
        f"Checked: {latest_pointer}, {settings.full_model_path}"
    )


def is_model_loaded() -> bool:
    """Check if model is loaded successfully."""
    global _model_loaded
    if not _model_loaded:
        try:
            load_model()
        except FileNotFoundError:
            return False
    return _model_loaded


def get_model():
    """Get the loaded model, loading it if necessary."""
    global _model
    if _model is None:
        load_model()
    return _model


TEMPLATE_PATH = settings.full_template_path
if not TEMPLATE_PATH.exists():
    raise FileNotFoundError(f"Template file not found at {TEMPLATE_PATH}")
df_template = pd.read_csv(TEMPLATE_PATH)


def predict(applicant_data: dict) -> tuple:
    """
    Predicts credit risk for a single applicant.
    Uses custom threshold from config (default 0.40 for conservative risk assessment).
    """
    model = get_model()
    
    template_cols = df_template.drop(columns=["loan_status"]).columns.tolist()
    
    row_data = {col: applicant_data.get(col, np.nan) for col in template_cols}
    input_df = pd.DataFrame([row_data])

    probability = model.predict_proba(input_df)[0][1]
    prediction_class = 1 if probability >= settings.PREDICTION_THRESHOLD else 0

    logger.debug(f"Prediction: class={prediction_class}, prob={probability:.4f}, threshold={settings.PREDICTION_THRESHOLD}")
    return prediction_class, probability


if __name__ == "__main__":
    print("Running prediction for sample applicant...")

    applicant_data = {
        "loan_amnt": 15000.0,
        "term": "36 months",
        "installment": 450.00,
        "purpose": "debt_consolidation",
        "issue_d": "Dec-2023",  
        "emp_length": "10+ years",
        "home_ownership": "MORTGAGE",
        "annual_inc": 85000.0,
        "verification_status": "Source Verified",
        "zip_code": "940xx",    
        "addr_state": "CA",
        "dti": 18.5,         
        "revol_bal": 14000.0,
        "revol_util": 55.0,   
        "earliest_cr_line": "Jan-2005",
        "fico_range_low": 700.0,
        "fico_range_high": 704.0,
        "inq_last_6mths": 1.0,
        "open_acc": 12.0,      
        "total_acc": 24.0,      
        "mort_acc": 1.0,       
        "delinq_2yrs": 0.0,
        "pub_rec": 0.0,     
        "pub_rec_bankruptcies": 0.0,
        "mths_since_last_delinq": 999.0
    }

    try:
        is_default, prob = predict(applicant_data)
        print(f"\nDefault Probability: {prob:.2%}")
        print(f"Decision: {'REJECT (High Risk)' if is_default == 1 else 'APPROVE (Low Risk)'}")
    except Exception as e:
        print(f"\nError: {e}")
