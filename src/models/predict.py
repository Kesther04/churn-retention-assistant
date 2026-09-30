from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

# Suppress unpickling version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

MODEL_DIR = Path(__file__).resolve().parent
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
MODEL_PATH = MODEL_DIR / "best_model_logistic_regression.joblib"

DEFAULT_CUSTOMER_FEATURES = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 70.0,
}


def _deep_unwrap(val):
    """Recursively extracts scalar primitives out of nested dicts or lists."""
    while isinstance(val, (dict, list, tuple)):
        if isinstance(val, dict):
            if not val:
                return None
            val = next(iter(val.values()))
        elif isinstance(val, (list, tuple)):
            if not val:
                return None
            val = val[0]
    return val


def _extract_numeric(val, default: float = 0.0) -> float:
    """Safely coerces unpacked values to float."""
    unwrapped = _deep_unwrap(val)
    try:
        if unwrapped is None:
            return float(default)
        return float(unwrapped)
    except (ValueError, TypeError):
        return float(default)


def load_artifacts():
    if not PREPROCESSOR_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Artifacts not found in {MODEL_DIR}. Please run training first."
        )

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocessor, model


def predict_churn_risk(customer_data: dict) -> dict:
    """Predicts churn probability and top risk factors for a single customer."""
    preprocessor, model = load_artifacts()

    # 1. Flatten all nested values from incoming dict
    cleaned_input = {}
    if isinstance(customer_data, dict):
        for k, v in customer_data.items():
            cleaned_input[k] = _deep_unwrap(v)

    # 2. Populate complete dataset schema
    full_customer_data = DEFAULT_CUSTOMER_FEATURES.copy()
    full_customer_data.update({k: v for k, v in cleaned_input.items() if v is not None})

    # 3. Clean numeric values
    tenure_val = _extract_numeric(full_customer_data.get("tenure"), default=1.0)
    monthly_val = _extract_numeric(full_customer_data.get("MonthlyCharges"), default=70.0)

    # Calculate TotalCharges cleanly without type errors
    raw_total = cleaned_input.get("TotalCharges")
    if raw_total is not None:
        total_val = _extract_numeric(raw_total, default=tenure_val * monthly_val)
    else:
        total_val = tenure_val * monthly_val

    full_customer_data["tenure"] = tenure_val
    full_customer_data["MonthlyCharges"] = monthly_val
    full_customer_data["TotalCharges"] = total_val
    full_customer_data["SeniorCitizen"] = int(_extract_numeric(full_customer_data.get("SeniorCitizen"), default=0))

    # 4. Predict
    df_customer = pd.DataFrame([full_customer_data])
    X_transformed = preprocessor.transform(df_customer)
    churn_prob = float(model.predict_proba(X_transformed)[0][1])

    if churn_prob < 0.30:
        risk_level = "Low"
    elif churn_prob < 0.60:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # 5. Extract top risk factors
    risk_factors = []
    if full_customer_data.get("Contract") == "Month-to-month":
        risk_factors.append("Month-to-month contract (High churn volatility)")

    if tenure_val <= 12:
        risk_factors.append(f"Low tenure ({int(tenure_val)} months - Early lifecycle)")

    if full_customer_data.get("InternetService") == "Fiber optic":
        risk_factors.append("Fiber optic service (Statistically higher churn rate)")

    if full_customer_data.get("PaymentMethod") == "Electronic check":
        risk_factors.append("Electronic check payment method")

    if (
        full_customer_data.get("OnlineSecurity") == "No"
        and full_customer_data.get("InternetService") != "No"
    ):
        risk_factors.append("Missing Online Security add-on")

    if (
        full_customer_data.get("TechSupport") == "No"
        and full_customer_data.get("InternetService") != "No"
    ):
        risk_factors.append("Missing Tech Support add-on")

    if monthly_val > 70:
        risk_factors.append(f"High monthly charges (${monthly_val:.2f}/mo)")

    top_risk_factors = (
        risk_factors[:3]
        if risk_factors
        else ["No major high-risk indicators detected."]
    )

    return {
        "churn_probability": round(churn_prob, 4),
        "risk_level": risk_level,
        "top_risk_factors": top_risk_factors,
    }