# src/predict.py

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Define paths relative to src/
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "src" / "models"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
# Adjust filename if you chose random_forest instead
MODEL_PATH = MODEL_DIR / "best_model_logistic_regression.joblib"


def load_artifacts():
    """Loads saved preprocessor and model artifacts."""
    if not PREPROCESSOR_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Artifacts not found in {MODEL_DIR}. Please run training first."
        )

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocessor, model


def predict_churn_risk(customer_data: dict) -> dict:
    """Predicts churn probability and top risk factors for a single customer.

    Args:
        customer_data (dict): Dictionary containing single customer attributes.

    Returns:
        dict: {
            'churn_probability': float,
            'risk_level': str ('Low', 'Medium', 'High'),
            'top_risk_factors': list of str
        }
    """
    preprocessor, model = load_artifacts()

    # Convert single dictionary to DataFrame
    df_customer = pd.DataFrame([customer_data])

    # Transform features using trained preprocessor
    X_transformed = preprocessor.transform(df_customer)

    # 1. Calculate Churn Probability (class index 1 is 'Yes' / Churn)
    churn_prob = float(model.predict_proba(X_transformed)[0][1])

    # Assign qualitative risk level
    if churn_prob < 0.30:
        risk_level = "Low"
    elif churn_prob < 0.60:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # 2. Extract Top 3 Risk Factors dynamically
    risk_factors = []

    # Domain rule checks based on key features
    if customer_data.get("Contract") == "Month-to-month":
        risk_factors.append("Month-to-month contract (High churn volatility)")

    if customer_data.get("tenure", 0) <= 12:
        risk_factors.append(
            f"Low tenure ({customer_data.get('tenure')} months - Early lifecycle)"
        )

    if customer_data.get("InternetService") == "Fiber optic":
        risk_factors.append("Fiber optic service (Statistically higher churn rate)")

    if customer_data.get("PaymentMethod") == "Electronic check":
        risk_factors.append("Electronic check payment method")

    if (
        customer_data.get("OnlineSecurity") == "No"
        and customer_data.get("InternetService") != "No"
    ):
        risk_factors.append("Missing Online Security add-on")

    if (
        customer_data.get("TechSupport") == "No"
        and customer_data.get("InternetService") != "No"
    ):
        risk_factors.append("Missing Tech Support add-on")

    if customer_data.get("MonthlyCharges", 0) > 70:
        risk_factors.append(
            f"High monthly charges (${customer_data.get('MonthlyCharges')}/mo)"
        )

    # Return top 3 risk factors (or default message if low risk)
    top_risk_factors = risk_factors[:3] if risk_factors else ["No major high-risk indicators detected."]

    return {
        "churn_probability": round(churn_prob, 4),
        "risk_level": risk_level,
        "top_risk_factors": top_risk_factors,
    }