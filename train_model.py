"""
ML Pipeline for Loan Eligibility Prediction.
Trains on full_data[:-200], saves model artifacts for the Streamlit dashboard.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# ── Load & Split ──────────────────────────────────────────────────────────────
df_full = pd.read_csv("data/input.csv")
df = df_full.iloc[:-200].copy()  # training data
holdout = df_full.iloc[-200:].copy()  # never used for training/evaluation

print(f"Full dataset: {len(df_full)} rows")
print(f"Training set: {len(df)} rows")
print(f"Hold-out set: {len(holdout)} rows (reserved, not used)")

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(data, fit_encoders=None):
    """Clean and encode features. Returns X, y (if target present), encoders."""
    df = data.copy()

    # Drop Loan_ID
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)

    # Fill missing values
    df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
    df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
    df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
    df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())
    df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].median())

    # Feature engineering
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["TotalIncome_Log"] = np.log1p(df["TotalIncome"])
    df["LoanAmount_Log"] = np.log1p(df["LoanAmount"])
    df["EMI"] = df["LoanAmount"] / (df["Loan_Amount_Term"] + 1)
    df["Balance_Income"] = df["TotalIncome"] - (df["EMI"] * 1000)

    # Encode categoricals
    cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
    encoders = fit_encoders or {}

    for col in cat_cols:
        df[col] = df[col].astype(str)
        if col not in encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].map(lambda x, _le=le: _le.transform([x])[0] if x in _le.classes_ else -1)

    # Target encoding
    y = None
    if "Loan_Status" in df.columns:
        df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
        y = df["Loan_Status"]
        df = df.drop("Loan_Status", axis=1)

    return df, y, encoders


X_train, y_train, encoders = preprocess(df)
feature_names = list(X_train.columns)
print(f"Features ({len(feature_names)}): {feature_names}")

# ── Train Models ──────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    y_proba = model.predict_proba(X_train)[:, 1]

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    # Metrics
    cm = confusion_matrix(y_train, y_pred)
    fpr, tpr, thresholds = roc_curve(y_train, y_proba)

    results[name] = {
        "accuracy": float(accuracy_score(y_train, y_pred)),
        "precision": float(precision_score(y_train, y_pred)),
        "recall": float(recall_score(y_train, y_pred)),
        "f1": float(f1_score(y_train, y_pred)),
        "roc_auc": float(roc_auc_score(y_train, y_proba)),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "cv_scores": [float(s) for s in cv_scores],
        "confusion_matrix": cm.tolist(),
        "fpr": [float(x) for x in fpr],
        "tpr": [float(x) for x in tpr],
        "classification_report": classification_report(y_train, y_pred, output_dict=True),
    }

    # Feature importance
    if hasattr(model, "feature_importances_"):
        results[name]["feature_importance"] = dict(zip(feature_names, [float(x) for x in model.feature_importances_]))
    elif hasattr(model, "coef_"):
        results[name]["feature_importance"] = dict(zip(feature_names, [float(x) for x in np.abs(model.coef_[0])]))
    elif isinstance(model, Pipeline) and hasattr(model.named_steps.get("clf", None), "coef_"):
        coefs = np.abs(model.named_steps["clf"].coef_[0])
        results[name]["feature_importance"] = dict(zip(feature_names, [float(x) for x in coefs]))

    print(f"  Accuracy: {results[name]['accuracy']:.4f} | CV: {results[name]['cv_mean']:.4f} ± {results[name]['cv_std']:.4f} | AUC: {results[name]['roc_auc']:.4f}")

# ── Pick Best Model ───────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["cv_mean"])
best_model = models[best_name]
print(f"\nBest model (by CV accuracy): {best_name}")

# ── Save Artifacts ────────────────────────────────────────────────────────────
os.makedirs("artifacts", exist_ok=True)

joblib.dump(best_model, "artifacts/best_model.pkl")
joblib.dump(encoders, "artifacts/encoders.pkl")
joblib.dump(feature_names, "artifacts/feature_names.pkl")
joblib.dump(models, "artifacts/all_models.pkl")

with open("artifacts/results.json", "w") as f:
    json.dump(results, f, indent=2)

with open("artifacts/best_model_name.txt", "w") as f:
    f.write(best_name)

print("\nArtifacts saved to artifacts/")
print("Done!")
