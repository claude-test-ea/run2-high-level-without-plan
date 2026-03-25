"""
Streamlit Dashboard for Loan Eligibility Prediction.
Uses the FULL dataset for all visualizations.
Model artifacts are loaded from artifacts/ (produced by train_model.py).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import joblib
import os

st.set_page_config(page_title="Loan Eligibility Predictor", layout="wide")
st.title("Loan Eligibility Prediction Dashboard")

# ── Load Data & Artifacts ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/input.csv")

@st.cache_resource
def load_artifacts():
    results = json.load(open("artifacts/results.json"))
    best_model = joblib.load("artifacts/best_model.pkl")
    encoders = joblib.load("artifacts/encoders.pkl")
    feature_names = joblib.load("artifacts/feature_names.pkl")
    all_models = joblib.load("artifacts/all_models.pkl")
    with open("artifacts/best_model_name.txt") as f:
        best_name = f.read().strip()
    return results, best_model, encoders, feature_names, all_models, best_name

df_full = load_data()

if not os.path.exists("artifacts/results.json"):
    st.error("Model artifacts not found. Run `python train_model.py` first.")
    st.stop()

results, best_model, encoders, feature_names, all_models, best_model_name = load_artifacts()

# ── Helper: preprocess a single row for prediction ────────────────────────────
def preprocess_input(input_dict):
    """Preprocess a single input dictionary for prediction."""
    row = pd.DataFrame([input_dict])
    row["Gender"] = row["Gender"].fillna("Male")
    row["Married"] = row["Married"].fillna("Yes")
    row["Dependents"] = row["Dependents"].fillna("0")
    row["Self_Employed"] = row["Self_Employed"].fillna("No")
    row["LoanAmount"] = row["LoanAmount"].astype(float)
    row["Loan_Amount_Term"] = row["Loan_Amount_Term"].astype(float)
    row["Credit_History"] = row["Credit_History"].astype(float)

    row["TotalIncome"] = row["ApplicantIncome"] + row["CoapplicantIncome"]
    row["TotalIncome_Log"] = np.log1p(row["TotalIncome"])
    row["LoanAmount_Log"] = np.log1p(row["LoanAmount"])
    row["EMI"] = row["LoanAmount"] / (row["Loan_Amount_Term"] + 1)
    row["Balance_Income"] = row["TotalIncome"] - (row["EMI"] * 1000)

    cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
    for col in cat_cols:
        row[col] = row[col].astype(str)
        le = encoders[col]
        row[col] = row[col].map(lambda x, _le=le: _le.transform([x])[0] if x in _le.classes_ else -1)

    return row[feature_names]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Data Explorer", "Model Performance", "Model Deep Dive", "Predictions"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Data Explorer")
    st.markdown(f"**Full dataset:** {len(df_full)} rows, {len(df_full.columns)} columns")

    # Dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df_full, use_container_width=True, height=300)

    # Summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(df_full.describe(include="all").T, use_container_width=True)

    # Missing values
    st.subheader("Missing Values")
    missing = df_full.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        fig_missing = px.bar(
            x=missing.index, y=missing.values,
            labels={"x": "Column", "y": "Missing Count"},
            title="Missing Values by Column",
            color=missing.values, color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("No missing values!")

    # Target distribution
    st.subheader("Loan Status Distribution")
    col1, col2 = st.columns(2)
    with col1:
        status_counts = df_full["Loan_Status"].value_counts()
        fig_target = px.pie(
            values=status_counts.values, names=status_counts.index,
            title="Loan Approval Rate", color_discrete_sequence=["#2ecc71", "#e74c3c"]
        )
        st.plotly_chart(fig_target, use_container_width=True)

    with col2:
        fig_bar = px.bar(
            x=status_counts.index, y=status_counts.values,
            labels={"x": "Loan Status", "y": "Count"},
            title="Loan Status Counts", color=status_counts.index,
            color_discrete_map={"Y": "#2ecc71", "N": "#e74c3c"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Categorical feature distributions
    st.subheader("Feature Distributions")
    cat_features = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
    cols = st.columns(3)
    for i, feat in enumerate(cat_features):
        with cols[i % 3]:
            counts = df_full[feat].value_counts()
            fig = px.bar(
                x=counts.index.astype(str), y=counts.values,
                title=feat, labels={"x": feat, "y": "Count"},
                color=counts.index.astype(str)
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Numerical distributions
    st.subheader("Numerical Distributions")
    num_features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    cols = st.columns(2)
    for i, feat in enumerate(num_features):
        with cols[i % 2]:
            fig = px.histogram(
                df_full, x=feat, color="Loan_Status", barmode="overlay",
                title=f"{feat} by Loan Status", opacity=0.7,
                color_discrete_map={"Y": "#2ecc71", "N": "#e74c3c"}
            )
            st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df_full.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        title="Feature Correlations", color_continuous_scale="RdBu_r"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Model Performance Comparison")
    st.info(f"Best model (by CV accuracy): **{best_model_name}**")

    # Metrics comparison table
    st.subheader("Metrics Overview")
    metrics_df = pd.DataFrame({
        name: {
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1 Score": r["f1"],
            "ROC AUC": r["roc_auc"],
            "CV Mean": r["cv_mean"],
            "CV Std": r["cv_std"],
        }
        for name, r in results.items()
    }).T
    st.dataframe(metrics_df.style.format("{:.4f}").highlight_max(axis=0, color="#2ecc71"), use_container_width=True)

    # Bar chart comparison
    st.subheader("Metrics Comparison")
    metrics_long = metrics_df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
    metrics_long = metrics_long.rename(columns={"index": "Model"})
    fig_comp = px.bar(
        metrics_long, x="Metric", y="Score", color="Model", barmode="group",
        title="Model Metrics Comparison"
    )
    fig_comp.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig_comp, use_container_width=True)

    # ROC Curves
    st.subheader("ROC Curves")
    fig_roc = go.Figure()
    colors = {"Logistic Regression": "#3498db", "Random Forest": "#2ecc71", "Gradient Boosting": "#e74c3c"}
    for name, r in results.items():
        fig_roc.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"], mode="lines",
            name=f"{name} (AUC={r['roc_auc']:.3f})",
            line=dict(color=colors.get(name, "#333"))
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(dash="dash", color="gray")
    ))
    fig_roc.update_layout(
        title="ROC Curves", xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # Cross-validation scores
    st.subheader("Cross-Validation Scores (5-Fold)")
    cv_data = []
    for name, r in results.items():
        for fold, score in enumerate(r["cv_scores"], 1):
            cv_data.append({"Model": name, "Fold": fold, "Accuracy": score})
    cv_df = pd.DataFrame(cv_data)
    fig_cv = px.line(
        cv_df, x="Fold", y="Accuracy", color="Model", markers=True,
        title="CV Accuracy per Fold"
    )
    st.plotly_chart(fig_cv, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: MODEL DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Model Deep Dive")
    selected_model = st.selectbox("Select Model", list(results.keys()))
    r = results[selected_model]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{r['accuracy']:.4f}")
    col2.metric("Precision", f"{r['precision']:.4f}")
    col3.metric("Recall", f"{r['recall']:.4f}")
    col4.metric("F1 Score", f"{r['f1']:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = np.array(r["confusion_matrix"])
    fig_cm = px.imshow(
        cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
        x=["Rejected (0)", "Approved (1)"], y=["Rejected (0)", "Approved (1)"],
        color_continuous_scale="Blues", title=f"Confusion Matrix — {selected_model}"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # Classification Report
    st.subheader("Classification Report")
    report = r["classification_report"]
    report_df = pd.DataFrame({
        k: v for k, v in report.items() if k not in ["accuracy"]
    }).T
    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

    # Feature Importance
    if "feature_importance" in r:
        st.subheader("Feature Importance")
        fi = r["feature_importance"]
        fi_df = pd.DataFrame({
            "Feature": list(fi.keys()),
            "Importance": list(fi.values())
        }).sort_values("Importance", ascending=True)
        fig_fi = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            title=f"Feature Importance — {selected_model}",
            color="Importance", color_continuous_scale="Viridis"
        )
        fig_fi.update_layout(height=500)
        st.plotly_chart(fig_fi, use_container_width=True)

    # ROC for selected model
    st.subheader("ROC Curve")
    fig_roc_single = go.Figure()
    fig_roc_single.add_trace(go.Scatter(
        x=r["fpr"], y=r["tpr"], mode="lines",
        name=f"AUC = {r['roc_auc']:.3f}", line=dict(color="#3498db")
    ))
    fig_roc_single.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(dash="dash", color="gray")
    ))
    fig_roc_single.update_layout(
        title=f"ROC Curve — {selected_model}",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate"
    )
    st.plotly_chart(fig_roc_single, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Predict Loan Eligibility")
    st.markdown(f"Using **{best_model_name}** for predictions.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])

        with col2:
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            applicant_income = st.number_input("Applicant Income", min_value=0, value=5000, step=100)
            coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0, step=100)

        with col3:
            loan_amount = st.number_input("Loan Amount (thousands)", min_value=1, value=150, step=5)
            loan_term = st.selectbox("Loan Amount Term (months)", [360, 180, 240, 120, 60, 36, 12])
            credit_history = st.selectbox("Credit History", [1.0, 0.0])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        input_data = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": float(loan_amount),
            "Loan_Amount_Term": float(loan_term),
            "Credit_History": credit_history,
            "Property_Area": property_area,
        }

        X_input = preprocess_input(input_data)
        prediction = best_model.predict(X_input)[0]
        proba = best_model.predict_proba(X_input)[0]

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.success("**Loan Approved!**")
            else:
                st.error("**Loan Rejected**")

        with col2:
            st.metric("Approval Probability", f"{proba[1]:.1%}")
            st.metric("Rejection Probability", f"{proba[0]:.1%}")

        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba[1] * 100,
            title={"text": "Approval Confidence (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2ecc71" if prediction == 1 else "#e74c3c"},
                "steps": [
                    {"range": [0, 50], "color": "#fadbd8"},
                    {"range": [50, 100], "color": "#d5f5e3"},
                ],
                "threshold": {"line": {"color": "black", "width": 2}, "value": 50}
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Show predictions from all models
        st.subheader("All Model Predictions")
        all_preds = {}
        for name, model in all_models.items():
            p = model.predict_proba(X_input)[0]
            all_preds[name] = {"Prediction": "Approved" if p[1] >= 0.5 else "Rejected", "Approval Prob": f"{p[1]:.1%}"}
        st.dataframe(pd.DataFrame(all_preds).T, use_container_width=True)
