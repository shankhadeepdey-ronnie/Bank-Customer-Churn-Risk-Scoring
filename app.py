# ============================================================
# STREAMLIT DASHBOARD
# Predictive Modeling and Risk Scoring for Bank Customer Churn
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bank Churn Risk Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main { background-color: #f7f9fc; }
    .risk-low {
        background-color: #e8f8ef; color: #117a37; padding: 15px;
        border-radius: 12px; font-size: 22px; font-weight: 700; text-align: center;
    }
    .risk-medium {
        background-color: #fff8e1; color: #b7791f; padding: 15px;
        border-radius: 12px; font-size: 22px; font-weight: 700; text-align: center;
    }
    .risk-high {
        background-color: #fff0e6; color: #c05621; padding: 15px;
        border-radius: 12px; font-size: 22px; font-weight: 700; text-align: center;
    }
    .risk-very-high {
        background-color: #fdecec; color: #c53030; padding: 15px;
        border-radius: 12px; font-size: 22px; font-weight: 700; text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOADERS
# ============================================================

@st.cache_resource
def load_model_files():
    model_path = "models/best_churn_model_calibrated.pkl"
    threshold_path = "models/calibrated_best_threshold.pkl"
    deployment_path = "models/deployment_info.pkl"

    if not os.path.exists(model_path):
        st.error("Model file not found: models/best_churn_model_calibrated.pkl")
        st.stop()
    if not os.path.exists(threshold_path):
        st.error("Threshold file not found: models/calibrated_best_threshold.pkl")
        st.stop()

    model = joblib.load(model_path)
    threshold = joblib.load(threshold_path)
    deployment_info = joblib.load(deployment_path) if os.path.exists(deployment_path) else None
    return model, threshold, deployment_info


@st.cache_data
def load_dataset():
    possible_paths = [
        "European_Bank.csv",
        "dataset/European_Bank.csv",
        "outputs/European_Bank_Feature_Engineered.csv",
        "dataset/European_Bank_Feature_Engineered.csv",
        "dataset/European_Bank_Clean_Featured.csv"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


@st.cache_data
def load_model_comparison():
    path = "outputs/final_test_model_comparison.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame({
        "Model": [
            "Tuned Gradient Boosting", "Tuned XGBoost", "Tuned Random Forest",
            "Tuned Hist Gradient Boosting", "Tuned Extra Trees",
            "Tuned Decision Tree", "Tuned Logistic Regression"
        ],
        "Threshold": [0.33, 0.65, 0.34, 0.31, 0.30, 0.19, 0.60],
        "Accuracy": [0.8525, 0.8540, 0.8540, 0.8505, 0.8410, 0.7750, 0.7770],
        "Precision": [0.6451, 0.6494, 0.6602, 0.6337, 0.6023, 0.4669, 0.4624],
        "Recall": [0.6118, 0.6143, 0.5823, 0.6290, 0.6437, 0.7445, 0.5897],
        "F1 Score": [0.6280, 0.6313, 0.6188, 0.6313, 0.6223, 0.5739, 0.5184],
        "ROC-AUC": [0.8682, 0.8668, 0.8661, 0.8645, 0.8636, 0.8425, 0.7926],
        "PR-AUC": [0.7113, 0.7127, 0.7021, 0.7118, 0.6998, 0.6527, 0.5397],
        "Predicted Churners": [386, 385, 359, 404, 435, 649, 519]
    })


@st.cache_data
def load_feature_importance():
    path = "outputs/final_feature_importance.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame({
        "Feature": [
            "Age", "NumOfProducts", "Engagement_Product", "IsActiveMember",
            "Geography_Germany", "Balance", "Age_Tenure", "Balance_to_Salary",
            "Gender_Male", "Product_Density", "EstimatedSalary", "CreditScore",
            "Senior_Customer_Flag", "Zero_Balance_Flag", "High_Balance_Flag",
            "Tenure", "Geography_Spain", "HasCrCard", "Low_CreditScore_Flag"
        ],
        "Importance": [
            0.318838, 0.261072, 0.091346, 0.063453,
            0.048965, 0.042263, 0.036429, 0.027350,
            0.020053, 0.018609, 0.014804, 0.013665,
            0.012677, 0.010221, 0.007771,
            0.007351, 0.002216, 0.002060, 0.000859
        ]
    })

# ============================================================
# PREDICTION HELPERS
# ============================================================

def engineer_features(input_df):
    df = input_df.copy()
    salary_safe = df["EstimatedSalary"].replace(0, np.nan)

    df["Balance_to_Salary"] = df["Balance"] / salary_safe
    df["Balance_to_Salary"] = df["Balance_to_Salary"].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["Product_Density"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["Engagement_Product"] = df["IsActiveMember"] * df["NumOfProducts"]
    df["Age_Tenure"] = df["Age"] * df["Tenure"]
    df["High_Balance_Flag"] = np.where(df["Balance"] > 97198.54, 1, 0)
    df["Zero_Balance_Flag"] = np.where(df["Balance"] == 0, 1, 0)
    df["Senior_Customer_Flag"] = np.where(df["Age"] >= 60, 1, 0)
    df["Low_CreditScore_Flag"] = np.where(df["CreditScore"] < 652, 1, 0)
    return df


def assign_risk_category(probability):
    if probability < 0.30:
        return "Low Risk"
    if probability < 0.60:
        return "Medium Risk"
    if probability < 0.80:
        return "High Risk"
    return "Very High Risk"


def risk_html(risk_category):
    css_class = {
        "Low Risk": "risk-low",
        "Medium Risk": "risk-medium",
        "High Risk": "risk-high",
        "Very High Risk": "risk-very-high"
    }.get(risk_category, "risk-medium")
    return f"<div class='{css_class}'>{risk_category}</div>"


def retention_action(risk_category):
    actions = {
        "Low Risk": "Maintain regular engagement and standard service.",
        "Medium Risk": "Send personalized engagement message or relevant product recommendation.",
        "High Risk": "Offer retention benefit, relationship-manager call, or product bundle.",
        "Very High Risk": "Immediate retention intervention with personalized offer and priority support."
    }
    return actions.get(risk_category, "Review customer profile and plan targeted intervention.")


def probability_gauge(probability):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": "Churn Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 30], "color": "#dff5e3"},
                    {"range": [30, 60], "color": "#fff3cd"},
                    {"range": [60, 80], "color": "#ffe0b2"},
                    {"range": [80, 100], "color": "#f8d7da"},
                ],
            }
        )
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def predict_customer(model, customer_df):
    featured_df = engineer_features(customer_df)
    probability = model.predict_proba(featured_df)[:, 1][0]
    risk = assign_risk_category(probability)
    action = retention_action(risk)
    return probability, risk, action, featured_df

# ============================================================
# LOAD DATA
# ============================================================

model, threshold, deployment_info = load_model_files()
df = load_dataset()
model_comparison_df = load_model_comparison()
feature_importance_df = load_feature_importance()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏦 Churn Intelligence")
st.sidebar.markdown("**Predictive Modeling and Risk Scoring for Bank Customer Churn**")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home", "Customer Risk Calculator", "What-if Simulator",
        "Analytics Dashboard", "Model Performance", "Feature Importance", "About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Final Model")
st.sidebar.success("Calibrated Tuned Gradient Boosting")
st.sidebar.metric("Decision Threshold", f"{float(threshold):.2f}")
st.sidebar.metric("Final ROC-AUC", "0.8667")
st.sidebar.metric("Final Recall", "66.34%")

# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":
    st.title("🏦 Bank Customer Churn Risk Scoring Dashboard")
    st.markdown(
        """
        This dashboard predicts whether a bank customer may leave the bank.
        It converts customer-level information into a churn probability score,
        risk category, and retention recommendation.
        """
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Size", "10,000")
    c2.metric("Churn Rate", "20.37%")
    c3.metric("Best ROC-AUC", "0.8682")
    c4.metric("Calibrated ROC-AUC", "0.8667")

    st.markdown("### Project Objective")
    st.info(
        "The goal is to move bank churn analysis from reactive reporting to proactive retention intelligence."
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("#### 🎯 Risk Calculator")
        st.write("Predict churn probability for an individual customer.")
    with m2:
        st.markdown("#### 🔁 What-if Simulator")
        st.write("Modify customer behavior and observe churn probability changes.")
    with m3:
        st.markdown("#### 📊 Explainability")
        st.write("Understand the top drivers behind churn risk.")

# ============================================================
# CUSTOMER RISK CALCULATOR
# ============================================================

elif page == "Customer Risk Calculator":
    st.title("🎯 Customer Churn Risk Calculator")
    st.markdown("Enter customer details below to estimate churn probability.")

    with st.form("risk_calculator_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            credit_score = st.slider("Credit Score", 350, 850, 650)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col2:
            age = st.slider("Age", 18, 92, 40)
            tenure = st.slider("Tenure with Bank", 0, 10, 5)
            balance = st.number_input("Account Balance", min_value=0.0, value=75000.0, step=1000.0)
        with col3:
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
            has_cr_card = st.selectbox("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            is_active = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            estimated_salary = st.number_input("Estimated Salary", min_value=1.0, value=100000.0, step=1000.0)
        submitted = st.form_submit_button("Predict Churn Risk")

    if submitted:
        customer_df = pd.DataFrame({
            "CreditScore": [credit_score],
            "Geography": [geography],
            "Gender": [gender],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active],
            "EstimatedSalary": [estimated_salary]
        })
        probability, risk, action, featured_customer = predict_customer(model, customer_df)

        st.markdown("---")
        st.markdown("## Prediction Result")
        r1, r2, r3 = st.columns([1.3, 1, 1.3])
        with r1:
            st.plotly_chart(probability_gauge(probability), use_container_width=True)
        with r2:
            st.metric("Churn Probability", f"{probability * 100:.2f}%")
            st.metric("Model Threshold", f"{float(threshold):.2f}")
        with r3:
            st.markdown("### Risk Category")
            st.markdown(risk_html(risk), unsafe_allow_html=True)
            st.markdown("### Recommended Action")
            st.success(action)
        st.markdown("### Engineered Customer Features")
        st.dataframe(featured_customer, use_container_width=True)

# ============================================================
# WHAT-IF SIMULATOR
# ============================================================

elif page == "What-if Simulator":
    st.title("🔁 What-if Churn Scenario Simulator")
    st.markdown("Compare a customer's current churn probability with a modified scenario.")

    col1, col2, col3 = st.columns(3)
    with col1:
        base_credit_score = st.slider("Current Credit Score", 350, 850, 650)
        base_geography = st.selectbox("Current Geography", ["France", "Germany", "Spain"])
        base_gender = st.selectbox("Current Gender", ["Male", "Female"])
    with col2:
        base_age = st.slider("Current Age", 18, 92, 40)
        base_tenure = st.slider("Current Tenure", 0, 10, 5)
        base_balance = st.number_input("Current Balance", min_value=0.0, value=75000.0, step=1000.0)
    with col3:
        base_products = st.selectbox("Current Products", [1, 2, 3, 4])
        base_card = st.selectbox("Current Credit Card Status", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        base_active = st.selectbox("Current Active Status", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        base_salary = st.number_input("Current Estimated Salary", min_value=1.0, value=100000.0, step=1000.0)

    st.markdown("---")
    st.markdown("## Modified Scenario")
    col4, col5, col6 = st.columns(3)
    with col4:
        new_products = st.selectbox("Scenario: Number of Products", [1, 2, 3, 4], index=max(base_products - 1, 0))
    with col5:
        new_active = st.selectbox("Scenario: Active Member Status", [1, 0], index=0 if base_active == 1 else 1, format_func=lambda x: "Yes" if x == 1 else "No")
    with col6:
        new_balance = st.number_input("Scenario: Balance", min_value=0.0, value=float(base_balance), step=1000.0)

    if st.button("Run What-if Simulation"):
        base_customer = pd.DataFrame({
            "CreditScore": [base_credit_score], "Geography": [base_geography], "Gender": [base_gender],
            "Age": [base_age], "Tenure": [base_tenure], "Balance": [base_balance],
            "NumOfProducts": [base_products], "HasCrCard": [base_card], "IsActiveMember": [base_active],
            "EstimatedSalary": [base_salary]
        })
        scenario_customer = base_customer.copy()
        scenario_customer["NumOfProducts"] = new_products
        scenario_customer["IsActiveMember"] = new_active
        scenario_customer["Balance"] = new_balance

        base_prob, base_risk, _, _ = predict_customer(model, base_customer)
        new_prob, new_risk, _, _ = predict_customer(model, scenario_customer)
        change = new_prob - base_prob

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Current Churn Probability", f"{base_prob * 100:.2f}%")
            st.markdown(risk_html(base_risk), unsafe_allow_html=True)
        with c2:
            st.metric("Scenario Churn Probability", f"{new_prob * 100:.2f}%")
            st.markdown(risk_html(new_risk), unsafe_allow_html=True)
        with c3:
            st.metric("Probability Change", f"{change * 100:.2f}%")
            if change < 0:
                st.success("Risk reduced under this scenario.")
            elif change > 0:
                st.warning("Risk increased under this scenario.")
            else:
                st.info("No major change in churn risk.")

        compare_df = pd.DataFrame({"Scenario": ["Current", "Modified"], "Churn Probability": [base_prob * 100, new_prob * 100]})
        fig = px.bar(compare_df, x="Scenario", y="Churn Probability", text="Churn Probability", title="Current vs Modified Churn Probability")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# ANALYTICS DASHBOARD
# ============================================================

elif page == "Analytics Dashboard":
    st.title("📊 Customer Churn Analytics")
    if df is None:
        st.warning("Dataset not found. Place European_Bank.csv inside the project folder or dataset folder.")
    else:
        analysis_df = df.copy()
        st.markdown("### Dataset Preview")
        st.dataframe(analysis_df.head(), use_container_width=True)

        if "Exited" in analysis_df.columns:
            c1, c2, c3 = st.columns(3)
            total_customers = len(analysis_df)
            churned_customers = int(analysis_df["Exited"].sum())
            churn_rate = analysis_df["Exited"].mean() * 100
            c1.metric("Total Customers", f"{total_customers:,}")
            c2.metric("Churned Customers", f"{churned_customers:,}")
            c3.metric("Churn Rate", f"{churn_rate:.2f}%")

            churn_counts = analysis_df["Exited"].value_counts().reset_index()
            churn_counts.columns = ["Exited", "Count"]
            churn_counts["Status"] = churn_counts["Exited"].map({0: "Retained", 1: "Churned"})
            fig = px.pie(churn_counts, names="Status", values="Count", title="Customer Churn Distribution", hole=0.45)
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                if "Geography" in analysis_df.columns:
                    geo_churn = analysis_df.groupby("Geography")["Exited"].mean().reset_index()
                    geo_churn["Churn Rate"] = geo_churn["Exited"] * 100
                    fig = px.bar(geo_churn, x="Geography", y="Churn Rate", text="Churn Rate", title="Churn Rate by Geography")
                    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                if "Gender" in analysis_df.columns:
                    gender_churn = analysis_df.groupby("Gender")["Exited"].mean().reset_index()
                    gender_churn["Churn Rate"] = gender_churn["Exited"] * 100
                    fig = px.bar(gender_churn, x="Gender", y="Churn Rate", text="Churn Rate", title="Churn Rate by Gender")
                    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                if "IsActiveMember" in analysis_df.columns:
                    active_churn = analysis_df.groupby("IsActiveMember")["Exited"].mean().reset_index()
                    active_churn["Status"] = active_churn["IsActiveMember"].map({0: "Inactive", 1: "Active"})
                    active_churn["Churn Rate"] = active_churn["Exited"] * 100
                    fig = px.bar(active_churn, x="Status", y="Churn Rate", text="Churn Rate", title="Churn Rate by Active Membership")
                    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)
            with c4:
                if "NumOfProducts" in analysis_df.columns:
                    product_churn = analysis_df.groupby("NumOfProducts")["Exited"].mean().reset_index()
                    product_churn["Churn Rate"] = product_churn["Exited"] * 100
                    fig = px.bar(product_churn, x="NumOfProducts", y="Churn Rate", text="Churn Rate", title="Churn Rate by Number of Products")
                    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":
    st.title("📈 Model Performance Comparison")
    st.markdown("The final experiment used tuned models with cross-validation, threshold optimization, and probability calibration.")
    st.markdown("### Final Test Model Comparison")
    st.dataframe(model_comparison_df, use_container_width=True)

    metric_choice = st.selectbox("Select metric to compare", ["ROC-AUC", "PR-AUC", "F1 Score", "Recall", "Precision", "Accuracy"])
    sorted_df = model_comparison_df.sort_values(metric_choice, ascending=False)
    fig = px.bar(sorted_df, x="Model", y=metric_choice, text=metric_choice, title=f"Model Comparison by {metric_choice}")
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "Calibrated GB")
    c2.metric("ROC-AUC", "0.8667")
    c3.metric("Recall", "0.6634")
    c4.metric("F1 Score", "0.6279")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif page == "Feature Importance":
    st.title("🧠 Churn Driver Explainability")
    st.markdown("Feature importance shows which customer characteristics contributed most to churn prediction.")
    st.dataframe(feature_importance_df, use_container_width=True)

    top_n = st.slider("Number of top features to display", 5, min(20, len(feature_importance_df)), 15)
    plot_df = feature_importance_df.head(top_n).sort_values("Importance")
    fig = px.bar(plot_df, x="Importance", y="Feature", orientation="h", title=f"Top {top_n} Churn Drivers")
    st.plotly_chart(fig, use_container_width=True)

    st.success("The strongest churn drivers are Age, Number of Products, Engagement-Product interaction, Active Membership, Geography, and Balance behavior.")

# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "About Project":
    st.title("📌 About the Project")
    st.markdown(
        """
        ## Predictive Modeling and Risk Scoring for Bank Customer Churn

        This project builds a machine learning-based churn intelligence system that predicts whether a bank customer is likely to leave and converts that prediction into a probability-based risk score.

        ### Methodology
        1. Data cleaning and preprocessing
        2. Feature engineering
        3. Stratified train-validation-test split
        4. One-hot encoding and scaling
        5. Hyperparameter tuning using RandomizedSearchCV
        6. Threshold optimization
        7. Probability calibration
        8. Streamlit deployment
        """
    )

    performance_summary = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"],
        "Value": [0.8400, 0.5960, 0.6634, 0.6279, 0.8667, 0.7095]
    })
    st.dataframe(performance_summary, use_container_width=True)
    st.info("The dashboard helps banks identify high-risk customers early and take targeted retention actions.")
