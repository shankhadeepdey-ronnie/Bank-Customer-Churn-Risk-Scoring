# ============================================================
# BANK CUSTOMER CHURN RISK SCORING DASHBOARD
# Streamlit App
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="Bank Churn Risk Scoring",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# Helper Functions
# ============================================================

@st.cache_resource
def load_model():
    model_path = "models/best_churn_model_calibrated.pkl"
    threshold_path = "models/calibrated_best_threshold.pkl"

    if not os.path.exists(model_path):
        st.error("Model file not found: models/best_churn_model_calibrated.pkl")
        st.stop()

    if not os.path.exists(threshold_path):
        st.error("Threshold file not found: models/calibrated_best_threshold.pkl")
        st.stop()

    model = joblib.load(model_path)
    threshold = joblib.load(threshold_path)

    return model, float(threshold)


@st.cache_data
def load_dataset():
    possible_paths = [
        "dataset/European_Bank.csv",
        "European_Bank.csv",
        "dataset/European_Bank_Feature_Engineered.csv",
        "outputs/European_Bank_Feature_Engineered.csv"
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
            "Tuned Gradient Boosting",
            "Tuned XGBoost",
            "Tuned Random Forest",
            "Tuned Hist Gradient Boosting",
            "Tuned Extra Trees",
            "Tuned Decision Tree",
            "Tuned Logistic Regression"
        ],
        "Threshold": [0.33, 0.65, 0.34, 0.31, 0.30, 0.19, 0.60],
        "Accuracy": [0.8525, 0.8540, 0.8540, 0.8505, 0.8410, 0.7750, 0.7770],
        "Precision": [0.6451, 0.6494, 0.6602, 0.6337, 0.6023, 0.4669, 0.4624],
        "Recall": [0.6118, 0.6143, 0.5823, 0.6290, 0.6437, 0.7445, 0.5897],
        "F1 Score": [0.6280, 0.6313, 0.6188, 0.6313, 0.6223, 0.5739, 0.5184],
        "ROC-AUC": [0.8682, 0.8668, 0.8661, 0.8645, 0.8636, 0.8425, 0.7926],
        "PR-AUC": [0.7113, 0.7127, 0.7021, 0.7118, 0.6998, 0.6527, 0.5397],
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


def engineer_features(df):
    df = df.copy()

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


def assign_risk(probability):
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.60:
        return "Medium Risk"
    elif probability < 0.80:
        return "High Risk"
    else:
        return "Very High Risk"


def recommended_action(risk):
    if risk == "Low Risk":
        return "Maintain regular engagement and standard service."
    elif risk == "Medium Risk":
        return "Send personalized engagement message or relevant product recommendation."
    elif risk == "High Risk":
        return "Offer retention benefit, relationship-manager call, or product bundle."
    else:
        return "Immediate retention intervention with personalized offer and priority support."


def predict_churn(customer_df):
    customer_featured = engineer_features(customer_df)
    probability = model.predict_proba(customer_featured)[:, 1][0]
    risk = assign_risk(probability)
    action = recommended_action(risk)

    return probability, risk, action, customer_featured


def create_gauge(probability):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Churn Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 30], "color": "#d8f3dc"},
                    {"range": [30, 60], "color": "#fff3cd"},
                    {"range": [60, 80], "color": "#ffe5b4"},
                    {"range": [80, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": probability * 100
                },
            },
        )
    )

    fig.update_layout(height=330, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def risk_box(risk):
    if risk == "Low Risk":
        color = "#d8f3dc"
        text = "#1b7f3a"
    elif risk == "Medium Risk":
        color = "#fff3cd"
        text = "#9a6a00"
    elif risk == "High Risk":
        color = "#ffe5b4"
        text = "#b45309"
    else:
        color = "#f8d7da"
        text = "#b91c1c"

    st.markdown(
        f"""
        <div style="
            background-color:{color};
            color:{text};
            padding:20px;
            border-radius:15px;
            text-align:center;
            font-size:26px;
            font-weight:800;">
            {risk}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# Load Files
# ============================================================

model, threshold = load_model()
df = load_dataset()
model_comparison_df = load_model_comparison()
feature_importance_df = load_feature_importance()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("🏦 Churn Risk Scoring")
st.sidebar.write("Bank Customer Churn Prediction Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Risk Calculator",
        "What-if Simulator",
        "Analytics",
        "Model Performance",
        "Feature Importance",
        "About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.metric("Final Model", "Calibrated GB")
st.sidebar.metric("Threshold", f"{threshold:.2f}")
st.sidebar.metric("ROC-AUC", "0.8667")
st.sidebar.metric("Recall", "66.34%")


# ============================================================
# Home Page
# ============================================================

if page == "Home":
    st.title("🏦 Bank Customer Churn Risk Scoring Dashboard")

    st.write(
        """
        This dashboard predicts whether a bank customer is likely to churn.
        It generates a churn probability score, assigns a risk category,
        and recommends a suitable retention action.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Dataset Size", "10,000")
    col2.metric("Churn Rate", "20.37%")
    col3.metric("Best ROC-AUC", "0.8682")
    col4.metric("Deployment ROC-AUC", "0.8667")

    st.markdown("### Project Objective")
    st.info(
        """
        The objective is to convert customer-level banking data into a proactive
        churn risk intelligence system for targeted customer retention.
        """
    )

    st.markdown("### Dashboard Modules")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("🎯 Risk Calculator")
        st.write("Predict churn probability for a customer.")

    with c2:
        st.subheader("🔁 What-if Simulator")
        st.write("Test how customer changes affect churn risk.")

    with c3:
        st.subheader("📊 Explainability")
        st.write("Analyze churn drivers and model performance.")


# ============================================================
# Risk Calculator Page
# ============================================================

elif page == "Risk Calculator":
    st.title("🎯 Customer Churn Risk Calculator")

    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            credit_score = st.slider("Credit Score", 350, 850, 650)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])

        with col2:
            age = st.slider("Age", 18, 92, 40)
            tenure = st.slider("Tenure", 0, 10, 5)
            balance = st.number_input("Balance", min_value=0.0, value=75000.0, step=1000.0)

        with col3:
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
            has_cr_card = st.selectbox(
                "Has Credit Card?",
                [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
            is_active = st.selectbox(
                "Is Active Member?",
                [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
            estimated_salary = st.number_input(
                "Estimated Salary",
                min_value=1.0,
                value=100000.0,
                step=1000.0
            )

        submit = st.form_submit_button("Predict Churn Risk")

    if submit:
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

        probability, risk, action, featured_df = predict_churn(customer_df)

        st.markdown("---")
        st.subheader("Prediction Result")

        r1, r2, r3 = st.columns([1.4, 1, 1.3])

        with r1:
            st.plotly_chart(create_gauge(probability), use_container_width=True)

        with r2:
            st.metric("Churn Probability", f"{probability * 100:.2f}%")
            st.metric("Model Threshold", f"{threshold:.2f}")

        with r3:
            st.markdown("### Risk Category")
            risk_box(risk)
            st.markdown("### Recommended Action")
            st.success(action)

        st.markdown("### Engineered Customer Features")
        st.dataframe(featured_df, use_container_width=True)


# ============================================================
# What-if Simulator Page
# ============================================================

elif page == "What-if Simulator":
    st.title("🔁 What-if Churn Scenario Simulator")

    st.write(
        """
        Compare a customer's current churn probability with a modified scenario.
        This helps understand how engagement, product usage, and balance changes affect churn risk.
        """
    )

    st.markdown("## Current Customer Profile")

    c1, c2, c3 = st.columns(3)

    with c1:
        base_credit_score = st.slider("Current Credit Score", 350, 850, 650)
        base_geography = st.selectbox("Current Geography", ["France", "Germany", "Spain"])
        base_gender = st.selectbox("Current Gender", ["Male", "Female"])

    with c2:
        base_age = st.slider("Current Age", 18, 92, 40)
        base_tenure = st.slider("Current Tenure", 0, 10, 5)
        base_balance = st.number_input("Current Balance", min_value=0.0, value=75000.0, step=1000.0)

    with c3:
        base_products = st.selectbox("Current Number of Products", [1, 2, 3, 4])
        base_card = st.selectbox(
            "Current Credit Card Status",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        base_active = st.selectbox(
            "Current Active Status",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        base_salary = st.number_input("Current Estimated Salary", min_value=1.0, value=100000.0, step=1000.0)

    st.markdown("---")
    st.markdown("## Modified Scenario")

    s1, s2, s3 = st.columns(3)

    with s1:
        new_products = st.selectbox("Scenario Number of Products", [1, 2, 3, 4], index=base_products - 1)

    with s2:
        new_active = st.selectbox(
            "Scenario Active Status",
            [1, 0],
            index=0 if base_active == 1 else 1,
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with s3:
        new_balance = st.number_input("Scenario Balance", min_value=0.0, value=float(base_balance), step=1000.0)

    if st.button("Run Simulation"):
        base_customer = pd.DataFrame({
            "CreditScore": [base_credit_score],
            "Geography": [base_geography],
            "Gender": [base_gender],
            "Age": [base_age],
            "Tenure": [base_tenure],
            "Balance": [base_balance],
            "NumOfProducts": [base_products],
            "HasCrCard": [base_card],
            "IsActiveMember": [base_active],
            "EstimatedSalary": [base_salary]
        })

        scenario_customer = base_customer.copy()
        scenario_customer["NumOfProducts"] = new_products
        scenario_customer["IsActiveMember"] = new_active
        scenario_customer["Balance"] = new_balance

        base_prob, base_risk, _, _ = predict_churn(base_customer)
        new_prob, new_risk, _, _ = predict_churn(scenario_customer)

        change = new_prob - base_prob

        st.markdown("## Simulation Result")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric("Current Probability", f"{base_prob * 100:.2f}%")
            risk_box(base_risk)

        with r2:
            st.metric("Scenario Probability", f"{new_prob * 100:.2f}%")
            risk_box(new_risk)

        with r3:
            st.metric("Change", f"{change * 100:.2f}%")
            if change < 0:
                st.success("Risk reduced under this scenario.")
            elif change > 0:
                st.warning("Risk increased under this scenario.")
            else:
                st.info("No major change.")

        compare_df = pd.DataFrame({
            "Scenario": ["Current", "Modified"],
            "Churn Probability": [base_prob * 100, new_prob * 100]
        })

        fig = px.bar(
            compare_df,
            x="Scenario",
            y="Churn Probability",
            text="Churn Probability",
            title="Current vs Modified Churn Probability"
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(yaxis_range=[0, 100])

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Analytics Page
# ============================================================

elif page == "Analytics":
    st.title("📊 Customer Churn Analytics")

    if df is None:
        st.warning("Dataset not found. Please upload dataset/European_Bank.csv.")
    else:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        if "Exited" in df.columns:
            total_customers = len(df)
            churned_customers = int(df["Exited"].sum())
            churn_rate = df["Exited"].mean() * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Customers", f"{total_customers:,}")
            c2.metric("Churned Customers", f"{churned_customers:,}")
            c3.metric("Churn Rate", f"{churn_rate:.2f}%")

            churn_df = df["Exited"].value_counts().reset_index()
            churn_df.columns = ["Exited", "Count"]
            churn_df["Status"] = churn_df["Exited"].map({0: "Retained", 1: "Churned"})

            fig = px.pie(
                churn_df,
                names="Status",
                values="Count",
                hole=0.45,
                title="Churn Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

            a1, a2 = st.columns(2)

            with a1:
                if "Geography" in df.columns:
                    geo = df.groupby("Geography")["Exited"].mean().reset_index()
                    geo["Churn Rate"] = geo["Exited"] * 100

                    fig = px.bar(
                        geo,
                        x="Geography",
                        y="Churn Rate",
                        text="Churn Rate",
                        title="Churn Rate by Geography"
                    )
                    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)

            with a2:
                if "Gender" in df.columns:
                    gender = df.groupby("Gender")["Exited"].mean().reset_index()
                    gender["Churn Rate"] = gender["Exited"] * 100

                    fig = px.bar(
                        gender,
                        x="Gender",
                        y="Churn Rate",
                        text="Churn Rate",
                        title="Churn Rate by Gender"
                    )
                    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)

            b1, b2 = st.columns(2)

            with b1:
                if "IsActiveMember" in df.columns:
                    active = df.groupby("IsActiveMember")["Exited"].mean().reset_index()
                    active["Status"] = active["IsActiveMember"].map({0: "Inactive", 1: "Active"})
                    active["Churn Rate"] = active["Exited"] * 100

                    fig = px.bar(
                        active,
                        x="Status",
                        y="Churn Rate",
                        text="Churn Rate",
                        title="Churn Rate by Active Membership"
                    )
                    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)

            with b2:
                if "NumOfProducts" in df.columns:
                    product = df.groupby("NumOfProducts")["Exited"].mean().reset_index()
                    product["Churn Rate"] = product["Exited"] * 100

                    fig = px.bar(
                        product,
                        x="NumOfProducts",
                        y="Churn Rate",
                        text="Churn Rate",
                        title="Churn Rate by Number of Products"
                    )
                    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Model Performance Page
# ============================================================

elif page == "Model Performance":
    st.title("📈 Model Performance Comparison")

    st.write(
        """
        The final experiment used tuned machine learning models with stratified splitting,
        cross-validation, threshold optimization, and probability calibration.
        """
    )

    st.subheader("Final Test Model Comparison")
    st.dataframe(model_comparison_df, use_container_width=True)

    metric = st.selectbox(
        "Select Metric",
        ["ROC-AUC", "PR-AUC", "F1 Score", "Recall", "Precision", "Accuracy"]
    )

    sorted_df = model_comparison_df.sort_values(metric, ascending=False)

    fig = px.bar(
        sorted_df,
        x="Model",
        y=metric,
        text=metric,
        title=f"Model Comparison by {metric}"
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(xaxis_tickangle=-35)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Final Deployment Model")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Model", "Calibrated GB")
    c2.metric("Accuracy", "0.8400")
    c3.metric("Recall", "0.6634")
    c4.metric("ROC-AUC", "0.8667")

    st.info(
        """
        Tuned Gradient Boosting achieved the best raw ROC-AUC.
        The calibrated version was selected for deployment because risk scoring requires reliable probability outputs.
        """
    )


# ============================================================
# Feature Importance Page
# ============================================================

elif page == "Feature Importance":
    st.title("🧠 Feature Importance and Churn Drivers")

    st.write(
        """
        Feature importance helps identify the customer attributes that contributed most to churn prediction.
        """
    )

    st.dataframe(feature_importance_df, use_container_width=True)

    top_n = st.slider("Top features to display", 5, min(20, len(feature_importance_df)), 10)

    plot_df = feature_importance_df.head(top_n).sort_values("Importance")

    fig = px.bar(
        plot_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top {top_n} Churn Drivers"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success(
        """
        The most important churn drivers are Age, Number of Products,
        Engagement-Product interaction, Active Membership, Geography Germany,
        Balance, and Balance-to-Salary ratio.
        """
    )


# ============================================================
# About Project Page
# ============================================================

elif page == "About Project":
    st.title("📌 About the Project")

    st.markdown(
        """
        ## Predictive Modeling and Risk Scoring for Bank Customer Churn

        This project builds a machine learning-based churn intelligence system for retail banking.
        The system predicts customer churn probability, assigns customer-level risk categories,
        and recommends targeted retention actions.

        ### Methodology

        1. Data cleaning  
        2. Feature engineering  
        3. Exploratory data analysis  
        4. Stratified train-validation-test split  
        5. Hyperparameter tuning using RandomizedSearchCV  
        6. Threshold optimization  
        7. Probability calibration  
        8. Streamlit dashboard deployment  

        ### Final Model

        **Calibrated Tuned Gradient Boosting**

        | Metric | Value |
        |---|---:|
        | Accuracy | 0.8400 |
        | Precision | 0.5960 |
        | Recall | 0.6634 |
        | F1 Score | 0.6279 |
        | ROC-AUC | 0.8667 |
        | PR-AUC | 0.7095 |

        ### Business Value

        The dashboard helps banks identify high-risk customers early and take proactive retention actions.
        """
    )
