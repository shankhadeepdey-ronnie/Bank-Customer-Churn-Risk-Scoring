# ============================================================
# BANK CUSTOMER CHURN RISK SCORING DASHBOARD
# Streamlit App with Live Training Fallback
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier


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
    .main {
        background-color: #f7f9fc;
    }
    .big-title {
        font-size: 40px;
        font-weight: 800;
    }
    .subtitle {
        color: #4a5568;
        font-size: 18px;
    }
    .info-card {
        background-color: white;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATASET LOADING
# ============================================================

@st.cache_data
def load_dataset():
    possible_paths = [
        "dataset/European_Bank.csv",
        "European_Bank.csv",
        "dataset/European_Bank_Feature_Engineered.csv",
        "outputs/European_Bank_Feature_Engineered.csv",
        "European_Bank_Feature_Engineered.csv",
        "dataset/European_Bank_Clean_Featured.csv",
        "European_Bank_Clean_Featured.csv"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path), path

    return None, None


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(input_df):
    df = input_df.copy()

    required_cols = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure",
        "Balance", "NumOfProducts", "HasCrCard",
        "IsActiveMember", "EstimatedSalary"
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    salary_safe = df["EstimatedSalary"].replace(0, np.nan)

    df["Balance_to_Salary"] = df["Balance"] / salary_safe
    df["Balance_to_Salary"] = (
        df["Balance_to_Salary"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    df["Product_Density"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["Engagement_Product"] = df["IsActiveMember"] * df["NumOfProducts"]
    df["Age_Tenure"] = df["Age"] * df["Tenure"]

    df["High_Balance_Flag"] = np.where(df["Balance"] > 97198.54, 1, 0)
    df["Zero_Balance_Flag"] = np.where(df["Balance"] == 0, 1, 0)
    df["Senior_Customer_Flag"] = np.where(df["Age"] >= 60, 1, 0)
    df["Low_CreditScore_Flag"] = np.where(df["CreditScore"] < 652, 1, 0)

    return df


# ============================================================
# LIVE MODEL TRAINING FALLBACK
# ============================================================

@st.cache_resource
def train_live_gradient_boosting_model():
    """
    Trains a tuned Gradient Boosting model live from the dataset.
    This avoids pickle/joblib version mismatch problems on Streamlit Cloud.
    """

    df_train, data_path = load_dataset()

    if df_train is None:
        st.error("Dataset not found. Please upload `European_Bank.csv` inside the dataset folder.")
        st.stop()

    drop_cols = []

    for col in ["CustomerId", "Surname"]:
        if col in df_train.columns:
            drop_cols.append(col)

    if "Year" in df_train.columns and df_train["Year"].nunique() == 1:
        drop_cols.append("Year")

    df_train = df_train.drop(columns=drop_cols, errors="ignore")

    if "Exited" not in df_train.columns:
        st.error("Target column `Exited` not found in dataset.")
        st.stop()

    X = df_train.drop(columns=["Exited"])
    y = df_train["Exited"]

    X = engineer_features(X)

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    try:
        onehot = OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        onehot = OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse=False
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", onehot, categorical_features),
            ("num", StandardScaler(), numerical_features)
        ],
        remainder="drop"
    )

    gb_model = GradientBoostingClassifier(
        n_estimators=507,
        learning_rate=0.014768002316749563,
        max_depth=4,
        min_samples_leaf=39,
        min_samples_split=59,
        subsample=0.969634193394765,
        max_features="log2",
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", gb_model)
        ]
    )

    pipeline.fit(X, y)

    return pipeline, data_path


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():
    """
    Tries saved pickle model first.
    If loading fails, trains model live from dataset.
    """

    possible_model_paths = [
        "models/best_churn_model_calibrated.pkl",
        "best_churn_model_calibrated.pkl",
        "models/Tuned_Gradient_Boosting_best_model.pkl",
        "Tuned_Gradient_Boosting_best_model.pkl",
        "models/best_churn_model_uncalibrated.pkl",
        "best_churn_model_uncalibrated.pkl"
    ]

    possible_threshold_paths = [
        "models/calibrated_best_threshold.pkl",
        "calibrated_best_threshold.pkl",
        "models/best_threshold.pkl",
        "best_threshold.pkl"
    ]

    threshold = 0.33
    threshold_path = None

    for path in possible_threshold_paths:
        if os.path.exists(path):
            threshold_path = path
            break

    if threshold_path is not None:
        try:
            threshold = float(joblib.load(threshold_path))
        except Exception:
            threshold = 0.33

    for model_path in possible_model_paths:
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                return model, threshold, model_path
            except Exception:
                pass

    model, data_path = train_live_gradient_boosting_model()
    return model, threshold, f"Live trained Gradient Boosting from {data_path}"


# ============================================================
# MODEL COMPARISON TABLE
# ============================================================

@st.cache_data
def load_model_comparison():
    possible_paths = [
        "outputs/final_test_model_comparison.csv",
        "final_test_model_comparison.csv",
        "model_comparison_step7.csv",
        "outputs/model_comparison_step7.csv"
    ]

    for path in possible_paths:
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
        "Predicted Churners": [386, 385, 359, 404, 435, 649, 519]
    })


# ============================================================
# FEATURE IMPORTANCE TABLE
# ============================================================

@st.cache_data
def load_feature_importance():
    possible_paths = [
        "outputs/final_feature_importance.csv",
        "final_feature_importance.csv",
        "outputs/random_forest_feature_importance.csv",
        "random_forest_feature_importance.csv"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)

    return pd.DataFrame({
        "Feature": [
            "Age",
            "NumOfProducts",
            "Engagement_Product",
            "IsActiveMember",
            "Geography_Germany",
            "Balance",
            "Age_Tenure",
            "Balance_to_Salary",
            "Gender_Male",
            "Product_Density",
            "EstimatedSalary",
            "CreditScore",
            "Senior_Customer_Flag",
            "Zero_Balance_Flag",
            "High_Balance_Flag",
            "Tenure",
            "Geography_Spain",
            "HasCrCard",
            "Low_CreditScore_Flag"
        ],
        "Importance": [
            0.318838,
            0.261072,
            0.091346,
            0.063453,
            0.048965,
            0.042263,
            0.036429,
            0.027350,
            0.020053,
            0.018609,
            0.014804,
            0.013665,
            0.012677,
            0.010221,
            0.007771,
            0.007351,
            0.002216,
            0.002060,
            0.000859
        ]
    })


# ============================================================
# RISK LOGIC
# ============================================================

def assign_risk_category(probability):
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.60:
        return "Medium Risk"
    elif probability < 0.80:
        return "High Risk"
    else:
        return "Very High Risk"


def retention_action(risk_category):
    if risk_category == "Low Risk":
        return "Maintain regular engagement and standard service."
    elif risk_category == "Medium Risk":
        return "Send personalized engagement message or relevant product recommendation."
    elif risk_category == "High Risk":
        return "Offer retention benefit, relationship-manager call, or product bundle."
    else:
        return "Immediate retention intervention with personalized offer and priority support."


def risk_box(risk_category):
    if risk_category == "Low Risk":
        bg_color = "#d8f3dc"
        text_color = "#1b7f3a"
    elif risk_category == "Medium Risk":
        bg_color = "#fff3cd"
        text_color = "#9a6a00"
    elif risk_category == "High Risk":
        bg_color = "#ffe5b4"
        text_color = "#b45309"
    else:
        bg_color = "#f8d7da"
        text_color = "#b91c1c"

    st.markdown(
        f"""
        <div style="
            background-color:{bg_color};
            color:{text_color};
            padding:22px;
            border-radius:15px;
            text-align:center;
            font-size:28px;
            font-weight:800;">
            {risk_category}
        </div>
        """,
        unsafe_allow_html=True
    )


def create_gauge(probability):
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


def predict_customer(customer_df):
    featured_df = engineer_features(customer_df)
    probability = model.predict_proba(featured_df)[:, 1][0]
    risk = assign_risk_category(probability)
    action = retention_action(risk)

    return probability, risk, action, featured_df


# ============================================================
# LOAD FILES
# ============================================================

model, threshold, loaded_model_path = load_model()
df, dataset_path = load_dataset()
model_comparison_df = load_model_comparison()
feature_importance_df = load_feature_importance()


# ============================================================
# SIDEBAR
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
st.sidebar.metric("Final Model", "Gradient Boosting")
st.sidebar.metric("Threshold", f"{threshold:.2f}")
st.sidebar.metric("ROC-AUC", "0.8667")
st.sidebar.metric("Recall", "66.34%")

with st.sidebar.expander("Loaded files"):
    st.write("Model:", loaded_model_path)
    st.write("Dataset:", dataset_path)


# ============================================================
# HOME PAGE
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
        st.write("Predict churn probability for an individual customer.")

    with c2:
        st.subheader("🔁 What-if Simulator")
        st.write("Test how customer changes affect churn risk.")

    with c3:
        st.subheader("📊 Explainability")
        st.write("Analyze churn drivers and model performance.")


# ============================================================
# RISK CALCULATOR PAGE
# ============================================================

elif page == "Risk Calculator":
    st.title("🎯 Customer Churn Risk Calculator")

    st.write("Enter customer details below to calculate churn risk.")

    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            credit_score = st.slider("Credit Score", 350, 850, 650)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])

        with col2:
            age = st.slider("Age", 18, 92, 40)
            tenure = st.slider("Tenure", 0, 10, 5)
            balance = st.number_input(
                "Balance",
                min_value=0.0,
                value=75000.0,
                step=1000.0
            )

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

        probability, risk, action, featured_df = predict_customer(customer_df)

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
# WHAT-IF SIMULATOR PAGE
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
        base_balance = st.number_input(
            "Current Balance",
            min_value=0.0,
            value=75000.0,
            step=1000.0
        )

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
        base_salary = st.number_input(
            "Current Estimated Salary",
            min_value=1.0,
            value=100000.0,
            step=1000.0
        )

    st.markdown("---")
    st.markdown("## Modified Scenario")

    s1, s2, s3 = st.columns(3)

    with s1:
        new_products = st.selectbox(
            "Scenario Number of Products",
            [1, 2, 3, 4],
            index=base_products - 1
        )

    with s2:
        new_active = st.selectbox(
            "Scenario Active Status",
            [1, 0],
            index=0 if base_active == 1 else 1,
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with s3:
        new_balance = st.number_input(
            "Scenario Balance",
            min_value=0.0,
            value=float(base_balance),
            step=1000.0
        )

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

        base_prob, base_risk, _, _ = predict_customer(base_customer)
        new_prob, new_risk, _, _ = predict_customer(scenario_customer)

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
# ANALYTICS PAGE
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
            churn_df["Status"] = churn_df["Exited"].map({
                0: "Retained",
                1: "Churned"
            })

            fig = px.pie(
                churn_df,
                names="Status",
                values="Count",
                hole=0.45,
                title="Customer Churn Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

            a1, a2 = st.columns(2)

            with a1:
                if "Geography" in df.columns:
                    geo_df = df.groupby("Geography")["Exited"].mean().reset_index()
                    geo_df["Churn Rate"] = geo_df["Exited"] * 100

                    fig = px.bar(
                        geo_df,
                        x="Geography",
                        y="Churn Rate",
                        text="Churn Rate",
                        title="Churn Rate by Geography"
                    )
                    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)

            with a2:
                if "Gender" in df.columns:
                    gender_df = df.groupby("Gender")["Exited"].mean().reset_index()
                    gender_df["Churn Rate"] = gender_df["Exited"] * 100

                    fig = px.bar(
                        gender_df,
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
                    active_df = df.groupby("IsActiveMember")["Exited"].mean().reset_index()
                    active_df["Status"] = active_df["IsActiveMember"].map({
                        0: "Inactive",
                        1: "Active"
                    })
                    active_df["Churn Rate"] = active_df["Exited"] * 100

                    fig = px.bar(
                        active_df,
                        x="Status",
                        y="Churn Rate",
                        text="Churn Rate",
                        title="Churn Rate by Active Membership"
                    )
                    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)

            with b2:
                if "NumOfProducts" in df.columns:
                    product_df = df.groupby("NumOfProducts")["Exited"].mean().reset_index()
                    product_df["Churn Rate"] = product_df["Exited"] * 100

                    fig = px.bar(
                        product_df,
                        x="NumOfProducts",
                        y="Churn Rate",
                        text="Churn Rate",
                        title="Churn Rate by Number of Products"
                    )
                    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MODEL PERFORMANCE PAGE
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

    numeric_metrics = [
        "ROC-AUC",
        "PR-AUC",
        "F1 Score",
        "Recall",
        "Precision",
        "Accuracy"
    ]

    available_metrics = [
        col for col in numeric_metrics if col in model_comparison_df.columns
    ]

    metric = st.selectbox("Select Metric", available_metrics)

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

    c1.metric("Model", "Gradient Boosting")
    c2.metric("Accuracy", "0.8400")
    c3.metric("Recall", "0.6634")
    c4.metric("ROC-AUC", "0.8667")

    st.info(
        """
        Tuned Gradient Boosting achieved the best raw ROC-AUC.
        Due to pickle version mismatch on deployment, the dashboard safely trains
        the tuned Gradient Boosting model live from the dataset.
        """
    )


# ============================================================
# FEATURE IMPORTANCE PAGE
# ============================================================

elif page == "Feature Importance":
    st.title("🧠 Feature Importance and Churn Drivers")

    st.write(
        """
        Feature importance helps identify the customer attributes that contributed most to churn prediction.
        """
    )

    st.dataframe(feature_importance_df, use_container_width=True)

    importance_col = "Importance"

    if importance_col not in feature_importance_df.columns:
        numeric_cols = feature_importance_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        if len(numeric_cols) > 0:
            importance_col = numeric_cols[0]

    top_n = st.slider(
        "Top features to display",
        5,
        min(20, len(feature_importance_df)),
        min(10, len(feature_importance_df))
    )

    plot_df = feature_importance_df.head(top_n).sort_values(importance_col)

    fig = px.bar(
        plot_df,
        x=importance_col,
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
# ABOUT PROJECT PAGE
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

        **Tuned Gradient Boosting**

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
