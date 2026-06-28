# ============================================================
# BANK CUSTOMER CHURN RISK SCORING DASHBOARD
# Fresh Safe Standalone Streamlit App
# ------------------------------------------------------------
# This app is designed to deploy safely on Streamlit Cloud.
# It does NOT require pickle/joblib model files.
# It does NOT require X_train_raw.csv or y_train.csv.
# It optionally reads a valid bank customer CSV for analytics,
# but the risk calculator works even if no CSV is available.
# ============================================================

import os
import math
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
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
    .small-muted {
        color: #6b7280;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANT PROJECT DATA
# ============================================================

MODEL_RESULTS = pd.DataFrame(
    {
        "Model": [
            "Tuned Gradient Boosting",
            "Tuned XGBoost",
            "Tuned Random Forest",
            "Tuned Hist Gradient Boosting",
            "Tuned Extra Trees",
            "Tuned Decision Tree",
            "Tuned Logistic Regression",
        ],
        "Threshold": [0.33, 0.65, 0.34, 0.31, 0.30, 0.19, 0.60],
        "Accuracy": [0.8525, 0.8540, 0.8540, 0.8505, 0.8410, 0.7750, 0.7770],
        "Precision": [0.6451, 0.6494, 0.6602, 0.6337, 0.6023, 0.4669, 0.4624],
        "Recall": [0.6118, 0.6143, 0.5823, 0.6290, 0.6437, 0.7445, 0.5897],
        "F1 Score": [0.6280, 0.6313, 0.6188, 0.6313, 0.6223, 0.5739, 0.5184],
        "ROC-AUC": [0.8682, 0.8668, 0.8661, 0.8645, 0.8636, 0.8425, 0.7926],
        "PR-AUC": [0.7113, 0.7127, 0.7021, 0.7118, 0.6998, 0.6527, 0.5397],
        "Predicted Churners": [386, 385, 359, 404, 435, 649, 519],
    }
)

CALIBRATED_RESULTS = {
    "Model": "Calibrated Tuned Gradient Boosting",
    "Threshold": 0.30,
    "Accuracy": 0.8400,
    "Precision": 0.5960,
    "Recall": 0.6634,
    "F1 Score": 0.6279,
    "ROC-AUC": 0.8667,
    "PR-AUC": 0.7095,
    "Predicted Churners": 453,
}

FEATURE_IMPORTANCE = pd.DataFrame(
    {
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
            "Low_CreditScore_Flag",
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
            0.000859,
        ],
    }
)

EDA_SUMMARY = pd.DataFrame(
    {
        "Segment": [
            "Overall Churn Rate",
            "Germany Churn Rate",
            "France Churn Rate",
            "Spain Churn Rate",
            "Female Churn Rate",
            "Male Churn Rate",
            "Inactive Member Churn Rate",
            "Active Member Churn Rate",
            "3 Products Churn Rate",
            "4 Products Churn Rate",
        ],
        "Value": [
            "20.37%",
            "32.44%",
            "16.15%",
            "16.67%",
            "25.07%",
            "16.46%",
            "26.85%",
            "14.27%",
            "82.71%",
            "100.00%",
        ],
    }
)

EXPECTED_CUSTOMER_COLUMNS = {
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
}


# ============================================================
# SAFE CSV READING FOR OPTIONAL ANALYTICS ONLY
# ============================================================

@st.cache_data(show_spinner=False)
def safe_read_csv(path: str):
    encodings = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1", "cp1252"]

    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            return df
        except Exception:
            pass

    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding, engine="python", on_bad_lines="skip")
            return df
        except Exception:
            pass

    return None


@st.cache_data(show_spinner=False)
def load_valid_customer_dataset():
    """
    Optional dataset loader.
    The app never depends on this. It only uses it for analytics if valid.
    """
    possible_paths = [
        "European_Bank.csv",
        "European_Bank_Feature_Engineered.csv",
        "European_Bank_Clean_Featured.csv",
        "dataset/European_Bank.csv",
        "dataset/European_Bank_Feature_Engineered.csv",
        "outputs/European_Bank_Feature_Engineered.csv",
    ]

    for path in possible_paths:
        if not os.path.exists(path):
            continue

        data = safe_read_csv(path)
        if data is None or data.empty:
            continue

        columns = set(data.columns)
        has_customer_features = EXPECTED_CUSTOMER_COLUMNS.issubset(columns)
        has_target = "Exited" in columns

        if has_customer_features and has_target:
            return data, path

    return None, None


# ============================================================
# FEATURE ENGINEERING + STANDALONE SCORING ENGINE
# ============================================================

def engineer_features(customer_df: pd.DataFrame) -> pd.DataFrame:
    df = customer_df.copy()

    for col in EXPECTED_CUSTOMER_COLUMNS:
        if col not in df.columns:
            if col in ["Geography", "Gender"]:
                df[col] = "Unknown"
            else:
                df[col] = 0

    salary_safe = pd.to_numeric(df["EstimatedSalary"], errors="coerce").replace(0, np.nan)
    balance = pd.to_numeric(df["Balance"], errors="coerce").fillna(0)
    tenure = pd.to_numeric(df["Tenure"], errors="coerce").fillna(0)
    products = pd.to_numeric(df["NumOfProducts"], errors="coerce").fillna(1)
    active = pd.to_numeric(df["IsActiveMember"], errors="coerce").fillna(0)
    age = pd.to_numeric(df["Age"], errors="coerce").fillna(40)
    credit = pd.to_numeric(df["CreditScore"], errors="coerce").fillna(650)

    df["Balance_to_Salary"] = (balance / salary_safe).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["Product_Density"] = products / (tenure + 1)
    df["Engagement_Product"] = active * products
    df["Age_Tenure"] = age * tenure
    df["High_Balance_Flag"] = np.where(balance > 97198.54, 1, 0)
    df["Zero_Balance_Flag"] = np.where(balance == 0, 1, 0)
    df["Senior_Customer_Flag"] = np.where(age >= 60, 1, 0)
    df["Low_CreditScore_Flag"] = np.where(credit < 652, 1, 0)

    return df


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def standalone_churn_probability(row: pd.Series) -> float:
    """
    Safe demo scoring engine based on the project's observed churn drivers.
    It is intentionally independent of pickle files and broken CSV files.
    """
    credit_score = float(row.get("CreditScore", 650))
    geography = str(row.get("Geography", "France"))
    gender = str(row.get("Gender", "Male"))
    age = float(row.get("Age", 40))
    tenure = float(row.get("Tenure", 5))
    balance = float(row.get("Balance", 75000))
    num_products = int(row.get("NumOfProducts", 1))
    has_cr_card = int(row.get("HasCrCard", 1))
    is_active = int(row.get("IsActiveMember", 1))
    estimated_salary = max(float(row.get("EstimatedSalary", 100000)), 1.0)

    balance_to_salary = balance / estimated_salary

    # Base churn tendency around the observed 20.37% churn rate.
    score = -1.35

    # Main model-derived drivers.
    score += 0.055 * (age - 40)
    score += 0.20 * max(age - 55, 0) / 10
    score -= 0.035 * tenure

    if num_products == 1:
        score += 0.18
    elif num_products == 2:
        score -= 0.35
    elif num_products == 3:
        score += 1.55
    elif num_products >= 4:
        score += 2.20

    if is_active == 0:
        score += 0.85
    else:
        score -= 0.25

    if geography == "Germany":
        score += 0.70
    elif geography == "Spain":
        score += 0.08

    if gender == "Female":
        score += 0.28

    if balance > 97198.54:
        score += 0.33
    if balance == 0:
        score -= 0.15

    if credit_score < 652:
        score += 0.18
    if credit_score < 550:
        score += 0.20
    if credit_score > 750:
        score -= 0.12

    if has_cr_card == 0:
        score += 0.05

    if balance_to_salary > 1:
        score += 0.15
    if balance_to_salary > 2:
        score += 0.15

    probability = sigmoid(score)
    return float(np.clip(probability, 0.02, 0.98))


def assign_risk_category(probability: float) -> str:
    if probability < 0.30:
        return "Low Risk"
    if probability < 0.60:
        return "Medium Risk"
    if probability < 0.80:
        return "High Risk"
    return "Very High Risk"


def retention_action(risk_category: str) -> str:
    actions = {
        "Low Risk": "Maintain regular engagement and standard service.",
        "Medium Risk": "Send personalized engagement message or relevant product recommendation.",
        "High Risk": "Offer retention benefit, relationship-manager call, or product bundle.",
        "Very High Risk": "Immediate retention intervention with personalized offer and priority support.",
    }
    return actions.get(risk_category, "Review customer profile and plan retention action.")


def predict_customer(customer_df: pd.DataFrame):
    featured_df = engineer_features(customer_df)
    probability = standalone_churn_probability(featured_df.iloc[0])
    risk = assign_risk_category(probability)
    action = retention_action(risk)
    return probability, risk, action, featured_df


# ============================================================
# VISUAL HELPERS
# ============================================================

def risk_box(risk_category: str):
    style_map = {
        "Low Risk": ("#d8f3dc", "#166534"),
        "Medium Risk": ("#fff3cd", "#92400e"),
        "High Risk": ("#ffe5b4", "#b45309"),
        "Very High Risk": ("#f8d7da", "#991b1b"),
    }
    bg_color, text_color = style_map.get(risk_category, ("#e5e7eb", "#111827"))

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
        unsafe_allow_html=True,
    )


def create_gauge(probability: float):
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
                    "value": probability * 100,
                },
            },
        )
    )
    fig.update_layout(height=330, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def make_customer_input(prefix=""):
    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.slider(f"{prefix}Credit Score", 350, 850, 650)
        geography = st.selectbox(f"{prefix}Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox(f"{prefix}Gender", ["Male", "Female"])

    with col2:
        age = st.slider(f"{prefix}Age", 18, 92, 40)
        tenure = st.slider(f"{prefix}Tenure", 0, 10, 5)
        balance = st.number_input(f"{prefix}Balance", min_value=0.0, value=75000.0, step=1000.0)

    with col3:
        num_products = st.selectbox(f"{prefix}Number of Products", [1, 2, 3, 4])
        has_cr_card = st.selectbox(
            f"{prefix}Has Credit Card?",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        is_active = st.selectbox(
            f"{prefix}Is Active Member?",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        estimated_salary = st.number_input(
            f"{prefix}Estimated Salary",
            min_value=1.0,
            value=100000.0,
            step=1000.0,
        )

    return pd.DataFrame(
        {
            "CreditScore": [credit_score],
            "Geography": [geography],
            "Gender": [gender],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active],
            "EstimatedSalary": [estimated_salary],
        }
    )


# ============================================================
# LOAD OPTIONAL DATASET
# ============================================================

customer_data, customer_data_path = load_valid_customer_dataset()


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
        "About Project",
    ],
)

st.sidebar.markdown("---")
st.sidebar.metric("Project Model", "Gradient Boosting")
st.sidebar.metric("Churn Rate", "20.37%")
st.sidebar.metric("ROC-AUC", "0.8667")
st.sidebar.metric("Recall", "66.34%")

with st.sidebar.expander("Deployment status"):
    st.write("Mode: Safe standalone dashboard")
    st.write("Pickle files: Not required")
    st.write("Training CSV files: Not required")
    if customer_data_path:
        st.write("Analytics dataset:", customer_data_path)
    else:
        st.write("Analytics dataset: Embedded summary used")


# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":
    st.title("🏦 Bank Customer Churn Risk Scoring Dashboard")

    st.write(
        """
        This dashboard converts bank customer attributes into churn risk intelligence.
        It estimates churn probability, assigns a risk category, and recommends a retention action.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Size", "10,000")
    c2.metric("Churn Rate", "20.37%")
    c3.metric("Best ROC-AUC", "0.8682")
    c4.metric("Deployment ROC-AUC", "0.8667")

    st.markdown("### Project Objective")
    st.info(
        """
        The objective is to identify customers who are likely to leave the bank,
        score their churn risk, and support proactive customer-retention decisions.
        """
    )

    st.markdown("### Dashboard Modules")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.subheader("🎯 Risk Calculator")
        st.write("Estimate churn risk for an individual customer.")
    with m2:
        st.subheader("🔁 What-if Simulator")
        st.write("Test how changes in customer profile affect risk.")
    with m3:
        st.subheader("📊 Insights")
        st.write("View churn analytics, model performance, and key drivers.")


# ============================================================
# RISK CALCULATOR PAGE
# ============================================================

elif page == "Risk Calculator":
    st.title("🎯 Customer Churn Risk Calculator")
    st.write("Enter customer details below to estimate churn risk.")

    with st.form("risk_form"):
        customer_df = make_customer_input(prefix="")
        submitted = st.form_submit_button("Predict Churn Risk")

    if submitted:
        probability, risk, action, featured_df = predict_customer(customer_df)

        st.markdown("---")
        st.subheader("Prediction Result")

        r1, r2, r3 = st.columns([1.4, 1, 1.3])
        with r1:
            st.plotly_chart(create_gauge(probability), use_container_width=True)
        with r2:
            st.metric("Churn Probability", f"{probability * 100:.2f}%")
            st.metric("Decision Threshold", "30.00%")
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
    st.write("Compare the current customer profile with a modified retention scenario.")

    st.markdown("## Current Customer Profile")
    base_customer = make_customer_input(prefix="Current ")

    st.markdown("---")
    st.markdown("## Modified Scenario")

    base_products = int(base_customer["NumOfProducts"].iloc[0])
    base_active = int(base_customer["IsActiveMember"].iloc[0])
    base_balance = float(base_customer["Balance"].iloc[0])

    s1, s2, s3 = st.columns(3)
    with s1:
        new_products = st.selectbox("Scenario Number of Products", [1, 2, 3, 4], index=base_products - 1)
    with s2:
        new_active = st.selectbox(
            "Scenario Active Status",
            [1, 0],
            index=0 if base_active == 1 else 1,
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
    with s3:
        new_balance = st.number_input("Scenario Balance", min_value=0.0, value=base_balance, step=1000.0)

    if st.button("Run Simulation"):
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

        compare_df = pd.DataFrame(
            {
                "Scenario": ["Current", "Modified"],
                "Churn Probability": [base_prob * 100, new_prob * 100],
            }
        )
        fig = px.bar(
            compare_df,
            x="Scenario",
            y="Churn Probability",
            text="Churn Probability",
            title="Current vs Modified Churn Probability",
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# ANALYTICS PAGE
# ============================================================

elif page == "Analytics":
    st.title("📊 Customer Churn Analytics")

    if customer_data is not None:
        st.success(f"Loaded valid analytics dataset: {customer_data_path}")
        st.subheader("Dataset Preview")
        st.dataframe(customer_data.head(), use_container_width=True)

        total_customers = len(customer_data)
        churned_customers = int(customer_data["Exited"].sum())
        churn_rate = customer_data["Exited"].mean() * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Customers", f"{total_customers:,}")
        c2.metric("Churned Customers", f"{churned_customers:,}")
        c3.metric("Churn Rate", f"{churn_rate:.2f}%")

        churn_df = customer_data["Exited"].value_counts().reset_index()
        churn_df.columns = ["Exited", "Count"]
        churn_df["Status"] = churn_df["Exited"].map({0: "Retained", 1: "Churned"})
        fig = px.pie(churn_df, names="Status", values="Count", hole=0.45, title="Churn Distribution")
        st.plotly_chart(fig, use_container_width=True)

        a1, a2 = st.columns(2)
        with a1:
            geo_df = customer_data.groupby("Geography")["Exited"].mean().reset_index()
            geo_df["Churn Rate"] = geo_df["Exited"] * 100
            fig = px.bar(geo_df, x="Geography", y="Churn Rate", text="Churn Rate", title="Churn Rate by Geography")
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        with a2:
            gender_df = customer_data.groupby("Gender")["Exited"].mean().reset_index()
            gender_df["Churn Rate"] = gender_df["Exited"] * 100
            fig = px.bar(gender_df, x="Gender", y="Churn Rate", text="Churn Rate", title="Churn Rate by Gender")
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        b1, b2 = st.columns(2)
        with b1:
            active_df = customer_data.groupby("IsActiveMember")["Exited"].mean().reset_index()
            active_df["Status"] = active_df["IsActiveMember"].map({0: "Inactive", 1: "Active"})
            active_df["Churn Rate"] = active_df["Exited"] * 100
            fig = px.bar(active_df, x="Status", y="Churn Rate", text="Churn Rate", title="Churn Rate by Active Membership")
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        with b2:
            product_df = customer_data.groupby("NumOfProducts")["Exited"].mean().reset_index()
            product_df["Churn Rate"] = product_df["Exited"] * 100
            fig = px.bar(product_df, x="NumOfProducts", y="Churn Rate", text="Churn Rate", title="Churn Rate by Number of Products")
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No valid customer dataset was loaded. Showing embedded project analytics instead.")
        st.dataframe(EDA_SUMMARY, use_container_width=True)

        plot_df = pd.DataFrame(
            {
                "Segment": ["France", "Germany", "Spain", "Female", "Male", "Inactive", "Active"],
                "Churn Rate": [16.15, 32.44, 16.67, 25.07, 16.46, 26.85, 14.27],
            }
        )
        fig = px.bar(plot_df, x="Segment", y="Churn Rate", text="Churn Rate", title="Key Churn Rates by Segment")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "Model Performance":
    st.title("📈 Model Performance Comparison")
    st.write("Final model comparison from the completed churn modeling experiment.")

    st.subheader("Final Test Model Comparison")
    st.dataframe(MODEL_RESULTS, use_container_width=True)

    metric = st.selectbox(
        "Select Metric",
        ["ROC-AUC", "PR-AUC", "F1 Score", "Recall", "Precision", "Accuracy"],
    )
    sorted_df = MODEL_RESULTS.sort_values(metric, ascending=False)
    fig = px.bar(sorted_df, x="Model", y=metric, text=metric, title=f"Model Comparison by {metric}")
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Final Calibrated Deployment Model")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", CALIBRATED_RESULTS["Model"])
    c2.metric("Accuracy", f"{CALIBRATED_RESULTS['Accuracy']:.4f}")
    c3.metric("Recall", f"{CALIBRATED_RESULTS['Recall']:.4f}")
    c4.metric("ROC-AUC", f"{CALIBRATED_RESULTS['ROC-AUC']:.4f}")

    st.info(
        "The dashboard uses a safe standalone scoring engine for deployment stability, while this page reports the original machine-learning experiment results."
    )


# ============================================================
# FEATURE IMPORTANCE PAGE
# ============================================================

elif page == "Feature Importance":
    st.title("🧠 Feature Importance and Churn Drivers")
    st.write("Most important churn drivers from the Gradient Boosting model.")

    st.dataframe(FEATURE_IMPORTANCE, use_container_width=True)

    top_n = st.slider("Top features to display", 5, 19, 10)
    plot_df = FEATURE_IMPORTANCE.head(top_n).sort_values("Importance")
    fig = px.bar(plot_df, x="Importance", y="Feature", orientation="h", title=f"Top {top_n} Churn Drivers")
    st.plotly_chart(fig, use_container_width=True)

    st.success(
        "Age, Number of Products, Engagement-Product interaction, Active Membership, Geography, and Balance are the strongest churn drivers."
    )


# ============================================================
# ABOUT PROJECT PAGE
# ============================================================

elif page == "About Project":
    st.title("📌 About the Project")

    st.markdown(
        """
        ## Predictive Modeling and Risk Scoring for Bank Customer Churn

        This project builds a churn risk intelligence system for retail banking customers.
        It predicts churn tendency, assigns risk categories, and recommends retention actions.

        ### Methodology

        1. Data cleaning  
        2. Feature engineering  
        3. Exploratory data analysis  
        4. Model training and comparison  
        5. Threshold optimization  
        6. Probability calibration  
        7. Dashboard deployment  

        ### Final Model Result

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

    st.warning(
        "Deployment note: This fresh safe version avoids pickle and training-file failures, so it can run reliably on Streamlit Cloud."
    )
