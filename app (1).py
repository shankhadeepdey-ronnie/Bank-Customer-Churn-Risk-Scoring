# ============================================================
# BANK CUSTOMER CHURN RISK SCORING DASHBOARD
# Fresh Safe Streamlit App
# - Does NOT load any .pkl model
# - Trains a fresh Gradient Boosting model from CSV files
# - Safely reads CSV files with encoding and bad-line protection
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bank Customer Churn Risk Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GLOBAL SETTINGS
# ============================================================

DEFAULT_THRESHOLD = 0.33

BASE_DEFAULTS = {
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Male",
    "Age": 40,
    "Tenure": 5,
    "Balance": 75000.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 100000.0,
}

NUMERIC_BASE_COLUMNS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

CATEGORICAL_BASE_COLUMNS = ["Geography", "Gender"]

DROP_COLUMNS = [
    "RowNumber",
    "CustomerId",
    "Surname",
    "Exited",
    "Churn",
    "Target",
    "target",
    "y",
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
    .small-note {
        color: #64748b;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SAFE FILE HELPERS
# ============================================================

def first_existing_path(candidate_paths):
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None


@st.cache_data(show_spinner=False)
def safe_read_csv(path):
    """
    Read CSV safely. This handles common encoding issues and skips broken lines.
    Returns None if the file cannot be read.
    """
    encodings = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1", "cp1252"]

    for enc in encodings:
        try:
            df = pd.read_csv(
                path,
                encoding=enc,
                engine="python",
                on_bad_lines="skip"
            )
            if df is not None and len(df.columns) > 0:
                df.columns = [str(col).strip() for col in df.columns]
                unnamed_cols = [col for col in df.columns if col.lower().startswith("unnamed")]
                df = df.drop(columns=unnamed_cols, errors="ignore")
                return df
        except Exception:
            continue

    return None


def normalize_target(y):
    """
    Converts target values to 0/1 safely.
    """
    y = pd.Series(y).reset_index(drop=True)

    if y.dtype == "object":
        y_clean = y.astype(str).str.strip().str.lower()
        mapping = {
            "1": 1,
            "0": 0,
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "churn": 1,
            "churned": 1,
            "exited": 1,
            "left": 1,
            "retained": 0,
            "stayed": 0,
            "active": 0,
        }
        y_mapped = y_clean.map(mapping)
        if y_mapped.notna().sum() > 0:
            return y_mapped

    y_numeric = pd.to_numeric(y, errors="coerce")

    unique_vals = sorted(y_numeric.dropna().unique().tolist())
    if set(unique_vals).issubset({0, 1}):
        return y_numeric

    if len(unique_vals) == 2:
        low, high = unique_vals[0], unique_vals[1]
        return y_numeric.map({low: 0, high: 1})

    return y_numeric


def extract_y_from_file(y_df):
    """
    Finds target column from y_train.csv safely.
    """
    if y_df is None or y_df.empty:
        return None

    preferred_names = ["Exited", "Churn", "target", "Target", "y", "label", "Label"]
    for col in preferred_names:
        if col in y_df.columns:
            return y_df[col]

    # If only one column exists, use it.
    if len(y_df.columns) == 1:
        return y_df.iloc[:, 0]

    # Otherwise find a binary-looking column.
    for col in y_df.columns:
        candidate = normalize_target(y_df[col])
        valid_unique = set(candidate.dropna().unique().tolist())
        if valid_unique.issubset({0, 1}) and len(valid_unique) >= 2:
            return y_df[col]

    return y_df.iloc[:, -1]


# ============================================================
# TRAINING DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_training_data():
    """
    Priority:
    1. X_train_raw.csv + y_train.csv
    2. A full dataset containing Exited
    """
    x_path = first_existing_path([
        "X_train_raw.csv",
        "dataset/X_train_raw.csv",
        "data/X_train_raw.csv",
        "outputs/X_train_raw.csv",
    ])

    y_path = first_existing_path([
        "y_train.csv",
        "dataset/y_train.csv",
        "data/y_train.csv",
        "outputs/y_train.csv",
    ])

    if x_path is not None and y_path is not None:
        X = safe_read_csv(x_path)
        y_df = safe_read_csv(y_path)
        y_raw = extract_y_from_file(y_df)

        if X is not None and y_raw is not None:
            n = min(len(X), len(y_raw))
            X = X.iloc[:n].reset_index(drop=True)
            y = normalize_target(y_raw.iloc[:n]).reset_index(drop=True)

            valid_mask = y.notna()
            X = X.loc[valid_mask].reset_index(drop=True)
            y = y.loc[valid_mask].astype(int).reset_index(drop=True)

            if len(X) > 50 and y.nunique() == 2:
                return X, y, f"{x_path} + {y_path}"

    full_dataset_path = first_existing_path([
        "European_Bank_Clean_Featured.csv",
        "dataset/European_Bank_Clean_Featured.csv",
        "European_Bank_Feature_Engineered.csv",
        "dataset/European_Bank_Feature_Engineered.csv",
        "outputs/European_Bank_Feature_Engineered.csv",
        "European_Bank.csv",
        "dataset/European_Bank.csv",
        "data/European_Bank.csv",
    ])

    if full_dataset_path is not None:
        df = safe_read_csv(full_dataset_path)
        if df is not None and "Exited" in df.columns:
            y = normalize_target(df["Exited"])
            X = df.drop(columns=["Exited"], errors="ignore")

            valid_mask = y.notna()
            X = X.loc[valid_mask].reset_index(drop=True)
            y = y.loc[valid_mask].astype(int).reset_index(drop=True)

            if len(X) > 50 and y.nunique() == 2:
                return X, y, full_dataset_path

    return None, None, None


@st.cache_data(show_spinner=False)
def load_display_dataset():
    """
    Loads data for analytics page. It uses training files first because the full
    European_Bank.csv may be corrupted in GitHub.
    """
    X, y, source = load_training_data()
    if X is None or y is None:
        return None, None

    display_df = X.copy()
    display_df["Exited"] = y.values
    return display_df, source


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def prepare_features(input_df):
    """
    Cleans and engineers features safely for both training and prediction.
    """
    df = input_df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    unnamed_cols = [col for col in df.columns if col.lower().startswith("unnamed")]
    df = df.drop(columns=unnamed_cols, errors="ignore")

    # Drop ID/target columns if present.
    df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns], errors="ignore")

    # Drop Year only if it is constant.
    if "Year" in df.columns:
        try:
            if df["Year"].nunique(dropna=True) <= 1:
                df = df.drop(columns=["Year"], errors="ignore")
        except Exception:
            df = df.drop(columns=["Year"], errors="ignore")

    # Ensure base columns exist.
    for col, default_value in BASE_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default_value

    # Clean categorical base columns.
    for col in CATEGORICAL_BASE_COLUMNS:
        df[col] = df[col].fillna(BASE_DEFAULTS[col]).astype(str).str.strip()
        df[col] = df[col].replace({"": BASE_DEFAULTS[col], "nan": BASE_DEFAULTS[col]})

    # Clean numeric base columns.
    for col in NUMERIC_BASE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(BASE_DEFAULTS[col])

    # Feature engineering.
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

    # Final cleaning for every column.
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown").astype(str)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def align_prediction_features(df, training_columns, categorical_columns):
    """
    Ensures prediction dataframe has exactly the columns used during training.
    """
    aligned = df.copy()

    for col in training_columns:
        if col not in aligned.columns:
            if col in categorical_columns:
                aligned[col] = "Unknown"
            else:
                aligned[col] = 0

    aligned = aligned[training_columns]
    return aligned


# ============================================================
# MODEL TRAINING
# ============================================================

@st.cache_resource(show_spinner="Training safe Gradient Boosting model from CSV files...")
def train_model_bundle():
    X_raw, y, data_source = load_training_data()

    if X_raw is None or y is None:
        st.error(
            "No valid training data found. Please keep `X_train_raw.csv` and `y_train.csv` "
            "in the root of your GitHub repo."
        )
        st.stop()

    X = prepare_features(X_raw)

    training_columns = X.columns.tolist()
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = [col for col in training_columns if col not in categorical_columns]

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

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_columns),
            ("num", numeric_transformer, numeric_columns),
        ],
        remainder="drop"
    )

    gb_model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        min_samples_leaf=25,
        min_samples_split=50,
        subsample=0.90,
        max_features="sqrt",
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", gb_model),
        ]
    )

    pipeline.fit(X, y)

    return {
        "model": pipeline,
        "threshold": DEFAULT_THRESHOLD,
        "source": data_source,
        "training_columns": training_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "n_rows": len(X),
        "n_features": len(training_columns),
    }


# ============================================================
# TABLES
# ============================================================

@st.cache_data(show_spinner=False)
def load_model_comparison():
    possible_paths = [
        "final_test_model_comparison.csv",
        "outputs/final_test_model_comparison.csv",
        "model_comparison_step7.csv",
        "outputs/model_comparison_step7.csv",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            df = safe_read_csv(path)
            if df is not None and not df.empty:
                return df

    return pd.DataFrame({
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
    })


@st.cache_data(show_spinner=False)
def load_feature_importance():
    possible_paths = [
        "final_feature_importance.csv",
        "outputs/final_feature_importance.csv",
        "random_forest_feature_importance.csv",
        "outputs/random_forest_feature_importance.csv",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            df = safe_read_csv(path)
            if df is not None and not df.empty:
                return df

    return pd.DataFrame({
        "Feature": [
            "Age", "NumOfProducts", "Engagement_Product", "IsActiveMember",
            "Geography_Germany", "Balance", "Age_Tenure", "Balance_to_Salary",
            "Gender_Male", "Product_Density", "EstimatedSalary", "CreditScore",
            "Senior_Customer_Flag", "Zero_Balance_Flag", "High_Balance_Flag",
            "Tenure", "Geography_Spain", "HasCrCard", "Low_CreditScore_Flag",
        ],
        "Importance": [
            0.318838, 0.261072, 0.091346, 0.063453,
            0.048965, 0.042263, 0.036429, 0.027350,
            0.020053, 0.018609, 0.014804, 0.013665,
            0.012677, 0.010221, 0.007771,
            0.007351, 0.002216, 0.002060, 0.000859,
        ],
    })


# ============================================================
# RISK AND PREDICTION HELPERS
# ============================================================

def assign_risk_category(probability):
    if probability < 0.30:
        return "Low Risk"
    if probability < 0.60:
        return "Medium Risk"
    if probability < 0.80:
        return "High Risk"
    return "Very High Risk"


def retention_action(risk_category):
    if risk_category == "Low Risk":
        return "Maintain regular engagement and standard customer service."
    if risk_category == "Medium Risk":
        return "Send personalized engagement message and relevant product recommendation."
    if risk_category == "High Risk":
        return "Offer retention benefit, relationship-manager call, or suitable product bundle."
    return "Immediate retention intervention with personalized offer and priority support."


def create_gauge(probability):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": "Churn Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
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


def risk_box(risk_category):
    colors = {
        "Low Risk": ("#d8f3dc", "#166534"),
        "Medium Risk": ("#fff3cd", "#92400e"),
        "High Risk": ("#ffe5b4", "#b45309"),
        "Very High Risk": ("#f8d7da", "#991b1b"),
    }
    bg_color, text_color = colors.get(risk_category, ("#e5e7eb", "#111827"))

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


def predict_customer(customer_df, bundle):
    prepared = prepare_features(customer_df)
    aligned = align_prediction_features(
        prepared,
        bundle["training_columns"],
        bundle["categorical_columns"],
    )

    probability = bundle["model"].predict_proba(aligned)[:, 1][0]
    risk = assign_risk_category(probability)
    action = retention_action(risk)

    return probability, risk, action, aligned


# ============================================================
# LOAD APP OBJECTS
# ============================================================

model_bundle = train_model_bundle()
df, dataset_source = load_display_dataset()
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
        "About Project",
    ],
)

st.sidebar.markdown("---")
st.sidebar.metric("Model", "Gradient Boosting")
st.sidebar.metric("Threshold", f"{model_bundle['threshold']:.2f}")
st.sidebar.metric("Training Rows", f"{model_bundle['n_rows']:,}")
st.sidebar.metric("Features", f"{model_bundle['n_features']:,}")

with st.sidebar.expander("Loaded safely"):
    st.write("Training source:", model_bundle["source"])
    st.write("Display source:", dataset_source)
    st.write("No `.pkl` model is being loaded.")


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

    if df is not None and "Exited" in df.columns:
        total_customers = len(df)
        churn_rate = df["Exited"].mean() * 100
    else:
        total_customers = model_bundle["n_rows"]
        churn_rate = 20.37

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers Used", f"{total_customers:,}")
    col2.metric("Observed Churn Rate", f"{churn_rate:.2f}%")
    col3.metric("Model Type", "Gradient Boosting")
    col4.metric("Risk Threshold", f"{model_bundle['threshold']:.2f}")

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

    st.success(
        "This fresh version avoids the broken pickle model and trains safely from CSV files."
    )


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
            balance = st.number_input("Balance", min_value=0.0, value=75000.0, step=1000.0)

        with col3:
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
            has_cr_card = st.selectbox(
                "Has Credit Card?",
                [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
            is_active = st.selectbox(
                "Is Active Member?",
                [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
            estimated_salary = st.number_input(
                "Estimated Salary",
                min_value=1.0,
                value=100000.0,
                step=1000.0,
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
            "EstimatedSalary": [estimated_salary],
        })

        probability, risk, action, prepared_customer = predict_customer(customer_df, model_bundle)

        st.markdown("---")
        st.subheader("Prediction Result")

        r1, r2, r3 = st.columns([1.4, 1, 1.3])

        with r1:
            st.plotly_chart(create_gauge(probability), use_container_width=True)

        with r2:
            st.metric("Churn Probability", f"{probability * 100:.2f}%")
            st.metric("Risk Threshold", f"{model_bundle['threshold']:.2f}")

        with r3:
            st.markdown("### Risk Category")
            risk_box(risk)
            st.markdown("### Recommended Action")
            st.success(action)

        st.markdown("### Prepared Customer Features")
        st.dataframe(prepared_customer, use_container_width=True)


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
        base_balance = st.number_input("Current Balance", min_value=0.0, value=75000.0, step=1000.0)

    with c3:
        base_products = st.selectbox("Current Number of Products", [1, 2, 3, 4])
        base_card = st.selectbox(
            "Current Credit Card Status",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        base_active = st.selectbox(
            "Current Active Status",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
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
            format_func=lambda x: "Yes" if x == 1 else "No",
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
            "EstimatedSalary": [base_salary],
        })

        scenario_customer = base_customer.copy()
        scenario_customer["NumOfProducts"] = new_products
        scenario_customer["IsActiveMember"] = new_active
        scenario_customer["Balance"] = new_balance

        base_prob, base_risk, _, _ = predict_customer(base_customer, model_bundle)
        new_prob, new_risk, _, _ = predict_customer(scenario_customer, model_bundle)
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
            "Churn Probability": [base_prob * 100, new_prob * 100],
        })

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

    if df is None:
        st.warning("No readable dataset found for analytics.")
    else:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

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
                title="Customer Churn Distribution",
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
                        title="Churn Rate by Geography",
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
                        title="Churn Rate by Gender",
                    )
                    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)

            b1, b2 = st.columns(2)

            with b1:
                if "IsActiveMember" in df.columns:
                    active_df = df.groupby("IsActiveMember")["Exited"].mean().reset_index()
                    active_df["Status"] = active_df["IsActiveMember"].map({0: "Inactive", 1: "Active"})
                    active_df["Churn Rate"] = active_df["Exited"] * 100
                    fig = px.bar(
                        active_df,
                        x="Status",
                        y="Churn Rate",
                        text="Churn Rate",
                        title="Churn Rate by Active Membership",
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
                        title="Churn Rate by Number of Products",
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
        This dashboard uses a safe live-trained Gradient Boosting model for deployment.
        The table below summarizes the model comparison from the project experiment.
        """
    )

    st.subheader("Final Test Model Comparison")
    st.dataframe(model_comparison_df, use_container_width=True)

    numeric_metrics = ["ROC-AUC", "PR-AUC", "F1 Score", "Recall", "Precision", "Accuracy"]
    available_metrics = [col for col in numeric_metrics if col in model_comparison_df.columns]

    if available_metrics and "Model" in model_comparison_df.columns:
        metric = st.selectbox("Select Metric", available_metrics)
        sorted_df = model_comparison_df.sort_values(metric, ascending=False)

        fig = px.bar(
            sorted_df,
            x="Model",
            y=metric,
            text=metric,
            title=f"Model Comparison by {metric}",
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Deployment Model")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "Gradient Boosting")
    c2.metric("Training Rows", f"{model_bundle['n_rows']:,}")
    c3.metric("Features", f"{model_bundle['n_features']:,}")
    c4.metric("Threshold", f"{model_bundle['threshold']:.2f}")

    st.info(
        "The app avoids pickle/joblib loading errors by training the model live from CSV files."
    )


# ============================================================
# FEATURE IMPORTANCE PAGE
# ============================================================

elif page == "Feature Importance":
    st.title("🧠 Feature Importance and Churn Drivers")

    st.write("Feature importance helps identify the customer attributes most related to churn prediction.")
    st.dataframe(feature_importance_df, use_container_width=True)

    importance_col = "Importance"
    if importance_col not in feature_importance_df.columns:
        numeric_cols = feature_importance_df.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns.tolist()
        if numeric_cols:
            importance_col = numeric_cols[0]

    if "Feature" in feature_importance_df.columns and importance_col in feature_importance_df.columns:
        top_n = st.slider(
            "Top features to display",
            5,
            min(20, len(feature_importance_df)),
            min(10, len(feature_importance_df)),
        )

        plot_df = feature_importance_df.head(top_n).sort_values(importance_col)
        fig = px.bar(
            plot_df,
            x=importance_col,
            y="Feature",
            orientation="h",
            title=f"Top {top_n} Churn Drivers",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.success(
        "The key churn drivers include Age, Number of Products, Active Membership, Geography, Balance, and Engagement behavior."
    )


# ============================================================
# ABOUT PAGE
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
        4. Model training  
        5. Risk thresholding  
        6. Customer-level churn prediction  
        7. What-if simulation  
        8. Streamlit dashboard deployment  

        ### Final Deployment Choice

        The dashboard uses a fresh live-trained **Gradient Boosting Classifier**.
        This was done to avoid deployment errors caused by incompatible `.pkl` model files.

        ### Business Value

        The dashboard helps banks identify high-risk customers early and take proactive retention actions.
        """
    )
