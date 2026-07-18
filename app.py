import os
import pickle

import pandas as pd
import streamlit as st


# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide",
)


# -------------------------------------------------
# Project and model paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "loan_default_model.pkl",
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl",
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_columns.pkl",
)


# -------------------------------------------------
# Load trained files
# -------------------------------------------------
@st.cache_resource
def load_prediction_files():
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    with open(SCALER_PATH, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    with open(FEATURES_PATH, "rb") as features_file:
        feature_columns = pickle.load(features_file)

    return model, scaler, feature_columns


try:
    model, scaler, feature_columns = load_prediction_files()

except FileNotFoundError:
    st.error(
        "Model files were not found. "
        "Run `python src/train_model.py` first."
    )
    st.stop()

except Exception as error:
    st.error(f"Could not load the prediction files: {error}")
    st.stop()


# -------------------------------------------------
# Custom styling
# -------------------------------------------------
st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 44px;
            font-weight: 800;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #8a8a8a;
            margin-bottom: 30px;
        }

        .result-safe {
            padding: 20px;
            border-radius: 12px;
            background-color: rgba(40, 167, 69, 0.15);
            border: 1px solid #28a745;
            text-align: center;
        }

        .result-risk {
            padding: 20px;
            border-radius: 12px;
            background-color: rgba(220, 53, 69, 0.15);
            border: 1px solid #dc3545;
            text-align: center;
        }

        div.stButton > button {
            width: 100%;
            height: 50px;
            font-size: 18px;
            font-weight: 700;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------
# Heading
# -------------------------------------------------
st.markdown(
    '<div class="main-title">🏦 Loan Default Prediction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Enter the applicant's financial and personal information
        to estimate loan-default risk.
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------
# Input form
# -------------------------------------------------
with st.form("loan_default_form"):

    st.subheader("Applicant Financial Information")

    column1, column2, column3 = st.columns(3)

    with column1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1,
        )

        income = st.number_input(
            "Annual Income",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=15000.0,
            step=500.0,
        )

    with column2:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=650,
            step=1,
        )

        months_employed = st.number_input(
            "Months Employed",
            min_value=0,
            max_value=600,
            value=36,
            step=1,
        )

        number_credit_lines = st.number_input(
            "Number of Credit Lines",
            min_value=0,
            max_value=50,
            value=3,
            step=1,
        )

    with column3:
        interest_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.1,
        )

        loan_term = st.number_input(
            "Loan Term (Months)",
            min_value=1,
            max_value=480,
            value=36,
            step=1,
        )

        dti_ratio = st.number_input(
            "Debt-to-Income Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.01,
            format="%.2f",
        )

    st.divider()
    st.subheader("Personal and Loan Details")

    column4, column5, column6 = st.columns(3)

    with column4:
        education = st.selectbox(
            "Education",
            [
                "Bachelor's",
                "High School",
                "Master's",
                "PhD",
            ],
        )

        employment_type = st.selectbox(
            "Employment Type",
            [
                "Full-time",
                "Part-time",
                "Self-employed",
                "Unemployed",
            ],
        )

        marital_status = st.selectbox(
            "Marital Status",
            [
                "Divorced",
                "Married",
                "Single",
            ],
        )

    with column5:
        has_mortgage = st.selectbox(
            "Has Mortgage?",
            ["No", "Yes"],
        )

        has_dependents = st.selectbox(
            "Has Dependents?",
            ["No", "Yes"],
        )

        has_cosigner = st.selectbox(
            "Has Co-Signer?",
            ["No", "Yes"],
        )

    with column6:
        loan_purpose = st.selectbox(
            "Loan Purpose",
            [
                "Auto",
                "Business",
                "Education",
                "Home",
                "Other",
            ],
        )

    consent = st.checkbox(
        "I confirm that the provided information is correct."
    )

    predict_button = st.form_submit_button(
        "Predict Default Risk"
    )


# -------------------------------------------------
# Prediction
# -------------------------------------------------
if predict_button:

    if not consent:
        st.warning(
            "Please confirm that the provided information is correct."
        )

    elif income <= 0:
        st.error("Annual income must be greater than zero.")

    elif loan_amount <= 0:
        st.error("Loan amount must be greater than zero.")

    else:
        input_data = {
            "Age": age,
            "Income": income,
            "LoanAmount": loan_amount,
            "CreditScore": credit_score,
            "MonthsEmployed": months_employed,
            "NumCreditLines": number_credit_lines,
            "InterestRate": interest_rate,
            "LoanTerm": loan_term,
            "DTIRatio": dti_ratio,
        }

        # Start all encoded feature values from zero
        applicant_df = pd.DataFrame(
            0,
            index=[0],
            columns=feature_columns,
            dtype=float,
        )

        # Add numerical values
        for feature_name, value in input_data.items():
            applicant_df.loc[0, feature_name] = value

        # Education encoding
        if education != "Bachelor's":
            education_column = f"Education_{education}"

            if education_column in applicant_df.columns:
                applicant_df.loc[0, education_column] = 1

        # Employment encoding
        if employment_type != "Full-time":
            employment_column = (
                f"EmploymentType_{employment_type}"
            )

            if employment_column in applicant_df.columns:
                applicant_df.loc[0, employment_column] = 1

        # Marital-status encoding
        if marital_status != "Divorced":
            marital_column = (
                f"MaritalStatus_{marital_status}"
            )

            if marital_column in applicant_df.columns:
                applicant_df.loc[0, marital_column] = 1

        # Yes/No encoding
        if has_mortgage == "Yes":
            applicant_df.loc[0, "HasMortgage_Yes"] = 1

        if has_dependents == "Yes":
            applicant_df.loc[0, "HasDependents_Yes"] = 1

        if has_cosigner == "Yes":
            applicant_df.loc[0, "HasCoSigner_Yes"] = 1

        # Loan-purpose encoding
        if loan_purpose != "Auto":
            purpose_column = f"LoanPurpose_{loan_purpose}"

            if purpose_column in applicant_df.columns:
                applicant_df.loc[0, purpose_column] = 1

        try:
            scaled_data = scaler.transform(applicant_df)

            prediction = int(model.predict(scaled_data)[0])

            default_probability = float(
                model.predict_proba(scaled_data)[0][1]
            )

            st.divider()
            st.subheader("Prediction Result")

            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="result-risk">
                        <h2>⚠️ High Default Risk</h2>
                        <p>
                            The model predicts that this applicant
                            may default on the loan.
                        </p>
                        <h3>
                            Estimated Default Probability:
                            {default_probability:.2%}
                        </h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:
                st.markdown(
                    f"""
                    <div class="result-safe">
                        <h2>✅ Low Default Risk</h2>
                        <p>
                            The model predicts that this applicant
                            is less likely to default on the loan.
                        </p>
                        <h3>
                            Estimated Default Probability:
                            {default_probability:.2%}
                        </h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.progress(
                min(max(default_probability, 0.0), 1.0)
            )

        except Exception as error:
            st.error(f"Prediction could not be completed: {error}")


# -------------------------------------------------
# Footer
# -------------------------------------------------
st.divider()

st.caption(
    "Disclaimer: This application is an educational machine-learning "
    "project. Its prediction should not be used as the sole basis for "
    "real financial or lending decisions."
)