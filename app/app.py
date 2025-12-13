import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Credit Risk Assessment Platform",
    layout="wide"
)

# =====================================================
# Load Model
# =====================================================
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "..", "model", "model.pkl")
    return joblib.load(model_path)

model = load_model()

# =====================================================
# Prediction Logic
# =====================================================
def predict_risk(df: pd.DataFrame, threshold: float):
    probabilities = model.predict_proba(df)
    prob_good = float(probabilities[:, 2][0])
    risk_label = "Low Risk" if prob_good >= threshold else "High Risk"
    return risk_label, prob_good

# =====================================================
# Header
# =====================================================
st.markdown("# Credit Risk Assessment Platform")
st.markdown(
    "This platform provides **decision support** for retail credit evaluation "
    "using statistical learning models trained on historical financial behavior."
)

st.divider()

# =====================================================
# System Documentation (Inline)
# =====================================================
with st.expander("System Documentation", expanded=True):
    st.markdown("""
    ### Purpose
    This application evaluates applicant financial characteristics and produces
    a **binary risk classification** to support underwriting decisions.

    ### Model Overview
    - **Training Objective**: Multiclass classification
        - Poor
        - Standard
        - Good
    - **Exposed Output**:
        - Low Risk (Good)
        - High Risk (Poor + Standard)

    ### Decision Logic
    The model estimates the probability that an applicant belongs to the
    *Good* credit class. A configurable policy threshold is applied to determine
    the final risk category.

    ### Intended Use
    - Internal credit assessment
    - Application pre-screening
    - Decision support for human underwriters

    ### Prohibited Use
    - Fully automated approvals or rejections
    - Regulatory decisions without human oversight
    """)

# =====================================================
# Risk Policy Configuration
# =====================================================
st.markdown("## Risk Policy Configuration")

with st.expander("Policy Definition and Governance"):
    st.markdown("""
    The approval threshold defines the **minimum probability** required for an
    applicant to be classified as *Low Risk*.

    **Higher thresholds**
    - Reduce approval rates
    - Lower portfolio risk
    - Increase manual review volume

    **Lower thresholds**
    - Increase approval rates
    - Increase portfolio risk
    - Reduce manual intervention
    """)

threshold = st.slider(
    "Approval Probability Threshold",
    0.40, 0.80, 0.60, 0.05
)

st.divider()

# =====================================================
# Applicant Input
# =====================================================
st.markdown("## Applicant Financial Profile")

with st.expander("Input Data Definitions", expanded=False):
    st.markdown("""
    **Age**  
    Applicant age in years.

    **Annual Income**  
    Total gross annual income from all reported sources.

    **Monthly In-hand Salary**  
    Net monthly income after deductions.

    **Spend Level**  
    Normalized indicator of discretionary spending behavior.

    **Credit Mix**  
    Overall quality and diversity of existing credit products.

    **Minimum Payment Behavior**  
    Consistency in meeting minimum payment obligations.

    **Credit Utilization Ratio**  
    Percentage of available credit currently utilized.

    **Credit History Length**  
    Duration of recorded credit history in months.
    """)

with st.form("credit_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 30)
        annual_income = st.number_input("Annual Income", 0, 1_000_000, 60000)
        monthly_salary = st.number_input("Monthly In-hand Salary", 0, 200000, 5000)
        spend_level = st.slider("Spend Level", 0.0, 1.0, 0.5)

    with col2:
        credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good", "Unknown"])
        payment_min = st.selectbox("Minimum Payment Behavior", ["No", "Unknown", "Yes"])
        credit_util = st.slider("Credit Utilization Ratio (%)", 0.0, 100.0, 30.0)
        credit_history = st.number_input("Credit History Length (months)", 0, 500, 120)

    submitted = st.form_submit_button("Evaluate Risk")

# =====================================================
# Risk Assessment Outcome (FINAL, FIXED)
# =====================================================
if submitted:
    applicant_df = pd.DataFrame([{
        "Age": age,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_salary,
        "Num_Bank_Accounts": 2,
        "Num_Credit_Card": 2,
        "Interest_Rate": 8,
        "Num_of_Loan": 1,
        "Num_of_Delayed_Payment": 0,
        "Changed_Credit_Limit": 5,
        "Num_Credit_Inquiries": 2,
        "Credit_Mix": credit_mix,
        "Credit_Utilization_Ratio": credit_util,
        "Credit_History_Age": credit_history,
        "Payment_of_Min_Amount": payment_min,
        "Total_EMI_per_month": 300,
        "Amount_invested_monthly": 400,
        "Monthly_Balance": 3000,
        "Spend_Level": spend_level
    }])

    risk, confidence = predict_risk(applicant_df, threshold)

    st.divider()
    st.markdown("## Risk Assessment Outcome")

    st.markdown(f"""
    **Risk Classification:** {risk}  
    **Probability of Good Credit Standing:** {confidence:.2%}
    """)

    st.progress(confidence)

    with st.expander("Decision Interpretation, Recommended Action, and Governance", expanded=True):

        st.markdown("""
        ### Decision Interpretation
        The probability score represents the model’s estimated likelihood that the
        applicant demonstrates credit behavior consistent with historically
        low-risk profiles.

        The final risk classification is derived by applying the institution’s
        policy-defined approval threshold to this probability.
        """)

        if risk == "High Risk":
            st.markdown("""
            ### Assessment Outcome
            The applicant does **not meet the automated approval criteria**
            under the current credit policy configuration.

            This indicates elevated risk relative to approved portfolio benchmarks.
            """)
        else:
            st.markdown("""
            ### Assessment Outcome
            The applicant meets the automated approval criteria under the
            current credit policy configuration and is classified as low risk.
            """)

        st.markdown("""
        ### Recommended Action
        - Route the application for manual credit review
        - Perform additional verification of income stability and liabilities
        - Apply enhanced due-diligence checks where required

        This result does **not** represent an automatic approval or rejection.
        """)

        st.markdown("""
        ### Model Scope and Limitations
        - Unreported liabilities are not captured
        - Sudden life events are not reflected
        - Predictions are based on historical patterns and may not fully
          represent future behavior
        """)

        st.markdown("""
        ### Governance and Compliance Notice
        This output is provided strictly as a **decision-support signal**.

        Final credit decisions must be made in accordance with internal credit
        policy, regulatory requirements, and qualified human judgment.
        """)

# =====================================================
# Footer
# =====================================================
st.divider()
st.caption(
    "Confidential Internal System • Credit Risk Assessment Platform • 2025"
)
