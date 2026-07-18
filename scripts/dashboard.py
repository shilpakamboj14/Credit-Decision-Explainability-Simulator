import streamlit as st
import pandas as pd
import joblib
from scripts.explain import get_decision_text

# Load our already-trained model and scaler (no retraining happens here)
model = joblib.load("/Users/shilpa/credit_project/data/model.pkl")
scaler = joblib.load("/Users/shilpa/credit_project/data/scaler.pkl")

# Load the cleaned dataset — needed as SHAP's "baseline" for comparison
data = pd.read_csv("/Users/shilpa/credit_project/data/cleaned_data.csv")
features = data.drop(["SeriousDlqin2yrs", "cust_id"], axis=1)
features_scaled = scaler.transform(features)

# --- Page header ---
st.set_page_config(layout="wide")
st.title("💳 Credit Decision Explainability Simulator")
st.caption("An ML-based credit decisioning tool that explains every decision in plain English — built to reflect ECOA adverse action notice requirements.")

st.markdown("---")

# --- Applicant details section ---
st.subheader("Applicant Details")
st.write("Enter the applicant's financial details below, then click **Check Application**.")

col1, col2 = st.columns(2)

with col1:
    utilization = st.number_input("Revolving Utilization (0 to 1)", min_value=0.0, max_value=1.0, value=0.3)
    late_30_59 = st.number_input("Number of Times 30-59 Days Late", min_value=0, max_value=20, value=0)
    monthly_income = st.number_input("Monthly Income", min_value=0, max_value=100000, value=4000)

with col2:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    debt_ratio = st.number_input("Debt Ratio", min_value=0.0, max_value=5.0, value=0.3)
    late_90 = st.number_input("Number of Times 90+ Days Late", min_value=0, max_value=20, value=0)

st.markdown("")

# Everything below only runs when the button is clicked
if st.button("🔍 Check Application", use_container_width=True):

    # Build a one-row table from the user's inputs, in the same
    # column order the model was trained on
    new_applicant = pd.DataFrame({
        "RevolvingUtilizationOfUnsecuredLines": [utilization],
        "age": [age],
        "NumberOfTime30-59DaysPastDueNotWorse": [late_30_59],
        "DebtRatio": [debt_ratio],
        "MonthlyIncome": [monthly_income],
        "NumberOfTimes90DaysLate": [late_90]
    })

    # Scale the applicant's data the same way training data was scaled
    applicant_scaled = scaler.transform(new_applicant)

    # 0 = will not default (approve), 1 = will default (decline)
    prediction = model.predict(applicant_scaled)

    # If declined, this function uses SHAP to find the top reasons
    # and turns them into a plain-English sentence
    result = get_decision_text(model, features_scaled, features.columns, applicant_scaled, prediction)

    st.markdown("---")
    st.subheader("Decision")

    if prediction[0] == 1:
        st.error(result)
    else:
        st.success(result)

# --- Methodology note, shown to anyone using the app ---
st.markdown("---")
with st.expander("📋 How this works"):
    st.markdown("""
1. **Model:** A logistic regression model, trained on historical credit data,
   predicts whether an applicant is likely to default.

2. **Class imbalance handling:** Since very few applicants in the training data
   actually default, the model is weighted (`class_weight = {0: 1, 1: 9}`) so it
   pays more attention to catching real defaulters, not just chasing overall accuracy.

3. **Explainability (SHAP):** For every declined application, SHAP calculates how
   much each factor (late payments, income, age, etc.) pushed the decision toward
   "high risk," compared to a typical applicant in the dataset.

4. **Top reasons only:** In line with real adverse action notice requirements
   (ECOA), only the top 3 contributing factors are shown — not a technical dump
   of all inputs.

5. **Approved applications:** No reason is shown for approvals, since regulations
   only require lenders to justify **declines**, not approvals.
""")