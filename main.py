import streamlit as st
from prediction_helper import predict

# Reduce space after title
st.markdown("<h1 style='margin-bottom: -60px;'>Lauki Finance: Credit Risk Modeling</h1>", unsafe_allow_html=True)

# User inputs
row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)
row5 = st.columns(1)

with row1[0]:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
with row1[1]:
    income = st.number_input("Income", min_value=1, value=50000)  # Prevents division by zero
with row1[2]:
    loan_amount = st.number_input("Loan Amount", min_value=1000, value=2560000)  # Prevents 0 input

loan_to_income_ratio = loan_amount / income if income > 0 else 0
with row2[0]:
    st.text("Loan_to_income_ratio:")
    st.text(f"{loan_to_income_ratio:.2f}")

with row2[1]:
    loan_tenure_months = st.number_input("Loan Tenure (months)", min_value=1, max_value=360, value=36)
with row2[2]:
    delinquent_ratio = st.number_input("Delinquent Ratio (%)", min_value=0.0, max_value=100.0, value=5.0)

with row3[0]:
    sanction_amount = st.number_input("Sanction Amount", min_value=1000, value=500000)
with row3[1]:
    loan_purpose = st.selectbox("Loan Purpose", ["Education", "Home", "Personal", "Auto"])
with row3[2]:
    loan_type = st.selectbox("Loan Type", ["Unsecured", "Secured"])

with row4[0]:
    residence_type = st.selectbox("Residence Type", ["Owned", "Rented", "Mortgage"])
with row4[1]:
    avg_dpd = st.number_input("Avg DPD", min_value=0, max_value=100, value=2)
with row4[2]:
    credit_utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0,
                                               value=30.0)

with row5[0]:
    number_of_open_accounts = st.number_input("Number of Open Accounts", min_value=0, max_value=50, value=1)

# Fix argument order
if st.button("Calculate Risk"):
    try:
        probability, credit_score, rating = predict(
            age, loan_amount, income, loan_tenure_months, delinquent_ratio, sanction_amount,
            loan_purpose, loan_type, residence_type, avg_dpd, credit_utilization_ratio,
            number_of_open_accounts
        )

        st.write(f"**Probability of default:** {probability:.2%}")
        st.write(f"**Credit Score:** {credit_score}")
        st.write(f"**Rating:** {rating}")

    except Exception as e:
        st.error(f"Error: {e}")
