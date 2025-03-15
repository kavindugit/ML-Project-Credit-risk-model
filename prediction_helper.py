import joblib
import pandas as pd
import numpy as np
import streamlit as st
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

# Path to the saved model
MODEL_PATH = 'artifacts/model_data.joblib'

# Load the model and its components
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']  # Ensure this matches model training
cols_to_scale = model_data['cols_to_scale']


# Ensure all features exist
def prepare_df(age, loan_amount, income, loan_tenure_months,
               delinquent_ratio, sanction_amount, loan_purpose, loan_type,
               residence_type, avg_dpd, credit_utilization_ratio, number_of_open_accounts):
    # Construct input data dictionary
    input_data = {
        'age': age,
        'sanction_amount': sanction_amount,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': number_of_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_amount / income if income > 0 else 0,
        'delinquent_ratio': delinquent_ratio,
        'avg_dpd_per_delinquent_month': avg_dpd,

        # Encoding categorical variables as one-hot
        'residence_type_Owned': int(residence_type == "Owned"),
        'residence_type_Rented': int(residence_type == "Rented"),

        'loan_purpose_Education': int(loan_purpose == "Education"),
        'loan_purpose_Home': int(loan_purpose == "Home"),
        'loan_purpose_Personal': int(loan_purpose == "Personal"),

        'loan_type_Unsecured': int(loan_type == "Unsecured"),

        # Additional Features (dummy values to maintain structure)
        'number_of_dependants': 1,
        'years_at_current_address': 1,
        'zipcode': 1,
        'processing_fee': 1,
        'gst': 1,
        'net_disbursement': 1,
        'principal_outstanding': 1,
        'bank_balance_at_application': 1,
        'number_of_closed_accounts': 1,
        'enquiry_count': 1,
        'loan_amount': 1
    }

    df = pd.DataFrame([input_data])

    # Ensure only existing columns are scaled
    for col in cols_to_scale:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default values

    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Ensure order matches model training
    df = df[features]

    return df


def calculate_credit_score(input_df, base_score=300, scale_length=600):
    # Ensure input shape matches model expectations
    x = np.dot(input_df.values, model.coef_.T) + model.intercept_

    default_probability = expit(x)
    non_default_probability = 1 - default_probability

    credit_score = base_score + non_default_probability.flatten() * scale_length

    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score < 900:
            return 'Excellent'
        else:
            return 'Undefined'

    rating = get_rating(credit_score[0])

    return default_probability.flatten()[0], int(credit_score[0]), rating


def predict(age, loan_amount, income, loan_tenure_months,
            delinquent_ratio, sanction_amount, loan_purpose, loan_type,
            residence_type, avg_dpd, credit_utilization_ratio, number_of_open_accounts):
    input_df = prepare_df(age, loan_amount, income, loan_tenure_months,
                          delinquent_ratio, sanction_amount, loan_purpose, loan_type,
                          residence_type, avg_dpd, credit_utilization_ratio, number_of_open_accounts)

    probability, credit_score, rating = calculate_credit_score(input_df)

    return probability, credit_score, rating
