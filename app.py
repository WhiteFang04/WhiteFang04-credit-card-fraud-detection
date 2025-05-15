# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Real-Time Credit Card Fraud Detection")

# Load model and scalers
model = joblib.load("logistic_model.pkl")
scaler_amount = joblib.load("scaler_amount.pkl")
scaler_time = joblib.load("scaler_time.pkl")

st.sidebar.header("Transaction Input Features")

# Define correct feature order
feature_names = joblib.load("feature_columns.pkl")

def user_input_features():
    st.sidebar.markdown("### Input Transaction Details")

    input_data = {}
    for col in feature_names:
        if col == 'Time':
            input_data[col] = st.sidebar.number_input('Time (seconds)', min_value=0.0, format="%.2f")
        elif col == 'Amount':
            input_data[col] = st.sidebar.number_input('Amount', min_value=0.0, format="%.2f")
        else:
            input_data[col] = st.sidebar.slider(col, -5.0, 5.0, 0.0, step=0.01)

    return pd.DataFrame([input_data])
    st.sidebar.markdown("### Input Transaction Details")

    time = st.sidebar.number_input('Time (seconds)', min_value=0.0, format="%.2f")
    amount = st.sidebar.number_input('Amount', min_value=0.0, format="%.2f")

    v_features = {}
    for i in range(1, 29):
        v_features[f'V{i}'] = st.sidebar.slider(f'V{i}', -5.0, 5.0, 0.0, step=0.01)

    data = {'Time': time, 'Amount': amount}
    data.update(v_features)

    # Create dataframe with correct column order
    return pd.DataFrame([data])[feature_names]

input_df = user_input_features()

# Scale Time and Amount
input_df[['Time']] = scaler_time.transform(input_df[['Time']])
input_df[['Amount']] = scaler_amount.transform(input_df[['Amount']])

# Now safe to predict
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)[:, 1]

st.subheader('Prediction')
fraud_label = np.array(['Not Fraud', 'Fraud'])
st.write(fraud_label[prediction][0])

st.subheader('Prediction Probability')
st.write(f"Fraud Probability: {prediction_proba[0]:.4f}")
