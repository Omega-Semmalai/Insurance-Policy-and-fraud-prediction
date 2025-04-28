import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

# Load the trained model (Make sure this is the correct .h5 or .keras model file)
model = load_model("C:\\Omega\\Semester 5\\Machine Learning\\Project\\lstm_fraud_model.h5")

# Load the dataset and fit the scaler
data = pd.read_csv('C:\\Omega\\Semester 5\\Machine Learning\\Project\\final.csv')
top_5_numerical_features = ['Age', 'ClaimAmount', 'PastNumberOfClaims', 'DriverRating', 'Deductible']
scaler = StandardScaler()
scaler.fit(data[top_5_numerical_features])

# Define the function to get user inputs in Streamlit
def get_user_input():
    user_input = {}
    user_input['Age'] = st.number_input("Enter value for Age:", min_value=0, max_value=100, value=25)
    user_input['ClaimAmount'] = st.number_input("Enter value for ClaimAmount:", min_value=0, value=10000)
    user_input['PastNumberOfClaims'] = st.number_input("Enter value for PastNumberOfClaims:", min_value=0, max_value=100, value=2)
    user_input['DriverRating'] = st.number_input("Enter value for DriverRating:", min_value=1, max_value=5, value=3)
    user_input['Deductible'] = st.number_input("Enter value for Deductible:", min_value=0, value=500)
    
    return pd.DataFrame([user_input])

# Custom decision-making based on input values
def custom_prediction(user_input):
    if user_input['ClaimAmount'].values[0] > 25000 and user_input['PastNumberOfClaims'].values[0] > 5:
        return "Fraud"
    elif user_input['DriverRating'].values[0] <= 2 and user_input['Deductible'].values[0] < 500:
        return "Fraud"
    elif user_input['Age'].values[0] < 25 and user_input['ClaimAmount'].values[0] > 10000:
        return "Fraud"
    else:
        return "Not Fraud"

# Streamlit app layout
st.title('INSURANCE PREDICTION')

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Home page with two buttons
if st.session_state.page == 'home':
    st.write("Select an option:")
    if st.button("Fraud Detection"):
        st.session_state.page = 'fraud_detection'
    if st.button("Insurance Policy"):
        st.write("Insurance policy page will be implemented later.")

# Fraud Detection page
if st.session_state.page == 'fraud_detection':
    st.header("Fraud Detection")

    # Collect user input
    user_input = get_user_input()

    # Button to trigger prediction
    if st.button("Detect"):
        # Scale the input
        user_scaled_input = scaler.transform(user_input[top_5_numerical_features])

        # Reshape input for the LSTM model
        user_input_scaled = user_scaled_input.reshape(1, 1, len(top_5_numerical_features))  # (1, 1, 5)

        # Predict using the trained LSTM model
        raw_prediction = model.predict(user_input_scaled)
        st.write(f"Raw Prediction Probability of Fraud: {raw_prediction[0][0]:.4f}")

        # Get decision based on custom rules
        decision = custom_prediction(user_input)
        st.write(f"Decision based on input values: {decision}")