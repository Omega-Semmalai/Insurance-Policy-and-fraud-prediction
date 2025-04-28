import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model
model = load_model("C:\\Omega\\Semester 5\\Machine Learning\\Project\\policyfinal.h5")

# Load the data to fit the scaler (use the same data or similar structure as during training)
file_path = 'C:\\Omega\\Semester 5\\Machine Learning\\Project\\final.csv'
data = pd.read_csv(file_path)

# Clean column names and fit the scaler using relevant columns
data.columns = data.columns.str.strip()

# Select only the columns used during training for scaling
features = ['WeekOfMonthClaimed', 'DayOfWeekClaimed', 'MonthClaimed', 'AgeOfPolicyHolder', 
            'ClaimAmount', 'AgeOfVehicle', 'Year']

# Filter the data to match these features
X = data[features].values

# Scale the input features
scaler = StandardScaler()
scaler.fit(X)

# Encode the target variable using LabelEncoder
label_encoder = LabelEncoder()
data['PolicyType'] = label_encoder.fit_transform(data['PolicyType'])

# Streamlit UI
st.title('Insurance Policy Prediction')

# Back button in top right
if st.button("Back", key="back_button"):
    st.session_state.page = 'home'

# Input fields for policy prediction based on 7 features
st.write("Enter the details for policy prediction:")
week_of_month = st.number_input("Week of the Month of the Claim:", min_value=1, max_value=5)
day_of_week = st.number_input("Day of the Week of the Claim:", min_value=1, max_value=7)
month_of_year = st.number_input("Month of the Year of the Claim:", min_value=1, max_value=12)
age = st.number_input("Age of the Policy Holder:", min_value=0, max_value=100, value=30)
claim_amount = st.number_input("Claim Amount:", min_value=0, value=250000)
vehicle_age = st.number_input("Age of the Vehicle:", min_value=0, value=5)
claim_year = st.number_input("Year of the Claim:", min_value=1990, max_value=2024, value=2024)

# Prepare input for prediction
input_features = np.array([week_of_month, day_of_week, month_of_year, age, claim_amount, vehicle_age, claim_year]).reshape(1, -1)

# Apply scaling to the input features
input_features_scaled = scaler.transform(input_features)

# Button to trigger prediction
if st.button("Predict Policy"):
    prediction_proba = model.predict(input_features_scaled)
    predicted_class = np.argmax(prediction_proba)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    st.write(f"Predicted Policy Type: {predicted_label}")