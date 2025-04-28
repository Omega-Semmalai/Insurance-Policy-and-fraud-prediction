import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 1: Load the CSV data into a DataFrame
# Replace 'your_data.csv' with the path to your CSV file
data = pd.read_csv("C:\\Omega\\Semester 5\\Machine Learning\\Project\\final.csv")

# Step 2: Display basic information and statistics about the dataset
print(data.info())
print(data.describe())

# Step 3: Check for missing values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Step 4: Encode categorical variables
# Label encoding for binary columns like 'Sex', 'MaritalStatus', 'AccidentArea', 'FraudFound'
label_encoders = {}
for column in ['Sex', 'MaritalStatus', 'AccidentArea', 'FraudFound', 'PoliceReportFiled', 'WitnessPresent', 'AgentType', 'BasePolicy']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# One-hot encode columns with more than 2 categories (e.g., 'Make', 'PolicyType', 'VehicleCategory')
data = pd.get_dummies(data, columns=['Make', 'PolicyType', 'VehicleCategory'], drop_first=True)

# Step 5: Handle missing values (if any)
# Filling missing values with mean for numerical columns
data.fillna(data.mean(), inplace=True)

# Step 6: Scale numerical features for better model performance
# Standardize numerical features like 'Age', 'ClaimAmount', 'VehiclePrice', etc.
scaler = StandardScaler()
numerical_cols = ['Age', 'ClaimAmount', 'VehiclePrice', 'AgeOfVehicle', 'AgeOfPolicyHolder', 'DriverRating', 'NumberOfSuppliments']
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Step 7: Save the preprocessed data as a single CSV file
# Assuming 'FraudFound' is the target variable
# Reorder the columns to keep the target column 'FraudFound' as the last column
data = data[[col for col in data.columns if col != 'FraudFound'] + ['FraudFound']]

# Step 8: Save the entire preprocessed dataset to a CSV file
data.to_csv('final.csv', index=False)

print("Data preprocessing complete and saved as 'final.csv'.")