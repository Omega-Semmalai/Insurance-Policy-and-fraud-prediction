import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Define possible values for categorical columns
policy_type = ['Liability', 'Collision', 'All Perils']
vehicle_category = ['Sport', 'Sedan', 'Utility']
sex = ['Male', 'Female']
marital_status = ['Single', 'Married', 'Divorced']
fault = ['Policy Holder', 'Third Party']
make = ['Toyota', 'Ford', 'Honda', 'Chevrolet', 'Nissan']
accident_area = ['Urban', 'Rural', 'Suburban']
base_policy = ['Liability', 'Collision', 'All Perils']

# Generate time-related features
months = list(range(1, 13))  # 1 to 12
days_of_week = list(range(1, 8))  # 1 to 7
week_of_month = list(range(1, 5))  # 1 to 4
years = list(range(1990, 2024))  # 1990 to 2023

# Generate vehicle prices as continuous variables
vehicle_price = np.random.uniform(10000, 80000, size=150000)  # Prices between 10,000 and 80,000

# Create correlated features
driver_age = np.random.randint(16, 66, size=150000)  # Drivers aged 16 to 65
past_claims = np.clip(np.random.poisson(lam=driver_age / 10), 0, 5)  # Correlate with driver age

# Adjust the driver rating based on past claims
driver_rating = np.where(past_claims == 0, np.random.choice([4, 5], size=150000),
                         np.random.randint(1, 4, size=150000))

# Generate other features
policy_number = np.random.randint(1, 500001, size=150000)
rep_number = np.random.randint(1, 21, size=150000)
deductible = np.random.choice([300, 400], size=150000)
policy_accident_days = np.random.choice(['more than 30', 'less than 30'], size=150000)
policy_claim_days = np.random.choice(['more than 30', 'less than 30'], size=150000)
agent_type = np.random.choice(['External', 'Internal'], size=150000)
witness_present = np.random.choice(['Yes', 'No'], size=150000)
police_report_filed = np.random.choice(['Yes', 'No'], size=150000)
number_of_supplements = np.random.randint(0, 6, size=150000)
address_change_claim = np.random.choice(['no change', '1 year', '4 to 8 years'], size=150000)
number_of_cars = np.random.randint(1, 4, size=150000)
month = np.random.choice(months, size=150000)
week_of_month = np.random.choice(week_of_month, size=150000)
day_of_week = np.random.choice(days_of_week, size=150000)
year = np.random.choice(years, size=150000)
base_policy = np.random.choice(base_policy, size=150000)
fraud_found = np.random.choice(['Yes', 'No'], size=150000)

# Generate Geographical Location: urban or rural
geographical_location = np.random.choice(['Urban', 'Rural'], size=150000, p=[0.7, 0.3])

# Create DataFrame
data = {
    'Month': month,
    'WeekOfMonth': week_of_month,
    'DayOfWeek': day_of_week,
    'Make': np.random.choice(make, size=150000),
    'AccidentArea': np.random.choice(accident_area, size=150000),
    'DayOfWeekClaimed': np.random.choice(days_of_week, size=150000),
    'MonthClaimed': np.random.choice(months, size=150000),
    'WeekOfMonthClaimed': np.random.choice(week_of_month, size=150000),
    'Sex': np.random.choice(sex, size=150000),
    'MaritalStatus': np.random.choice(marital_status, size=150000),
    'Age': driver_age,  # Reusing driver_age for Age
    'Fault': np.random.choice(fault, size=150000),
    'PolicyType': np.random.choice(policy_type, size=150000),
    'VehicleCategory': np.random.choice(vehicle_category, size=150000),
    'VehiclePrice': vehicle_price,
    'PolicyNumber': policy_number,
    'RepNumber': rep_number,
    'Deductible': deductible,
    'DriverRating': driver_rating,
    'Days:Policy-Accident': policy_accident_days,
    'Days:Policy-Claim': policy_claim_days,
    'PastNumberOfClaims': past_claims,
    'AgeOfVehicle': np.random.randint(1, 11, size=150000),  # Vehicle age between 1 to 10 years
    'AgeOfPolicyHolder': np.random.randint(16, 66, size=150000),  # Age of policyholder
    'PoliceReportFiled': police_report_filed,
    'WitnessPresent': witness_present,
    'AgentType': agent_type,
    'NumberOfSuppliments': number_of_supplements,
    'AddressChange-Claim': address_change_claim,
    'NumberOfCars': number_of_cars,
    'Year': year,
    'BasePolicy': base_policy,
    'FraudFound': fraud_found,
}

# Create the DataFrame
df = pd.DataFrame(data)

# File path to save the CSV
file_path = 'C:/Omega/Semester 5/Machine Learning/Project/correct 2.csv'

# Export to CSV
df.to_csv(file_path, index=False)

file_path
