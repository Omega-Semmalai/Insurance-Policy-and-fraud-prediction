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

# Price ranges based on vehicle make and category
price_ranges = {
    ('Toyota', 'Sedan'): (600000, 1200000),
    ('Toyota', 'Utility'): (800000, 1500000),
    ('Toyota', 'Sport'): (1000000, 1800000),
    ('Ford', 'Sedan'): (600000, 1200000),
    ('Ford', 'Utility'): (800000, 1500000),
    ('Ford', 'Sport'): (1000000, 1800000),
    ('Honda', 'Sedan'): (600000, 1200000),
    ('Honda', 'Utility'): (800000, 1500000),
    ('Honda', 'Sport'): (1000000, 1800000),
    ('Chevrolet', 'Sedan'): (500000, 1100000),
    ('Chevrolet', 'Utility'): (700000, 1500000),
    ('Chevrolet', 'Sport'): (1000000, 1800000),
    ('Nissan', 'Sedan'): (600000, 1200000),
    ('Nissan', 'Utility'): (700000, 1400000),
    ('Nissan', 'Sport'): (1000000, 1800000)
}

# Generate time-related features
months = list(range(1, 13))  # 1 to 12
days_of_week = list(range(1, 8))  # 1 to 7
week_of_month = list(range(1, 5))  # 1 to 4
years = list(range(1990, 2024))  # 1990 to 2023

# Function to generate vehicle price based on make and category
def generate_vehicle_price(make, category):
    price_range = price_ranges.get((make, category), (400000, 15000000))
    return np.random.uniform(price_range[0], price_range[1])

# Generate other features
n_samples = 150000  # Define number of samples
policy_number = np.random.randint(1, 500001, size=n_samples)
rep_number = np.random.randint(1, 21, size=n_samples)
deductible = np.random.choice([300, 400], size=n_samples)
policy_accident_days = np.random.choice(['more than 30', 'less than 30'], size=n_samples)
policy_claim_days = np.random.choice(['more than 30', 'less than 30'], size=n_samples)
agent_type = np.random.choice(['External', 'Internal'], size=n_samples)
witness_present = np.random.choice(['Yes', 'No'], size=n_samples)
police_report_filed = np.random.choice(['Yes', 'No'], size=n_samples)
number_of_supplements = np.random.randint(0, 6, size=n_samples)
address_change_claim = np.random.choice(['no change', '1 year', '4 to 8 years'], size=n_samples)
number_of_cars = np.random.randint(1, 4, size=n_samples)
month = np.random.choice(months, size=n_samples)
week_of_month = np.random.choice(week_of_month, size=n_samples)
day_of_week = np.random.choice(days_of_week, size=n_samples)
year = np.random.choice(years, size=n_samples)

# Generate random driver ages (for example, between 16 and 65)
driver_age = np.random.randint(16, 66, size=n_samples)  # Driver age between 16 and 65
# Generate driver ratings (example, between 1 and 5)
driver_rating = np.random.randint(1, 6, size=n_samples)  # Driver rating between 1 and 5
# Generate past claims (for example, between 0 and 5)
past_claims = np.random.randint(0, 6, size=n_samples)  # Past number of claims between 0 and 5

# Generate make and vehicle category first to use them for vehicle price generation
vehicle_make = np.random.choice(make, size=n_samples)
vehicle_category = np.random.choice(vehicle_category, size=n_samples)

# Generate vehicle prices based on the generated make and category
vehicle_price = [generate_vehicle_price(v_make, v_category) for v_make, v_category in zip(vehicle_make, vehicle_category)]

# Generate claims based on vehicle price (simple linear relationship)
claim_amount = np.clip(vehicle_price * np.random.uniform(0.1, 0.3, size=n_samples), 0, 20000)

# Generate fraud detection based on previous data patterns
fraud_found = np.where(claim_amount > 15000, np.random.choice(['Yes', 'No'], size=n_samples, p=[0.3, 0.7]), np.random.choice(['Yes', 'No'], size=n_samples, p=[0.1, 0.9]))

# Create DataFrame
data = {
    'Month': month,
    'WeekOfMonth': week_of_month,
    'DayOfWeek': day_of_week,
    'Make': vehicle_make,
    'AccidentArea': np.random.choice(accident_area, size=n_samples),
    'DayOfWeekClaimed': np.random.choice(days_of_week, size=n_samples),
    'MonthClaimed': np.random.choice(months, size=n_samples),
    'WeekOfMonthClaimed': np.random.choice(week_of_month, size=n_samples),
    'Sex': np.random.choice(sex, size=n_samples),
    'MaritalStatus': np.random.choice(marital_status, size=n_samples),
    'Age': driver_age,  # Reusing driver_age for Age
    'Fault': np.random.choice(fault, size=n_samples),
    'PolicyType': np.random.choice(policy_type, size=n_samples),
    'VehicleCategory': vehicle_category,
    'VehiclePrice': vehicle_price,
    'ClaimAmount': claim_amount,
    'PolicyNumber': policy_number,
    'RepNumber': rep_number,
    'Deductible': deductible,
    'DriverRating': driver_rating,
    'Days:Policy-Accident': policy_accident_days,
    'Days:Policy-Claim': policy_claim_days,
    'PastNumberOfClaims': past_claims,
    'AgeOfVehicle': np.random.randint(1, 11, size=n_samples),  # Vehicle age between 1 to 10 years
    'AgeOfPolicyHolder': np.random.randint(16, 66, size=n_samples),  # Age of policyholder
    'PoliceReportFiled': police_report_filed,
    'WitnessPresent': witness_present,
    'AgentType': agent_type,
    'NumberOfSuppliments': number_of_supplements,
    'AddressChange-Claim': address_change_claim,
    'NumberOfCars': number_of_cars,
    'Year': year,
'BasePolicy': np.random.choice(base_policy, size=n_samples),
    'FraudFound': fraud_found,
}

# Assert that all arrays have the same length
for key, value in data.items():
    assert len(value) == n_samples, f"Length mismatch for {key}: {len(value)}"

df = pd.DataFrame(data)

# Introduce missing values randomly in selected columns
missing_indices_policy_type = np.random.choice(df.index, size=5000, replace=False)
df.loc[missing_indices_policy_type, 'PolicyType'] = np.nan

missing_indices_claim_amount = np.random.choice(df.index, size=5000, replace=False)
df.loc[missing_indices_claim_amount, 'ClaimAmount'] = np.nan

missing_indices_age = np.random.choice(df.index, size=5000, replace=False)
df.loc[missing_indices_age, 'Age'] = np.nan

missing_indices_vehicle_price = np.random.choice(df.index, size=5000, replace=False)
df.loc[missing_indices_vehicle_price, 'VehiclePrice'] = np.nan

missing_indices_deductible = np.random.choice(df.index, size=5000, replace=False)
df.loc[missing_indices_deductible, 'Deductible'] = np.nan

missing_indices_driver_rating = np.random.choice(df.index, size=5000, replace=False)
df.loc[missing_indices_driver_rating, 'DriverRating'] = np.nan

missing_indices_number_of_cars = np.random.choice(df.index, size=5000, replace=False)
df.loc[missing_indices_number_of_cars, 'NumberOfCars'] = np.nan

# Check for missing values in the DataFrame
print(df.isnull().sum())


# You can include more columns as needed, following the same pattern.

# Create the DataFrame
df = pd.DataFrame(data)

# File path to save the CSV
file_path = 'C:/Omega/Semester 5/Machine Learning/Project/Fraud detection1.csv'

# Export to CSV
df.to_csv(file_path, index=False)

file_path
