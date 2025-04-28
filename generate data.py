import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Define possible values for each column
policy_type = ['Liability', 'Collision', 'All Perils']
vehicle_category = ['Sport', 'Sedan', 'Utility']
base_policy = ['Liability', 'Collision', 'All Perils']

# Generate vehicle prices as continuous variables
vehicle_price = np.random.uniform(10000, 80000, size=150000)  # Prices between 10,000 and 80,000

# Create correlated features
driver_age = np.random.randint(16, 66, size=150000)  # Drivers aged 16 to 65
past_claims = np.clip(np.random.poisson(lam=driver_age / 10), 0, 5)  # Correlate with driver age

# Adjust the driver rating based on past claims and age
driver_rating = np.where(past_claims == 0, np.random.choice([4, 5], size=150000),
                         np.random.randint(1, 4, size=150000))

# Generate other features
deductible = np.random.choice([300, 400], size=150000)
policy_accident_days = np.random.choice(['more than 30', 'less than 30'], size=150000)
policy_claim_days = np.random.choice(['more than 30', 'less than 30'], size=150000)
agent_type = np.random.choice(['External', 'Internal'], size=150000)

# Additional features
policy_number = np.random.randint(1, 500001, size=150000)  # Random policy numbers
rep_number = np.random.randint(1, 21, size=150000)  # Random rep numbers
age_of_vehicle = np.random.randint(1, 11, size=150000)  # Vehicle ages between 1 and 10 years
age_of_policy_holder = np.random.choice(['16-17', '18-25', '26-30', '31-35', '36-40', '41-50', '51-65', 'over 65'], size=150000)
police_report_filed = np.random.choice(['Yes', 'No'], size=150000)
witness_present = np.random.choice(['Yes', 'No'], size=150000)
number_of_supplements = np.random.randint(0, 6, size=150000)  # Number of supplements between 0 and 5
address_change_claim = np.random.choice(['no change', '1 year', '4 to 8 years'], size=150000)
number_of_cars = np.random.randint(1, 4, size=150000)  # Number of cars (1 to 3)
year = np.random.randint(1990, 2024, size=150000)  # Random years between 1990 and 2024
fraud_found = np.random.choice(['Yes', 'No'], size=150000)

# Create DataFrame
data = {
    'Fault': np.random.choice(['Policy Holder', 'Third Party'], size=150000),
    'PolicyType': np.random.choice(policy_type, 150000),
    'VehicleCategory': np.random.choice(vehicle_category, 150000),
    'VehiclePrice': vehicle_price,
    'PolicyNumber': policy_number,
    'RepNumber': rep_number,
    'Deductible': deductible,
    'DriverRating': driver_rating,
    'Days:Policy-Accident': policy_accident_days,
    'Days:Policy-Claim': policy_claim_days,
    'PastNumberOfClaims': past_claims,
    'AgeOfVehicle': age_of_vehicle,
    'AgeOfPolicyHolder': age_of_policy_holder,
    'PoliceReportFiled': police_report_filed,
    'WitnessPresent': witness_present,
    'AgentType': agent_type,
    'NumberOfSuppliments': number_of_supplements,
    'AddressChange-Claim': address_change_claim,
    'NumberOfCars': number_of_cars,
    'Year': year,
    'BasePolicy': np.random.choice(base_policy, 150000),
    'FraudFound': fraud_found,
}

# Create the DataFrame
df = pd.DataFrame(data)

# File path to save the CSV
file_path = 'C:/Omega/Semester 5/Machine Learning/Project/correct.csv'

# Export to CSV
df.to_csv(file_path, index=False)

file_path

'''import pandas as pd
import numpy as np

# Define possible values for each column
fault = ['Policy Holder', 'Third Party']
policy_type = ['Sport - Liability', 'Sport - Collision', 'Sedan - Liability', 'Sedan - Collision', 'Sedan - All Perils', 'Utility - All Perils']
vehicle_category = ['Sport', 'Sedan', 'Utility']
vehicle_price = ['less than 20,000', '20,000 to 29,000', '30,000 to 39,000', 'more than 69,000']
deductible = [300, 400]
driver_rating = [1, 2, 3, 4, 5]
policy_accident_days = ['more than 30', 'less than 30']
policy_claim_days = ['more than 30', 'less than 30']
past_claims = [0, 1, 2, 3, 4, 5]
vehicle_age = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
policyholder_age = ['16-17', '18-25', '26-30', '31-35', '36-40', '41-50', '51-65', 'over 65']
police_report = ['Yes', 'No']
witness_present = ['Yes', 'No']
agent_type = ['External', 'Internal']
num_supplements = [0, 1, 2, 3, 4, 5]
address_change = ['no change', '1 year', '4 to 8 years']
num_cars = [1, 2, 3]
year = list(range(1990, 2024))
base_policy = ['Liability', 'Collision', 'All Perils']
fraud_found = ['Yes', 'No']

# Generate 150,000 rows of synthetic data
np.random.seed(42)
data = {
    'Fault': np.random.choice(fault, 150000),
    'PolicyType': np.random.choice(policy_type, 150000),
    'VehicleCategory': np.random.choice(vehicle_category, 150000),
    'VehiclePrice': np.random.choice(vehicle_price, 150000),
    'PolicyNumber': np.random.randint(1, 500001, size=150000),
    'RepNumber': np.random.randint(1, 21, size=150000),
    'Deductible': np.random.choice(deductible, 150000),
    'DriverRating': np.random.choice(driver_rating, 150000),
    'Days:Policy-Accident': np.random.choice(policy_accident_days, 150000),
    'Days:Policy-Claim': np.random.choice(policy_claim_days, 150000),
    'PastNumberOfClaims': np.random.choice(past_claims, 150000),
    'AgeOfVehicle': np.random.choice(vehicle_age, 150000),
    'AgeOfPolicyHolder': np.random.choice(policyholder_age, 150000),
    'PoliceReportFiled': np.random.choice(police_report, 150000),
    'WitnessPresent': np.random.choice(witness_present, 150000),
    'AgentType': np.random.choice(agent_type, 150000),
    'NumberOfSuppliments': np.random.choice(num_supplements, 150000),
    'AddressChange-Claim': np.random.choice(address_change, 150000),
    'NumberOfCars': np.random.choice(num_cars, 150000),
    'Year': np.random.choice(year, 150000),
    'BasePolicy': np.random.choice(base_policy, 150000),
    'FraudFound': np.random.choice(fraud_found, 150000),
}

# Create the DataFrame
df = pd.DataFrame(data)

# File path to save the CSV
file_path = 'C:/Omega/Semester 5/Machine Learning/Project/synthetic_insurance_data.csv'

# Export to CSV
df.to_csv(file_path, index=False)

file_path
'''