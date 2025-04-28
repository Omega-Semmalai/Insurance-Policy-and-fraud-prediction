'''import pandas as pd
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

# Define price ranges based on make and category
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

# Generate vehicle prices based on make and category
vehicle_price = np.array([
    np.random.uniform(*price_ranges[(m, c)]) 
    for m, c in zip(np.random.choice(make, size=150000), np.random.choice(vehicle_category, size=150000))
])

# Generate driver ages with normal distribution
driver_age = np.random.normal(loc=40, scale=15, size=150000).astype(int)
driver_age = np.clip(driver_age, 18, 80)  # Ensure age is between 18 and 80

# Generate years of driving experience based on driver age
driving_experience = np.clip(driver_age - 18, 1, 60)  # Ensure experience is between 1 and 60 years

# Generate engine sizes
engine_size = np.random.choice([1.6, 2.0, 2.5, 3.0, 3.5, 4.0], size=150000)

# Generate past claims
past_claims = np.clip(np.random.poisson(lam=1, size=150000), 0, 5)

# Generate the fault type
fault_type = np.random.choice(fault, size=150000)

# Generate other features
year_of_manufacture = np.random.randint(1995, 2024, size=150000)
number_of_cars = np.random.choice([1, 2], size=150000)
sex = np.random.choice(sex, size=150000)
marital_status = np.random.choice(marital_status, size=150000)
policy_type = np.random.choice(policy_type, size=150000)
vehicle_category = np.random.choice(vehicle_category, size=150000)
deductible = np.random.choice([300, 400, 500], size=150000)
accident_area = np.random.choice(accident_area, size=150000)
vehicle_age = np.random.randint(0, 31, size=150000)  # Vehicle age between 0 to 30 years

# Calculate vehicle value depreciation
maximum_vehicle_age = 30
depreciation = vehicle_price * (1 - (vehicle_age / maximum_vehicle_age))

# Policy type multiplier
policy_type_multiplier = np.where(policy_type == 'Liability', 1.0, np.where(policy_type == 'Collision', 1.2, 1.5))

# Marital status discount (5% discount for married drivers)
marital_status_factor = np.where(marital_status == 'Married', 0.95, 1.0)

# Deductible effect on premium (higher deductibles result in lower premiums)
deductible_factor = np.where(deductible == 300, 1.1, np.where(deductible == 400, 1.05, 1.0))

# Logic for Premium Prediction
# Base premium depends on vehicle price (adjusted for depreciation) and engine size
base_premium = (depreciation * 0.05) + (engine_size * 500)

# Risk factors and premium adjustments
age_risk_factor = np.where(driver_age < 25, 1.2, np.where(driver_age > 65, 1.3, 1.0))  # Younger and older drivers have higher premiums
experience_discount = np.where(driving_experience > 10, 0.9, 1.0)  # More experienced drivers get a discount
claims_penalty = np.where(past_claims > 0, 1.5, 1.0)  # More claims lead to higher premiums
accident_area_factor = np.where(accident_area == 'Urban', 1.2, np.where(accident_area == 'Rural', 1.0, 1.1))  # Urban areas have higher premiums

# Final premium calculation with added factors
final_premium = base_premium * age_risk_factor * experience_discount * claims_penalty * accident_area_factor
final_premium *= policy_type_multiplier * marital_status_factor * deductible_factor

# Generate fraud detection
fraud_found = np.where(final_premium > 3000, np.random.choice(['Yes', 'No'], size=150000, p=[0.2, 0.8]), np.random.choice(['Yes', 'No'], size=150000, p=[0.05, 0.95]))

# Create DataFrame
data = {
    'Make': np.random.choice(make, size=150000),
    'VehicleCategory': vehicle_category,
    'YearOfManufacture': year_of_manufacture,
    'VehiclePrice': vehicle_price,
    'DepreciatedVehiclePrice': depreciation,
    'EngineSize': engine_size,
    'NumberOfCars': number_of_cars,
    'DriverAge': driver_age,
    'Sex': sex,
    'MaritalStatus': marital_status,
    'DrivingExperience': driving_experience,
    'PastNumberOfClaims': past_claims,
    'Fault': fault_type,
    'PolicyType': policy_type,
    'Deductible': deductible,
    'AccidentArea': accident_area,
    'VehicleAge': vehicle_age,
    'PremiumAmount': final_premium,
    'FraudFound': fraud_found
}

# Create the DataFrame
df = pd.DataFrame(data)

# File path to save the CSV
file_path = 'C:/Omega/Semester 5/Machine Learning/Project/vehicle_insurance_premium.csv'

# Export to CSV
df.to_csv(file_path, index=False)

file_path

import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
random.seed(42)

# Define constants for the dataset
num_records = 150000
vehicle_makes = ['Toyota', 'Honda', 'Ford', 'BMW', 'Audi', 'Hyundai', 'Kia', 'Nissan', 'Volkswagen', 'Chevrolet']
vehicle_models = ['Model A', 'Model B', 'Model C', 'Model D', 'Model E', 'Model F', 'Model G', 'Model H', 'Model I', 'Model J']
vehicle_types = ['Sedan', 'SUV', 'Hatchback', 'Coupe', 'Convertible']
engine_sizes = [1000, 1200, 1500, 1800, 2000, 2500]
locations = ['Urban', 'Suburban', 'Rural']
usage_patterns = ['Personal', 'Commercial', 'Rental']
claims_histories = [0, 1, 2, 3, 4]  # Number of claims
security_features = ['Basic', 'Standard', 'Advanced']
coverage_preferences = ['Personal Accident', 'Full Coverage', 'Third Party Only', 'Comprehensive']
previous_insurance_types = ['Third Party', 'Comprehensive', 'Zero Depreciation']
credit_scores = np.random.choice(range(300, 851), num_records)  # Credit scores between 300 and 850
marital_statuses = ['Single', 'Married']

# Generate random data
data = {
    'Vehicle Make': [random.choice(vehicle_makes) for _ in range(num_records)],
    'Vehicle Model': [random.choice(vehicle_models) for _ in range(num_records)],
    'Vehicle Age': np.random.randint(1, 21, num_records),  # Vehicle age between 1 to 20 years
    'Engine Size': [random.choice(engine_sizes) for _ in range(num_records)],
    'Vehicle Type': [random.choice(vehicle_types) for _ in range(num_records)],
    'Geographical Location': [random.choice(locations) for _ in range(num_records)],
    'Usage Pattern': [random.choice(usage_patterns) for _ in range(num_records)],
    'Driver Age': np.random.randint(18, 65, num_records),  # Driver age between 18 to 65 years
    'Driving Experience': np.random.randint(0, 50, num_records),  # Driving experience in years
    'Claims History': [random.choice(claims_histories) for _ in range(num_records)],
    'Annual Mileage': np.random.randint(5000, 30000, num_records),  # Annual mileage between 5000 to 30000 km
    'Security Features': [random.choice(security_features) for _ in range(num_records)],
    'Coverage Preferences': [random.choice(coverage_preferences) for _ in range(num_records)],
    'Previous Insurance Type': [random.choice(previous_insurance_types) for _ in range(num_records)],
    'Credit Score': credit_scores,
    'Marital Status': [random.choice(marital_statuses) for _ in range(num_records)],
}

# Create DataFrame
insurance_data = pd.DataFrame(data)

# Assign premium amounts based on coverage preference
def calculate_premium(row):
    if row['Coverage Preferences'] == 'Personal Accident':
        return random.randint(1000, 5000)  # Personal Accident cover
    elif row['Coverage Preferences'] == 'Full Coverage':
        return random.randint(10000, 30000)  # Comprehensive Insurance
    elif row['Coverage Preferences'] == 'Third Party Only':
        return random.randint(3000, 7000)  # Third-Party Liability Insurance
    elif row['Coverage Preferences'] == 'Comprehensive':
        return random.randint(10000, 30000)  # Comprehensive Insurance
    else:
        return 0

# Update the Premium Amount calculation
insurance_data['Premium Amount (INR)'] = insurance_data.apply(calculate_premium, axis=1)


# Save the DataFrame to a CSV file
file_path = 'C:/Omega/Semester 5/Machine Learning/Project/new_insurance_premium.csv'
insurance_data.to_csv(file_path, index=False)

# Display the first few records of the generated dataset
insurance_data.head()
'''
import numpy as np
import pandas as pd

# Sample DataFrame structure
np.random.seed(0)  # For reproducibility
num_records = 150000

# Sample Data
vehicle_makes = ['Toyota', 'Honda', 'Ford', 'BMW', 'Audi', 'Hyundai', 'Kia', 'Nissan', 'Volkswagen', 'Chevrolet']
vehicle_models = ['Model A', 'Model B', 'Model C', 'Model D', 'Model E']
vehicle_types = ['Sedan', 'SUV', 'Hatchback', 'Coupe', 'Convertible']
engine_sizes = [1000, 1200, 1500, 1800, 2000, 2500]
locations = ['Urban', 'Suburban', 'Rural']
usage_patterns = ['Personal', 'Commercial', 'Rental']
claims_histories = [0, 1, 2, 3, 4]  # Number of claims
security_features = ['Basic', 'Standard', 'Advanced']
coverage_preferences = ['Personal Accident', 'Full Coverage', 'Third Party Only', 'Comprehensive']
previous_insurance_types = ['Third Party', 'Comprehensive', 'Zero Depreciation']
credit_scores = np.random.choice(range(300, 851), num_records)  # Credit scores between 300 and 850
marital_statuses = ['Single', 'Married']

# Simulating a dataset
data = {
    'Vehicle Make': np.random.choice(vehicle_makes, num_records),
    'Vehicle Model': np.random.choice(vehicle_models, num_records),
    'Vehicle Type': np.random.choice(vehicle_types, num_records),
    'Engine Size': np.random.choice(engine_sizes, num_records),
    'Usage Pattern': np.random.choice(usage_patterns, num_records),
    'Claims History': np.random.choice(claims_histories, num_records),
    'Security Features': np.random.choice(security_features, num_records),
    'Coverage Preference': np.random.choice(coverage_preferences, num_records),
    'Credit Score': np.random.choice(credit_scores, num_records),
    'Marital Status': np.random.choice(marital_statuses, num_records)
}

insurance_data = pd.DataFrame(data)

# Function to calculate premium based on realistic logic
def calculate_realistic_premium(row):
    # Set base premium based on vehicle make, type, and engine size
    base_premium = 2000  # Base starting premium in INR

    # Vehicle make premium adjustment
    luxury_brands = ['BMW', 'Audi']
    if row['Vehicle Make'] in luxury_brands:
        base_premium += 10000  # Luxury brands have higher premiums
    elif row['Vehicle Make'] in ['Ford', 'Chevrolet']:
        base_premium += 3000  # Mid-range brands

    # Engine size adjustment (higher engine size -> higher premium)
    if row['Engine Size'] >= 2000:
        base_premium += 3000
    elif row['Engine Size'] >= 1500:
        base_premium += 2000

    # Vehicle type adjustment
    if row['Vehicle Type'] == 'SUV':
        base_premium += 3000  # SUVs typically attract higher premiums
    elif row['Vehicle Type'] == 'Convertible':
        base_premium += 5000  # Convertibles are considered riskier

    # Usage pattern adjustment (commercial and rental vehicles attract higher premiums)
    if row['Usage Pattern'] == 'Commercial':
        base_premium += 4000
    elif row['Usage Pattern'] == 'Rental':
        base_premium += 5000

    # Claims history adjustment
    base_premium += min(row['Claims History'] * 2000, 10000)  # Cap at ₹10,000

    # Security features adjustment (better security reduces premium)
    if row['Security Features'] == 'Standard':
        base_premium *= 0.95  # 5% reduction for standard security
    elif row['Security Features'] == 'Advanced':
        base_premium *= 0.90  # 10% reduction for advanced security

    # Coverage preference adjustment
    if row['Coverage Preference'] == 'Full Coverage':
        base_premium += 3000
    elif row['Coverage Preference'] == 'Comprehensive':
        base_premium += 4000
    elif row['Coverage Preference'] == 'Third Party Only':
        base_premium -= 1000  # Third party only is cheaper

    # Credit score adjustment (lower credit score increases premium)
    if row['Credit Score'] < 600:
        base_premium += 3000  # Higher premium for bad credit
    elif row['Credit Score'] > 750:
        base_premium *= 0.90  # 10% discount for good credit

    # Marital status adjustment (married drivers are less risky)
    if row['Marital Status'] == 'Married':
        base_premium *= 0.95  # 5% discount for married individuals

    return base_premium

# Function to determine policy type based on premium
def determine_policy_type(premium):
    if premium < 3000:
        return 'Third Party Liability Insurance'
    elif 3000 <= premium < 12000:
        return 'Comprehensive Insurance'
    elif 12000 <= premium < 25000:
        return 'Full Coverage Insurance'
    elif 25000 <= premium < 30000:
        return 'Zero Depreciation Insurance'
    elif premium >= 30000:
        return 'Luxury Insurance'
    else:
        return 'Unknown'

# Apply the function to each row in the DataFrame
insurance_data['Premium Amount (INR)'] = insurance_data.apply(calculate_realistic_premium, axis=1)
insurance_data['Policy Type'] = insurance_data['Premium Amount (INR)'].apply(determine_policy_type)

# Save the DataFrame to a CSV file
file_path = 'C:/Omega/Semester 5/Machine Learning/Project/insurance_premium.csv'
insurance_data.to_csv(file_path, index=False)

# Display the first few records of the generated dataset
insurance_data.head()
