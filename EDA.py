import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import randint

# Load the data
file_path = "C:\\Omega\\Semester 5\\Machine Learning\\Project\\final.csv"
data = pd.read_csv(file_path)

# Display basic information and initial rows of the dataset
print("Dataset Information:")
print(data.info())
print("\nFirst 5 rows of the dataset:")
print(data.head())

# Handling missing values
print("\nMissing Values:")
print(data.isnull().sum())
data.dropna(inplace=True)

# Exploratory Data Analysis (EDA)
# 1. Distribution of the target variable
sns.countplot(x='FraudFound', data=data)
plt.title('Distribution of Fraud Found')
plt.xlabel('Fraud Found (0: No, 1: Yes)')
plt.ylabel('Count')
plt.show()

# 2. Distribution of numerical features
numerical_cols = data.select_dtypes(include=[np.number]).columns
data[numerical_cols].hist(bins=15, figsize=(15, 10))
plt.suptitle('Distribution of Numerical Features')
plt.show()

# 3. Outlier Detection using box plots
num_cols = len(numerical_cols)
cols_per_row = 4
rows = (num_cols // cols_per_row) + (num_cols % cols_per_row > 0)

plt.figure(figsize=(20, 5 * rows))
for i, col in enumerate(numerical_cols):
    plt.subplot(rows, cols_per_row, i + 1)
    sns.boxplot(x=data[col])
    plt.title(f'Box Plot of {col}')
    plt.xlim(data[col].min() - 1, data[col].max() + 1)

plt.subplots_adjust(hspace=0.95, top=0.933)
plt.show()

# Prepare the data for training
X = data.drop('FraudFound', axis=1)  # Features (all columns except target)
y = data['FraudFound']  # Target variable

# Encoding categorical variables if any
le = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = le.fit_transform(X[col])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the numerical features (optional but recommended for some models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Gradient Boosting Classifier
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train, y_train)

# Make predictions
y_pred = gbc.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Visualizing the Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

