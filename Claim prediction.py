import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Load the data
file_path = 'C:\\Omega\\Semester 5\\Machine Learning\\Project\\final.csv'  # Ensure the correct file path
data = pd.read_csv(file_path)

# Clean column names
data.columns = data.columns.str.strip()

# Encode the target variable (PolicyType) using LabelEncoder
label_encoder = LabelEncoder()
data['PolicyType'] = label_encoder.fit_transform(data['PolicyType'])

# One-hot encode categorical features
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
if categorical_columns:
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Separate features and target variable
X = data.drop(columns=['PolicyType']).values
y = data['PolicyType'].values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection using RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances and select top 10 features
selector = SelectFromModel(rf, max_features=10, prefit=True)
X_train_top = selector.transform(X_train)
X_test_top = selector.transform(X_test)

top_features_indices = selector.get_support(indices=True)
top_feature_names = data.drop(columns=['PolicyType']).columns[top_features_indices]
print(f"Top 10 features: {top_feature_names.tolist()}")

# Standardize the selected top 10 features
scaler = StandardScaler()
X_train_top = scaler.fit_transform(X_train_top)
X_test_top = scaler.transform(X_test_top)

# One-hot encode the target variable for multiclass classification
y_train_onehot = pd.get_dummies(y_train).values
y_test_onehot = pd.get_dummies(y_test).values

# Define a function to create the neural network for multiclass classification
def create_feedforward_nn():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=X_train_top.shape[1]))  # Input layer
    model.add(Dense(16, activation='relu'))  # Hidden layer
    model.add(Dense(y_train_onehot.shape[1], activation='softmax'))  # Output layer for multiclass classification
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create and train the feedforward neural network using the top 10 features
model = create_feedforward_nn()
model.fit(X_train_top, y_train_onehot, epochs=10, batch_size=32, verbose=0)

# Make predictions on the test set
y_test_pred_proba = model.predict(X_test_top)

# Convert predicted probabilities to class labels
y_test_pred = np.argmax(y_test_pred_proba, axis=1)

# Evaluate the model using accuracy, MSE, MAE, and ROC AUC
accuracy = accuracy_score(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test_onehot, y_test_pred_proba, multi_class='ovr')

# Print evaluation metrics
print(f"Feedforward Neural Networks - Accuracy for Policy Type: {accuracy:.4f}")
print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, ROC AUC: {roc_auc:.4f}")

# Function to make a prediction for a single input
def make_prediction(input_features):
    # Ensure the input is a numpy array and reshape it
    input_features = np.array(input_features).reshape(1, -1)

    # Standardize the input features using the same scaler as the training data
    input_features = scaler.transform(input_features)

    # Get prediction probabilities from the model
    prediction_proba = model.predict(input_features)

    # Get the class with the highest probability
    predicted_class = np.argmax(prediction_proba)

    # Map the predicted class back to the original label
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    # Return the prediction
    return predicted_label

# Function to get user input for each feature
def get_user_input():
    user_input = []
    for feature_name in top_feature_names:
        value = float(input(f"Enter the value for {feature_name}: "))
        user_input.append(value)

    return user_input

# Get input from the user
print("\nPlease enter the values for the following top features:")
user_input = get_user_input()

# Make a prediction based on user input
predicted_policy_type = make_prediction(user_input)

print(f"\nPredicted Policy Type: {predicted_policy_type}")