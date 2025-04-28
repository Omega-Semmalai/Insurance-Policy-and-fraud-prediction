import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load the data
file_path = 'C:\\Omega\\Semester 5\\Machine Learning\\Project\\final.csv'
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

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode the target variable (y_train and y_test) for multiclass classification
y_train_onehot = pd.get_dummies(y_train).values
y_test_onehot = pd.get_dummies(y_test).values

# Define a function to create the neural network for multiclass classification
def create_feedforward_nn():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))  # Input layer
    model.add(Dense(16, activation='relu'))  # Hidden layer
    model.add(Dense(y_train_onehot.shape[1], activation='softmax'))  # Output layer for multiclass classification
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create and train the feedforward neural network
model = create_feedforward_nn()
model.fit(X_train, y_train_onehot, epochs=10, batch_size=32, verbose=0)

# Save the trained model to an .h5 file
model.save("policyfinal.h5")
print("Model saved to policy_type_model.h5")

# Make predictions on the test set
y_test_pred_proba = model.predict(X_test)

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