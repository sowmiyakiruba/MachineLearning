import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = '/mnt/data/laptop.csv'  # Update the path if running locally
data = pd.read_csv(file_path)

# Data preprocessing
# Drop unnecessary columns
data_cleaned = data.drop(columns=["Unnamed: 0", "Name", "Brand"])
data_cleaned = data_cleaned.dropna()  # Remove rows with missing values

# Separate features and target variable
X = data_cleaned.drop(columns=["Price"])  # Features
y = data_cleaned["Price"]  # Target variable

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Standardize the features (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Support Vector Machine model for Regression
svm_model = SVR(kernel='rbf')  # Radial Basis Function kernel is commonly used
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)
