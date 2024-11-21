import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
file_path = r'laptop.csv'
data = pd.read_csv(file_path)

# Data preprocessing
# Drop unnecessary columns and handle missing values
cleaned_data = data.drop(columns=["Unnamed: 0", "Name", "Brand"])  # Remove non-informative columns
cleaned_data = cleaned_data.dropna()  # Remove rows with missing values

# Separate features and target variable (Price)
X = cleaned_data.drop(columns=["Price"])
y = cleaned_data["Price"]

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the Decision Tree model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Root Mean Squared Error (RMSE):", rmse)
