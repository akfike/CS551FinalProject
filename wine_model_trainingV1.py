import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Correctly loading the data with semicolon separators
red_wine_data = pd.read_csv('./data/winequality-red.csv', sep=';')
white_wine_data = pd.read_csv('./data/winequality-white.csv', sep=';')

# Display the first few rows of each dataset again to confirm correct loading
print(red_wine_data.head())
print(white_wine_data.head())

# Add a new column to each dataset to indicate wine type (1 for red, 0 for white)
red_wine_data['type'] = 1
white_wine_data['type'] = 0

# Combine the datasets
wine_data = pd.concat([red_wine_data, white_wine_data], ignore_index=True)
wine_data.to_csv('./data/combined_wine_data.csv', index=False)

# Check for missing values
missing_values = wine_data.isnull().sum()

print("Missing: ", missing_values)

# Splitting the data into features and target
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Calculate the metrics
# Average squared difference between the actual and predicted values
# Lower = better
mse = mean_squared_error(y_test, y_pred)
# Proportion of the variance in the dependent variable that is predictable from the independent variables
# Closer to 1 = better
r2 = r2_score(y_test, y_pred)

print("MSE: ", mse) # MSE: 0.3705
print("R2: ", r2) # R2: 0.4982
