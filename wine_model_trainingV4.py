import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle

# Loading the data
red_wine_data = pd.read_csv('./data/winequality-red.csv', sep=';')
white_wine_data = pd.read_csv('./data/winequality-white.csv', sep=';')

# Add a column to each dataset to indicate wine type (1 for red, 0 for white)
red_wine_data['type'] = 1
white_wine_data['type'] = 0

# Combine the datasets
wine_data = pd.concat([red_wine_data, white_wine_data], ignore_index=True)

# Splitting the data into features and target
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Serialize the scaler to a file
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize TPOT
tpot = TPOTRegressor(generations=10, population_size=100, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)

# Export the best pipeline
tpot.export('./best_pipeline.py')

# Predict on the test set using the best found model
y_pred = tpot.predict(X_test)

# Calculate the metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE: ", mse) # MSE: 0.3405
print("R2: ", r2) # R2: 0.5388

# Serialize the trained model to a file
with open('tpot_trained_model.pkl', 'wb') as model_file:
    pickle.dump(tpot.fitted_pipeline_, model_file)

