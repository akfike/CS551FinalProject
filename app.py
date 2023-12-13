from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model and scaler
with open('tpot_trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:  # Make sure you have saved this during your model training
    scaler = pickle.load(scaler_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json(force=True)
    print(data)
    
    # Check for missing data
    for key in data:
        if data[key] == '':
            data[key] = None  # Or an appropriate value for your context
    
    # Convert JSON data to DataFrame with correct feature names
    input_data = pd.DataFrame([data], columns=[
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'type'
    ])

    print(input_data)

    
    # Check for NaN values and handle them if necessary
    if input_data.isnull().values.any():
        # Handle NaNs here. For example, you could impute missing values:
        # input_data.fillna(value=imputed_value, inplace=True)
        return jsonify({'error': 'Input contains NaN values.'}), 400
    
    # Feature scaling using the loaded scaler
    input_scaled = scaler.transform(input_data)
    
    # Predict with the loaded model
    prediction = model.predict(input_scaled)
    
    # Return prediction as JSON
    return jsonify(prediction.tolist())


if __name__ == '__main__':
    app.run(debug=True)
