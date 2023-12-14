from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model and scaler
with open('tpot_trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json(force=True)

    # Check and convert data
    validated_data = {}
    for key, value in data.items():
        try:
            if key == 'type':
                if value not in [0, 1]:
                    return jsonify({'error': f"Invalid input for 'type'. Expected 0 or 1, got {value}"}), 400
                validated_data[key] = int(value)
            else:
                validated_data[key] = float(value)
        except ValueError:
            return jsonify({'error': f"Invalid input for '{key}'. Expected a numeric value."}), 400

    # Convert validated data to DataFrame
    input_data = pd.DataFrame([validated_data], columns=[
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'type'
    ])

    # Check for NaN values (should not happen with the above validation, but still good to check)
    if input_data.isnull().values.any():
        return jsonify({'error': 'Input contains NaN values.'}), 400

    # Feature scaling using the loaded scaler
    input_scaled = scaler.transform(input_data)

    # Predict with the loaded model
    prediction = model.predict(input_scaled)

    # Return prediction as JSON
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)
