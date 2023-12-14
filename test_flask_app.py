import unittest
import json

from app import app


class FlaskTestCase(unittest.TestCase):

    # Set up the Flask test client
    def setUp(self):
        app.testing = True
        self.client = app.test_client()

    def test_predict_full_data_red_wine(self):
        sample_payload = {
            'fixed acidity': 7.4,
            'volatile acidity': 0.700,
            'citric acid': 0,
            'residual sugar': 1.9,
            'chlorides': 0.076,
            'free sulfur dioxide': 11,
            'total sulfur dioxide': 34,
            'density': 0.9978,
            'pH': 3.51,
            'sulphates': 0.56,
            'alcohol': 9.4,
            'type': 1  
        }

        # Convert to JSON string
        sample_payload_json = json.dumps(sample_payload)

        # Make a POST request to the predict endpoint
        response = self.client.post('/predict', data=sample_payload_json, content_type='application/json')

        # Parse the response data
        data = json.loads(response.data)

        # Assert that the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Assert that the response contains the expected prediction
        expected_prediction = [5]  
        self.assertEqual(data, expected_prediction)

    def test_predict_full_data_white_wine(self):
        sample_payload = {
            'fixed acidity': 7,
            'volatile acidity': 0.27,
            'citric acid': 0.36,
            'residual sugar': 20.7,
            'chlorides': 0.045,
            'free sulfur dioxide': 45,
            'total sulfur dioxide': 170,
            'density': 1.001,
            'pH': 3,
            'sulphates': 0.45,
            'alcohol': 8.8,
            'type': 0  
        }

        # Convert to JSON string
        sample_payload_json = json.dumps(sample_payload)

        # Make a POST request to the predict endpoint
        response = self.client.post('/predict', data=sample_payload_json, content_type='application/json')

        # Parse the response data
        data = json.loads(response.data)

        # Assert that the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Assert that the response contains the expected prediction
        expected_prediction = [6]  
        self.assertEqual(data, expected_prediction)


    def test_predict_missing_data(self):
        sample_payload_with_missing = {
            'fixed acidity': '',
            'volatile acidity': 0.700,
            'citric acid': 0,
            'residual sugar': 1.9,
            'chlorides': 0.076,
            'free sulfur dioxide': 11,
            'total sulfur dioxide': 34,
            'density': 0.9978,
            'pH': 3.51,
            'sulphates': 0.56,
            'alcohol': 9.4,
            'type': 1  
        }

        # Convert to JSON string
        sample_payload_json = json.dumps(sample_payload_with_missing)
        # Make a POST request to the predict endpoint
        response = self.client.post('/predict', data=sample_payload_json, content_type='application/json')

        self.assertEqual(response.status_code, 400)

        data = json.loads(response.data)
        # Update the expected error message
        expected_error_message = {'error': "Invalid input for 'fixed acidity'. Expected a numeric value."}
        self.assertEqual(data, expected_error_message)
    
    def test_predict_invalid_numerical_data(self):
        # Test with a non-numerical value for a parameter
        sample_payload = {
            'fixed acidity': 'non-numeric',  # Invalid input
            'volatile acidity': 0.700,
            'citric acid': 0,
            'residual sugar': 1.9,
            'chlorides': 0.076,
            'free sulfur dioxide': 11,
            'total sulfur dioxide': 34,
            'density': 0.9978,
            'pH': 3.51,
            'sulphates': 0.56,
            'alcohol': 9.4,
            'type': 1
        }

        sample_payload_json = json.dumps(sample_payload)
        response = self.client.post('/predict', data=sample_payload_json, content_type='application/json')

        self.assertEqual(response.status_code, 400)

        data = json.loads(response.data)
        expected_error_message = {'error': "Invalid input for 'fixed acidity'. Expected a numeric value."}
        self.assertEqual(data, expected_error_message)
    
    def test_predict_invalid_type_data(self):
        # Test with an invalid value for 'type'
        sample_payload = {
            'fixed acidity': 7.4,
            'volatile acidity': 0.700,
            'citric acid': 0,
            'residual sugar': 1.9,
            'chlorides': 0.076,
            'free sulfur dioxide': 11,
            'total sulfur dioxide': 34,
            'density': 0.9978,
            'pH': 3.51,
            'sulphates': 0.56,
            'alcohol': 9.4,
            'type': 3  # Invalid input
        }

        sample_payload_json = json.dumps(sample_payload)
        response = self.client.post('/predict', data=sample_payload_json, content_type='application/json')

        self.assertEqual(response.status_code, 400)

        data = json.loads(response.data)
        expected_error_message = {'error': "Invalid input for 'type'. Expected 0 or 1, got 3"}
        self.assertEqual(data, expected_error_message)

if __name__ == '__main__':
    unittest.main(verbosity=2)
