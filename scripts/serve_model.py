# Step 1: Import required libraries
import pickle
import numpy as np
import pandas as pd
import shap
import logging
from flask import Flask, request, jsonify

# Step 2: Initialize Flask app
app = Flask(__name__)

# Step 3: Configure logging for tracking requests & errors
logging.basicConfig(filename='api.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Step 4: Load the trained model and scaler
model_path = "fraud_model.pkl"
scaler_path = "scaler.pkl"

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Step 5: Define API route to predict fraud
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON request data
        data = request.get_json()

        # Convert data into DataFrame
        input_data = pd.DataFrame([data])

        # Scale the input
        scaled_input = scaler.transform(input_data)

        # Make prediction (returns probability for fraud class 1)
        fraud_probability = model.predict_proba(scaled_input)[0][1]

        # Log prediction
        logging.info(f"Prediction: {fraud_probability}, Input Data: {data}")

        # Return fraud probability as JSON response
        return jsonify({'fraud_probability': fraud_probability})
    
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Step 6: Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
