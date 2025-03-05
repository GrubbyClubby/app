import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Ensure model file exists
model_path = os.path.abspath('temp/surrogate_model.pkl')
try:
    model = joblib.load(model_path)
except Exception as e:
    app.logger.error(f"Error loading model: {e}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        app.logger.debug(f"Received data: {data}")

        if not data or 'variables' not in data:
            return jsonify({'error': 'Invalid data format'}), 400

        variables = data['variables']
        temperature = float(variables['temperature'])
        humidity = float(variables['humidity'])
        occupancy = int(variables['occupancy'])

        # Generate weekly predictions for a year (52 weeks)
        weekly_predictions = []
        for week in range(52):
            prediction = model.predict([[temperature, humidity, occupancy]])
            weekly_predictions.append(prediction[0])
            
            # Example temperature drift (optional, can be refined)
            temperature += np.random.uniform(-1, 1)  # Simulating seasonal changes
            humidity += np.random.uniform(-2, 2)  # Minor humidity variations
            occupancy += np.random.randint(-1, 2)  # Simulating occupancy fluctuations
            occupancy = max(1, occupancy)  # Ensure occupancy stays at least 1

        return jsonify({'heating_demand': weekly_predictions})
    except Exception as e:
        app.logger.error(f"Error in /predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
