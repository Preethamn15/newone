from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = load('random_forest_model.joblib')
scaler = load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON data from the request
        data = request.get_json()
        high = float(data['high'])
        low = float(data['low'])
        open_val = float(data['open'])
        volume = float(data['volume'])

        # Prepare the features
        features = np.array([[high, low, open_val, volume]])

        # Scale the features using the scaler
        scaled_features = scaler.transform(features)

        # Make a prediction
        prediction = model.predict(scaled_features)

        # Return the prediction as JSON
        return jsonify({'prediction': round(prediction[0], 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
