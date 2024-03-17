from flask import Flask, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)

# Load your trained model
# Make sure to replace 'path/to/your/model.joblib' with the actual path to your model file
model = load('random_forest_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Extracting the input features from the request
    player = data.get('player')
    opposition = data.get('opposition')
    bf = float(data.get('balls_faced', 0))  # Default to 0 if not provided
    ov = float(data.get('overs', 0))        # Default to 0 if not provided

    # Assuming 'test_X' is prepared and available. You might need to adjust this part
    # based on how your model was trained and how your data needs to be prepared.
    # This example assumes that the input features for the prediction are only 'bf' and 'ov'
    # and that the model expects a 2D array-like structure with these features.

    # Prepare the features for prediction
    features = np.array([[1, 1, bf, ov]])  # Assuming the model expects a 2D array-like structure

    # Make prediction
    preds = model.predict(features)
    predicted_runs = preds.astype(int).tolist()

    # Return the prediction result
    return jsonify({
        'player': player,
        'opposition': opposition,
        'predicted_runs': predicted_runs,
        'message': f"{player}'s overall run predicted is {predicted_runs} Against {opposition}"
    })

if __name__ == '__main__':
    app.run(port=5000,debug=True)
