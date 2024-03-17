

from flask import Flask, jsonify, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model_filename = 'rf_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    # Convert data into DataFrame
    try:
        input_data = pd.DataFrame([data])
        # Ensure the input data matches the model features
        features = ['matches', 'innings', 'not_out', 'runs', 'average_score', 'ball_faced', 'strike_rate', '100s', '50', '0s', '4s', '6s']
        input_data = input_data[features]

        # Make prediction using the model
        prediction = model.predict(input_data)

        # Return the prediction
        return jsonify(prediction.tolist())
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
