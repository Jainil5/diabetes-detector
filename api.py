from flask import Flask, request, jsonify
import torch
import numpy as np

# Create a Flask app
app = Flask(__name__)


# Define a prediction endpoint
@app.route('/predict', methods=['POST'])
def predict_diabetes():
    try:
        # Get input data from the request
        input_data = request.get_json
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
