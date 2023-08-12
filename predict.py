from diabetes import *

# Example new inputs for prediction
new_inputs = [1, 108, 60, 46, 178, 35.5, 0.415, 24]

# Predict the outcome using the loaded model
predicted_outcome = predict_diabetes(new_inputs)
print(f"According to your data you are: {predicted_outcome}")
