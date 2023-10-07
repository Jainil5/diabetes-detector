import torch
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiabetesModel(nn.Module):
    def __init__(self):
        super(DiabetesModel, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)  # 2 output classes (0 or 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load the saved model
loaded_model = DiabetesModel()
loaded_model.load_state_dict(torch.load("diabetes_model.pt"))
loaded_model.to(device)
loaded_model.eval()
# print("Loaded Model: ",model)


def predict_diabetes(input_data):

    loaded_model.eval()  # Set the model to evaluation mode

    # Convert input_data to a PyTorch tensor
    input_data_tensor = torch.FloatTensor(input_data)

    # Ensure input_data has the same shape as the model expects (8 features)
    if input_data_tensor.shape != (8,):
        raise ValueError("Input data must have 8 features.")

    with torch.no_grad():
        # Forward pass to get predictions
        predictions = loaded_model(input_data_tensor)

        # Assuming class 1 corresponds to index 1
        sigmoid = torch.nn.Sigmoid()
        probability_class_1 = sigmoid(predictions[1]).item()

        # Get the predicted class (0 or 1) by rounding the probability
        predicted_class = torch.round(torch.tensor(probability_class_1))
        result = 'Diabetic' if predicted_class == 1 else 'Not Diabetic'

    return result

# Example usage:
# Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
# new_data = [0,120,50,35,168,43.1,2.288,33]
# outcome = predict_diabetes(new_data)
# print(outcome)