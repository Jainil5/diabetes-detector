import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("diabetes.csv")
column_names = df.columns.tolist()

X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values
scaler = StandardScaler()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)


# Define the neural network model
class DiabetesModel(nn.Module):
    def __init__(self, input_size):
        super(DiabetesModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


# Initialize the model
input_size = X_train_tensor.shape[1]
model = DiabetesModel(input_size).to(device)
#
# # Define loss function and optimizer
# criterion = nn.BCELoss()  # Binary Cross Entropy loss for binary classification
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # Training loop
# num_epochs = 500
# batch_size = 32
#
# for epoch in range(num_epochs):
#     for i in range(0, len(X_train_tensor), batch_size):
#         batch_X = X_train_tensor[i:i + batch_size].to(device)
#         batch_y = y_train_tensor[i:i + batch_size].unsqueeze(1).to(device)  # Add .unsqueeze(1)
#
#         # Zero the parameter gradients
#         optimizer.zero_grad()
#
#         # Forward pass
#         outputs = model(batch_X)
#         loss = criterion(outputs, batch_y)
#
#         # Backpropagation and optimization
#         loss.backward()
#         optimizer.step()
#
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
# # Save the trained model
# torch.save(model.state_dict(), 'diabetes_model.pt')
# print("Model saved.")

# Load the saved model
loaded_model = DiabetesModel(input_size)
loaded_model.load_state_dict(torch.load('diabetes_model.pt'))
loaded_model.to(device)
loaded_model.eval()


# Function to predict diabetes outcome from new inputs using the loaded model
def predict_diabetes(inputs):
    inputs_scaled = scaler.transform(np.array(inputs).reshape(1, -1))
    inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        prediction = loaded_model(inputs_tensor)
        probability = prediction.item()
    outcome = "Diabetic" if probability >= 0.5 else "Non-Diabetic"
    return outcome


# # Example new inputs for prediction
# new_inputs = [1, 108, 60, 46, 178, 35.5, 0.415, 24]
#
# # Predict the outcome using the loaded model
# predicted_outcome = predict_diabetes(new_inputs)
# print(f"According to your data you are: {predicted_outcome}")
