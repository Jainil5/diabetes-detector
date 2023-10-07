import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
df = pd.read_csv("diabetes.csv")
column_names = df.columns.tolist()

X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values
#scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.LongTensor(y_val)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# Step 3: Define your neural network model
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

# Step 4: Train the model on the training set

model = DiabetesModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
epochs = 2000

# Lists to store training and validation loss and accuracy
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Calculate training loss and accuracy for the epoch
    train_loss = total_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    print(f'Epoch [{epoch + 1}/{epochs}] Training Loss: {train_loss:.4f} Training Accuracy: {train_accuracy * 100:.2f}%')

# Step 5: Evaluate the model on the validation set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    y_pred = model(X_val)
    _, predicted = torch.max(y_pred, 1)
    val_loss = criterion(y_pred, y_val)
    val_accuracy = (predicted == y_val).sum().item() / y_val.size(0)
    val_losses.append(val_loss.item())
    val_accuracies.append(val_accuracy)

print(f'Validation Loss: {val_loss.item():.4f} Validation Accuracy: {val_accuracy * 100:.2f}%')

# Step 6: Evaluate the model on the test set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    y_pred = model(X_test)
    _, predicted = torch.max(y_pred, 1)
    test_loss = criterion(y_pred, y_test)
    test_accuracy = (predicted == y_test).sum().item() / y_test.size(0)

print(f'Test Loss: {test_loss.item():.4f} Test Accuracy: {test_accuracy * 100:.2f}%')

# Save the trained model
torch.save(model.state_dict(), 'diabetes_model.pt')
print("Model saved.")

# # Load the saved model
# loaded_model = DiabetesModel()
# loaded_model.load_state_dict(torch.load('diabetes_model.pt'))
# loaded_model.to(device)
# loaded_model.eval()
# print("Loaded Model: ",loaded_model)


def predict_diabetes(input_data):

    model.eval()  # Set the model to evaluation mode

    # Convert input_data to a PyTorch tensor
    input_data_tensor = torch.FloatTensor(input_data)

    # Ensure input_data has the same shape as the model expects (8 features)
    if input_data_tensor.shape != (8,):
        raise ValueError("Input data must have 8 features.")

    with torch.no_grad():
        # Forward pass to get predictions
        predictions = model(input_data_tensor)

        # Assuming class 1 corresponds to index 1
        sigmoid = torch.nn.Sigmoid()
        probability_class_1 = sigmoid(predictions[1]).item()

        # Get the predicted class (0 or 1) by rounding the probability
        predicted_class = torch.round(torch.tensor(probability_class_1))
        print(predicted_class)
        result = 'Diabetic' if predicted_class == 1 else 'Not Diabetic'

    return result


# # Example new inputs for prediction
# new_inputs = [6,148,72,35,0,33.6,0.627,50]
# # non - 0
# # yes - 1
# # Predict the outcome using the loaded model
# predicted_outcome = predict_diabetes(new_inputs)
# print(f"According to your data you are: {predicted_outcome}")
