import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the dataset (example dataset for demonstration)
data = [
    [0.15, 0.8, 0],  # [meanfun, IQR, label (female)]
    [0.12, 0.9, 1],  # [meanfun, IQR, label (male)]
    [0.13, 0.6, 0],  # [meanfun, IQR, label (female)]
    [0.16, 0.5, 0],  # [meanfun, IQR, label (female)]
    [0.11, 0.7, 1],  # [meanfun, IQR, label (male)]
]

# Convert dataset to PyTorch tensors
inputs = torch.tensor([row[:2] for row in data], dtype=torch.float32)
labels = torch.tensor([row[2] for row in data], dtype=torch.long)

# Create DataLoader
dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define the model
class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 4),  # 2 inputs (meanfun, IQR) to 4 hidden units
            nn.ReLU(),
            nn.Linear(4, 2),  # 4 hidden units to 2 outputs (male, female)
            nn.Softmax(dim=1)  # Softmax for probabilities
        )

    def forward(self, x):
        return self.fc(x)

# Initialize the model, loss function, and optimizer
model = GenderClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    for batch_inputs, batch_labels in dataloader:
        # Forward pass
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Testing the model
def predict(model, meanfun, IQR):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([[meanfun, IQR]], dtype=torch.float32)
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        return "Male" if prediction == 1 else "Female"

# Example predictions
print(predict(model, 0.13, 0.6))  # Expected Female
print(predict(model, 0.11, 0.8))  # Expected Male
