import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def generate_complex_1d_data(num_samples=1000):
    # x from 0 to 10
    x = np.linspace(0, 10, num_samples)
    # Non-linear function with minor noise
    y = x * np.sin(2 * x) + 0.1 * np.random.randn(num_samples)
    return x.astype(np.float32), y.astype(np.float32)

class Simple1DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)  # Extra layer
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

if __name__ == "__main__":
    # Generate data
    x, y = generate_complex_1d_data(num_samples=1000)
    X_tensor = torch.from_numpy(x).unsqueeze(1)
    Y_tensor = torch.from_numpy(y).unsqueeze(1)

    # Split into train and test
    train_size = int(0.8 * len(X_tensor))
    X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
    Y_train, Y_test = Y_tensor[:train_size], Y_tensor[train_size:]

    # Create model
    model = Simple1DNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Lower learning rate

    # Train
    for epoch in range(1000):  # Increase epochs
        model.train()
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, Y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {loss.item():.6f}")

    # Evaluate and visualize
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy().flatten()
    x_test_np = X_test.numpy().flatten()
    y_test_np = Y_test.numpy().flatten()

    # Print a few samples to check changes
    print("Sample predictions vs actual:")
    for i in range(5):
        print(f"x={x_test_np[i]:.2f}, predicted={predictions[i]:.2f}, actual={y_test_np[i]:.2f}")



    plt.plot(x_test_np, y_test_np, label='Actual', color='blue')
    plt.plot(x_test_np, predictions, label='Predicted', color='red')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Complex Single-Dim Function: Actual vs. Predicted')
    plt.show()
