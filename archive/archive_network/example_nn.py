import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic non-linear data: y = sin(x1) + 0.5*cos(x2)
# def generate_data(num_samples=2000):
#     x1 = 2 * np.pi * np.random.rand(num_samples)
#     x2 = 2 * np.pi * np.random.rand(num_samples)
#     y = np.sin(x1) + 0.5 * np.cos(x2) + 0.1 * np.random.randn(num_samples)
#     X = np.column_stack((x1, x2))
#     return X.astype(np.float32), y.astype(np.float32)

def generate_complex_1d_data(num_samples=1000):
    # x from 0 to 10
    x = np.linspace(0, 10, num_samples)
    # Non-linear function with minor noise
    y = x * np.sin(2 * x) + 0.1 * np.random.randn(num_samples)
    return x.astype(np.float32), y.astype(np.float32)

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)

if __name__ == "__main__":
    # Create data
    X, Y = generate_data()
    X_tensor = torch.from_numpy(X)
    Y_tensor = torch.from_numpy(Y).unsqueeze(1)

    # Split data
    train_size = int(0.8 * len(X))
    X_train, Y_train = X_tensor[:train_size], Y_tensor[:train_size]
    X_val, Y_val = X_tensor[train_size:], Y_tensor[train_size:]

    # Model setup
    model = DeepNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, Y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val)
                val_loss = criterion(val_preds, Y_val)
            print(f"Epoch {epoch+1}: Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}")

    # Evaluation and visualization
    model.eval()
    with torch.no_grad():
        predictions = model(X_val)
    preds_np = predictions.numpy().flatten()
    Y_val_np = Y_val.numpy().flatten()

    plt.figure()
    plt.scatter(range(len(Y_val_np)), Y_val_np, label='Actual', alpha=0.6)
    plt.scatter(range(len(preds_np)), preds_np, label='Predicted', alpha=0.6)
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Comparison of Actual vs. Predicted')
    plt.show()
