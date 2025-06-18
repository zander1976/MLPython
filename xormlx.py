import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# XOR data
X = mx.array([[0., 0.],
              [0., 1.],
              [1., 0.],
              [1., 1.]])
y = mx.array([[0.], [1.], [1.], [0.]])

# Define model
class XORModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def __call__(self, x):
        x = mx.tanh(self.fc1(x))
        x = mx.sigmoid(self.fc2(x))
        return x

model = XORModel()
_ = model(X)  # Trigger parameter initialization

# Binary cross-entropy loss
def bce_loss(y_pred, y_true):
    eps = 1e-7
    return -mx.mean(y_true * mx.log(y_pred + eps) + (1 - y_true) * mx.log(1 - y_pred + eps))

# Optimizer setup
optimizer = optim.Adam(learning_rate=0.1)
optimizer.init(model.parameters())  # ✅ Initialize optimizer

# Training loop
for epoch in range(1000):
    def loss_fn(x, y_true):  # <-- Accepting inputs as arguments
        y_pred = model(x)  # Forward pass
        return bce_loss(y_pred, y_true)  # Return loss

    # Call value_and_grad with the arguments passed correctly
    loss, grads = mx.value_and_grad(loss_fn)(X, y)  # Provide data to loss_fn

    # Update model parameters
    optimizer.update(model.parameters(), grads)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Test predictions
predictions = model(X)
print("Predictions:")
print(mx.round(predictions).tolist())
