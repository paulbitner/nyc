# load_model_example_with_plot.py

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random

# --------------------------
# Device selection
# --------------------------
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = get_device()
print(f"Using device: {device}")

# --------------------------
# Model definition (same as training)
# --------------------------
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

# --------------------------
# Load weights
# --------------------------
try:
    state = torch.load("model.pth", weights_only=True, map_location=device)
except TypeError:
    state = torch.load("model.pth", map_location=device)

model.load_state_dict(state)
print("<All keys matched successfully>")

# --------------------------
# Classes
# --------------------------
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# --------------------------
# Pick a random test image
# --------------------------
test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

idx = random.randint(0, len(test_data) - 1)
x, y = test_data[idx]
print(f"Index {idx}: actual label = {classes[y]}")

# --------------------------
# Predict
# --------------------------
model.eval()
with torch.no_grad():
    pred = model(x.to(device).unsqueeze(0))
    predicted_idx = pred.argmax(1).item()
    predicted = classes[predicted_idx]

print(f"Predicted: {predicted}")

# --------------------------
# Visualize the image
# --------------------------
plt.imshow(x.squeeze(), cmap="gray")
plt.title(f"Predicted: {predicted}\nActual: {classes[y]}")
plt.axis("off")
plt.show()
