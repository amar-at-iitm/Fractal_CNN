# test_model.py

import os
import sys
from pathlib import Path

# Ensure both script directory and project root are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import wandb

from model import CNNModel
from best_config import best_config

# Normalization constants for CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Initialize wandb
wandb.init(
    project="classical_CNN_cifar10",
    name="classical-cifar10-test-evaluation",
    config=best_config
)

# Loading test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

data_dir = PROJECT_ROOT / "cifar10"
test_dataset = datasets.ImageFolder(str(data_dir / "test"), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=best_config["batch_size"], shuffle=False, num_workers=2)

# Loading model
model = CNNModel(
    filters=best_config["filters_per_layer"],
    kernel_size=3,
    activation=best_config["activation"],
    dropout=best_config["dropout_rate"],
    use_batchnorm=best_config["use_batchnorm"],
    input_shape=best_config.get("input_shape", (3, 32, 32)),
    dense_units=best_config.get("dense_units", 256),
    num_classes=best_config.get("num_classes", len(test_dataset.classes))
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_weights_path = Path(__file__).resolve().parent / best_config.get("model_path", "best_model.pth")
model.load_state_dict(torch.load(str(model_weights_path), map_location=device))
model.to(device)
model.eval()

# Evaluating on test data
correct = 0
total = 0
all_preds = []
all_images = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_images.extend(inputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Log test accuracy to wandb
wandb.log({"test_accuracy": accuracy})

# Class labels
class_names = test_dataset.classes

# Displaying 10x3 prediction grid
fig, axes = plt.subplots(10, 3, figsize=(12, 30))
fig.suptitle("Sample Predictions from Test Set", fontsize=20, y=1.02)

for i, ax in enumerate(axes.flat):
    if i >= len(all_images):
        break
    img = np.transpose(all_images[i], (1, 2, 0))
    # Denormalize image for correct RGB display
    img = img * np.array(CIFAR10_STD) + np.array(CIFAR10_MEAN)
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    pred = class_names[all_preds[i]]
    true = class_names[all_labels[i]]
    ax.set_title(f"Pred: {pred}\nTrue: {true}", fontsize=9)
    ax.axis("off")

plt.tight_layout()

# Logging image grid to wandb
wandb.log({"prediction_grid": wandb.Image(fig)})
plt.close(fig)

# Finish the wandb run
wandb.finish()

