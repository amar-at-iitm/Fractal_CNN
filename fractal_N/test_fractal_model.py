# test_fractal_model.py

import sys
from pathlib import Path

# Ensure the project root is on sys.path so we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import matplotlib.pyplot as plt
import numpy as np
import wandb

from fractal_model import CNNModel  # Reusing the CNNModel from fractal_model.py
from best_config import best_config  # Importing the best hyperparameters

wandb.login(key="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")
# Initialize wandb
wandb.init(
    project="Fractal_CNN",
    name="fractal-test-evaluation",
    config=best_config
)

# Loading test data
transform = transforms.Compose([
    transforms.ToTensor(),
])
data_dir = str(PROJECT_ROOT / "inaturalist_12K")
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=best_config["batch_size"], shuffle=False, num_workers=2)

# Loading model
model = CNNModel(
    filters=best_config["filters_per_layer"],
    kernel_size=3,
    activation=best_config["activation"],
    dropout=best_config["dropout_rate"],
    use_batchnorm=best_config["use_batchnorm"],
    input_shape=best_config["input_shape"]
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(best_config["model_path"], map_location=device))
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
    img = np.transpose(all_images[i], (1, 2, 0))
    ax.imshow(img)
    pred = class_names[all_preds[i]]
    true = class_names[all_labels[i]]
    ax.set_title(f"Pred: {pred}\nTrue: {true}", fontsize=9)
    ax.axis("off")

plt.tight_layout()

# Logging image grid to wandb
wandb.log({"prediction_grid": wandb.Image(fig)})

plt.show()

# Finish the wandb run
wandb.finish()
