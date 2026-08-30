# fractal_train.py

import sys
from pathlib import Path

# Ensure the project root is on sys.path so we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import wandb
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import os
from fractal_model import CNNModel
from fractal_sweep_config import sweep_config
from src.tools import get_transforms

# Enable cuDNN benchmark for faster convolutions on fixed input resolutions
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Training function
def train():
    # Initialize wandb
    wandb.init()
    config = wandb.config
    # Generating a meaningful run name using config values
    run_name = f"run_filters-{config.filters_per_layer}_act-{config.activation}_bs-{config.batch_size}_lr-{config.learning_rate}_do-{config.dropout_rate}_bn-{config.use_batchnorm}_aug-{config.augmentation}"
    wandb.run.name = run_name
    # wandb.run.save()

    # Transforms
    train_tf, val_tf = get_transforms(config.augmentation)

    # Loading datasets
    data_root = PROJECT_ROOT / "inaturalist_12K"
    train_data = datasets.ImageFolder(str(data_root / "train"), transform=train_tf)
    val_data = datasets.ImageFolder(str(data_root / "val"), transform=val_tf)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    num_workers = 2

    train_loader = DataLoader(
        train_data, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0)
    )

    # Preparing model
    filters = config.filters_per_layer
    model = CNNModel(
        filters=filters,
        kernel_size=3,
        activation=config.activation,
        dropout=config.dropout_rate,
        use_batchnorm=config.use_batchnorm,
        input_shape=(3, 192, 192)
    )
    model.to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Automatic Mixed Precision (AMP) scaler
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    # To track best validation accuracy
    best_val_acc = 0.0

    # Training loop
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        print("-" * 60)
        model.train()
        total_loss, correct, total = 0, 0, 0

        for inputs, labels in tqdm(train_loader, desc="Training Progress", ncols=100, colour="magenta"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_cuda):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation loop
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation Progress", ncols=100, colour="cyan"):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=use_cuda):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        print("-" * 60)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

    # Saving model if it's the best across all sweeps
    global_best_path = "best_accuracy.txt"
    current_best = 0.0

    # Reading global best accuracy if file exists
    if os.path.exists(global_best_path):
        with open(global_best_path, "r") as f:
            try:
                current_best = float(f.read().strip())
            except:
                current_best = 0.0

    # Saving model only if it's better than global best
    if val_acc > current_best:
        torch.save(model.state_dict(), "best_model.pth")
        with open(global_best_path, "w") as f:
            f.write(str(val_acc))
        print(f"New global best model saved with val_acc: {val_acc:.4f}")

    wandb.finish()
    print("Training run complete.")


# Run wandb agent with sweep
if __name__ == "__main__":
    wandb.login(key="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")
    sweep_id = wandb.sweep(sweep_config, project="fractal_CNN")
    wandb.agent(sweep_id, function=train, count=2)
    wandb.finish()
    print("Sweep complete")
