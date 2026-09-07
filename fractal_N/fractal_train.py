# fractal_train.py

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

import wandb
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from fractal_model import CNNModel
from fractal_sweep_config import sweep_config

# Enable cuDNN benchmark for faster convolutions on fixed input resolutions
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# CIFAR-10 dataset statistics for normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Transforms
def get_transforms(augmentation):
    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_transform, val_transform

# Training function
def train():
    # Initialize wandb
    wandb.init()
    config = wandb.config

    # Read alpha values with safe defaults
    alpha1 = getattr(config, 'alpha1', 0.2)
    alpha2 = getattr(config, 'alpha2', 0.2)

    # Generating a meaningful run name using config values
    run_name = f"run_a1-{alpha1}_a2-{alpha2}_filters-{config.filters_per_layer}_act-{config.activation}_bs-{config.batch_size}_lr-{config.learning_rate}_do-{config.dropout_rate}_bn-{config.use_batchnorm}_aug-{config.augmentation}"
    wandb.run.name = run_name

    # Transforms
    train_tf, val_tf = get_transforms(config.augmentation)

    # Loading datasets
    data_root = PROJECT_ROOT / "cifar10"
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
        alpha1=alpha1,
        alpha2=alpha2,
        input_shape=(3, 32, 32),
        dense_units=config.dense_units,
        num_classes=len(train_data.classes)  # 10 for CIFAR-10
    )
    model.to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    # Automatic Mixed Precision (AMP) scaler
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

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
            "alpha1": alpha1,
            "alpha2": alpha2,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

    # Saving model and tracking best hyperparameters across all sweeps
    global_best_path = SCRIPT_DIR / "best_accuracy.txt"
    best_config_path = SCRIPT_DIR / "best_config.py"
    current_best = 0.0

    # Reading global best accuracy if file exists
    if global_best_path.exists():
        try:
            with open(global_best_path, "r") as f:
                content = f.read().strip()
                if "val_acc:" in content:
                    current_best = float(content.split("val_acc:")[1].split()[0])
                else:
                    current_best = float(content.splitlines()[0].strip())
        except Exception:
            current_best = 0.0

    # Saving model only if it strictly outperforms previous best
    if val_acc > current_best:
        model_save_path = SCRIPT_DIR / "best_model.pth"
        torch.save(model.state_dict(), str(model_save_path))

        # 1. Store full hyperparameters and metrics in best_accuracy.txt
        with open(global_best_path, "w") as f:
            f.write(f"val_acc: {val_acc:.4f}\n")
            f.write(f"train_acc: {train_acc:.4f}\n")
            f.write(f"alpha1: {alpha1}\n")
            f.write(f"alpha2: {alpha2}\n")
            f.write(f"filters_per_layer: {list(config.filters_per_layer)}\n")
            f.write(f"activation: {config.activation}\n")
            f.write(f"dense_units: {config.dense_units}\n")
            f.write(f"learning_rate: {config.learning_rate}\n")
            f.write(f"batch_size: {config.batch_size}\n")
            f.write(f"dropout_rate: {config.dropout_rate}\n")
            f.write(f"use_batchnorm: {config.use_batchnorm}\n")
            f.write(f"augmentation: {config.augmentation}\n")
            f.write(f"epochs: {config.epochs}\n")

        # 2. Store programmatic dictionary in best_config.py for test_fractal_model.py
        with open(best_config_path, "w") as f:
            f.write("# Auto-generated best configuration from sweep\n")
            f.write("best_config = {\n")
            f.write(f"    'val_acc': {val_acc:.4f},\n")
            f.write(f"    'alpha1': {alpha1},\n")
            f.write(f"    'alpha2': {alpha2},\n")
            f.write(f"    'filters_per_layer': {list(config.filters_per_layer)},\n")
            f.write(f"    'activation': '{config.activation}',\n")
            f.write(f"    'dense_units': {config.dense_units},\n")
            f.write(f"    'learning_rate': {config.learning_rate},\n")
            f.write(f"    'batch_size': {config.batch_size},\n")
            f.write(f"    'dropout_rate': {config.dropout_rate},\n")
            f.write(f"    'use_batchnorm': {config.use_batchnorm},\n")
            f.write(f"    'augmentation': {config.augmentation},\n")
            f.write(f"    'epochs': {config.epochs},\n")
            f.write("    'input_shape': (3, 32, 32),\n")
            f.write(f"    'num_classes': {len(train_data.classes)},\n")
            f.write("    'model_path': 'best_model.pth'\n")
            f.write("}\n")

        print(f"★ New global best model saved! Validation Accuracy: {val_acc*100:.2f}%")
        print(f"Hyperparameters recorded in: {global_best_path}")

    wandb.finish()
    print("Training run complete.")


# Run wandb agent with sweep
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="fractal_CNN_cifar10")
    wandb.agent(sweep_id, function=train)
    wandb.finish()
    print("Sweep complete")

