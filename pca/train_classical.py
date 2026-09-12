# train_classical.py - Training 1D CNN with Classical Activations on PCA Dataset using WandB Sweep

import os
import sys
import json
from pathlib import Path

# Ensure script directory and project root are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CNN1DModel
from sweep_config import classical_sweep_config
from pca_data import PCADataset, process_and_save_pca_dataset

# Enable cuDNN benchmark for faster convolutions on fixed input resolutions
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ==============================================================================
# Configuration
# ==============================================================================
DATASET_NAME = "inaturalist_12K"           # Dataset: "cifar10", "cifar100", "svhn", "tiny_imagenet", "inaturalist_12K"
DATASET_DIR_NAME = f"{DATASET_NAME}_pcs"
DATA_ROOT = PROJECT_ROOT / DATASET_DIR_NAME

# WandB Project name
WANDB_PROJECT = f"pca_1d_classical_{DATASET_NAME}"

# Maximum number of sweep iterations (set to an integer like 10, or None to run all grid combinations)
SWEEP_COUNT = None


def train():
    # Initialize wandb for this sweep run
    wandb.init()
    config = wandb.config

    # Read hyperparameters from wandb config
    filters = config.filters_per_layer
    kernel_size = getattr(config, "kernel_size", 5)
    activation = config.activation
    dropout_rate = config.dropout_rate
    use_batchnorm = config.use_batchnorm
    dense_units = config.dense_units
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    epochs = config.epochs

    # Generate descriptive run name
    run_name = f"classical_1d_act-{activation}_ks-{kernel_size}_filters-{filters}_bs-{batch_size}_lr-{learning_rate}_do-{dropout_rate}_bn-{use_batchnorm}"
    wandb.run.name = run_name

    # Detect GPU / CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print(f"\n[DEVICE] Hardware acceleration: {device} (CUDA: {use_cuda})")

    # Ensure PCA dataset exists; if not, generate it
    if not DATA_ROOT.exists():
        print(f"[DATA] '{DATA_ROOT}' not found. Generating 1D PCA dataset automatically...")
        process_and_save_pca_dataset(dataset_name=DATASET_NAME, root_dir=PROJECT_ROOT)

    # Read dataset metadata
    meta_path = DATA_ROOT / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        input_length = metadata.get("output_1d_shape", [1024])[0]
        num_classes = metadata.get("num_classes", 10)
    else:
        input_length = 1024
        num_classes = 10

    print(f"[DATA] Input 1D Length: {input_length} | Num Classes: {num_classes}")

    # Load 1D PCA dataset splits
    train_dataset = PCADataset(str(DATA_ROOT), split="train")
    val_dataset = PCADataset(str(DATA_ROOT), split="val")

    num_workers = 2 if os.name != 'nt' else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda
    )

    # Instantiate 1D CNN with Classical Activation
    model = CNN1DModel(
        filters=filters,
        kernel_size=kernel_size,
        approach="classical",
        activation=activation,
        dropout=dropout_rate,
        use_batchnorm=use_batchnorm,
        input_length=input_length,
        dense_units=dense_units,
        num_classes=num_classes
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Classical 1D CNN initialized with '{activation}' ({total_params:,} trainable params).")

    # Loss, optimizer, and mixed precision scaler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    # ==========================================================================
    # Training Loop
    # ==========================================================================
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 65)
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Training Run ({activation})", ncols=100, colour="magenta"):
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

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation", ncols=100, colour="cyan"):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=use_cuda):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch + 1:02d} Summary | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

    # ==========================================================================
    # Global Best Checkpoint Tracking Across Sweeps
    # ==========================================================================
    global_best_path = SCRIPT_DIR / "best_classical_accuracy.txt"
    best_config_path = SCRIPT_DIR / "best_classical_config.py"
    current_best = 0.0

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

    if val_acc > current_best:
        model_save_path = SCRIPT_DIR / "best_classical_model.pth"
        torch.save(model.state_dict(), str(model_save_path))

        # 1. Store full hyperparameters and metrics in best_classical_accuracy.txt
        with open(global_best_path, "w") as f:
            f.write(f"val_acc: {val_acc:.4f}\n")
            f.write(f"train_acc: {train_acc:.4f}\n")
            f.write(f"filters_per_layer: {list(filters)}\n")
            f.write(f"kernel_size: {kernel_size}\n")
            f.write(f"activation: {activation}\n")
            f.write(f"dense_units: {dense_units}\n")
            f.write(f"learning_rate: {learning_rate}\n")
            f.write(f"batch_size: {batch_size}\n")
            f.write(f"dropout_rate: {dropout_rate}\n")
            f.write(f"use_batchnorm: {use_batchnorm}\n")
            f.write(f"epochs: {epochs}\n")
            f.write(f"input_length: {input_length}\n")
            f.write(f"num_classes: {num_classes}\n")

        # 2. Store programmatic dictionary in best_classical_config.py for test_model.py
        with open(best_config_path, "w") as f:
            f.write("# Auto-generated best configuration from classical 1D sweep\n")
            f.write("best_config = {\n")
            f.write(f"    'val_acc': {val_acc:.4f},\n")
            f.write(f"    'filters_per_layer': {list(filters)},\n")
            f.write(f"    'kernel_size': {kernel_size},\n")
            f.write(f"    'activation': '{activation}',\n")
            f.write(f"    'dense_units': {dense_units},\n")
            f.write(f"    'learning_rate': {learning_rate},\n")
            f.write(f"    'batch_size': {batch_size},\n")
            f.write(f"    'dropout_rate': {dropout_rate},\n")
            f.write(f"    'use_batchnorm': {use_batchnorm},\n")
            f.write(f"    'epochs': {epochs},\n")
            f.write(f"    'input_length': {input_length},\n")
            f.write(f"    'num_classes': {num_classes},\n")
            f.write("    'model_path': 'best_classical_model.pth'\n")
            f.write("}\n")

        print(f"★ New global best classical model saved! Validation Accuracy: {val_acc*100:.2f}%")
        print(f"Hyperparameters recorded in: {global_best_path}")

    wandb.finish()
    print("Sweep run complete.")


# ==============================================================================
# WandB Sweep Entrypoint
# ==============================================================================

if __name__ == "__main__":
    wandb.login(key="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")    
    sweep_id = wandb.sweep(classical_sweep_config, project=WANDB_PROJECT)
    wandb.agent(sweep_id, function=train)
    wandb.finish()
    print("Classical 1D CNN sweep completed successfully.")
