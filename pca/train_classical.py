# train_classical.py - Training 1D CNN with Classical Activations on PCA Dataset

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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CNN1DModel
from pca_data import PCADataset, process_and_save_pca_dataset

# Enable cuDNN benchmark for faster convolutions on fixed input resolutions
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ==============================================================================
# Hyperparameters and Configuration (Hardcoded for direct execution)
# ==============================================================================
DATASET_NAME = "inaturalist12K"           # Dataset name
DATASET_DIR_NAME = f"{DATASET_NAME}_pcs"
DATA_ROOT = PROJECT_ROOT / DATASET_DIR_NAME

# Model architecture
FILTERS = [64, 128, 256]          # 1D Conv layer filter sizes
KERNEL_SIZE = 5                    # 1D kernel size
ACTIVATION = "relu"                # "relu", "squared_relu", or "cubic_relu"
DROPOUT = 0.2                      # Dropout rate
USE_BATCHNORM = True               # BatchNorm1d
DENSE_UNITS = 256                  # Dense layer units

# Optimization
BATCH_SIZE = 64                    # Batch size
LEARNING_RATE = 1e-3               # Learning rate
WEIGHT_DECAY = 1e-4                # Weight decay (L2 penalty)
EPOCHS = 15                        # Number of training epochs

# Optional wandb logging (set USE_WANDB = True if wandb is installed and logged in)
USE_WANDB = False
WANDB_PROJECT = "pca_1d_classical"


def train():
    # Detect GPU / CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print(f"\n[DEVICE] Using device: {device} | CUDA Available: {use_cuda}")

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
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda
    )

    # Instantiate 1D CNN with Classical Activation
    model = CNN1DModel(
        filters=FILTERS,
        kernel_size=KERNEL_SIZE,
        approach="classical",
        activation=ACTIVATION,
        dropout=DROPOUT,
        use_batchnorm=USE_BATCHNORM,
        input_length=input_length,
        dense_units=DENSE_UNITS,
        num_classes=num_classes
    )
    model.to(device)
    print(f"[MODEL] Classical 1D CNN initialized with '{ACTIVATION}' activation.")

    # Loss, optimizer, and mixed precision scaler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    # Optional wandb initialization
    if USE_WANDB:
        try:
            import wandb
            wandb.init(
                project=WANDB_PROJECT,
                name=f"classical_1d_{ACTIVATION}_bs{BATCH_SIZE}_lr{LEARNING_RATE}",
                config={
                    "approach": "classical",
                    "activation": ACTIVATION,
                    "filters": FILTERS,
                    "kernel_size": KERNEL_SIZE,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "epochs": EPOCHS,
                    "input_length": input_length,
                    "num_classes": num_classes
                }
            )
        except Exception as e:
            print(f"[WANDB] Failed to initialize wandb ({e}). Proceeding without wandb.")

    best_val_acc = 0.0
    model_save_path = SCRIPT_DIR / "best_classical_model.pth"

    # ==========================================================================
    # Training Loop
    # ==========================================================================
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 65)
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc="Classical Training", ncols=100, colour="magenta"):
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

        if USE_WANDB:
            try:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                })
            except Exception:
                pass

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "config": {
                    "approach": "classical",
                    "activation": ACTIVATION,
                    "filters": FILTERS,
                    "kernel_size": KERNEL_SIZE,
                    "input_length": input_length,
                    "num_classes": num_classes
                }
            }, str(model_save_path))
            print(f"  ★ New best classical model saved! Val Acc: {val_acc*100:.2f}%")

    print("\n" + "=" * 65)
    print(f"Classical 1D CNN Training Completed. Best Val Acc: {best_val_acc*100:.2f}%")
    print(f"Best model weights: {model_save_path}")
    print("=" * 65)


if __name__ == "__main__":
    train()

