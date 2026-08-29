# train.py 

import wandb
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import os
from model import CNNModel  
from sweep_config import sweep_config

# Enable cuDNN benchmark for faster convolutions on fixed input resolutions
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Defining training and validation transforms
def get_transforms(augmentation):
    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return train_transform, val_transform

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
    train_data = datasets.ImageFolder("../inaturalist_12K/train", transform=train_tf)
    val_data = datasets.ImageFolder("../inaturalist_12K/val", transform=val_tf)

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

      
    print("Training run complete.")

# Run wandb agent with sweep
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="Fractal_CNN")
    wandb.agent(sweep_id, function=train)
    wandb.finish()
    print("Sweep complete")
