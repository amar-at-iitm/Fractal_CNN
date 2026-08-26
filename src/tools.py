from torchvision import transforms


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