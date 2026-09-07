# sweep_config.py

sweep_config = {
    "method": "grid",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "filters_per_layer": {
            "values": [
                [64, 128, 256, 512],       # High-capacity 4-layer CNN (optimal spatial resolution 2x2)
                [32, 64, 128, 256, 512]     # Deep 5-layer CNN (spatial resolution 1x1)
            ]
        },
        "activation": {
            "values": ["relu"]             # Classical baseline
        },
        "use_batchnorm": {
            "values": [True]               # BatchNorm is critical for fast convergence & high accuracy
        },
        "dropout_rate": {
            "values": [0.2, 0.3]           # Optimal regularization with BatchNorm
        },
        "dense_units": {
            "values": [256, 512]           # Classifier head capacity
        },
        "augmentation": {
            "values": [True]               # Mandatory on CIFAR-10 (RandomCrop + Flip)
        },
        "batch_size": {
            "values": [64]                 # Optimal batch size for 40k training images
        },
        "learning_rate": {
            "values": [1e-3, 5e-4]         # Sweet spot for Adam with BatchNorm
        },
        "epochs": {
            "values": [15]                 # 15 epochs allows convergence to 80-86%+ on CIFAR-10
        }
    }
}

