# sweep_config.py for 1D PCA models

# Sweep configuration for classical 1D CNN
classical_sweep_config = {
    "method": "grid",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "filters_per_layer": {
            "values": [
                [64, 128, 256],          # 3-layer 1D CNN
                [64, 128, 256, 512],     # 4-layer 1D CNN
            ]
        },
        "kernel_size": {
            "values": [5, 7]             # 1D kernel size (5 or 7 gives wide receptive field)
        },
        "activation": {
            "values": ["relu"]
        },
        "use_batchnorm": {
            "values": [True]
        },
        "dropout_rate": {
            "values": [0.2, 0.3]
        },
        "dense_units": {
            "values": [256, 512]
        },
        "batch_size": {
            "values": [32]
        },
        "learning_rate": {
            "values": [1e-3, 5e-4]
        },
        "epochs": {
            "values": [15]
        }
    }
}

# Sweep configuration for fractal 1D CNN
fractal_sweep_config = {
    "method": "grid",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "alpha1": {
            "values": [0.1, 0.2, 0.3]
        },
        "alpha2": {
            "values": [0.1, 0.2, 0.3]
        },
        "filters_per_layer": {
            "values": [
                [64, 128, 256],
                [64, 128, 256, 512],
            ]
        },
        "kernel_size": {
            "values": [5, 7]
        },
        "activation": {
            "values": ["f_relu"]
        },
        "use_batchnorm": {
            "values": [True]
        },
        "dropout_rate": {
            "values": [0.2, 0.3]
        },
        "dense_units": {
            "values": [256, 512]
        },
        "batch_size": {
            "values": [64, 128]
        },
        "learning_rate": {
            "values": [1e-3, 5e-4]
        },
        "epochs": {
            "values": [15]
        }
    }
}

