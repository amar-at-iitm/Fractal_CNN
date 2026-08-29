sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "filters_per_layer": {
            "values": [
                [32, 32, 32, 32, 32],
                [32, 64, 64, 128, 128],
                [64, 64, 128, 128, 256],
                [32, 64, 128, 256, 512]
            ]
        },
        "activation": {
            "values": ["relu", "squared_relu", "cubic_relu"]
        },
        "use_batchnorm": {
            "values": [True, False]
        },
        "dropout_rate": {
            "values": [0.2, 0.3]
        },
        "dense_units": {
            "values": [128, 256, 512]
        },
        "augmentation": {
            "values": [True, False]
        },
        "batch_size": {
            "values": [32, 64]
        },
        "learning_rate": {
            "values": [1e-3, 3e-4, 1e-4]
        },
        "epochs": {
            "values": [10, 15, 20]
        }
    }
}
