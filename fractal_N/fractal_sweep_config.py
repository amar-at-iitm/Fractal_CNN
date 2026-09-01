sweep_config = {
    "method": "grid",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "alpha1" : {
            "values": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]}, 
        "alpha2" : {
            "values": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]},
        "filters_per_layer": {
            "values": [
                [32, 64, 128, 256, 512]
            ]
        },
        "activation": {
            "values": ["f_relu"]
        },
        "use_batchnorm": {
            "values": [True, False]
        },
        "dropout_rate": {
            "values": [0.2]
        },
        "dense_units": {
            "values": [128, 256, 512]
        },
        "augmentation": {
            "values": [True, False]
        },
        "batch_size": {
            "values": [32]
        },
        "learning_rate": {
            "values": [1e-3, 1e-4, 1e-5]
        },
        "epochs": {
            "values": [10]
        }
    }
}
