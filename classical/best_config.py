# best_config.py
# Update these parameters with your best sweep run results

best_config = {
    "filters_per_layer": [32, 64, 128, 256, 512],
    "activation": "relu",
    "dropout_rate": 0.2,
    "use_batchnorm": True,
    "input_shape": (3, 192, 192),
    "batch_size": 32,
    "model_path": "best_model.pth"
}

