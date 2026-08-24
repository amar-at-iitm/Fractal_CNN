# Fractal_CNN: 
---
### Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/amar-at-iitm/Fractal_CNN
   cd Fractal_CNN
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure WandB:
   ```bash
   wandb login
   ```

### Project Structure 
```
.
├── requirements.txt              # List of Python dependencies
├── README.md                     # Root README with project overview
├── datapreparation.py            # Downloads and prepares the data

├── inaturalist_12K/              # Dataset folder (train, val, test)
   ├── train/                     # Currently, 80% of original 
   ├── val/                       # 20% split from train
   └── test/                      # originally 'val/'

├── model.py                      # Custom CNN model with 5 conv layers
├── train.py                      # Training with wandb sweeps
├── test_model.py                 # Evaluate best model and show predictions
├── sweep_config.py               # Hyperparameter sweep config for wandb
├── best_config.py                # Best configuration from sweep
├── best_model.pth                # Saved best model weights
├── best_accuracy.txt             # Validation accuracy of best model
└── README.md                     # README for Part A

```
### `data_preparation.py`
- Downloads the Nature 12K dataset
- Unzips the downloaded file
- Renames the original val/ folder to test/
- Creates a new `val` folder containing 20% of images randomly moved from the `train` folder.
- Resizes all the images in each folder to 256*256
- Deletes unnecessary files( if exist) to avoid errors while training, validation and testing
#### Run the Script:
   ```bash
   python data_preparation.py
   ```
#### Dataset Structure After Processing
```
.
inaturalist_12K/
├── train/    # Currently, 80% of original 
├── val/      # 20% split from train
└── test/     # originally 'val/'
```

## CNN based image classifiers using a subset of the iNaturalist dataset.

- Build a CNN with:
   - 5 Conv → Activation → MaxPool blocks
   - Customizable dense & output layers (10 classes)
   - Flexible filters, kernel sizes, activations, and neurons
- Compute total parameters & operations (based on m, k×k, n)
- Train on iNaturalist:
   - Use 80/20 train-validation split (balanced by class)
   - Apply WandB sweeps for hyperparameter tuning:
      - Filters, activations, dropout, batch norm, etc.
- Include:
   - Accuracy vs experiments plot
   - Parallel coordinates & correlation summary
- Report test accuracy
- Show results in a creative 10×3 prediction grid.

### `model.py`
The model is designed for multi-class image classification tasks and is tailored to work with the iNaturalist dataset (10 classes). It provides flexibility regarding layer configuration, activation functions, and regularization techniques.
- Modular CNN architecture with 5 convolutional blocks
- Customizable filter configuration, kernel size, and activation functions
- Optional Batch Normalization and Dropout
- Automatically handles flattening based on input image size
- Designed to work with inputs of shape (3, 256, 256) (iNaturalist dataset standard)

### `train.py`
- Run the Script:
   ```bash
   python train.py
   ```
This script trains a configurable Convolutional Neural Network (CNN) on the iNaturalist 12K dataset using PyTorch. It integrates Weights & Biases (wandb) for experiment tracking and supports sweep-based hyperparameter tuning. It also saves the best-performing model based on validation accuracy.
- Leverages the modular CNN model from `Question 1: model.py`
- Uses wandb sweeps to perform hyperparameter optimization
- Implements optional data augmentation, batch normalization, and dropout
- Tracks training/validation loss and accuracy across epochs
- Saves the best model (based on validation accuracy) to `best_model.pth`

### `test_model.py`
- Run the Script:
   ```bash
   python test_model.py
   ```
The `test_model.py` script is used to evaluate the performance of the best-trained CNN model on the **test split** of the iNaturalist_12K dataset. This script also logs final metrics and predictions to **Weights & Biases (wandb)** for visualization and reporting.


- Loads and applies the best configuration from `best_config.py`
- Evaluates test accuracy using the saved model (`model_path`)
- Visualizes predictions on a 10x3 image grid with true vs. predicted labels
- Logs:
  - Test accuracy
  - Sample prediction grid (image panel) `10*3`