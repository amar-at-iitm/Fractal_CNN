# Fractal_CNN

A PyTorch deep learning framework implementing Convolutional Neural Networks (CNNs) on a subset of the **iNaturalist 12K** dataset, comparing standard classical activations with novel **Fractal Activation Functions** constructed via Read-Bajraktarević (RB) operator fractal interpolation.

---

<!-- ## Table of Contents
1. [Installation & Setup](#installation--setup)
2. [Project Structure](#project-structure)
3. [Dataset Preparation](#dataset-preparation)
4. [Classical CNN Pipeline (`classical/`)](#classical-cnn-pipeline-classical)
5. [Fractal CNN Pipeline (`fractal_N/`)](#fractal-cnn-pipeline-fractal_n)
   - [Theoretical Background](#theoretical-background)
   - [Activation Layer Implementation](#activation-layer-implementation)
   - [Fractal CNN Architecture](#fractal-cnn-architecture)
   - [Automated Hyperparameter Sweep & Grid Search](#automated-hyperparameter-sweep--grid-search)
   - [Global Best Model Tracking](#global-best-model-tracking)
   - [Evaluation on Test Set](#evaluation-on-test-set)
6. [Usage Guide](#usage-guide) -->

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/amar-at-iitm/Fractal_CNN
   cd Fractal_CNN
   ```

2. **Set up a virtual environment:**
   ```bash
   python3 -m venv frctl_env
   source frctl_env/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Authenticate Weights & Biases (WandB):**
   ```bash
   wandb login
   ```

---

## Project Structure

```text
.
├── requirements.txt                 # List of Python dependencies
├── datapreparation.py               # Downloads, splits, and prepares the iNaturalist dataset
├── inaturalist_12K/                 # Dataset directory
│   ├── train/                       # 80% train split (from original train)
│   ├── val/                         # 20% validation split (from original train)
│   └── test/                        # Evaluation split (originally 'val/')
│
├── src/                             # Core library utilities & fractal math
│   ├── activations.py               # Classical activation functions and derivatives
│   ├── base_functions.py            # Hermite interpolation base functions (H5)
│   ├── fractal_functions.py         # RB alpha-fractalization algorithms & derivatives
│   ├── fractal_activation.py        # Fractal activation module for [-1, 1] domain
│   ├── fractal_activation_N.py      # Nonzero-part fractal activation module for [0, 1] domain
│   └── tools.py                     # Data augmentation and transform pipelines
│
├── classical/                       # Classical CNN pipeline
│   ├── model.py                     # 5-block CNN supporting standard activations (ReLU, GELU, etc.)
│   ├── sweep_config.py              # Hyperparameter sweep configuration (Bayesian/Grid)
│   ├── train.py                     # Training script with WandB sweep integration
│   ├── test_model.py                # Test set evaluation and prediction grid visualization
│   ├── best_config.py               # Best configuration recorded from sweep
│   ├── best_model.pth               # Saved weights of the best classical model
│   └── best_accuracy.txt            # Highest validation accuracy achieved
│
├── fractal_N/                       # Fractal CNN pipeline (Nonzero-part-only on [0, 1])
│   ├── fractal_model.py             # 5-block CNN using FractalActivationN
│   ├── fractal_train.py             # Script-based training with WandB sweep integration
│   ├── fractal_train.ipynb          # Interactive notebook with automated (alpha1, alpha2) grid search
│   ├── fractal_sweep_config.py      # Architecture and training hyperparameter sweep configuration
│   ├── check_activation.ipynb       # Activation plotting & verification notebook
│   ├── test_fractal_model.py        # Test set evaluation and 10x3 prediction grid visualization
│   ├── best_config.py               # Optimal hyperparameters for evaluation
│   ├── best_model.pth               # Saved weights of the best fractal model
│   ├── best_accuracy.txt            # Top validation accuracy & winning hyperparameters
│   └── LAYERS_GUIDE.md              # Architectural reference and guide for modifying layers
│
└── README.md                        # Project documentation
```

---

## Dataset Preparation

The dataset used is a 10-class subset of the **iNaturalist 12K** dataset. Run `datapreparation.py` to prepare the directory structure:

```bash
python datapreparation.py
```

### What `datapreparation.py` does:
- Downloads and extracts the iNaturalist dataset.
- Renames the original `val/` folder to `test/` (to serve as an unseen benchmark).
- Splits the original `train/` folder into an **80/20 train-validation split** (stratified across all 10 classes).
- Resizes images and verifies integrity to ensure consistent input tensors.

---

## Classical CNN Pipeline (`classical/`)

The classical pipeline implements a modular 5-block convolutional architecture:
- **Feature Extractor:** 5 sequential blocks of `Conv2d(3x3, pad=1)` → `BatchNorm2d` (optional) → `Activation` → `MaxPool2d(2x2)` → `Dropout` (optional).
- **Supported Activations:** `ReLU`, `GELU`, `SiLU`, `Mish`.
- **Classifier Head:** `Linear(flatten_size, dense_units)` → `Activation` → `Dropout` → `Linear(dense_units, 10)`.
- **Hyperparameter Sweeps (`sweep_config.py` & `train.py`):**
  Uses WandB sweeps to optimize filters per layer, activation function, dense units, learning rate, batch size, dropout rate, batch normalization, and data augmentation.
- **Evaluation (`test_model.py`):**
  Loads `best_model.pth` and `best_config.py`, evaluates on the test split, and logs a 10×3 sample prediction grid to WandB.

---

## Fractal CNN Pipeline (`fractal_N/`)

The `fractal_N` directory explores a novel **fractal perturbation approach** to neural network activations.

### Theoretical Background
Standard activations like ReLU set negative inputs to zero ($x < 0 \to 0$) and apply an identity mapping for positive values. The `fractal_N` approach preserves the negative-suppression property of ReLU while replacing the linear positive segment with an **$\alpha$-fractal interpolation** on the domain $[0, b]$ (where $b = 1$):

$$\text{Activation}(x) = \begin{cases} 
0 & \text{for } x < 0 \\ 
f_\alpha(x) & \text{for } 0 \le x \le b \\ 
x & \text{for } x > b 
\end{cases}$$

Here, $f_\alpha(x)$ is constructed using the **Read-Bajraktarević (RB) operator** on $n_{\text{subintervals}} = 2$ and $n_{\text{iter}} = 2$ with scale factors $\alpha = [\alpha_1, \alpha_2]$. The base function $f(x) = x$ is perturbed by:

$$g(x) = f(x) + x^2(x - b)^2$$

Because $x^2(x - b)^2$ vanishes at $x = 0$ and $x = b$, boundary continuity is strictly preserved ($g(0) = f(0) = 0$ and $g(b) = f(b) = b$).

### Activation Layer Implementation
Implemented in [`src/fractal_activation_N.py`](src/fractal_activation_N.py) as `FractalActivationN(nn.Module)`:
- Precomputes a lookup table (LUT) of $(x_p, f_p)$ points from the fractal interpolation algorithm.
- Computes piecewise-linear interpolation natively in PyTorch via `torch.searchsorted`, ensuring full differentiability via PyTorch Autograd.
- Applies three-zone masking: $0$ for $x < 0$, fractal interpolation for $0 \le x \le 1$, and classical pass-through for $x > 1$.

### Fractal CNN Architecture
Implemented in [`fractal_N/fractal_model.py`](fractal_N/fractal_model.py):
- Input resolution: $(3, 192, 192)$ (halved spatial dimensions across 5 pooling stages to $6 \times 6$).
- Dynamically accepts `filters`, `kernel_size`, `alpha1`, `alpha2`, `dense_units`, `dropout`, and `use_batchnorm`.
- Automatic feature map flattening calculation via a dummy forward pass.

### Automated Hyperparameter Sweep & Grid Search
In [`fractal_N/fractal_train.ipynb`](fractal_N/fractal_train.ipynb), training is automated across all combinations of $\alpha_1$ and $\alpha_2$:

```python
ALPHA1 = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45] 
ALPHA2 = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]

for alpha1 in ALPHA1:
    for alpha2 in ALPHA2:
        alpha = [alpha1, alpha2]
        relu_fractal = alpha_fractalize(identity, g_identity, a, b, n_subintervals, alpha, n_iter, True)
        d_relu_fractal = alpha_fractalize_first_derivative(d_identity, dg_identity, a, b, n_subintervals, alpha, n_iter, True)
        relu_fractals[(alpha1, alpha2)] = relu_fractal
        d_relu_fractals[(alpha1, alpha2)] = d_relu_fractal

        # Set active alpha combination for train()
        current_alpha1 = alpha1
        current_alpha2 = alpha2

        # Run hyperparameter optimization sweep for this (alpha1, alpha2) pair
        sweep_id = wandb.sweep(sweep_config, project="fractal_CNN")
        wandb.agent(sweep_id, function=train, count=SWEEP_RUNS_PER_COMBO)
        wandb.finish()
```

- **WandB Tracking:** `wandb.config` is dynamically updated with `alpha1` and `alpha2`. Run names explicitly reflect both the alpha parameters and sweep architecture choices (e.g., `run_a1-0.2_a2-0.25_filters-[32,64,128,256,512]_lr-0.001`).
- **Mixed Precision:** Uses `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')` for accelerated training throughput and reduced GPU memory usage.

### Global Best Model Tracking
`best_accuracy.txt` and `best_model.pth` are **never reset** between different $(\alpha_1, \alpha_2)$ combinations.
- A global variable `GLOBAL_BEST_VAL_ACC` synchronizes in memory and against `best_accuracy.txt`.
- Checkpoints and metric files are updated **only** when a run strictly surpasses the global best validation accuracy achieved so far across all combinations.
- If a combination yields lower accuracy, `best_accuracy.txt` is retained intact.
- `best_accuracy.txt` stores the top accuracy and winning parameter configuration:
  <!-- ```text
  val_acc: <highest_val_acc>
  alpha1: <best_alpha1>
  alpha2: <best_alpha2>
  filters_per_layer: [32, 64, 128, 256, 512]
  activation: f_relu
  dense_units: 256
  learning_rate: 0.001
  batch_size: 32
  dropout_rate: 0.2
  use_batchnorm: True
  augmentation: True
  epochs: 10
  ``` -->

### Evaluation on Test Set
[`fractal_N/test_fractal_model.py`](fractal_N/test_fractal_model.py) evaluates `best_model.pth` against the unseen test dataset:
- Computes overall classification accuracy across the 10 classes.
- Generates a 10×3 sample prediction panel displaying true vs. predicted labels.
- Logs prediction grids and test metrics directly to WandB.

---

## Usage Guide

### 1. Running Classical Training & Evaluation
```bash
# Run WandB sweep training
python classical/train.py

# Evaluate the best classical model on the test split
python classical/test_model.py
```

### 2. Running Fractal CNN Training & Sweeps
```bash
# Option A: Interactive Notebook (Automated alpha search + hyperparameter sweeps)
jupyter notebook fractal_N/fractal_train.ipynb

# Option B: Script-based single sweep execution
python fractal_N/fractal_train.py

# Evaluate the best fractal model on the test split
python fractal_N/test_fractal_model.py
```

### 3. Visualizing Fractal Activations
Open [`fractal_N/check_activation.ipynb`](fractal_N/check_activation.ipynb) in Jupyter to inspect and plot the mathematical shape of $f_{\text{relu}}$, $f_{\text{squared\_relu}}$, and $f_{\text{cubic\_relu}}$ across domain intervals.
