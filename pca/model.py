import sys
from pathlib import Path

# Ensure the project root is on sys.path so we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fractal_functions import (alpha_fractalize, alpha_fractalize_first_derivative)
from src.fractal_activation_N import FractalActivationN


# ==============================================================================
# Classical Activation Functions
# ==============================================================================

class SquaredReLU(nn.Module):
    """Squared ReLU: f(x) = max(0, x)^2"""
    def forward(self, x):
        return torch.pow(F.relu(x), 2)


class CubicReLU(nn.Module):
    """Cubic ReLU: f(x) = max(0, x)^3"""
    def forward(self, x):
        return torch.pow(F.relu(x), 3)


def get_classical_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    if name == 'squared_relu':
        return SquaredReLU()
    if name == 'cubic_relu':
        return CubicReLU()
    raise ValueError(f"Unsupported classical activation: '{name}'. Choices: ['relu', 'squared_relu', 'cubic_relu']")


# ==============================================================================
# Fractal Activation Base Functions and Construction
# ==============================================================================

def identity(x):
    return np.asarray(x, dtype=float)

def d_identity(x):
    return np.ones_like(np.asarray(x, dtype=float))

def square(x):
    return np.asarray(x, dtype=float) ** 2

def d_square(x):
    return 2.0 * np.asarray(x, dtype=float)

def cube(x):
    return np.asarray(x, dtype=float) ** 3

def d_cube(x):
    return 3.0 * np.asarray(x, dtype=float) ** 2


# Domain and subinterval parameters
FRACTAL_A = 0
FRACTAL_B = 1
FRACTAL_N_SUBINTERVALS = 2
FRACTAL_N_ITER = 10


def _perturbation(x, b=FRACTAL_B):
    x = np.asarray(x, dtype=float)
    return x**2 * (x - b)**2


def _d_perturbation(x, b=FRACTAL_B):
    x = np.asarray(x, dtype=float)
    return 2.0 * x * (x - b) * (2.0 * x - b)


def g_identity(x):
    return identity(x) + _perturbation(x)

def g_square(x):
    return square(x) + _perturbation(x)

def g_cube(x):
    return cube(x) + _perturbation(x)

def dg_identity(x):
    return d_identity(x) + _d_perturbation(x)

def dg_square(x):
    return d_square(x) + _d_perturbation(x)

def dg_cube(x):
    return d_cube(x) + _d_perturbation(x)


def get_fractal_activation(name: str, alpha1: float = 0.2, alpha2: float = 0.2) -> nn.Module:
    """Instantiate a FractalActivationN module with specified alphas."""
    alpha = [alpha1, alpha2]
    name = name.lower()

    if name in ['f_relu', 'fractal_relu']:
        frac_func = alpha_fractalize(identity, g_identity, FRACTAL_A, FRACTAL_B, FRACTAL_N_SUBINTERVALS, alpha, FRACTAL_N_ITER, True)
        return FractalActivationN(frac_func, lambda x: x)
    elif name in ['f_squared_relu', 'fractal_squared_relu']:
        frac_func = alpha_fractalize(square, g_square, FRACTAL_A, FRACTAL_B, FRACTAL_N_SUBINTERVALS, alpha, FRACTAL_N_ITER, True)
        return FractalActivationN(frac_func, lambda x: x ** 2)
    elif name in ['f_cubic_relu', 'fractal_cubic_relu']:
        frac_func = alpha_fractalize(cube, g_cube, FRACTAL_A, FRACTAL_B, FRACTAL_N_SUBINTERVALS, alpha, FRACTAL_N_ITER, True)
        return FractalActivationN(frac_func, lambda x: x ** 3)
    else:
        raise ValueError(f"Unsupported fractal activation: '{name}'. Choices: ['f_relu', 'f_squared_relu', 'f_cubic_relu']")


# ==============================================================================
# 1D Convolutional Neural Network (CNN1D) for 1D PCA Datasets
# ==============================================================================

class CNN1DModel(nn.Module):
    """
    1D Convolutional Neural Network architecture designed for 1D PCA datasets.
    
    Supports both Classical and Fractal activations:
      - When approach == 'classical': uses ReLU, SquaredReLU, or CubicReLU
      - When approach == 'fractal'  : uses FractalActivationN with alpha1, alpha2
      
    Architecture:
      Input (B, 1, L) or (B, L)
      ├── [Conv1d -> (BatchNorm1d) -> Activation -> MaxPool1d(2) -> (Dropout)] x len(filters)
      ├── Flatten
      └── Classifier: Linear(flatten_size, dense_units) -> Activation -> Dropout -> Linear(dense_units, num_classes)
    """
    def __init__(
        self,
        filters=(64, 128, 256),
        kernel_size=5,
        approach="classical",
        activation="relu",
        dropout=0.2,
        use_batchnorm=True,
        alpha1=0.2,
        alpha2=0.2,
        input_length=1024,
        dense_units=256,
        num_classes=10
    ):
        super().__init__()
        self.approach = approach.lower()
        self.activation_name = activation
        self.input_length = input_length
        self.filters = list(filters)
        self.kernel_size = kernel_size

        # Factory to get activation instance
        def create_activation():
            if self.approach == "classical":
                return get_classical_activation(activation)
            elif self.approach == "fractal":
                return get_fractal_activation(activation, alpha1=alpha1, alpha2=alpha2)
            else:
                raise ValueError(f"Unknown approach: '{self.approach}'. Must be 'classical' or 'fractal'.")

        # Feature extractor layers
        layers = []
        in_channels = 1  # 1D PCA signal has 1 channel
        padding = kernel_size // 2

        for out_channels in self.filters:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(create_activation())
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        # Compute flattened dimension dynamically using a dummy tensor
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            out = self.features(dummy)
            flatten_size = out.view(1, -1).shape[1]

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, dense_units),
            create_activation(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dense_units, num_classes)
        )

    def forward(self, x):
        # Allow both (B, L) and (B, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[1] != 1 and x.shape[2] == 1:
            x = x.transpose(1, 2)

        feat = self.features(x)
        feat_flat = feat.view(feat.size(0), -1)
        out = self.classifier(feat_flat)
        return out


# Alias helper
def get_1d_model(
    approach: str = "classical",
    filters=(64, 128, 256),
    kernel_size: int = 5,
    activation: str = "relu",
    dropout: float = 0.2,
    use_batchnorm: bool = True,
    alpha1: float = 0.2,
    alpha2: float = 0.2,
    input_length: int = 1024,
    dense_units: int = 256,
    num_classes: int = 10
) -> CNN1DModel:
    """Helper factory function to create a 1D CNN model."""
    return CNN1DModel(
        filters=filters,
        kernel_size=kernel_size,
        approach=approach,
        activation=activation,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
        alpha1=alpha1,
        alpha2=alpha2,
        input_length=input_length,
        dense_units=dense_units,
        num_classes=num_classes
    )

