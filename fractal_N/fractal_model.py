# fractal_model.py  (fractal_N variant — nonzero-part-only fractalization)

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


# ============================================================
# Base functions for fractalization: x, x², x³
# (only the positive/nonzero part — negatives are handled by
#  FractalActivationN which outputs 0 for x < 0)
# ============================================================
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


# ============================================================
# Fractal precomputation parameters
# ============================================================
a = 0       # Fractal domain starts at 0 (negatives → 0)
b = 1       # Fractal domain ends at b (above b → classical)
n_subintervals = 3
n_iter = 2
alpha = [ 0.1, 0.2, 0.3]


# ============================================================
# Base functions g for the RB operator (must differ from f).
#
#   g(x) = f(x) + x²(x - b)²
#
# The perturbation term x²(x-b)² vanishes at x=0 and x=b,
# so g(0) = f(0) and g(b) = f(b)  —  boundary conditions OK.
#
# Derivative of the perturbation:
#   d/dx [x²(x-b)²] = 2x(x-b)(2x - b)
# ============================================================
def _perturbation(x):
    x = np.asarray(x, dtype=float)
    return x**2 * (x - b)**2

def _d_perturbation(x):
    x = np.asarray(x, dtype=float)
    return 2.0 * x * (x - b) * (2.0 * x - b)

# g functions (f + perturbation)
def g_identity(x):
    return identity(x) + _perturbation(x)

def g_square(x):
    return square(x) + _perturbation(x)

def g_cube(x):
    return cube(x) + _perturbation(x)

# g' functions (f' + perturbation')
def dg_identity(x):
    return d_identity(x) + _d_perturbation(x)

def dg_square(x):
    return d_square(x) + _d_perturbation(x)

def dg_cube(x):
    return d_cube(x) + _d_perturbation(x)


# ============================================================
# Precompute fractal versions of x, x², x³ on [0, b].
# ============================================================
identity_fractal = alpha_fractalize(identity, g_identity, a, b, n_subintervals, alpha, n_iter, True)
d_identity_fractal = alpha_fractalize_first_derivative(d_identity, dg_identity, a, b, n_subintervals, alpha, n_iter, True)

square_fractal = alpha_fractalize(square, g_square, a, b, n_subintervals, alpha, n_iter, True)
d_square_fractal = alpha_fractalize_first_derivative(d_square, dg_square, a, b, n_subintervals, alpha, n_iter, True)

cube_fractal = alpha_fractalize(cube, g_cube, a, b, n_subintervals, alpha, n_iter, True)
d_cube_fractal = alpha_fractalize_first_derivative(d_cube, dg_cube, a, b, n_subintervals, alpha, n_iter, True)


# ============================================================
# Helper function to get a fractal activation nn.Module by name.
# Returns an nn.Module that can be used in nn.Sequential.
#
# Three-zone behavior per module:
#   x < 0      → 0
#   0 ≤ x ≤ b  → fractal interpolation
#   x > b      → classical  x / x² / x³
# ============================================================
def get_activation(name):
    name = name.lower()
    if name == 'f_relu':
        return FractalActivationN(identity_fractal, lambda x: x)
    if name == 'f_squared_relu':
        return FractalActivationN(square_fractal, lambda x: x ** 2)
    if name == 'f_cubic_relu':
        return FractalActivationN(cube_fractal, lambda x: x ** 3)
    raise ValueError(f"Unsupported activation: {name}")


class CNNModel(nn.Module):
    def __init__(self,
                 filters,                    # List of filters => Controls number of conv layers and filter sizes
                 kernel_size,                # Size of filters
                 activation,                 # Activation function
                 dropout,                    # Dropout rate (optional)
                 use_batchnorm,              # Whether to use batch norm (optional)
                 input_shape=(3, 192, 192),  # Input shape compatible with iNaturalist dataset (192x192)
                 dense_units=256,            # Number of neurons in the dense (fully connected) layer
                 num_classes=10):            # Output layer with 10 neurons

        super().__init__()
        layers = []
        in_channels = input_shape[0]

        # Building conv-activation-maxpool blocks
        for out_channels in filters:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1))  # Conv layer
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))   # Optional BatchNorm
            layers.append(get_activation(activation))         # Fractal activation module
            layers.append(nn.MaxPool2d(2))                    # Max pooling
            if dropout > 0:
                layers.append(nn.Dropout(dropout))            # Optional dropout
            in_channels = out_channels

        # Feature extractor with conv-activation-maxpool blocks
        self.features = nn.Sequential(*layers)

        # Automatically calculate the output size after conv layers for the FC layer
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.features(dummy)
            flatten_size = out.view(1, -1).shape[1]

        # Classifier block with dense + activation + dropout + output
        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, dense_units),   # First dense layer
            get_activation(activation),             # Fractal activation in dense layer
            nn.Dropout(dropout),                    # Dropout
            nn.Linear(dense_units, num_classes)     # Output layer with 10 neurons
        )

    def forward(self, x):
        x = self.features(x)            # Pass through convolutional blocks
        x = torch.flatten(x, 1)         # Flatten before fully connected layers
        return self.classifier(x)       # Output logits for classification
