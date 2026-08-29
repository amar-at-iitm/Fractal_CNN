# Question 1

import torch
import torch.nn as nn
import torch.nn.functional as F

class SquaredReLU(nn.Module):
    def forward(self, x):
        return torch.pow(F.relu(x), 2)

class CubicReLU(nn.Module):
    def forward(self, x):
        return torch.pow(F.relu(x), 3)

# Helper function to get activation function by name
def get_activation(name):
    name = name.lower()
    if name == 'relu': return nn.ReLU()
    if name == 'squared_relu': return SquaredReLU()
    if name == 'cubic_relu': return CubicReLU()
    raise ValueError(f"Unsupported activation: {name}")

class CNNModel(nn.Module):      
    def __init__(self, 
                 filters,               # List of filters => Controls number of conv layers and filter sizes  
                 kernel_size,           # Size of filters  
                 activation,            # Activation function  
                 dropout,               # Dropout rate (optional)
                 use_batchnorm,         # Whether to use batch norm (optional)
                 input_shape=(3, 256, 256),  # Input shape compatible with iNaturalist dataset  
                 dense_units=256,       # Number of neurons in the dense (fully connected) layer  
                 num_classes=10):       # Output layer with 10 neurons  

        super().__init__()
        layers = []
        in_channels = input_shape[0]

        # Building conv-activation-maxpool blocks 
        for out_channels in filters:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1))  # Conv layer
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))  # Optional BatchNorm
            layers.append(get_activation(activation))         # Activation function 
            layers.append(nn.MaxPool2d(2))                    # Max pooling 
            if dropout > 0:
                layers.append(nn.Dropout(dropout))            # Optional dropout
            in_channels = out_channels

        # Feature extractor with 5 conv-activation-maxpool blocks 
        self.features = nn.Sequential(*layers)

        # Automatically calculate the output size after conv layers for the FC layer
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.features(dummy)
            flatten_size = out.view(1, -1).shape[1]

        # Classifier block with dense + activation + dropout + output 
        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, dense_units),   # First dense layer  
            get_activation(activation),             # Activation in dense layer  
            nn.Dropout(dropout),                    # Dropout
            nn.Linear(dense_units, num_classes)     # Output layer with 10 neurons  
        )

    def forward(self, x):
        x = self.features(x)            # Pass through convolutional blocks
        x = torch.flatten(x, 1)         # Flatten before fully connected layers
        return self.classifier(x)       # Output logits for classification
