import numpy as np 


# 1. Standard ReLU: max(0, x)
def relu(x):
    return np.maximum(0, x)

# 2. Quadratic ReLU (Squared ReLU): max(0, x^2)
def squared_relu(x):
    return np.maximum(0, x) ** 2
    # equivalently: np.maximum(0, x**2)

# 3. Cubic ReLU: max(0, x^3)
def cubic_relu(x):
    return np.maximum(0, x) ** 3


#==============================================================
# First Derivatives
#==============================================================

# First Derivative of ReLU is 1 for x > 0 and 0 for x <= 0
def d_relu(x):
    x = np.asarray(x, dtype=float)
    return (x > 0).astype(float)

# First Derivative of Squared ReLU is 2x for x > 0 and 0 for x <= 0
def d_squared_relu(x):
    return 2 * np.maximum(0, x)

# First Derivative of Cubic ReLU is 3x^2 for x > 0 and 0 for x <= 0
def d_cubic_relu(x):
    return 3 * np.maximum(0, x) ** 2


#==============================================================
# Second Derivatives
#==============================================================

# 2nd Derivative of ReLU is zero everywhere except at x=0, where it's undefined. For practical purposes, we can return zero.
def dd_relu(x):
    return np.zeros_like(np.asarray(x, dtype=float))

# 2nd Derivative of Squared ReLU is 2 for x > 0 and 0 for x <= 0
def dd_squared_relu(x):
    x = np.asarray(x, dtype=float)
    return 2 * (x > 0).astype(float)

# 2nd Derivative of Cubic ReLU is 6x for x > 0 and 0 for x <= 0
def dd_cubic_relu(x):
    return 6 * np.maximum(0, x)