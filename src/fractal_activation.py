import torch
import torch.nn as nn
import torch.nn.functional as F


class FractalActivation(nn.Module):
    """PyTorch-native fractal activation function.

    Uses a precomputed fractal lookup table (from alpha_fractalize) for inputs
    within the interpolation domain, and falls back to the classical activation
    for inputs outside the domain.

    Args:
        fractal_lut: dict with 'partition' (x-coords) and 'values' (y-coords)
                     from alpha_fractalize(..., dict=True).
        classical_fn: callable that takes a torch.Tensor and returns a
                      torch.Tensor — the classical activation to use outside
                      the fractal domain. e.g. lambda x: F.relu(x)
    """

    def __init__(self, fractal_lut, classical_fn):
        super().__init__()
        # Store the precomputed LUT as non-trainable buffers (auto-move with .to(device))
        self.register_buffer(
            'xp', torch.tensor(fractal_lut['partition'], dtype=torch.float32)
        )
        self.register_buffer(
            'fp', torch.tensor(fractal_lut['values'], dtype=torch.float32)
        )
        self.classical_fn = classical_fn

    def forward(self, x):
        # Mask: which elements fall inside the fractal domain
        in_domain = (x >= self.xp[0]) & (x <= self.xp[-1])

        # --- Fractal branch (piecewise-linear interpolation on the LUT) ---
        # Clamp so searchsorted doesn't go out-of-bounds; the clamp result is
        # only used where in_domain is True (via torch.where below).
        x_clamped = x.clamp(self.xp[0], self.xp[-1])

        # Find the left index of the interval containing each value
        idx = torch.searchsorted(self.xp, x_clamped) - 1
        idx = idx.clamp(0, len(self.xp) - 2)

        # Linear interpolation: y = y0 + t * (y1 - y0)
        x0 = self.xp[idx]
        x1 = self.xp[idx + 1]
        y0 = self.fp[idx]
        y1 = self.fp[idx + 1]
        t = (x_clamped - x0) / (x1 - x0)
        fractal_out = y0 + t * (y1 - y0)

        # --- Classical branch (for values outside the fractal domain) ---
        classical_out = self.classical_fn(x)

        # Combine: fractal inside domain, classical outside
        return torch.where(in_domain, fractal_out, classical_out)

