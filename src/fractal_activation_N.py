import torch
import torch.nn as nn


class FractalActivationN(nn.Module):
    """PyTorch-native fractal activation for the nonzero-part-only approach.

    Three-zone behavior:
        x < 0        → 0              (like ReLU: kill negatives)
        0 ≤ x ≤ b    → fractal interp (from precomputed LUT on [0, b])
        x > b        → classical_fn(x)(direct x, x², or x³)

    Args:
        fractal_lut: dict with 'partition' (x-coords) and 'values' (y-coords)
                     from alpha_fractalize(..., dict=True) on [0, b].
        classical_fn: callable for x > b, e.g. lambda x: x  (identity),
                      lambda x: x**2, lambda x: x**3.
    """

    def __init__(self, fractal_lut, classical_fn):
        super().__init__()
        self.register_buffer(
            'xp', torch.tensor(fractal_lut['partition'], dtype=torch.float32)
        )
        self.register_buffer(
            'fp', torch.tensor(fractal_lut['values'], dtype=torch.float32)
        )
        self.classical_fn = classical_fn

    def forward(self, x):
        # --- Zone masks ---
        negative = x < self.xp[0]                                  # x < 0
        in_domain = (x >= self.xp[0]) & (x <= self.xp[-1])         # 0 ≤ x ≤ b
        # above_domain (x > b) is the implicit default

        # --- Fractal branch (piecewise-linear interp on the LUT) ---
        x_clamped = x.clamp(self.xp[0], self.xp[-1])
        idx = torch.searchsorted(self.xp, x_clamped) - 1
        idx = idx.clamp(0, len(self.xp) - 2)
        x0 = self.xp[idx]
        x1 = self.xp[idx + 1]
        y0 = self.fp[idx]
        y1 = self.fp[idx + 1]
        t = (x_clamped - x0) / (x1 - x0)
        fractal_out = y0 + t * (y1 - y0)

        # --- Classical branch (for x > b: direct x, x², or x³) ---
        classical_out = self.classical_fn(x)

        # --- Combine: start with classical, overlay fractal, then zero ---
        result = classical_out                                      # x > b
        result = torch.where(in_domain, fractal_out, result)       # 0 ≤ x ≤ b
        result = torch.where(negative, torch.zeros_like(x), result) # x < 0

        return result

