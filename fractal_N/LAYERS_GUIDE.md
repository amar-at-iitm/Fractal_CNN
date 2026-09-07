# Guide: Understanding and Modifying Model Layers in `fractal_N`

This guide explains in detail:
1. **Where and how the model is defined** in `fractal_N`.
2. **How convolutional and dense layers are constructed dynamically**.
3. **Step-by-step instructions** to add or remove layers across all project files (`fractal_model.py`, `fractal_train.py`, `fractal_sweep_config.py`, `best_config.py`, and `test_fractal_model.py`).
4. **Critical precautions** regarding spatial dimension collapse, checkpoint loading, activation functions, and memory.

---

## 1. Where and How the Model is Defined

The model architecture is defined in:
📁 **[`fractal_N/fractal_model.py`](file:///home/amar-kumar/Desktop/Fractal/NN/Fractal_CNN/fractal_N/fractal_model.py)** under the class **`CNNModel(nn.Module)`** (lines 89–154).

### Key Components of `CNNModel`:

```
Input Image: (3, 192, 192)
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  self.features = nn.Sequential(*layers)                     │
│  Repeated for each value in `filters`:                       │
│    1. nn.Conv2d(in_channels, out_channels, kernel_size, p=1)│
│    2. nn.BatchNorm2d(out_channels)  [if use_batchnorm]      │
│    3. FractalActivationN(...)       [f_relu/squared/cubic]  │
│    4. nn.MaxPool2d(kernel_size=2, stride=2)  (Halves H & W) │
│    5. nn.Dropout(dropout)            [if dropout > 0]       │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
Flatten (Spatial dimension → 1D Vector of size `flatten_size`)
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  self.classifier = nn.Sequential(...)                       │
│    1. nn.Linear(flatten_size, dense_units)                  │
│    2. FractalActivationN(...)                               │
│    3. nn.Dropout(dropout)                                   │
│    4. nn.Linear(dense_units, num_classes=10)                │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
Output Logits (10 classes)
```

### Dynamic Layer Generation:
The convolutional depth is **not hardcoded**. It is dynamically constructed by iterating over the `filters` argument:

```python
layers = []
in_channels = input_shape[0]

# Building conv-activation-maxpool blocks
for out_channels in filters:
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1))
    if use_batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(get_activation(activation))
    layers.append(nn.MaxPool2d(2))
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    in_channels = out_channels

self.features = nn.Sequential(*layers)
```

The feature map output size is automatically calculated by running a dummy tensor through `self.features`:
```python
with torch.no_grad():
    dummy = torch.zeros(1, *input_shape)
    out = self.features(dummy)
    flatten_size = out.view(1, -1).shape[1]
```

---

## 2. How to Add or Remove Convolutional Layers

Because `CNNModel` generates convolutional blocks from the `filters` list, the number of convolutional blocks equals `len(filters)`.

### Example 1: Removing Layers (e.g., 5 layers → 3 layers)
- **Original (5 layers):** `filters = [32, 64, 128, 256, 512]`
- **3 layers:** `filters = [32, 64, 128]` or `[64, 128, 256]`

### Example 2: Adding Layers (e.g., 5 layers → 6 layers)
- **6 layers:** `filters = [32, 64, 128, 256, 512, 512]`

---

## 3. Files You Must Update When Changing Layers

When you modify the number of layers, you should update the configurations across the following files:

### A. [`fractal_sweep_config.py`](file:///home/amar-kumar/Desktop/Fractal/NN/Fractal_CNN/fractal_N/fractal_sweep_config.py)
In `sweep_config["parameters"]["filters_per_layer"]`, add or change the filter lists you want W&B to sweep over:
```python
"filters_per_layer": {
    "values": [
        [32, 64, 128],                 # 3 layers
        [32, 64, 128, 256],            # 4 layers
        [32, 64, 128, 256, 512],       # 5 layers (default)
        [32, 64, 128, 256, 512, 512]   # 6 layers
    ]
},
```

### B. [`best_config.py`](file:///home/amar-kumar/Desktop/Fractal/NN/Fractal_CNN/fractal_N/best_config.py)
Update `filters_per_layer` to match the trained architecture you wish to evaluate or test:
```python
best_config = {
    "filters_per_layer": [32, 64, 128, 256],  # updated to 4 layers
    "activation": "f_relu",
    "dropout_rate": 0.2,
    "use_batchnorm": True,
    "input_shape": (3, 192, 192),
    "batch_size": 32,
    "model_path": "best_model.pth"
}
```

### C. [`fractal_train.py`](file:///home/amar-kumar/Desktop/Fractal/NN/Fractal_CNN/fractal_N/fractal_train.py)
`fractal_train.py` reads `filters = config.filters_per_layer` automatically from W&B sweep.
> [!NOTE]
> In `fractal_train.py` (lines 67–76), `dense_units` is currently not passed from `config`. If you also want to tune dense units via sweep, pass `dense_units=config.dense_units` into `CNNModel(...)`.

### D. [`test_fractal_model.py`](file:///home/amar-kumar/Desktop/Fractal/NN/Fractal_CNN/fractal_N/test_fractal_model.py)
`test_fractal_model.py` imports `best_config["filters_per_layer"]` and passes it to `CNNModel`. As long as `best_config.py` is updated and matches the checkpoint saved in `best_config["model_path"]`, it works without further edits.

---

## 4. How to Add or Remove Fully Connected (Dense) Layers

Currently, `CNNModel` has **1 hidden dense layer** + **1 output layer**:
```python
self.classifier = nn.Sequential(
    nn.Linear(flatten_size, dense_units),   # Hidden layer
    get_activation(activation),             # Activation
    nn.Dropout(dropout),                    # Dropout
    nn.Linear(dense_units, num_classes)     # Output classification layer
)
```

### To add another hidden dense layer:
In [`fractal_N/fractal_model.py`](file:///home/amar-kumar/Desktop/Fractal/NN/Fractal_CNN/fractal_N/fractal_model.py), modify `self.classifier`:
```python
self.classifier = nn.Sequential(
    nn.Linear(flatten_size, dense_units),
    get_activation(activation),
    nn.Dropout(dropout),
    nn.Linear(dense_units, dense_units // 2),  # Extra hidden layer
    get_activation(activation),
    nn.Dropout(dropout),
    nn.Linear(dense_units // 2, num_classes)   # Final output layer
)
```

### To support an arbitrary number of dense layers dynamically:
You can accept `dense_units` as a list (e.g., `dense_units=[512, 256]`):
```python
fc_layers = []
in_dim = flatten_size
for units in dense_units:
    fc_layers.append(nn.Linear(in_dim, units))
    fc_layers.append(get_activation(activation))
    if dropout > 0:
        fc_layers.append(nn.Dropout(dropout))
    in_dim = units
fc_layers.append(nn.Linear(in_dim, num_classes))
self.classifier = nn.Sequential(*fc_layers)
```

---

## 5. Critical Precautions to Maintain When Changing Layers

### ⚠️ Precaution 1: Spatial Dimension Collapse (Maximum Conv Layers Limit)
Every convolutional block currently includes `nn.MaxPool2d(2)`, which divides both width and height by 2 ($W_{new} = \lfloor W / 2 \rfloor$).
For CIFAR-10 input size of **$32 \times 32$**:

| Layer Block | Spatial Dimension After Block | Status |
| :--- | :--- | :--- |
| Input | $32 \times 32$ | Initial size |
| Block 1 | $16 \times 16$ | OK |
| Block 2 | $8 \times 8$ | OK |
| Block 3 | $4 \times 4$ | OK |
| Block 4 | $2 \times 2$ | OK (Optimal 4-layer architecture: `[64, 128, 256, 512]`) |
| Block 5 | $1 \times 1$ | OK (Deep 5-layer architecture: `[32, 64, 128, 256, 512]`) |
| **Block 6** | **$0 \times 0$** | **CRASH! ($\lfloor 1 / 2 \rfloor = 0$)** |

> [!CAUTION]
> **CIFAR-10 Hard Limit:** You **cannot have 6 or more layers** on $32 \times 32$ if every layer has `nn.MaxPool2d(2)`. At layer 6, spatial dimensions collapse to $0 \times 0$, causing a PyTorch crash. Keep your architectures to 4 or 5 pooling blocks.

For high-resolution inputs (e.g. **$192 \times 192$**):

| Layer Block | Spatial Dimension After Block | Status |
| :--- | :--- | :--- |
| Input | $192 \times 192$ | Initial size |
| Block 1 | $96 \times 96$ | OK |
| Block 2 | $48 \times 48$ | OK |
| Block 3 | $24 \times 24$ | OK |
| Block 4 | $12 \times 12$ | OK |
| Block 5 | $6 \times 6$ | OK |
| Block 6 | $3 \times 3$ | OK |
| Block 7 | $1 \times 1$ | OK ($\lfloor 3 / 2 \rfloor = 1$) |
| **Block 8** | **$0 \times 0$** | **CRASH! ($\lfloor 1 / 2 \rfloor = 0$)** |


---

### ⚠️ Precaution 2: Checkpoint & State Dict Incompatibility (`best_model.pth`)
PyTorch state dicts store weights keyed by exact layer indices (e.g., `features.0.weight`, `features.4.weight`, `classifier.0.weight`).
- If you change the number of layers from 5 to 4 or 6, any attempt to load an old `best_model.pth` with `model.load_state_dict(...)` in `test_fractal_model.py` will fail with:
  ```text
  RuntimeError: Error(s) in loading state_dict for CNNModel:
  Missing key(s) in state_dict: ...
  Unexpected key(s) in state_dict: ...
  size mismatch for ...
  ```
- **Rule:** Whenever you change the number of layers, either:
  1. Train a new model and save to a unique filename (e.g., `best_model_4layers.pth`), or
  2. Overwrite `best_model.pth` by re-running training from scratch, and **reset `best_accuracy.txt`** to `0.0`.

---

### ⚠️ Precaution 3: Resetting `best_accuracy.txt`
In `fractal_train.py` (lines 151–168):
```python
global_best_path = "best_accuracy.txt"
if val_acc > current_best:
    torch.save(model.state_dict(), "best_model.pth")
    with open(global_best_path, "w") as f:
        f.write(str(val_acc))
```
If your previous 5-layer model achieved an accuracy of e.g. `0.65`, and your new 3-layer model achieves `0.58`, `best_model.pth` **will NOT be saved** unless you delete `best_accuracy.txt` or reset its value to `0.0`.

---

### ⚠️ Precaution 4: Fractal Activation Stability in Deeper Networks
Fractal activations (`FractalActivationN`: `f_relu`, `f_squared_relu`, `f_cubic_relu`) use custom RB fractal operators.
- Higher powers like $x^2$ (`f_squared_relu`) and $x^3$ (`f_cubic_relu`) magnify activations rapidly.
- As the network gets deeper (e.g. 5, 6, 7 layers), gradients can explode or vanish exponentially unless normalized.
- **Rule:** Always keep `use_batchnorm = True` when adding layers. BatchNorm stabilizes the input distributions between fractal activation stages.

---

### ⚠️ Precaution 5: GPU Memory (VRAM) and Batch Size
Adding layers increases the number of intermediate activation maps stored for backpropagation:
- If you increase the number of layers or channels (e.g., adding `512` or `1024`), monitor GPU memory.
- If you encounter CUDA Out-Of-Memory (`RuntimeError: CUDA out of memory`), reduce `batch_size` in `fractal_sweep_config.py` / `best_config.py` from `32` to `16`.

---

## 6. Summary Checklist Before Changing Layers

- [ ] Choose valid channel sizes (e.g., `[32, 64, 128]` for 3 layers, `[32, 64, 128, 256]` for 4 layers).
- [ ] Verify total pooling layers do not reduce $192 \times 192$ below $1 \times 1$ (maximum 7 consecutive `MaxPool2d(2)` layers).
- [ ] Keep `use_batchnorm=True` to prevent gradient instability with fractal activations.
- [ ] Update `fractal_sweep_config.py` and `best_config.py`.
- [ ] Delete or reset `best_accuracy.txt` before training so the new architecture can save its weights.
- [ ] Use a distinctive model filename (e.g., `best_model_4layers.pth`) to avoid overwriting or mixing up checkpoints.

