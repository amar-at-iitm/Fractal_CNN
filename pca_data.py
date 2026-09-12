import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import helpers and preparation routines from data_preparation.py
try:
    from data_preparation import (
        prepare_cifar10,
        prepare_cifar100,
        prepare_svhn,
        prepare_tiny_imagenet,
        prepare_inaturalist,
        verify_dataset_splits,
        remove_ds_store,
        download_file,
        unzip_file,
        crop_images,
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import from data_preparation.py: {e}. "
        "Ensure data_preparation.py is in the root directory."
    )


# ==============================================================================
# DEFAULT CONFIGURATION (Can be changed here or overridden via CLI)
# ==============================================================================
# Supported DATASET_NAME options:
#   - "cifar10"
#   - "cifar100"
#   - "svhn"
#   - "tiny-imagenet" (or "tiny_imagenet")
#   - "inaturalist"   (or "inaturalist_12K")
DATASET_NAME = "cifar10"

# Percentage of top features to retain (e.g. 20 for 20%, 10 for 10%).
# If set to None, N_COMPONENTS will be used directly.
TOP_FEATURES_PCT = 20.0

# Number of principal components (used if TOP_FEATURES_PCT is None):
N_COMPONENTS = None

# PCA Mode:
#   - "global" : Dataset-wide PCA on (C*H*W) flattened images -> extracts top x% features [Default]
#   - "channel": Per-image channel PCA: projects (C, HW) -> 1D sequence of length H*W
PCA_MODE = "global"

# Split ratio and random seed (used if data preparation is triggered)
VAL_RATIO = 0.2
SEED = 42

# Overwrite existing _pcs output directory if True
OVERWRITE = False

# Batch size for loading and PCA fitting
BATCH_SIZE = 64


# ==============================================================================
# Directory Mapping Helpers
# ==============================================================================

def get_dataset_dir_name(name: str) -> str:
    """Map user dataset name to its standard directory name in the project root."""
    name_clean = name.strip().lower().replace("-", "_")
    mapping = {
        "cifar10": "cifar10",
        "cifar_10": "cifar10",
        "cifar100": "cifar100",
        "cifar_100": "cifar100",
        "svhn": "svhn",
        "tiny_imagenet": "tiny_imagenet",
        "tiny_imagenet_200": "tiny_imagenet",
        "inaturalist": "inaturalist_12K",
        "inaturalist_12k": "inaturalist_12K",
    }
    return mapping.get(name_clean, name)


# ==============================================================================
# Dataset Existence, Zip, and Transformation Verification
# ==============================================================================

def check_splits_populated(dataset_dir: Path) -> bool:
    """Check if train, val, and test splits exist and contain image files."""
    for split in ["train", "val", "test"]:
        split_path = dataset_dir / split
        if not split_path.exists() or not split_path.is_dir():
            return False
        has_files = False
        for root, _, files in os.walk(split_path):
            if any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in files):
                has_files = True
                break
        if not has_files:
            return False
    return True


def check_inaturalist_transformed(dataset_dir: Path, target_size=(192, 192)) -> bool:
    """Check if iNaturalist images have been properly cropped to target_size."""
    if not check_splits_populated(dataset_dir):
        return False
    train_dir = dataset_dir / "train"
    image_extensions = (".jpg", ".jpeg", ".png")
    sample_count = 0
    for root, _, files in os.walk(train_dir):
        for f in files:
            if f.lower().endswith(image_extensions):
                try:
                    with Image.open(os.path.join(root, f)) as img:
                        if img.size != target_size:
                            return False
                    sample_count += 1
                    if sample_count >= 10:
                        return True
                except Exception:
                    pass
    return sample_count > 0


def check_and_prepare_dataset(dataset_name: str, root_dir: Path, val_ratio: float = 0.2, seed: int = 42) -> Path:
    """
    Verify if the dataset exists in root directory, check zip files and transform status,
    and trigger data_preparation.py routines if missing or incomplete.
    """
    dir_name = get_dataset_dir_name(dataset_name)
    dataset_path = root_dir / dir_name

    print("\n" + "=" * 70)
    print(f"Checking dataset status: '{dataset_name}' -> Directory: {dataset_path}")
    print("=" * 70)

    # CIFAR-10
    if dir_name == "cifar10":
        if check_splits_populated(dataset_path):
            print(f"✓ CIFAR-10 is already prepared and verified in {dataset_path}.")
        else:
            print(f"! CIFAR-10 splits not found or incomplete in {dataset_path}.")
            print("  Invoking prepare_cifar10 from data_preparation.py...")
            prepare_cifar10(root_dir=str(dataset_path), val_ratio=val_ratio, seed=seed)

    # CIFAR-100
    elif dir_name == "cifar100":
        if check_splits_populated(dataset_path):
            print(f"✓ CIFAR-100 is already prepared and verified in {dataset_path}.")
        else:
            print(f"! CIFAR-100 splits not found or incomplete in {dataset_path}.")
            print("  Invoking prepare_cifar100 from data_preparation.py...")
            prepare_cifar100(root_dir=str(dataset_path), val_ratio=val_ratio, seed=seed)

    # SVHN
    elif dir_name == "svhn":
        if check_splits_populated(dataset_path):
            print(f"✓ SVHN is already prepared and verified in {dataset_path}.")
        else:
            print(f"! SVHN splits not found or incomplete in {dataset_path}.")
            print("  Invoking prepare_svhn from data_preparation.py...")
            prepare_svhn(root_dir=str(dataset_path), val_ratio=val_ratio, seed=seed)

    # Tiny ImageNet
    elif dir_name == "tiny_imagenet":
        zip_in_root = root_dir / "tiny-imagenet-200.zip"
        zip_in_dir = dataset_path / "tiny-imagenet-200.zip"
        if zip_in_root.exists() and not zip_in_dir.exists():
            os.makedirs(dataset_path, exist_ok=True)
            zip_in_root.rename(zip_in_dir)

        if check_splits_populated(dataset_path):
            print(f"✓ Tiny ImageNet splits are already prepared in {dataset_path}.")
        else:
            if zip_in_dir.exists():
                print(f"✓ Found existing zip file: {zip_in_dir}")
            else:
                print(f"! Zip file not found. Download will be performed.")
            print("  Invoking prepare_tiny_imagenet from data_preparation.py...")
            prepare_tiny_imagenet(root_dir=str(dataset_path), val_ratio=val_ratio, seed=seed)

    # iNaturalist-12K
    elif dir_name == "inaturalist_12K":
        zip_in_root = root_dir / "nature_12K.zip"
        zip_in_dir = dataset_path / "nature_12K.zip"
        if zip_in_root.exists():
            print(f"✓ Found nature_12K.zip in root directory: {zip_in_root}")
        elif zip_in_dir.exists():
            print(f"✓ Found nature_12K.zip in {zip_in_dir}")

        if check_splits_populated(dataset_path) and check_inaturalist_transformed(dataset_path):
            print(f"✓ iNaturalist-12K is already prepared and cropped (192x192) in {dataset_path}.")
        else:
            if check_splits_populated(dataset_path) and not check_inaturalist_transformed(dataset_path):
                print("! iNaturalist splits found, but images are not cropped to 192x192.")
                print("  Applying center crop (192, 192)...")
                crop_images(str(dataset_path), target_size=(192, 192))
            else:
                print(f"! iNaturalist-12K not fully prepared in {dataset_path}.")
                print("  Invoking prepare_inaturalist from data_preparation.py...")
                prepare_inaturalist(root_dir=str(dataset_path), val_ratio=val_ratio, seed=seed)

    # Custom / Unknown Dataset
    else:
        if check_splits_populated(dataset_path):
            print(f"✓ Custom dataset '{dataset_name}' found and populated in {dataset_path}.")
        else:
            raise FileNotFoundError(
                f"Dataset directory '{dataset_path}' does not contain populated 'train/', 'val/', and 'test/' splits."
            )

    remove_ds_store(str(dataset_path))
    verify_dataset_splits(dataset_name, str(dataset_path))
    return dataset_path


# ==============================================================================
# PCA Algorithms
# ==============================================================================

def pca_channel_per_image(img_array: np.ndarray, n_components: int = 1) -> np.ndarray:
    """
    Per-Image Channel PCA:
    Converts a single 3D image array of shape (C, H, W) to 1D using PCA across channels.
    """
    if img_array.ndim == 2:
        img_array = img_array[np.newaxis, ...]

    C, H, W = img_array.shape
    HW = H * W

    if C == 1:
        return img_array.reshape(-1).astype(np.float32)

    X = img_array.reshape(C, HW).T.astype(np.float32)
    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean

    cov = np.dot(X_centered.T, X_centered) / max(HW - 1, 1)
    U, S, Vt = np.linalg.svd(cov)
    components = Vt[:n_components, :].T

    X_pca = np.dot(X_centered, components)
    out_1d = X_pca.reshape(-1)
    return out_1d.astype(np.float32)


class GlobalPCA:
    """
    Dataset-wide PCA on flattened images (C * H * W).
    Projects images onto the top n_components (top x% features) explaining the maximum variance.
    """
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, dataloader: DataLoader, desc: str = "Fitting Global PCA"):
        print(f"Fitting Global PCA (n_components={self.n_components})...")
        total_samples = 0
        mean_accum = None

        # Pass 1: Compute dataset mean vector
        for images, _ in tqdm(dataloader, desc=f"{desc} (Mean)"):
            B = images.size(0)
            X = images.view(B, -1).numpy().astype(np.float64)
            if mean_accum is None:
                mean_accum = np.sum(X, axis=0)
            else:
                mean_accum += np.sum(X, axis=0)
            total_samples += B

        self.mean_ = (mean_accum / total_samples).astype(np.float32)
        D = len(self.mean_)
        self.n_components = min(self.n_components, D, total_samples)

        # Pass 2: IncrementalPCA or Batch Covariance SVD
        try:
            from sklearn.decomposition import IncrementalPCA
            ipca = IncrementalPCA(n_components=self.n_components, batch_size=dataloader.batch_size)
            for images, _ in tqdm(dataloader, desc=f"{desc} (IncrementalPCA)"):
                B = images.size(0)
                X = images.view(B, -1).numpy().astype(np.float32)
                ipca.partial_fit(X)
            self.components_ = ipca.components_.astype(np.float32)
            self.explained_variance_ = ipca.explained_variance_.astype(np.float32)
            self.explained_variance_ratio_ = ipca.explained_variance_ratio_.astype(np.float32)
        except ImportError:
            print("  sklearn not found. Using PyTorch exact covariance SVD...")
            cov = torch.zeros((D, D), dtype=torch.float64)
            mean_tensor = torch.from_numpy(self.mean_).double()

            for images, _ in tqdm(dataloader, desc=f"{desc} (Covariance)"):
                B = images.size(0)
                X = images.view(B, -1).double() - mean_tensor
                cov += torch.mm(X.t(), X)

            cov /= max(total_samples - 1, 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            
            # Sort descending
            eigenvalues = eigenvalues.flip(dims=[0])
            eigenvectors = eigenvectors.flip(dims=[1])

            top_evals = eigenvalues[:self.n_components].float().numpy()
            top_evecs = eigenvectors[:, :self.n_components].t().float().numpy()

            self.components_ = top_evecs
            self.explained_variance_ = top_evals
            total_var = float(eigenvalues.sum())
            self.explained_variance_ratio_ = (top_evals / max(total_var, 1e-9)).astype(np.float32)

        cum_var = float(np.sum(self.explained_variance_ratio_)) * 100.0
        print(f"✓ Global PCA fitted: {self.n_components} components explain {cum_var:.2f}% of total dataset variance.")

    def transform(self, img_flat: np.ndarray) -> np.ndarray:
        """Transform a single flattened image (D,) to 1D vector (n_components,)."""
        X_centered = img_flat.astype(np.float32) - self.mean_
        out_1d = np.dot(self.components_, X_centered)
        return out_1d.astype(np.float32)


# ==============================================================================
# Main Processing & Dataset Conversion Function
# ==============================================================================

def process_and_save_pca_dataset(
    dataset_name: str = DATASET_NAME,
    root_dir: Path = PROJECT_ROOT,
    top_features_pct: float = TOP_FEATURES_PCT,
    n_components: int = N_COMPONENTS,
    pca_mode: str = PCA_MODE,
    output_dir_name: str = None,
    overwrite: bool = OVERWRITE,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
    batch_size: int = BATCH_SIZE
) -> Path:
    """
    Main function to:
      1. Check and prepare the source dataset using data_preparation.py if needed.
      2. Calculate the top x% of features (n_components).
      3. Create destination folder: '<dataset_dir>_pcs' in root directory.
      4. Convert each 3D (C, H, W) image to 1D with top x% features using PCA.
      5. Save per-sample .npy files, consolidated .npy arrays, and metadata.json.
    """
    dataset_dir = check_and_prepare_dataset(
        dataset_name=dataset_name,
        root_dir=root_dir,
        val_ratio=val_ratio,
        seed=seed
    )

    # Determine destination folder
    if output_dir_name is not None and output_dir_name.strip():
        output_dir = root_dir / output_dir_name.strip()
    else:
        output_dir = root_dir / f"{dataset_dir.name}_pcs"

    # Load splits using ImageFolder
    raw_splits = {}
    for split in ["train", "val", "test"]:
        split_path = dataset_dir / split
        if split_path.exists():
            raw_splits[split] = datasets.ImageFolder(str(split_path))
        else:
            raise FileNotFoundError(f"Missing required split directory: {split_path}")

    class_names = raw_splits["train"].classes
    class_to_idx = raw_splits["train"].class_to_idx

    # Inspect sample image dimensions (C, H, W)
    sample_img, _ = raw_splits["train"][0]
    sample_np = np.array(sample_img.convert("RGB")).transpose(2, 0, 1)  # (C, H, W)
    C, H, W = sample_np.shape
    total_original_features = C * H * W
    print(f"Original image shape: ({C}, {H}, {W}) -> Total Raw Features: {total_original_features}")

    # Calculate number of components based on top_features_pct or n_components
    if top_features_pct is not None:
        pct_val = top_features_pct / 100.0 if top_features_pct > 1.0 else top_features_pct
        pct_display = pct_val * 100.0
        if pca_mode == "global":
            calculated_k = max(1, min(total_original_features, int(round(total_original_features * pct_val))))
        else:
            calculated_k = max(1, min(H * W, int(round((H * W) * pct_val))))
        n_components = calculated_k
        print(f"[PCA CONFIG] Retaining top {pct_display:.1f}% of features -> {n_components} components (out of {total_original_features})")
    elif n_components is not None:
        n_components = max(1, min(total_original_features, n_components))
        pct_display = (n_components / total_original_features) * 100.0
        print(f"[PCA CONFIG] Using exact n_components = {n_components} ({pct_display:.1f}% of raw features)")
    else:
        # Default: 20%
        pct_display = 20.0
        n_components = max(1, int(round(total_original_features * 0.20)))
        print(f"[PCA CONFIG] Defaulting to top 20% of features -> {n_components} components")

    print("\n" + "=" * 70)
    print(f"Target Output Directory: {output_dir}")
    print(f"PCA Mode: '{pca_mode}' | Top Features: {pct_display:.1f}% ({n_components} components)")
    print("=" * 70)

    # Check if already computed
    if output_dir.exists() and not overwrite:
        metadata_file = output_dir / "metadata.json"
        if metadata_file.exists() and check_splits_populated(output_dir):
            try:
                with open(metadata_file, "r") as f:
                    meta = json.load(f)
                if meta.get("n_components") == n_components and meta.get("pca_mode") == pca_mode:
                    print(f"✓ PCA dataset already exists in {output_dir} with matching {n_components} components. Skipping.")
                    print("  (Pass --overwrite to force regeneration)")
                    return output_dir
            except Exception:
                pass

    os.makedirs(output_dir, exist_ok=True)

    # Fit PCA model
    global_pca = None
    if pca_mode == "global" or (pca_mode == "channel" and n_components < H * W):
        print(f"\n[PCA FITTING] Fitting PCA across training split to retain top {n_components} principal features...")
        
        # Generator dataset that yields appropriate representation for PCA fitting
        class PCAFitDataset(Dataset):
            def __init__(self, raw_ds, mode):
                self.raw_ds = raw_ds
                self.mode = mode
            def __len__(self):
                return len(self.raw_ds)
            def __getitem__(self, idx):
                img, _ = self.raw_ds[idx]
                arr = np.array(img.convert("RGB")).transpose(2, 0, 1).astype(np.float32) / 255.0
                if self.mode == "global":
                    return torch.from_numpy(arr.reshape(-1))
                else:
                    arr_chan = pca_channel_per_image(arr, n_components=1)
                    return torch.from_numpy(arr_chan)

        fit_dataset = PCAFitDataset(raw_splits["train"], mode=pca_mode)
        fit_loader = DataLoader(fit_dataset, batch_size=batch_size, shuffle=False)

        global_pca = GlobalPCA(n_components=n_components)
        global_pca.fit(fit_loader)
        with open(output_dir / "pca_model.pkl", "wb") as f:
            pickle.dump(global_pca, f)
        out_dim_1d = n_components
    else:
        out_dim_1d = H * W

    print(f"Projected 1D representation length: {out_dim_1d}")

    # Process and export splits
    split_counts = {}
    for split, dataset_obj in raw_splits.items():
        print(f"\nProcessing '{split}' split ({len(dataset_obj)} images)...")
        dest_split_dir = output_dir / split

        for cls in class_names:
            os.makedirs(dest_split_dir / cls, exist_ok=True)

        all_1d_data = []
        all_labels = []

        for idx in tqdm(range(len(dataset_obj)), desc=f"Exporting {split} to 1D PCA"):
            img_path, label = dataset_obj.samples[idx]
            cls_name = class_names[label]
            file_stem = Path(img_path).stem

            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                img_np = np.array(img_rgb).transpose(2, 0, 1).astype(np.float32) / 255.0

            if pca_mode == "global":
                arr_flat = img_np.reshape(-1)
                arr_1d = global_pca.transform(arr_flat)
            elif global_pca is not None:
                # Channel mode with top x% features: channel projection followed by top principal components
                arr_chan = pca_channel_per_image(img_np, n_components=1)
                arr_1d = global_pca.transform(arr_chan)
            else:
                # Channel mode with 100% features: full 1D channel-projected sequence
                arr_1d = pca_channel_per_image(img_np, n_components=1)

            dest_file = dest_split_dir / cls_name / f"{file_stem}.npy"
            np.save(dest_file, arr_1d)

            all_1d_data.append(arr_1d)
            all_labels.append(label)

        split_data_arr = np.stack(all_1d_data, axis=0).astype(np.float32)
        split_labels_arr = np.array(all_labels, dtype=np.int64)

        np.save(output_dir / f"{split}_data.npy", split_data_arr)
        np.save(output_dir / f"{split}_labels.npy", split_labels_arr)
        split_counts[split] = len(dataset_obj)

    cum_var = float(np.sum(global_pca.explained_variance_ratio_)) if global_pca else 1.0

    metadata = {
        "dataset_name": dataset_name,
        "original_shape": [C, H, W],
        "total_original_features": total_original_features,
        "top_features_pct": pct_display,
        "pca_mode": pca_mode,
        "n_components": n_components,
        "output_1d_shape": [out_dim_1d],
        "explained_variance_ratio_sum": round(cum_var, 4),
        "num_classes": len(class_names),
        "classes": class_names,
        "class_to_idx": class_to_idx,
        "split_counts": split_counts,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    remove_ds_store(str(output_dir))

    print("\n" + "=" * 70)
    print(f"✓ Successfully generated 1D PCA dataset at: {output_dir}")
    print(f"  • Original 3D shape: ({C}, {H}, {W}) -> {total_original_features} raw features")
    print(f"  • Top features kept: {pct_display:.1f}% ({n_components} components)")
    if global_pca:
        print(f"  • Total variance explained: {cum_var * 100:.2f}%")
    print(f"  • Converted 1D shape: ({out_dim_1d},)")
    print(f"  • Split counts: {split_counts}")
    print(f"  • Class subdirectories: {len(class_names)} classes in train/, val/, test/")
    print(f"  • Consolidated arrays: {split}_data.npy & {split}_labels.npy")
    print(f"  • Metadata saved to: {output_dir / 'metadata.json'}")
    print("=" * 70 + "\n")
    return output_dir


# ==============================================================================
# PyTorch Dataset Loader Class for 1D PCA Data
# ==============================================================================

class PCADataset(Dataset):
    """
    PyTorch Dataset for loading 1D PCA transformed data from the _pcs directory.
    Supports fast in-memory loading from consolidated .npy arrays or on-demand loading.
    
    Usage:
        from pca_data import PCADataset
        train_ds = PCADataset("cifar10_pcs", split="train")
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    """
    def __init__(self, data_root: str, split: str = "train", load_in_memory: bool = True):
        self.data_root = Path(data_root)
        self.split = split
        self.load_in_memory = load_in_memory

        meta_path = self.data_root / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
                self.classes = self.metadata.get("classes", [])
        else:
            self.metadata = {}
            self.classes = []

        data_file = self.data_root / f"{split}_data.npy"
        labels_file = self.data_root / f"{split}_labels.npy"

        if self.load_in_memory and data_file.exists() and labels_file.exists():
            self.data = np.load(data_file)
            self.labels = np.load(labels_file)
            self.use_consolidated = True
        else:
            self.use_consolidated = False
            split_dir = self.data_root / split
            self.samples = []
            if split_dir.exists():
                for cls_idx, cls_name in enumerate(sorted(os.listdir(split_dir))):
                    cls_dir = split_dir / cls_name
                    if cls_dir.is_dir():
                        for f in os.listdir(cls_dir):
                            if f.endswith(".npy"):
                                self.samples.append((cls_dir / f, cls_idx))

    def __len__(self):
        if self.use_consolidated:
            return len(self.data)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_consolidated:
            x = torch.from_numpy(self.data[idx]).float()
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y
        file_path, label = self.samples[idx]
        arr = np.load(file_path)
        return torch.from_numpy(arr).float(), torch.tensor(label, dtype=torch.long)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert 3D image datasets to 1D via PCA with top x% features."
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=DATASET_NAME,
        help=f"Dataset name: 'cifar10', 'cifar100', 'svhn', 'tiny-imagenet', 'inaturalist' (default: {DATASET_NAME})"
    )
    parser.add_argument(
        "--top_features_pct", "--top_features", "-p", "--pct",
        type=float,
        default=TOP_FEATURES_PCT,
        help="Percentage of top features to retain (e.g. 20 for 20%%, 10 for 10%%, 0.2 for 20%%) (default: 20.0)"
    )
    parser.add_argument(
        "--n_components", "-k",
        type=int,
        default=N_COMPONENTS,
        help="Exact number of principal components to retain (overrides top_features_pct if set)"
    )
    parser.add_argument(
        "--pca_mode", "-m",
        type=str,
        default=PCA_MODE,
        choices=["global", "channel"],
        help=f"PCA mode: 'global' (dataset-wide PCA to top x%% features) or 'channel' (per-image channel PCA) (default: {PCA_MODE})"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(PROJECT_ROOT),
        help="Base directory containing datasets (default: project root)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=None,
        help="Custom output directory name (defaults to <dataset>_pcs in root directory)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=VAL_RATIO,
        help="Validation split ratio if data preparation is triggered (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for IncrementalPCA and data loading (default: 64)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=OVERWRITE,
        help="Overwrite existing output directory if it already exists"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_and_save_pca_dataset(
        dataset_name=args.dataset,
        root_dir=Path(args.data_dir),
        top_features_pct=args.top_features_pct,
        n_components=args.n_components,
        pca_mode=args.pca_mode,
        output_dir_name=args.output_dir,
        overwrite=args.overwrite,
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_size=args.batch_size
    )
