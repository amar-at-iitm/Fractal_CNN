import os
import sys
import json
import pickle
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
# DATASET CONFIGURATION (Change dataset name and parameters here)
# ==============================================================================
# Supported DATASET_NAME options:
#   - "cifar10"
#   - "cifar100"
#   - "svhn"
#   - "tiny-imagenet" (or "tiny_imagenet")
#   - "inaturalist"   (or "inaturalist_12K")
DATASET_NAME = "cifar10"

# PCA Mode:
#   - "channel": Per-image PCA: projects (C, H, W) -> 1D sequence of length H*W (or H*W * n_components)
#   - "global" : Dataset-wide PCA: flattens (C*H*W) -> 1D feature vector of length n_components
PCA_MODE = "channel"

# Number of principal components:
#   - For "channel" mode: 1 yields a 1D vector of length H*W per image
#   - For "global" mode : e.g. 64 or 128 yields a 1D vector of length n_components
N_COMPONENTS = 1

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
        # Check if split contains at least one class directory with files
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
    # Sample a few images from train split to inspect resolution
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

    # --------------------------------------------------------------------------
    # 1. CIFAR-10
    # --------------------------------------------------------------------------
    if dir_name == "cifar10":
        if check_splits_populated(dataset_path):
            print(f"✓ CIFAR-10 is already prepared and verified in {dataset_path}.")
        else:
            print(f"! CIFAR-10 splits not found or incomplete in {dataset_path}.")
            print("  Invoking prepare_cifar10 from data_preparation.py...")
            prepare_cifar10(root_dir=str(dataset_path), val_ratio=val_ratio, seed=seed)

    # --------------------------------------------------------------------------
    # 2. CIFAR-100
    # --------------------------------------------------------------------------
    elif dir_name == "cifar100":
        if check_splits_populated(dataset_path):
            print(f"✓ CIFAR-100 is already prepared and verified in {dataset_path}.")
        else:
            print(f"! CIFAR-100 splits not found or incomplete in {dataset_path}.")
            print("  Invoking prepare_cifar100 from data_preparation.py...")
            prepare_cifar100(root_dir=str(dataset_path), val_ratio=val_ratio, seed=seed)

    # --------------------------------------------------------------------------
    # 3. SVHN
    # --------------------------------------------------------------------------
    elif dir_name == "svhn":
        if check_splits_populated(dataset_path):
            print(f"✓ SVHN is already prepared and verified in {dataset_path}.")
        else:
            print(f"! SVHN splits not found or incomplete in {dataset_path}.")
            print("  Invoking prepare_svhn from data_preparation.py...")
            prepare_svhn(root_dir=str(dataset_path), val_ratio=val_ratio, seed=seed)

    # --------------------------------------------------------------------------
    # 4. Tiny ImageNet
    # --------------------------------------------------------------------------
    elif dir_name == "tiny_imagenet":
        zip_in_root = root_dir / "tiny-imagenet-200.zip"
        zip_in_dir = dataset_path / "tiny-imagenet-200.zip"
        
        # Move zip from project root to dataset_path if present in root
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

    # --------------------------------------------------------------------------
    # 5. iNaturalist-12K
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # Custom / Unknown Dataset
    # --------------------------------------------------------------------------
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
    Converts a single 3D image array of shape (C, H, W) to a 1D sequence using PCA.
    
    Steps:
      1. Reshape (C, H, W) to (H*W, C).
      2. Compute mean of each channel and center the pixels: X_c = X - mean.
      3. Compute covariance matrix (C x C) and its SVD/eigendecomposition.
      4. Project the pixels onto the top principal component(s): (H*W, C) @ (C, n_components) -> (H*W, n_components).
      5. Flatten to 1D representation of length (H*W * n_components) [default: H*W].
    """
    if img_array.ndim == 2:  # Grayscale (H, W) -> treat as (1, H, W)
        img_array = img_array[np.newaxis, ...]

    C, H, W = img_array.shape
    HW = H * W

    # If already single channel, reshape directly to 1D
    if C == 1:
        return img_array.reshape(-1).astype(np.float32)

    # Reshape to (HW, C)
    X = img_array.reshape(C, HW).T.astype(np.float32)  # (HW, C)

    # Center channels
    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean

    # C x C Covariance matrix (typically 3 x 3 for RGB)
    cov = np.dot(X_centered.T, X_centered) / max(HW - 1, 1)

    # SVD on 3x3 covariance matrix
    U, S, Vt = np.linalg.svd(cov)
    # Principal components (eigenvectors): shape (C, n_components)
    components = Vt[:n_components, :].T

    # Project to 1D representation
    X_pca = np.dot(X_centered, components)  # shape (HW, n_components)

    # Flatten to 1D array
    out_1d = X_pca.reshape(-1)
    return out_1d.astype(np.float32)


class GlobalPCA:
    """
    Dataset-wide PCA using Incremental SVD / PCA on flattened images (C * H * W).
    Flattens each image from 3D (C, H, W) to a vector of length D = C*H*W,
    fits PCA across training images, and transforms each image to 1D feature vector of length n_components.
    """
    def __init__(self, n_components: int = 64):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None

    def fit(self, dataloader: DataLoader, desc: str = "Fitting Global PCA"):
        """Fit PCA using PyTorch SVD on mini-batches or accumulated covariance."""
        print(f"Fitting Global PCA (n_components={self.n_components})...")
        total_samples = 0
        mean_accum = None

        # First pass: compute global mean
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

        # Second pass: approximate covariance matrix via Gram matrix or Incremental SVD
        try:
            from sklearn.decomposition import IncrementalPCA
            ipca = IncrementalPCA(n_components=min(self.n_components, D), batch_size=dataloader.batch_size)
            for images, _ in tqdm(dataloader, desc=f"{desc} (IncrementalPCA)"):
                B = images.size(0)
                X = images.view(B, -1).numpy().astype(np.float32)
                ipca.partial_fit(X)
            self.components_ = ipca.components_.astype(np.float32)
            self.explained_variance_ = ipca.explained_variance_ratio_.astype(np.float32)
        except ImportError:
            # Fallback: PyTorch low-rank PCA on sample batch
            print("  sklearn not found. Using PyTorch SVD fallback...")
            sample_tensors = []
            max_sample_count = 5000
            collected = 0
            for images, _ in dataloader:
                B = images.size(0)
                sample_tensors.append(images.view(B, -1))
                collected += B
                if collected >= max_sample_count:
                    break
            X_all = torch.cat(sample_tensors, dim=0)[:max_sample_count].float()
            X_centered = X_all - torch.from_numpy(self.mean_)
            _, _, V = torch.pca_lowrank(X_centered, q=self.n_components)
            self.components_ = V.t().numpy().astype(np.float32)
            self.explained_variance_ = np.ones(self.n_components, dtype=np.float32) / self.n_components

        print("✓ Global PCA fitting complete.")

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
    pca_mode: str = PCA_MODE,
    n_components: int = N_COMPONENTS,
    overwrite: bool = OVERWRITE,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
    batch_size: int = BATCH_SIZE
) -> Path:
    """
    Main function to:
      1. Check and prepare the source dataset using data_preparation.py if needed.
      2. Create the destination folder: '<dataset_dir>_pcs' in the root directory.
      3. Convert each 3D (C, H, W) image to 1D using PCA.
      4. Save per-sample .npy in train/val/test class directories, consolidated .npy arrays,
         and metadata.json.
    """
    # Step 1: Ensure dataset exists and splits are populated
    dataset_dir = check_and_prepare_dataset(
        dataset_name=dataset_name,
        root_dir=root_dir,
        val_ratio=val_ratio,
        seed=seed
    )

    # Output directory in the root directory with '_pcs' suffix
    out_dir_name = f"{dataset_dir.name}_pcs"
    output_dir = root_dir / out_dir_name

    print("\n" + "=" * 70)
    print(f"Target Output Directory: {output_dir}")
    print(f"PCA Mode: '{pca_mode}' | Components: {n_components}")
    print("=" * 70)

    # Check if output already exists
    if output_dir.exists() and not overwrite:
        metadata_file = output_dir / "metadata.json"
        if metadata_file.exists() and check_splits_populated(output_dir):
            print(f"✓ PCA dataset already exists in {output_dir}. Skipping computation.")
            print("  (Set OVERWRITE = True to regenerate)")
            return output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Load splits using ImageFolder (to obtain classes, image paths, and labels)
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
    print(f"Original image shape: ({C}, {H}, {W}) [Channels, Height, Width]")

    # If global PCA mode is selected, fit global PCA on training split
    global_pca = None
    if pca_mode == "global":
        train_loader = DataLoader(
            raw_splits["train"],
            batch_size=batch_size,
            shuffle=False,
            transform=transforms.ToTensor()
        )
        global_pca = GlobalPCA(n_components=n_components)
        global_pca.fit(train_loader)
        # Save fitted PCA model
        with open(output_dir / "pca_model.pkl", "wb") as f:
            pickle.dump(global_pca, f)

    # Output 1D dimension
    if pca_mode == "channel":
        out_dim_1d = H * W * n_components
    else:
        out_dim_1d = n_components
    print(f"Projected 1D representation length: {out_dim_1d}")

    # Process and export splits
    split_counts = {}
    for split, dataset_obj in raw_splits.items():
        print(f"\nProcessing '{split}' split ({len(dataset_obj)} images)...")
        dest_split_dir = output_dir / split

        # Create class folders in destination
        for cls in class_names:
            os.makedirs(dest_split_dir / cls, exist_ok=True)

        all_1d_data = []
        all_labels = []

        # Iterate over all samples in dataset
        for idx in tqdm(range(len(dataset_obj)), desc=f"Exporting {split} to 1D PCA"):
            img_path, label = dataset_obj.samples[idx]
            cls_name = class_names[label]
            file_stem = Path(img_path).stem

            # Load PIL image and convert to RGB (C, H, W)
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                img_np = np.array(img_rgb).transpose(2, 0, 1).astype(np.float32) / 255.0

            # Convert to 1D using selected PCA mode
            if pca_mode == "channel":
                arr_1d = pca_channel_per_image(img_np, n_components=n_components)
            else:
                arr_flat = img_np.reshape(-1)
                arr_1d = global_pca.transform(arr_flat)

            # Save individual .npy file
            dest_file = dest_split_dir / cls_name / f"{file_stem}.npy"
            np.save(dest_file, arr_1d)

            all_1d_data.append(arr_1d)
            all_labels.append(label)

        # Save consolidated numpy arrays for this split
        split_data_arr = np.stack(all_1d_data, axis=0).astype(np.float32)
        split_labels_arr = np.array(all_labels, dtype=np.int64)

        np.save(output_dir / f"{split}_data.npy", split_data_arr)
        np.save(output_dir / f"{split}_labels.npy", split_labels_arr)
        split_counts[split] = len(dataset_obj)

    # Save metadata JSON
    metadata = {
        "dataset_name": dataset_name,
        "original_shape": [C, H, W],
        "pca_mode": pca_mode,
        "n_components": n_components,
        "output_1d_shape": [out_dim_1d],
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
    print(f"  • Original 3D shape: ({C}, {H}, {W})")
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
    Supports either fast in-memory loading from consolidated .npy arrays or on-demand loading.
    
    Usage:
        from pca_data import PCADataset
        train_ds = PCADataset("cifar10_pcs", split="train")
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    """
    def __init__(self, data_root: str, split: str = "train", load_in_memory: bool = True):
        self.data_root = Path(data_root)
        self.split = split
        self.load_in_memory = load_in_memory

        # Load metadata if present
        meta_path = self.data_root / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
                self.classes = self.metadata.get("classes", [])
        else:
            self.metadata = {}
            self.classes = []

        # Check for consolidated array
        data_file = self.data_root / f"{split}_data.npy"
        labels_file = self.data_root / f"{split}_labels.npy"

        if self.load_in_memory and data_file.exists() and labels_file.exists():
            self.data = np.load(data_file)
            self.labels = np.load(labels_file)
            self.use_consolidated = True
        else:
            self.use_consolidated = False
            # Find all .npy files in split directory
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
# Script Execution Entrypoint
# ==============================================================================

if __name__ == "__main__":
    process_and_save_pca_dataset(
        dataset_name=DATASET_NAME,
        root_dir=PROJECT_ROOT,
        pca_mode=PCA_MODE,
        n_components=N_COMPONENTS,
        overwrite=OVERWRITE,
        val_ratio=VAL_RATIO,
        seed=SEED,
        batch_size=BATCH_SIZE
    )

