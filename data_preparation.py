import os
import shutil
import zipfile
import random
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np

import torch
import torchvision
from torchvision import datasets


# ==============================================================================
# General Download & Utility Helpers
# ==============================================================================

def download_file(url, dest_path):
    """Download a file from a URL with a visual progress bar."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(dest_path)}",
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024 * 64):
            file.write(data)
            bar.update(len(data))


def unzip_file(zip_path, extract_to):
    """Extract all contents from a zip file."""
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction complete for {zip_path}.")


def remove_ds_store(data_dir):
    """Clean up macOS .DS_Store files recursively."""
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file == '.DS_Store':
                try:
                    os.remove(os.path.join(root, file))
                except Exception as e:
                    pass


def verify_dataset_splits(dataset_name, dataset_dir):
    """Verify and display counts of train, val, and test splits using ImageFolder."""
    print(f"\n--- Verifying {dataset_name} ImageFolder Splits ---")
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_dir, split)
        if os.path.exists(split_path):
            try:
                ds = datasets.ImageFolder(split_path)
                print(f"  • {split:5s} split: {len(ds):>6d} images across {len(ds.classes):>3d} classes")
            except Exception as e:
                print(f"  • {split:5s} split: Directory found, but failed to load via ImageFolder: {e}")
        else:
            print(f"  • {split:5s} split: MISSING ({split_path})")
    print("-" * 50)


# ==============================================================================
# 1. CIFAR-10: Export to train / val / test ImageFolder format
# ==============================================================================

def prepare_cifar10(root_dir="./data/cifar10", val_ratio=0.2, seed=42):
    """Download CIFAR-10 and export to standard ImageFolder directory structure:

        root_dir/
            ├── train/ (<class_name>/<idx>.png)  [40,000 images]
            ├── val/   (<class_name>/<idx>.png)  [10,000 images]
            └── test/  (<class_name>/<idx>.png)  [10,000 images]
    """
    print("\n" + "=" * 65)
    print("Preparing CIFAR-10 with train/val/test splits")
    print("=" * 65)
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    test_dir = os.path.join(root_dir, "test")

    if (os.path.exists(train_dir) and any(os.scandir(train_dir)) and
        os.path.exists(val_dir) and any(os.scandir(val_dir)) and
        os.path.exists(test_dir) and any(os.scandir(test_dir))):
        print(f"CIFAR-10 already prepared in {root_dir}. Skipping export.")
        verify_dataset_splits("CIFAR-10", root_dir)
        return

    # Download raw batches
    cache_dir = os.path.join(root_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    raw_train = datasets.CIFAR10(root=cache_dir, train=True, download=True)
    raw_test = datasets.CIFAR10(root=cache_dir, train=False, download=True)
    class_names = raw_train.classes

    # Create class directories
    for cls in class_names:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    # Stratified 80/20 split on raw_train
    random.seed(seed)
    class_indices = {i: [] for i in range(len(class_names))}
    for idx, (_, label) in enumerate(raw_train):
        class_indices[label].append(idx)

    train_indices = set()
    val_indices = set()
    for label, indices in class_indices.items():
        random.shuffle(indices)
        n_val = int(len(indices) * val_ratio)
        val_indices.update(indices[:n_val])
        train_indices.update(indices[n_val:])

    print(f"Exporting CIFAR-10 train ({len(train_indices)}) & val ({len(val_indices)}) images...")
    for idx in tqdm(range(len(raw_train)), desc="Exporting train & val"):
        img, label = raw_train[idx]
        cls_name = class_names[label]
        dest_split = val_dir if idx in val_indices else train_dir
        img.save(os.path.join(dest_split, cls_name, f"cifar10_{idx:05d}.png"))

    print(f"Exporting CIFAR-10 test ({len(raw_test)}) images...")
    for idx in tqdm(range(len(raw_test)), desc="Exporting test"):
        img, label = raw_test[idx]
        cls_name = class_names[label]
        img.save(os.path.join(test_dir, cls_name, f"cifar10_test_{idx:05d}.png"))

    # Cleanup cache tar
    remove_ds_store(root_dir)
    verify_dataset_splits("CIFAR-10", root_dir)


# ==============================================================================
# 2. CIFAR-100: Export to train / val / test ImageFolder format
# ==============================================================================

def prepare_cifar100(root_dir="./data/cifar100", val_ratio=0.2, seed=42):
    """Download CIFAR-100 and export to standard ImageFolder directory structure:

        root_dir/
            ├── train/ (<class_name>/<idx>.png)  [40,000 images]
            ├── val/   (<class_name>/<idx>.png)  [10,000 images]
            └── test/  (<class_name>/<idx>.png)  [10,000 images]
    """
    print("\n" + "=" * 65)
    print("Preparing CIFAR-100 with train/val/test splits")
    print("=" * 65)
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    test_dir = os.path.join(root_dir, "test")

    if (os.path.exists(train_dir) and any(os.scandir(train_dir)) and
        os.path.exists(val_dir) and any(os.scandir(val_dir)) and
        os.path.exists(test_dir) and any(os.scandir(test_dir))):
        print(f"CIFAR-100 already prepared in {root_dir}. Skipping export.")
        verify_dataset_splits("CIFAR-100", root_dir)
        return

    cache_dir = os.path.join(root_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    raw_train = datasets.CIFAR100(root=cache_dir, train=True, download=True)
    raw_test = datasets.CIFAR100(root=cache_dir, train=False, download=True)
    class_names = raw_train.classes

    for cls in class_names:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    random.seed(seed)
    class_indices = {i: [] for i in range(len(class_names))}
    for idx, (_, label) in enumerate(raw_train):
        class_indices[label].append(idx)

    train_indices = set()
    val_indices = set()
    for label, indices in class_indices.items():
        random.shuffle(indices)
        n_val = int(len(indices) * val_ratio)
        val_indices.update(indices[:n_val])
        train_indices.update(indices[n_val:])

    print(f"Exporting CIFAR-100 train ({len(train_indices)}) & val ({len(val_indices)}) images...")
    for idx in tqdm(range(len(raw_train)), desc="Exporting train & val"):
        img, label = raw_train[idx]
        cls_name = class_names[label]
        dest_split = val_dir if idx in val_indices else train_dir
        img.save(os.path.join(dest_split, cls_name, f"cifar100_{idx:05d}.png"))

    print(f"Exporting CIFAR-100 test ({len(raw_test)}) images...")
    for idx in tqdm(range(len(raw_test)), desc="Exporting test"):
        img, label = raw_test[idx]
        cls_name = class_names[label]
        img.save(os.path.join(test_dir, cls_name, f"cifar100_test_{idx:05d}.png"))

    remove_ds_store(root_dir)
    verify_dataset_splits("CIFAR-100", root_dir)


# ==============================================================================
# 3. SVHN: Export to train / val / test ImageFolder format
# ==============================================================================

def prepare_svhn(root_dir="./data/svhn", val_ratio=0.2, seed=42):
    """Download SVHN and export to standard ImageFolder directory structure:

        root_dir/
            ├── train/ (<digit>/<idx>.png)  [~58,605 images]
            ├── val/   (<digit>/<idx>.png)  [~14,652 images]
            └── test/  (<digit>/<idx>.png)  [ 26,032 images]
    """
    print("\n" + "=" * 65)
    print("Preparing SVHN with train/val/test splits")
    print("=" * 65)
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    test_dir = os.path.join(root_dir, "test")

    if (os.path.exists(train_dir) and any(os.scandir(train_dir)) and
        os.path.exists(val_dir) and any(os.scandir(val_dir)) and
        os.path.exists(test_dir) and any(os.scandir(test_dir))):
        print(f"SVHN already prepared in {root_dir}. Skipping export.")
        verify_dataset_splits("SVHN", root_dir)
        return

    cache_dir = os.path.join(root_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    raw_train = datasets.SVHN(root=cache_dir, split='train', download=True)
    raw_test = datasets.SVHN(root=cache_dir, split='test', download=True)

    digits = [str(i) for i in range(10)]
    for d in digits:
        os.makedirs(os.path.join(train_dir, d), exist_ok=True)
        os.makedirs(os.path.join(val_dir, d), exist_ok=True)
        os.makedirs(os.path.join(test_dir, d), exist_ok=True)

    random.seed(seed)
    class_indices = {i: [] for i in range(10)}
    for idx in range(len(raw_train)):
        _, label = raw_train[idx]
        class_indices[int(label)].append(idx)

    train_indices = set()
    val_indices = set()
    for label, indices in class_indices.items():
        random.shuffle(indices)
        n_val = int(len(indices) * val_ratio)
        val_indices.update(indices[:n_val])
        train_indices.update(indices[n_val:])

    print(f"Exporting SVHN train ({len(train_indices)}) & val ({len(val_indices)}) images...")
    for idx in tqdm(range(len(raw_train)), desc="Exporting train & val"):
        img, label = raw_train[idx]
        cls_name = str(int(label))
        dest_split = val_dir if idx in val_indices else train_dir
        img.save(os.path.join(dest_split, cls_name, f"svhn_{idx:06d}.png"))

    print(f"Exporting SVHN test ({len(raw_test)}) images...")
    for idx in tqdm(range(len(raw_test)), desc="Exporting test"):
        img, label = raw_test[idx]
        cls_name = str(int(label))
        img.save(os.path.join(test_dir, cls_name, f"svhn_test_{idx:06d}.png"))

    remove_ds_store(root_dir)
    verify_dataset_splits("SVHN", root_dir)


# ==============================================================================
# 4. Tiny ImageNet: Export to train / val / test ImageFolder format
# ==============================================================================

def prepare_tiny_imagenet(root_dir="./data/tiny_imagenet", val_ratio=0.2, seed=42):
    """Download Tiny ImageNet-200 and prepare into standard ImageFolder splits:

        root_dir/
            ├── train/ (<wnid>/<idx>.JPEG)  [80,000 images - 400 per class]
            ├── val/   (<wnid>/<idx>.JPEG)  [20,000 images - 100 per class]
            └── test/  (<wnid>/<idx>.JPEG)  [10,000 images - 50 per class, from annotated val]
    """
    print("\n" + "=" * 65)
    print("Preparing Tiny ImageNet-200 with train/val/test splits")
    print("=" * 65)
    os.makedirs(root_dir, exist_ok=True)
    dataset_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(root_dir, "tiny-imagenet-200.zip")
    extracted_path = os.path.join(root_dir, "tiny-imagenet-200")

    final_train = os.path.join(root_dir, "train")
    final_val = os.path.join(root_dir, "val")
    final_test = os.path.join(root_dir, "test")

    if (os.path.exists(final_train) and any(os.scandir(final_train)) and
        os.path.exists(final_val) and any(os.scandir(final_val)) and
        os.path.exists(final_test) and any(os.scandir(final_test))):
        print(f"Tiny ImageNet already prepared in {root_dir}. Skipping.")
        verify_dataset_splits("Tiny ImageNet-200", root_dir)
        return

    # Download & unzip if not extracted
    if not os.path.exists(extracted_path):
        if not os.path.exists(zip_path):
            print("Downloading Tiny ImageNet-200 (~248 MB)...")
            download_file(dataset_url, zip_path)
        unzip_file(zip_path, root_dir)

    # 1. Prepare TEST set from the original annotated 'val/'
    raw_val_dir = os.path.join(extracted_path, "val")
    annotations_file = os.path.join(raw_val_dir, "val_annotations.txt")
    raw_val_images = os.path.join(raw_val_dir, "images")

    os.makedirs(final_test, exist_ok=True)
    if os.path.exists(annotations_file) and os.path.exists(raw_val_images):
        print("Reorganizing annotated validation set into held-out test split...")
        with open(annotations_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                img_name, class_id = parts[0], parts[1]
                dst_class_dir = os.path.join(final_test, class_id)
                os.makedirs(dst_class_dir, exist_ok=True)
                src = os.path.join(raw_val_images, img_name)
                dst = os.path.join(dst_class_dir, img_name)
                if os.path.exists(src):
                    shutil.move(src, dst)

    # 2. Split original 'train/' into 80% train and 20% val
    raw_train_dir = os.path.join(extracted_path, "train")
    os.makedirs(final_train, exist_ok=True)
    os.makedirs(final_val, exist_ok=True)

    random.seed(seed)
    if os.path.exists(raw_train_dir):
        class_folders = [f for f in os.listdir(raw_train_dir) if os.path.isdir(os.path.join(raw_train_dir, f))]
        print(f"Splitting {len(class_folders)} classes into train ({int((1-val_ratio)*100)}%) and val ({int(val_ratio*100)}%)...")

        for class_id in tqdm(class_folders, desc="Processing Tiny-ImageNet classes"):
            src_images_dir = os.path.join(raw_train_dir, class_id, "images")
            if not os.path.exists(src_images_dir):
                src_images_dir = os.path.join(raw_train_dir, class_id)

            images = [img for img in os.listdir(src_images_dir) if img.lower().endswith(('.jpeg', '.jpg', '.png'))]
            random.shuffle(images)
            n_val = int(len(images) * val_ratio)
            val_imgs = images[:n_val]
            train_imgs = images[n_val:]

            # Move to final_val
            target_val_cls = os.path.join(final_val, class_id)
            os.makedirs(target_val_cls, exist_ok=True)
            for img in val_imgs:
                shutil.move(os.path.join(src_images_dir, img), os.path.join(target_val_cls, img))

            # Move to final_train
            target_train_cls = os.path.join(final_train, class_id)
            os.makedirs(target_train_cls, exist_ok=True)
            for img in train_imgs:
                shutil.move(os.path.join(src_images_dir, img), os.path.join(target_train_cls, img))

    # Clean up empty extracted folder
    if os.path.exists(extracted_path):
        shutil.rmtree(extracted_path, ignore_errors=True)

    remove_ds_store(root_dir)
    verify_dataset_splits("Tiny ImageNet-200", root_dir)


# ==============================================================================
# 5. iNaturalist-12K: Export to train / val / test ImageFolder format
# ==============================================================================

def rename_val_to_test(data_dir):
    val_path = os.path.join(data_dir, 'val')
    test_path = os.path.join(data_dir, 'test')
    if not os.path.exists(val_path):
        return
    if os.path.exists(test_path):
        return
    os.rename(val_path, test_path)
    print("Renamed original 'val' to held-out 'test'")


def create_val_split(train_dir, val_dir, split_ratio=0.2, seed=42):
    if os.path.exists(val_dir) and any(os.scandir(val_dir)):
        return
    os.makedirs(val_dir, exist_ok=True)
    random.seed(seed)
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        images = os.listdir(class_path)
        num_val = int(len(images) * split_ratio)
        val_images = random.sample(images, num_val)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(val_class_dir, exist_ok=True)
        for img_name in val_images:
            src = os.path.join(class_path, img_name)
            dst = os.path.join(val_class_dir, img_name)
            shutil.move(src, dst)
        print(f"[{class_name}] -> Moved {num_val} images to validation set.")


def crop_images(data_dir, target_size=(192, 192)):
    image_extensions = ('.jpg', '.jpeg', '.png')
    print(f"Center cropping images to {target_size}...")
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if not img_name.lower().endswith(image_extensions):
                    continue
                try:
                    with Image.open(img_path) as img:
                        if img.size != target_size:
                            img = img.convert("RGB")
                            img = ImageOps.fit(img, target_size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
                            img.save(img_path)
                except Exception as e:
                    pass
    print("Cropping completed.")


def prepare_inaturalist(root_dir="inaturalist_12K", val_ratio=0.2, target_size=(192, 192), seed=42):
    """Download and prepare iNaturalist-12K into train, val, and test splits."""
    print("\n" + "=" * 65)
    print("Preparing iNaturalist-12K Dataset with train/val/test splits")
    print("=" * 65)
    dataset_url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
    zip_filename = "nature_12K.zip"
    extracted_dir = root_dir

    if not os.path.exists(zip_filename):
        download_file(dataset_url, zip_filename)

    if not os.path.exists(extracted_dir):
        unzip_file(zip_filename, ".")

    rename_val_to_test(extracted_dir)
    train_path = os.path.join(extracted_dir, "train")
    val_path = os.path.join(extracted_dir, "val")
    create_val_split(train_path, val_path, split_ratio=val_ratio, seed=seed)
    crop_images(extracted_dir, target_size=target_size)
    remove_ds_store(extracted_dir)
    verify_dataset_splits("iNaturalist-12K", extracted_dir)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download, split, and prepare vision datasets with train/val/test splits for Fractal CNN."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["cifar10", "cifar100", "svhn", "tiny-imagenet", "inaturalist", "all"],
        help="Dataset to prepare: 'cifar10', 'cifar100', 'svhn', 'tiny-imagenet', 'inaturalist', or 'all' (default: all)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Base root directory for storing datasets (default: ./data)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation split ratio from training set (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/val splitting (default: 42)"
    )
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    if args.dataset in ["cifar10", "all"]:
        prepare_cifar10(root_dir=os.path.join(args.data_dir, "cifar10"), val_ratio=args.val_ratio, seed=args.seed)

    if args.dataset in ["cifar100", "all"]:
        prepare_cifar100(root_dir=os.path.join(args.data_dir, "cifar100"), val_ratio=args.val_ratio, seed=args.seed)

    if args.dataset in ["svhn", "all"]:
        prepare_svhn(root_dir=os.path.join(args.data_dir, "svhn"), val_ratio=args.val_ratio, seed=args.seed)

    if args.dataset in ["tiny-imagenet", "all"]:
        prepare_tiny_imagenet(root_dir=os.path.join(args.data_dir, "tiny_imagenet"), val_ratio=args.val_ratio, seed=args.seed)

    if args.dataset in ["inaturalist", "all"]:
        prepare_inaturalist(root_dir="inaturalist_12K", val_ratio=args.val_ratio, seed=args.seed)

    print("\n" + "=" * 65)
    print("All requested dataset preparation and split tasks completed!")
    print("Every prepared dataset now contains 'train/', 'val/', and 'test/'")
    print("subdirectories ready for PyTorch torchvision.datasets.ImageFolder.")
    print("=" * 65)
