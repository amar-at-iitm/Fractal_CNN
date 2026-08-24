import os
import shutil
import zipfile
import random
import requests
from tqdm import tqdm
from PIL import Image

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(dest_path)}",
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")


def rename_val_to_test(data_dir):
    val_path = os.path.join(data_dir, 'val')
    test_path = os.path.join(data_dir, 'test')
    
    if not os.path.exists(val_path):
        print("No 'val' folder found to rename.")
        return
    
    if os.path.exists(test_path):
        print("'test' folder already exists. Skipping rename.")
        return
    
    os.rename(val_path, test_path)
    print("Renamed 'val' to 'test'")

def create_val_split(train_dir, val_dir, split_ratio=0.2):
    if os.path.exists(val_dir) and any(os.scandir(val_dir)):
        print("Validation split already exists. Skipping split.")
        return
    
    os.makedirs(val_dir, exist_ok=True)

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



def resize_images(data_dir, target_size=(256, 256)):
    image_extensions = ('.jpg')  # Add more extensions if needed

    print("Resizing images...")
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
                
                # Skip files that are not images
                if not img_name.lower().endswith(image_extensions):
                    continue

                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")  # Ensure 3 channels
                        img = img.resize(target_size)
                        img.save(img_path)
                except Exception as e:
                    print(f"Failed to process {img_path}: {e}")
    print("Resizing completed.")


#  cleanup step
def remove_ds_store(data_dir):
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file == '.DS_Store':
                try:
                    os.remove(os.path.join(root, file))
                    print(f"Removed: {os.path.join(root, file)}")
                except Exception as e:
                    print(f"Failed to remove .DS_Store: {e}")

# Main function to execute the steps
if __name__ == "__main__":
    # Configurations
    dataset_url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
    zip_filename = "nature_12K.zip"
    extracted_dir = "inaturalist_12K"

    # Step 1: Downloading
    zip_path = zip_filename
    if not os.path.exists(zip_path):
        download_file(dataset_url, zip_path)
    else:
        print("Dataset already downloaded.")

    # Step 2: Unzipping
    if not os.path.exists(extracted_dir):
        unzip_file(zip_path, ".")
    else:
        print("Dataset already unzipped.")

    # Step 3: Renaming val -> test
    rename_val_to_test(os.path.join(extracted_dir))

    # Step 4: Creating new validation split from train/
    train_path = os.path.join(extracted_dir, "train")
    val_path = os.path.join(extracted_dir, "val")
    create_val_split(train_path, val_path, split_ratio=0.2)

    # Step 5: Resizing all images to 256x256
    resize_images(extracted_dir, target_size=(256, 256))

    # Step 6: Removing .DS_Store files
    remove_ds_store(extracted_dir)
    
