import os
import shutil
import random
import yaml
from ultralytics import YOLO


def setup_yolo_dataset(base_export_dir):
    yolo_dataset_dir = os.path.join(base_export_dir, "yolo_dataset")
    yaml_path = os.path.join(yolo_dataset_dir, "runway_pose.yaml")

    # If the dataset is already built, skip the heavy copying process
    if os.path.exists(yolo_dataset_dir):
        print("YOLO dataset folder already exists! Skipping copy phase...")
        return yaml_path

    print("Building unified YOLO dataset from weather folders. This will take a moment...")

    # 1. Create YOLO directory structure
    dirs_to_make = [
        os.path.join(yolo_dataset_dir, "images", "train"),
        os.path.join(yolo_dataset_dir, "images", "val"),
        os.path.join(yolo_dataset_dir, "labels", "train"),
        os.path.join(yolo_dataset_dir, "labels", "val")
    ]
    for d in dirs_to_make:
        os.makedirs(d, exist_ok=True)

    # 2. Collect all valid image and label paths from the 3 weather folders
    weather_dir = os.path.join(base_export_dir, "weather")
    weather_conditions = ["Broken_Clouds", "Heavy_Rain", "Rain"]

    all_data_pairs = []

    for weather in weather_conditions:
        img_folder = os.path.join(weather_dir, weather, "usable", "images")
        lbl_folder = os.path.join(weather_dir, weather, "usable", "labels_pose")

        if not os.path.exists(img_folder):
            continue

        images = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_name in images:
            label_name = os.path.splitext(img_name)[0] + ".txt"
            src_img = os.path.join(img_folder, img_name)
            src_lbl = os.path.join(lbl_folder, label_name)

            # Only add if the label exists
            if os.path.exists(src_lbl):
                all_data_pairs.append((src_img, src_lbl, img_name, label_name))

    # 3. Shuffle and Split (90% Train, 10% Validation)
    random.seed(42)
    random.shuffle(all_data_pairs)

    split_index = int(len(all_data_pairs) * 0.9)
    train_pairs = all_data_pairs[:split_index]
    val_pairs = all_data_pairs[split_index:]

    # 4. Copy files to YOLO structure
    def copy_pairs(pairs, split_name):
        for src_img, src_lbl, img_name, lbl_name in pairs:
            shutil.copy(src_img, os.path.join(yolo_dataset_dir, "images", split_name, img_name))
            shutil.copy(src_lbl, os.path.join(yolo_dataset_dir, "labels", split_name, lbl_name))

    print(f"Copying {len(train_pairs)} training files...")
    copy_pairs(train_pairs, "train")

    print(f"Copying {len(val_pairs)} validation files...")
    copy_pairs(val_pairs, "val")

    # 5. Create the YAML configuration file
    yaml_data = {
        'path': yolo_dataset_dir,
        'train': 'images/train',
        'val': 'images/val',
        'kpt_shape': [4, 3],  # 4 points (corners), 3 values per point (X, Y, Visibility)
        'names': {0: 'runway'}
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"Dataset completely unified! YAML saved to {yaml_path}")
    return yaml_path


def train_runway_model():
    # Dynamically locate the ScenerioExports folder
    base_export_dir = os.path.join(os.getcwd(), "ScenerioExports")

    # Run the dataset builder
    yaml_path = setup_yolo_dataset(base_export_dir)

    # Load the model
    model = YOLO('yolov8s-pose.pt')

    # --- LAB COMPUTER HARDWARE OPTIMIZATIONS ---
    # Since the lab PC likely has more RAM/VRAM, we can increase batch and workers.
    # If the lab PC crashes with OOM, simply drop batch back to 4 or 8.

    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=1280,
        batch=2,  # Increased for lab PC (Drop to 8 or 4 if VRAM crashes)
        close_mosaic=0,
        device=0,
        workers=4,  # Increased for lab PC CPU (Drop to 1 or 2 if System RAM crashes)

        perspective=0.001,
        pose=15.0,
        fliplr=0.0,  # Disabled to preserve [NR, NL, FL, FR] corner order
        flipud=0.0,
        patience=30,  # Auto-stops if mAP doesn't improve for 30 epochs

        project="Runway_Training",
        name="Yolo8s_Pose_Batch2"
    )


if __name__ == '__main__':
    train_runway_model()