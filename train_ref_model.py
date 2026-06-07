#!/usr/bin/env python3
"""
Train a custom YOLO ref detector from labeled dataset.

Usage:
    python train_ref_model.py --dataset ref_dataset_labeled.json --output ref_model
"""

import json
import argparse
import shutil
import os
from pathlib import Path
from PIL import Image
import random

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("ERROR: ultralytics not installed. Install with: pip install ultralytics")

class RefModelTrainer:
    def __init__(self, dataset_json: str, output_dir: str = "ref_model"):
        self.dataset_json = dataset_json
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load labeled dataset
        with open(dataset_json) as f:
            self.dataset = json.load(f)

        print(f"Loaded dataset: {self.dataset['total_refs']} refs, {self.dataset['total_players']} players")

    def create_yolo_dataset(self):
        """Create YOLO dataset structure from labeled detections."""
        print("\nCreating YOLO dataset structure...")

        yolo_dir = self.output_dir / "yolo_data"
        yolo_dir.mkdir(exist_ok=True)

        # Create directories
        for split in ["images/train", "images/val", "labels/train", "labels/val"]:
            (yolo_dir / split).mkdir(parents=True, exist_ok=True)

        # Split data: 70% train, 30% val
        refs = self.dataset["refs"]
        players = self.dataset["players"]

        random.shuffle(refs)
        random.shuffle(players)

        split_idx_ref = int(len(refs) * 0.7)
        split_idx_player = int(len(players) * 0.7)

        train_refs = refs[:split_idx_ref]
        val_refs = refs[split_idx_ref:]
        train_players = players[:split_idx_player]
        val_players = players[split_idx_player:]

        print(f"Train: {len(train_refs)} refs, {len(train_players)} players")
        print(f"Val:   {len(val_refs)} refs, {len(val_players)} players")

        # Copy files and create labels
        file_idx = 0

        for split_name, split_data in [
            ("train", train_refs + train_players),
            ("val", val_refs + val_players)
        ]:
            for detection in split_data:
                src_img = detection["image_path"]
                label = 0 if detection in (train_refs + val_refs) else 1  # 0=ref, 1=player

                if not Path(src_img).exists():
                    print(f"WARNING: Image not found: {src_img}")
                    continue

                # Copy image
                dst_img = yolo_dir / "images" / split_name / f"{file_idx:06d}.jpg"
                shutil.copy2(src_img, dst_img)

                # Create label file (YOLO format: class_id x_center y_center width height normalized)
                # Since we extracted crops, bbox is the full image (0, 0, 1, 1) with the label
                dst_label = yolo_dir / "labels" / split_name / f"{file_idx:06d}.txt"
                with open(dst_label, "w") as f:
                    f.write(f"{label} 0.5 0.5 1.0 1.0\n")  # Full image bounding box

                file_idx += 1

        # Create data.yaml
        data_yaml = f"""path: {yolo_dir}
train: images/train
val: images/val

nc: 2
names: ['ref', 'player']
"""

        yaml_path = yolo_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            f.write(data_yaml)

        print(f"YOLO dataset created at: {yolo_dir}")
        return yolo_dir

    def train(self, epochs: int = 50, imgsz: int = 640):
        """Train the YOLO model."""
        if not HAS_YOLO:
            print("ERROR: ultralytics not installed")
            return

        print(f"\nTraining YOLO ref detector for {epochs} epochs...")

        yolo_dir = self.create_yolo_dataset()
        data_yaml = yolo_dir / "data.yaml"

        # Load base model and train
        model = YOLO("yolov8n.pt")  # Nano model for faster training

        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            device=0,  # GPU device 0, use 'cpu' if no GPU
            patience=10,
            save=True,
            project=str(self.output_dir),
            name="train",
            verbose=True,
        )

        print(f"\nTraining complete!")
        print(f"Best model saved to: {self.output_dir / 'train' / 'weights' / 'best.pt'}")

        # Create inference script
        self.create_inference_script()

    def create_inference_script(self):
        """Create a script to test the trained model."""
        script = '''#!/usr/bin/env python3
"""Test the trained ref detector model."""

from ultralytics import YOLO
import cv2
import sys

model = YOLO("train/weights/best.pt")

# Test on an image or video
test_file = sys.argv[1] if len(sys.argv) > 1 else "left.mp4"

results = model(test_file, conf=0.5)

for result in results:
    print(f"Frame: {result.path}")
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = "REF" if cls == 0 else "PLAYER"
        print(f"  {label}: confidence={conf:.2f} bbox=({x1},{y1})-({x2},{y2})")
'''

        script_path = self.output_dir / "test_model.py"
        with open(script_path, "w") as f:
            f.write(script)

        print(f"Inference script created: {script_path}")

    def run(self, epochs: int = 50):
        """Run full training pipeline."""
        print("=" * 60)
        print("REF DETECTOR MODEL TRAINING")
        print("=" * 60)
        self.train(epochs)
        print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom ref detector YOLO model")
    parser.add_argument("--dataset", required=True, help="Path to labeled dataset JSON")
    parser.add_argument("--output", default="ref_model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")

    args = parser.parse_args()

    trainer = RefModelTrainer(args.dataset, args.output)
    trainer.run(args.epochs)
