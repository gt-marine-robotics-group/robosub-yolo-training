import os
import shutil
import zipfile
import albumentations as A
from ultralytics import YOLO, settings
from ultralytics.data.augment import Albumentations
from roboflow import Roboflow
import wandb
 
# Disable YOLO's native W&B to avoid conflicts with custom injector
settings.update({"wandb": False})
 
# ==========================================
# LIGHTING TOGGLE
# ==========================================
APPLY_EXTREME_LIGHTING = True
 
if APPLY_EXTREME_LIGHTING:
    print("[INFO] Injecting custom underwater albumentations pipeline")
 
    def custom_albumentations_init(self, p=1.0, transforms=None, **kwargs):
        self.p = p
        self.contains_spatial = False
        self.transform = A.Compose([
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.15, p=0.6),
            A.Blur(blur_limit=5, p=0.1),
            A.MedianBlur(blur_limit=5, p=0.1),
            A.ToGray(p=0.05)
        ])
    Albumentations.__init__ = custom_albumentations_init
else:
    print("[INFO] Using standard YOLO default augmentations")
 
 
# ==========================================
# ROBOFLOW DATASET DOWNLOAD
# ==========================================
rf = Roboflow(api_key="fP0H2pAoE5CYfySw9gCg")
project = rf.workspace("ucrt").project("robosub-comp-2026")
 
DATASET_VERSION = 5
LIGHTING_SUFFIX = "ExtLighting" if APPLY_EXTREME_LIGHTING else "DefaultLighting"
MODEL_VERSION = f"RoboSub_v{DATASET_VERSION}_FastSweep_{LIGHTING_SUFFIX}"
dataset_folder = f"RoboSub-Comp-2026-{DATASET_VERSION}"
 
print(f"[INFO] Cleaning old dataset folder ({dataset_folder})")
if os.path.exists(dataset_folder):
    shutil.rmtree(dataset_folder)
if os.path.exists("RoboSub-Comp-2026-1"):
    shutil.rmtree("RoboSub-Comp-2026-1")
 
print(f"[INFO] Downloading dataset v{DATASET_VERSION}")
try:
    dataset = project.version(DATASET_VERSION).download("yolov11")
except zipfile.BadZipFile:
    print("[WARN] Corrupted zip, retrying download...")
    if os.path.exists(dataset_folder):
        shutil.rmtree(dataset_folder)
    dataset = project.version(DATASET_VERSION).download("yolov11")
 
yaml_path = os.path.join(dataset.location, "data.yaml")
if not os.path.exists(yaml_path):
    raise FileNotFoundError("FATAL: data.yaml not found after download")
 
 
# ==========================================
# HYPERPARAMETERS
# ==========================================
BEST_PARAMS = {
    "epochs": 30,             # Shorter sweeps for faster iteration
    "optimizer": "SGD",
    "imgsz": 640,
    "batch_size": 16,
    "mixup": 0.0,
    "lr0": 0.015,             # Higher LR for new dataset
    "weight_decay": 0.0005,
    "mosaic": 0.0,
}
 
 
# ==========================================
# CUSTOM METRICS INJECTOR
# ==========================================
def on_fit_epoch_end(trainer):
    """Log metrics to W&B if available."""
    try:
        if hasattr(trainer, 'metrics') and trainer.metrics:
            metrics_to_log = {**trainer.metrics}
            if hasattr(trainer, 'lr'):
                metrics_to_log.update(trainer.lr)
            metrics_to_log["epoch"] = trainer.epoch
            wandb.log(metrics_to_log)
    except Exception as e:
        print(f"[WARN] Skipping W&B log epoch {trainer.epoch}: {e}")
 
 
# ==========================================
# TRAINING
# ==========================================
def train_best_model():
    lighting_tag = "extreme-lighting" if APPLY_EXTREME_LIGHTING else "default-lighting"
    wandb.init(
        project="robosub_project",
        group="rtx-5070-best-models",
        tags=["rtx-5070", "desktop", "final-model", lighting_tag]
    )
 
    wandb.define_metric("epoch")
    wandb.define_metric("metrics/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("train/*", step_metric="epoch")
 
    model = YOLO("yolo11n.pt")
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
 
    print("[INFO] Starting training run")
    model.train(
        data=yaml_path,
        epochs=BEST_PARAMS["epochs"],
        imgsz=BEST_PARAMS["imgsz"],
        batch=BEST_PARAMS["batch_size"],
        optimizer=BEST_PARAMS["optimizer"],
        lr0=BEST_PARAMS["lr0"],
        weight_decay=BEST_PARAMS["weight_decay"],
        mosaic=BEST_PARAMS["mosaic"],
        mixup=BEST_PARAMS["mixup"],
 
        project=f"runs/{MODEL_VERSION}",
        name=f"best_run_{lighting_tag}",
 
        device=0,
        workers=8,
        cache=True
    )
 
if __name__ == "__main__":
    train_best_model()
