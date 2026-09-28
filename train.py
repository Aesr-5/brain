"""
Training script for BRATS 2020 Brain Tumor Segmentation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json

from config import (
    TRAIN_CONFIG, MODEL_CONFIG, DATA_CONFIG, 
    CHECKPOINTS_DIR, LOGS_DIR
)
from utils import get_brats_transforms


class BRATSDataset(Dataset):
    """
    Placeholder BRATS dataset class.
    Implement this with actual data loading logic.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.files = []  # Load your files here
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load image and label from NIFTI files
        # Apply transforms
        # Return {"image": tensor, "label": tensor}
        pass


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, loss_fn, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    """Main training loop."""
    device = torch.device(TRAIN_CONFIG["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = UNet(
        spatial_dims=MODEL_CONFIG["spatial_dims"],
        in_channels=MODEL_CONFIG["in_channels"],
        out_channels=MODEL_CONFIG["out_channels"],
        channels=MODEL_CONFIG["channels"],
        strides=MODEL_CONFIG["strides"],
        num_res_units=MODEL_CONFIG["num_res_units"],
    ).to(device)
    
    # Loss and optimizer
    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"]
    )
    
    # Setup data loaders (placeholder - implement with actual BRATS data)
    # train_dataset = BRATSDataset(...)
    # train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False)
    
    print("⚠️  Note: Data loading not implemented. Update BRATSDataset class with actual paths.")
    print("📥 Download BRATS dataset: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation")
    
    # Training loop
    history = {"train_loss": [], "val_loss": []}
    best_loss = float("inf")
    
    for epoch in range(TRAIN_CONFIG["num_epochs"]):
        print(f"\n--- Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']} ---")
        
        # train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        # val_loss = validate(model, val_loader, loss_fn, device)
        
        # history["train_loss"].append(train_loss)
        # history["val_loss"].append(val_loss)
        
        # print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # # Save checkpoint
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     checkpoint_path = CHECKPOINTS_DIR / f"best_model_epoch{epoch+1}.pth"
        #     torch.save(model.state_dict(), checkpoint_path)
        #     print(f"✅ Model saved: {checkpoint_path}")
    
    # Save training history
    history_path = LOGS_DIR / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✅ Training complete! History saved to: {history_path}")


if __name__ == "__main__":
    main()
