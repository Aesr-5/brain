"""
Evaluation script for BRATS 2020 Brain Tumor Segmentation
Computes Dice, Hausdorff95, Sensitivity, and Specificity
"""

import torch
import numpy as np
from pathlib import Path
from monai.networks.nets import UNet
import json
from tqdm import tqdm

from config import MODEL_CONFIG, BRATS_CLASSES
from utils import dice_score, hausdorff_distance


def compute_metrics(pred, target, class_name="WT"):
    """
    Compute evaluation metrics for a single prediction.
    
    Args:
        pred: Predicted segmentation (3D array)
        target: Ground truth segmentation (3D array)
        class_name: Class to evaluate (WT, TC, ET)
        
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Get class labels
    class_labels = BRATS_CLASSES[class_name]
    
    # Convert to binary masks
    pred_binary = np.isin(pred, class_labels).astype(np.uint8)
    target_binary = np.isin(target, class_labels).astype(np.uint8)
    
    # Dice score
    metrics["dice"] = dice_score(pred_binary, target_binary)
    
    # Hausdorff distance
    if np.sum(pred_binary) > 0 and np.sum(target_binary) > 0:
        metrics["hausdorff95"] = hausdorff_distance(pred_binary, target_binary)
    else:
        metrics["hausdorff95"] = 0.0
    
    # Sensitivity (True Positive Rate)
    tp = np.sum(pred_binary * target_binary)
    fn = np.sum((1 - pred_binary) * target_binary)
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity (True Negative Rate)
    tn = np.sum((1 - pred_binary) * (1 - target_binary))
    fp = np.sum(pred_binary * (1 - target_binary))
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return metrics


def evaluate_model(model_path, val_data_dir, device="cuda"):
    """
    Evaluate model on validation set.
    
    Args:
        model_path: Path to saved model checkpoint
        val_data_dir: Directory containing validation data
        device: torch device
        
    Returns:
        Dictionary with aggregated metrics
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = UNet(
        spatial_dims=MODEL_CONFIG["spatial_dims"],
        in_channels=MODEL_CONFIG["in_channels"],
        out_channels=MODEL_CONFIG["out_channels"],
        channels=MODEL_CONFIG["channels"],
        strides=MODEL_CONFIG["strides"],
        num_res_units=MODEL_CONFIG["num_res_units"],
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Results storage
    results = {
        "WT": [],  # Whole tumor
        "TC": [],  # Tumor core
        "ET": []   # Enhancing tumor
    }
    
    # Evaluate (placeholder - implement with actual validation data)
    print("⚠️  Validation data loading not implemented.")
    print("Update evaluate_model() with actual prediction logic.")
    
    # Aggregate metrics
    aggregated = {}
    for class_name in results.keys():
        if results[class_name]:
            metrics_list = results[class_name]
            aggregated[class_name] = {
                "dice_mean": np.mean([m["dice"] for m in metrics_list]),
                "dice_std": np.std([m["dice"] for m in metrics_list]),
                "hausdorff95_mean": np.mean([m["hausdorff95"] for m in metrics_list]),
                "sensitivity_mean": np.mean([m["sensitivity"] for m in metrics_list]),
                "specificity_mean": np.mean([m["specificity"] for m in metrics_list]),
            }
    
    return aggregated


def print_results(results):
    """Print evaluation results in a nice format."""
    print("\n" + "="*80)
    print("BRATS 2020 EVALUATION RESULTS")
    print("="*80)
    
    for class_name, metrics in results.items():
        print(f"\n{class_name} (Whole Tumor)" if class_name == "WT" else f"\n{class_name}")
        print("-" * 40)
        for metric_name, value in metrics.items():
            print(f"  {metric_name:.<30} {value:.4f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <model_checkpoint> [val_data_dir]")
        print("Example: python evaluate.py checkpoints/best_model.pth data/val")
        sys.exit(1)
    
    model_path = sys.argv[1]
    val_data_dir = sys.argv[2] if len(sys.argv) > 2 else "data/val"
    
    print(f"📊 Evaluating model: {model_path}")
    results = evaluate_model(model_path, val_data_dir)
    print_results(results)
