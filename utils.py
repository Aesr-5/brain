"""
Utility functions for BRATS 2020 Brain Tumor Segmentation
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List
import nibabel as nib
from monai.transforms import Compose, EnsureChannelFirstd, NormalizeIntensityd, Spacingd, Orientationd


def load_nifti(file_path: str) -> Tuple[np.ndarray, dict]:
    """
    Load a NIfTI file and return data with metadata.
    
    Args:
        file_path: Path to the .nii or .nii.gz file
        
    Returns:
        data: 3D numpy array
        header: NIfTI header information
    """
    img = nib.load(file_path)
    data = img.get_fdata()
    return data, img.header


def save_nifti(data: np.ndarray, file_path: str, affine: np.ndarray = None) -> None:
    """
    Save a numpy array as a NIfTI file.
    
    Args:
        data: 3D numpy array to save
        file_path: Output path for the .nii.gz file
        affine: Affine transformation matrix (default: identity)
    """
    if affine is None:
        affine = np.eye(4)
    
    img = nib.Nifti1Image(data.astype(np.uint8), affine)
    nib.save(img, file_path)


def get_brats_transforms():
    """
    Return MONAI transforms for BRATS preprocessing.
    
    Returns:
        Compose object with transforms
    """
    return Compose([
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True),
    ])


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """
    Calculate Dice coefficient.
    
    Args:
        pred: Predicted segmentation (binary or multi-class)
        target: Ground truth segmentation
        smooth: Smoothing constant
        
    Returns:
        Dice coefficient value [0, 1]
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = np.sum(pred_flat * target_flat)
    return (2 * intersection + smooth) / (np.sum(pred_flat) + np.sum(target_flat) + smooth)


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate Hausdorff distance (simplified 95th percentile version).
    
    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        
    Returns:
        Hausdorff distance value
    """
    from scipy.spatial.distance import directed_hausdorff
    
    pred_coords = np.column_stack(np.where(pred))
    target_coords = np.column_stack(np.where(target))
    
    if len(pred_coords) == 0 or len(target_coords) == 0:
        return 0.0
    
    d1 = directed_hausdorff(pred_coords, target_coords)[0]
    d2 = directed_hausdorff(target_coords, pred_coords)[0]
    
    return max(d1, d2)


def get_file_list(data_dir: str, pattern: str = "*.nii.gz") -> List[Path]:
    """
    Get list of files matching pattern in directory.
    
    Args:
        data_dir: Root data directory
        pattern: File pattern to search
        
    Returns:
        List of Path objects
    """
    return sorted(Path(data_dir).glob(f"**/{pattern}"))


if __name__ == "__main__":
    print("BRATS utilities module loaded successfully!")
