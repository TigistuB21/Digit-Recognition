"""
Digit Recognition Dataset Loader
================================
Easily load the preprocessed handwritten digit dataset in:
  1. NumPy (.npz)
  2. PyTorch (torchvision.datasets.ImageFolder / DataLoader)
  3. TensorFlow / Keras (image_dataset_from_directory)
  4. Scikit-learn (Flattened feature vectors)
"""

import os
import numpy as np
import pandas as pd

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(DATASET_DIR, "digit_recognition_dataset.npz")
CSV_PATH = os.path.join(DATASET_DIR, "dataset_metadata.csv")


def load_numpy_splits():
    """
    Returns X_train, y_train, X_test, y_test as normalized float32 arrays in [0.0, 1.0].
    
    Returns:
        X_train: np.ndarray, shape (N_train, 28, 28), float32
        y_train: np.ndarray, shape (N_train,), int64
        X_test:  np.ndarray, shape (N_test, 28, 28), float32
        y_test:  np.ndarray, shape (N_test,), int64
    """
    data = np.load(NPZ_PATH)
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"]


def load_full_numpy():
    """
    Returns the complete dataset of all samples.
    
    Returns:
        X_all: np.ndarray, shape (N, 28, 28), float32
        y_all: np.ndarray, shape (N,), int64
    """
    data = np.load(NPZ_PATH)
    return data["X_all"], data["y_all"]


def load_metadata_dataframe():
    """
    Returns the pandas DataFrame with metadata (image_id, digit_label, split, source_file).
    """
    return pd.read_csv(CSV_PATH)


def get_pytorch_dataloaders(batch_size=32):
    """
    Creates PyTorch DataLoaders using standard torchvision ImageFolder.
    """
    try:
        import torch
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(), # converts to [0.0, 1.0] and adds channel dimension (1, 28, 28)
        ])
        
        train_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=transform)
        test_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"), transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    except ImportError:
        print("[INFO] PyTorch / torchvision not installed. Use load_numpy_splits() instead.")
        return None, None


if __name__ == "__main__":
    print("--- Loading Digit Recognition Dataset ---")
    X_train, y_train, X_test, y_test = load_numpy_splits()
    print(f"X_train shape: {X_train.shape} (min={X_train.min():.1f}, max={X_train.max():.1f})")
    print(f"y_train shape: {y_train.shape} (classes={np.unique(y_train)})")
    print(f"X_test shape:  {X_test.shape} (min={X_test.min():.1f}, max={X_test.max():.1f})")
    print(f"y_test shape:  {y_test.shape} (classes={np.unique(y_test)})")
    
    meta_df = load_metadata_dataframe()
    print(f"\nMetadata CSV Summary:")
    print(meta_df.groupby(["digit_label", "split"]).size().unstack().fillna(0).astype(int))
