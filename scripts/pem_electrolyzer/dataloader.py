"""
Data Loading Module for PEM Electrolyzer PINN.

Handles loading and preprocessing of NORCE experimental data:
- Test4: Training data with STRICT filtering (steady-state only)
- Test2/Test3: OOD evaluation with MINIMAL filtering

Supports both:
- MLOPS subset CSV files (test{N}_subset.csv)
- Original NORCE .txt files (test{N}/...txt)

Data Column Names:
- PS-I-MON: Current [A]
- H-P1: Hydrogen back pressure [bar]
- O-P1: Oxygen back pressure [bar]
- T-ELY-CH1: Stack temperature [°C]
- CV-mean: Mean cell voltage [V]
"""

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from typing import Tuple, Dict


def get_data_paths(data_dir: str = 'dataset/') -> Dict[str, Path]:
    """
    Get paths to data files. Auto-detect CSV subset or original .txt files.

    Args:
        data_dir: Directory containing data files

    Returns:
        data_paths: Dictionary mapping dataset names to file paths
    """
    data_dir = Path(data_dir)
    data_paths = {}

    # Try to find files for each test dataset
    for test_name in ['test2', 'test3', 'test4']:
        # Pattern 1: CSV subset files (MLOPS repo)
        csv_subset = data_dir / f"{test_name}_subset.csv"
        if csv_subset.exists():
            data_paths[test_name] = csv_subset
            continue

        # Pattern 2: Original NORCE .txt files in subdirectories
        # Example: data_dir/test2/2024_09_27_V8_VF_CF_test2_data.txt
        txt_dir = data_dir / test_name
        if txt_dir.exists() and txt_dir.is_dir():
            txt_files = list(txt_dir.glob("*.txt"))
            if txt_files:
                data_paths[test_name] = txt_files[0]  # Use first .txt file found
                continue

    return data_paths


def load_test4_training(
    data_dir: str = "dataset/",
    device: str = "cpu",
    batch_size: int = 4096,
    verbose: bool = True,
    seed: int = 42,
    use_keepout: bool = True
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Load Test4 with STRICT filtering and validation split.

    Supports two validation strategies:
    1. KEEP-OUT (default, use_keepout=True): Temperature-based split
       - Training: T < 76°C OR T > 80°C (75.7% of data)
       - Validation: 76°C ≤ T ≤ 80°C (24.3% of data)

    2. RANDOM (use_keepout=False): Standard 80/20 random split

    Args:
        data_dir: Directory containing data files
        device: torch device ('cpu' or 'cuda')
        batch_size: Batch size for DataLoaders
        verbose: Print loading statistics
        seed: Random seed for shuffling training data
        use_keepout: If True, use keep-out validation (76-80°C)

    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        stats: Dictionary with normalization statistics
    """
    # Get data paths
    data_paths = get_data_paths(data_dir)
    if 'test4' not in data_paths:
        raise FileNotFoundError(f"Test4 data not found in {data_dir}")

    file_path = data_paths['test4']

    if verbose:
        split_type = "KEEP-OUT" if use_keepout else "RANDOM"
        print("\n" + "="*60)
        print(f"Loading Test4 Training Data (STRICT + {split_type} split)")
        print("="*60)
        print(f"File: {file_path}")

    # Auto-detect file format based on extension
    if str(file_path).endswith('.csv'):
        df = pd.read_csv(file_path, sep=',')
    else:  # .txt files use semicolon separator
        df = pd.read_csv(file_path, sep=';', decimal='.')

    n_raw = len(df)

    if verbose:
        print(f"Raw samples: {n_raw:,}")

    # Drop NaN values in required columns
    required_cols = ["PS-I-MON", "H-P1", "O-P1", "T-ELY-CH1", "CV-mean"]
    df = df.dropna(subset=required_cols)

    # STRICT filtering for training (steady-state only)
    df = df.loc[
        (df["PS-I-MON"] >= 5.0) &
        (df["H-P1"] > 10.0) &
        (df["O-P1"] > 10.0) &
        (df["T-ELY-CH1"] > 70.0) &
        (df["T-ELY-CH1"] <= 85.0) &
        (df["CV-mean"] > 1.5) &
        (df["CV-mean"] < 2.0)
    ].reset_index(drop=True)

    n_filtered = len(df)

    if verbose:
        print(f"After strict filtering: {n_filtered:,} ({100*n_filtered/n_raw:.1f}%)")
        print(f"\nData ranges:")
        print(f"  Current: {df['PS-I-MON'].min():.2f} - {df['PS-I-MON'].max():.2f} A")
        print(f"  H2 Pressure: {df['H-P1'].min():.1f} - {df['H-P1'].max():.1f} bar")
        print(f"  O2 Pressure: {df['O-P1'].min():.1f} - {df['O-P1'].max():.1f} bar")
        print(f"  Temperature: {df['T-ELY-CH1'].min():.1f} - {df['T-ELY-CH1'].max():.1f} °C")
        print(f"  Voltage: {df['CV-mean'].min():.3f} - {df['CV-mean'].max():.3f} V")

    # Convert to tensors in ORIGINAL UNITS
    current_A = torch.tensor(df["PS-I-MON"].values, dtype=torch.float32)
    temperature_C = torch.tensor(df["T-ELY-CH1"].values, dtype=torch.float32)
    H2_pressure = torch.tensor(df["H-P1"].values, dtype=torch.float32)
    O2_pressure = torch.tensor(df["O-P1"].values, dtype=torch.float32)
    voltage = torch.tensor(df["CV-mean"].values, dtype=torch.float32)

    # Stack inputs: [current_A, H2_pressure, O2_pressure, temperature_C]
    X = torch.stack([current_A, H2_pressure, O2_pressure, temperature_C], dim=1)
    y = voltage

    # Compute average pressure for normalization stats
    pressure_avg = (H2_pressure + O2_pressure) / 2.0

    # Compute normalization statistics
    stats = {
        'i_mean': current_A.mean().item(),
        'i_std': current_A.std().item(),
        'T_mean': temperature_C.mean().item(),
        'T_std': temperature_C.std().item(),
        'P_mean': pressure_avg.mean().item(),
        'P_std': pressure_avg.std().item(),
        'V_mean': voltage.mean().item(),
        'V_std': voltage.std().item(),
        'n_samples': n_filtered,
    }

    if verbose:
        print(f"\nNormalization statistics (original units):")
        print(f"  Current: {stats['i_mean']:.2f} ± {stats['i_std']:.2f} A")
        print(f"  Temperature: {stats['T_mean']:.1f} ± {stats['T_std']:.1f} °C")
        print(f"  Avg Pressure: {stats['P_mean']:.1f} ± {stats['P_std']:.1f} bar")
        print(f"  Voltage: {stats['V_mean']:.4f} ± {stats['V_std']:.4f} V")

    # ====================================================================
    # VALIDATION SPLIT
    # ====================================================================
    if use_keepout:
        # KEEP-OUT VALIDATION
        T_KEEPOUT_LOW = 76.0
        T_KEEPOUT_HIGH = 80.0

        temp_C = df["T-ELY-CH1"].values
        val_mask_np = (temp_C >= T_KEEPOUT_LOW) & (temp_C <= T_KEEPOUT_HIGH)
        train_mask_np = ~val_mask_np

        val_mask = torch.tensor(val_mask_np, dtype=torch.bool)
        train_mask = torch.tensor(train_mask_np, dtype=torch.bool)

        train_X = X[train_mask]
        train_y = y[train_mask]
        val_X = X[val_mask]
        val_y = y[val_mask]

        n_train = len(train_X)
        n_val = len(val_X)

        if verbose:
            print(f"\nKEEP-OUT validation split (76-80°C):")
            print(f"  Training: {n_train:,} samples ({100*n_train/n_filtered:.1f}%)")
            print(f"  Validation: {n_val:,} samples ({100*n_val/n_filtered:.1f}%)")
    else:
        # RANDOM VALIDATION (standard 80/20 IID split)
        torch.manual_seed(seed)
        n_total = len(X)
        n_val = int(n_total * 0.2)
        n_train = n_total - n_val

        # Random permutation for split
        perm = torch.randperm(n_total)
        val_indices = perm[:n_val]
        train_indices = perm[n_val:]

        train_X = X[train_indices]
        train_y = y[train_indices]
        val_X = X[val_indices]
        val_y = y[val_indices]

        if verbose:
            print(f"\nRANDOM validation split (80/20):")
            print(f"  Training: {n_train:,} samples ({100*n_train/n_total:.1f}%)")
            print(f"  Validation: {n_val:,} samples ({100*n_val/n_total:.1f}%)")

    # Shuffle training data
    torch.manual_seed(seed)
    train_perm = torch.randperm(n_train)
    train_X = train_X[train_perm]
    train_y = train_y[train_perm]

    # Move to device
    train_X = train_X.to(device)
    train_y = train_y.to(device)
    val_X = val_X.to(device)
    val_y = val_y.to(device)

    # Create datasets and loaders
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if verbose:
        print(f"  Batch size: {batch_size}")

    return train_loader, val_loader, stats


def load_ood_minimal(
    dataset_name: str,
    data_dir: str = "dataset/",
    device: str = "cpu",
    batch_size: int = 4096,
    verbose: bool = True
) -> Tuple[DataLoader, Dict]:
    """
    Load Test2 or Test3 with MINIMAL filtering for OOD evaluation.

    Args:
        dataset_name: "test2" or "test3"
        data_dir: Directory containing data files
        device: torch device
        batch_size: Batch size for DataLoader
        verbose: Print loading statistics

    Returns:
        dataloader: DataLoader for evaluation
        info: Dictionary with dataset information
    """
    if dataset_name not in ["test2", "test3"]:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'test2' or 'test3'.")

    # Get data paths
    data_paths = get_data_paths(data_dir)
    if dataset_name not in data_paths:
        raise FileNotFoundError(f"{dataset_name} data not found in {data_dir}")

    file_path = data_paths[dataset_name]

    if verbose:
        print(f"\n" + "="*60)
        print(f"Loading {dataset_name.upper()} OOD Data (MINIMAL filtering)")
        print("="*60)
        print(f"File: {file_path}")

    # Auto-detect file format
    if str(file_path).endswith('.csv'):
        df = pd.read_csv(file_path, sep=',')
    else:
        df = pd.read_csv(file_path, sep=';', decimal='.')

    n_raw = len(df)

    if verbose:
        print(f"Raw samples: {n_raw:,}")

    # Drop NaN values in required columns
    required_cols = ["PS-I-MON", "H-P1", "O-P1", "T-ELY-CH1", "CV-mean"]
    df = df.dropna(subset=required_cols)

    # MINIMAL filtering for OOD evaluation
    df = df.loc[
        (df["H-P1"] >= 0) &
        (df["O-P1"] >= 0)
    ]

    # Remove cold startup (I < 1A AND T < 60°C together)
    cold_startup_mask = (df["PS-I-MON"] < 1.0) & (df["T-ELY-CH1"] < 60.0)
    df = df.loc[~cold_startup_mask]

    # Broader voltage sanity check
    df = df.loc[
        (df["CV-mean"] >= 1.0) &
        (df["CV-mean"] <= 2.5)
    ].reset_index(drop=True)

    n_filtered = len(df)

    if verbose:
        print(f"After minimal filtering: {n_filtered:,} ({100*n_filtered/n_raw:.1f}%)")
        print(f"\nData ranges:")
        print(f"  Current: {df['PS-I-MON'].min():.2f} - {df['PS-I-MON'].max():.2f} A")
        print(f"  H2 Pressure: {df['H-P1'].min():.1f} - {df['H-P1'].max():.1f} bar")
        print(f"  O2 Pressure: {df['O-P1'].min():.1f} - {df['O-P1'].max():.1f} bar")
        print(f"  Temperature: {df['T-ELY-CH1'].min():.1f} - {df['T-ELY-CH1'].max():.1f} °C")
        print(f"  Voltage: {df['CV-mean'].min():.3f} - {df['CV-mean'].max():.3f} V")

    # Convert to tensors
    current_A = torch.tensor(df["PS-I-MON"].values, dtype=torch.float32)
    temperature_C = torch.tensor(df["T-ELY-CH1"].values, dtype=torch.float32)
    H2_pressure = torch.tensor(df["H-P1"].values, dtype=torch.float32)
    O2_pressure = torch.tensor(df["O-P1"].values, dtype=torch.float32)
    voltage = torch.tensor(df["CV-mean"].values, dtype=torch.float32)

    # Stack inputs
    X = torch.stack([current_A, H2_pressure, O2_pressure, temperature_C], dim=1)
    y = voltage

    # Move to device
    X = X.to(device)
    y = y.to(device)

    # Create dataset and loader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Dataset info
    info = {
        'name': dataset_name,
        'n_samples': n_filtered,
        'n_raw': n_raw,
        'retention_rate': n_filtered / n_raw,
    }

    return dataloader, info


if __name__ == "__main__":
    """Quick test of data loading."""
    print("Testing data loaders...")

    # Test with default data_dir
    try:
        train_loader, val_loader, stats = load_test4_training(
            data_dir="dataset/",
            verbose=True
        )
        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
    except FileNotFoundError as e:
        print(f"Warning: {e}")

    print("\nData loading test complete!")
