#!/usr/bin/env python3
"""
Extract lightweight CSV subsets from full NORCE data files.

This script extracts 5 key columns from the semicolon-delimited NORCE data files
and saves them as comma-separated CSV files for easier distribution and use.

Source: NORCE experimental PEM electrolyzer data (Proton OnSite M400 stack)
Columns: PS-I-MON, H-P1, O-P1, T-ELY-CH1, CV-mean
"""

from pathlib import Path
import pandas as pd


def extract_subset(source_path: Path, output_path: Path, columns: list[str]) -> None:
    """
    Extract specified columns from a semicolon-delimited file and save as CSV.

    Args:
        source_path: Path to source .txt file (semicolon-delimited)
        output_path: Path to output .csv file (comma-delimited)
        columns: List of column names to extract
    """
    print(f"\nProcessing: {source_path.name}")

    # Check if source file exists
    if not source_path.exists():
        print(f"  ERROR: Source file not found at {source_path}")
        return

    # Read source file with semicolon delimiter
    df = pd.read_csv(source_path, sep=';')

    # Verify all columns exist
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        print(f"  ERROR: Missing columns: {missing_cols}")
        print(f"  Available columns: {df.columns.tolist()[:10]}...")
        return

    # Extract subset
    df_subset = df[columns]

    # Save as comma-separated CSV
    df_subset.to_csv(output_path, index=False)

    # Report statistics
    original_size = source_path.stat().st_size / (1024 * 1024)  # MB
    output_size = output_path.stat().st_size / (1024 * 1024)  # MB
    row_count = len(df_subset)

    print(f"  Rows: {row_count:,}")
    print(f"  Original size: {original_size:.2f} MB")
    print(f"  Output size: {output_size:.2f} MB")
    print(f"  Compression: {(1 - output_size/original_size)*100:.1f}%")
    print(f"  Saved to: {output_path.name}")


def main():
    """Extract subsets for all test datasets."""
    # Define paths relative to project root
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data"
    output_dir = Path(__file__).parent

    # Columns to extract
    columns = [
        "PS-I-MON",      # Current [A]
        "H-P1",          # Hydrogen pressure [bar]
        "O-P1",          # Oxygen pressure [bar]
        "T-ELY-CH1",     # Temperature [°C]
        "CV-mean",       # Cell voltage mean [V]
    ]

    # Dataset configurations
    datasets = [
        {
            "name": "test2",
            "source": data_dir / "test2" / "2024_09_27_V8_VF_CF_test2_data.txt",
            "output": output_dir / "test2_subset.csv",
            "description": "Current sweep experiment (2024-09-27) - OOD evaluation",
        },
        {
            "name": "test3",
            "source": data_dir / "test3" / "2024_09_30_V8_VF_CF_test3_data.txt",
            "output": output_dir / "test3_subset.csv",
            "description": "Pressure swap experiment (2024-09-30) - OOD evaluation",
        },
        {
            "name": "test4",
            "source": data_dir / "test4" / "2024_10_16_V8_VF_CF_test4_data.txt",
            "output": output_dir / "test4_subset.csv",
            "description": "Long-term stability test (2024-10-16) - training data",
        },
    ]

    print("=" * 70)
    print("NORCE Data Extraction")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Output directory: {output_dir}")
    print(f"Extracting columns: {', '.join(columns)}")

    # Process each dataset
    for dataset in datasets:
        print("\n" + "-" * 70)
        print(f"Dataset: {dataset['name']}")
        print(f"Description: {dataset['description']}")
        extract_subset(dataset["source"], dataset["output"], columns)

    print("\n" + "=" * 70)
    print("Extraction complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
