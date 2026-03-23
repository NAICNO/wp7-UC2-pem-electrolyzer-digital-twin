# PEM Electrolyzer Dataset

This directory contains lightweight CSV subsets extracted from the full NORCE experimental data files. These subsets include only the 5 key variables needed for model training and evaluation.

## Data Provenance

**Source**: NORCE experimental PEM electrolyzer tests
**Stack**: Proton OnSite M400 (4-cell stack)
**Active area**: 50 cm² (4 cells × 12.5 cm² each)
**Location**: NORCE Research Centre, Stavanger, Norway

## Datasets

### test2_subset.csv
- **Full name**: Current sweep experiment (2024-09-27)
- **Purpose**: Out-of-distribution (OOD) evaluation
- **Description**: Systematic current sweep to characterize stack performance across operating range
- **Use case**: Testing model generalization to different current densities

### test3_subset.csv
- **Full name**: Pressure swap experiment (2024-09-30)
- **Purpose**: Out-of-distribution (OOD) evaluation
- **Description**: Pressure variation experiment with cathode/anode pressure swaps
- **Use case**: Testing model generalization to different pressure conditions

### test4_subset.csv
- **Full name**: Long-term stability test (2024-10-16)
- **Purpose**: Training data
- **Description**: Extended operation test capturing degradation and dynamic behavior
- **Use case**: Primary dataset for training physics-informed neural network models

## Variables

Each CSV file contains 5 columns with the following physical quantities:

| Column | Description | Units | Physical Quantity |
|--------|-------------|-------|-------------------|
| `PS-I-MON` | Stack current | A | Total current through stack |
| `H-P1` | Hydrogen pressure | bar | Cathode side pressure |
| `O-P1` | Oxygen pressure | bar | Anode side pressure |
| `T-ELY-CH1` | Stack temperature | °C | Operating temperature (channel 1) |
| `CV-mean` | Mean cell voltage | V | Average voltage across all cells |

## Data Characteristics

- **Format**: Comma-separated values (CSV) with header row
- **Temporal resolution**: ~1 Hz (varies slightly by test)
- **File sizes**: ~2-10 MB per test (vs. ~40-230 MB for full data files)
- **Missing values**: None (cleaned data)
- **Coordinate system**: Time series (implicit index)

## Regenerating Subsets

If you have access to the full NORCE data files, you can regenerate these CSV subsets using:

```bash
python extract_data.py
```

The extraction script expects the full data files to be located at:
```
data/test2/2024_09_27_V8_VF_CF_test2_data.txt
data/test3/2024_09_30_V8_VF_CF_test3_data.txt
data/test4/2024_10_16_V8_VF_CF_test4_data.txt
```

The script will:
1. Read the semicolon-delimited source files
2. Extract the 5 key columns
3. Save as comma-separated CSV files
4. Report file sizes and compression ratios

## Usage Notes

### For Model Training
```python
import pandas as pd

# Load training data
df_train = pd.read_csv('dataset/test4_subset.csv')

# Extract features and targets
current = df_train['PS-I-MON'].values
pressure_h = df_train['H-P1'].values
pressure_o = df_train['O-P1'].values
temperature = df_train['T-ELY-CH1'].values
voltage = df_train['CV-mean'].values
```

### For OOD Evaluation
```python
# Load OOD test data
df_test2 = pd.read_csv('dataset/test2_subset.csv')  # Current sweep
df_test3 = pd.read_csv('dataset/test3_subset.csv')  # Pressure swap
```

### Important Constraints
- **Fixed parameters**: R_ohm = 0.15 Ω·cm² (from EIS measurements, NOT in data)
- **Active area**: 50 cm² (for current density calculations)
- **NO data leakage**: Keep train/test datasets completely separate
- **NO synthetic data**: Only use real NORCE experimental data

## Data Integrity

These subsets are extracted from the authoritative NORCE data files. If you need:
- Additional variables (e.g., flow rates, other temperatures)
- Raw sensor data (before processing)
- Full experimental metadata

Please refer to the original files in `/data/test{2,3,4}/` or contact the NORCE team.

## References

For detailed information about the experimental setup and data collection:
- See `references/290905_transcriptions.txt` for expert knowledge on parameter behavior
- See `README.md` (project root) for overall project architecture
- See `RULES.md` for data handling constraints and requirements

---

**Last updated**: 2026-01-27
**Contact**: NAIC Project Team
