"""
Unit tests for dataset/extract_data.py.

Tests extract_subset function: file I/O, column extraction, error
handling, and CSV output format.  Uses a temporary directory with
synthetic semicolon-delimited data so no real NORCE files are needed.
"""
import sys
import io
import pytest
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'dataset'))

from extract_data import extract_subset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def sample_txt(tmp_dir):
    """Write a semicolon-delimited text file with synthetic NORCE-style data."""
    data = pd.DataFrame({
        'PS-I-MON': [10.0, 20.0, 30.0],
        'H-P1': [15.0, 20.0, 25.0],
        'O-P1': [15.5, 20.5, 25.5],
        'T-ELY-CH1': [75.0, 77.0, 79.0],
        'CV-mean': [1.75, 1.78, 1.80],
        'EXTRA_COL': [1, 2, 3],
    })
    path = tmp_dir / 'test_data.txt'
    data.to_csv(path, sep=';', index=False)
    return path


REQUIRED_COLS = ['PS-I-MON', 'H-P1', 'O-P1', 'T-ELY-CH1', 'CV-mean']


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestExtractSubsetHappyPath:
    def test_creates_output_file(self, sample_txt, tmp_dir):
        out = tmp_dir / 'out.csv'
        extract_subset(sample_txt, out, REQUIRED_COLS)
        assert out.exists()

    def test_output_is_comma_separated(self, sample_txt, tmp_dir):
        out = tmp_dir / 'out.csv'
        extract_subset(sample_txt, out, REQUIRED_COLS)
        df = pd.read_csv(out, sep=',')
        assert len(df) == 3

    def test_output_contains_correct_columns(self, sample_txt, tmp_dir):
        out = tmp_dir / 'out.csv'
        extract_subset(sample_txt, out, REQUIRED_COLS)
        df = pd.read_csv(out)
        assert set(df.columns) == set(REQUIRED_COLS)

    def test_extra_column_excluded(self, sample_txt, tmp_dir):
        out = tmp_dir / 'out.csv'
        extract_subset(sample_txt, out, REQUIRED_COLS)
        df = pd.read_csv(out)
        assert 'EXTRA_COL' not in df.columns

    def test_values_preserved(self, sample_txt, tmp_dir):
        out = tmp_dir / 'out.csv'
        extract_subset(sample_txt, out, REQUIRED_COLS)
        df = pd.read_csv(out)
        assert df['PS-I-MON'].tolist() == [10.0, 20.0, 30.0]

    def test_row_count_matches_source(self, sample_txt, tmp_dir):
        out = tmp_dir / 'out.csv'
        extract_subset(sample_txt, out, REQUIRED_COLS)
        df = pd.read_csv(out)
        assert len(df) == 3


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestExtractSubsetErrors:
    def test_missing_source_file_no_raise(self, tmp_dir):
        """Function should print an error but not raise on missing source."""
        missing = tmp_dir / 'does_not_exist.txt'
        out = tmp_dir / 'out.csv'
        extract_subset(missing, out, REQUIRED_COLS)  # must not raise
        assert not out.exists()

    def test_missing_column_no_raise(self, sample_txt, tmp_dir):
        """Function should print an error but not raise for missing column."""
        out = tmp_dir / 'out.csv'
        extract_subset(sample_txt, out, REQUIRED_COLS + ['NON_EXISTENT_COL'])
        # Output should not have been created since columns are missing
        assert not out.exists()

    def test_subset_of_columns(self, sample_txt, tmp_dir):
        """Extracting a single column should work fine."""
        out = tmp_dir / 'out.csv'
        extract_subset(sample_txt, out, ['PS-I-MON'])
        df = pd.read_csv(out)
        assert list(df.columns) == ['PS-I-MON']
