"""
Unit tests for digital_twin/backend/mock_pinn.py.

Tests generate_mock_temperatures: output shape, value ranges, physical
plausibility, and boundary conditions.  The deprecation warning is
expected and suppressed in these tests.
"""
import sys
import warnings
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'digital_twin'))

# Import while suppressing the expected DeprecationWarning
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    from backend.mock_pinn import generate_mock_temperatures


# ---------------------------------------------------------------------------
# Deprecation warning
# ---------------------------------------------------------------------------

class TestDeprecationWarning:
    def test_module_emits_deprecation_warning(self):
        """Re-importing must surface a DeprecationWarning (stacklevel enforced)."""
        # The warning is issued at module import time.  We verify it was raised
        # by checking the recwarn fixture captures it on a fresh import.
        import importlib
        import backend.mock_pinn as _mod
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            importlib.reload(_mod)
        categories = [w.category for w in caught]
        assert DeprecationWarning in categories


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_default_grid_size_shape(self):
        t = generate_mock_temperatures()
        assert t.shape == (10, 10)

    def test_custom_grid_size(self):
        t = generate_mock_temperatures(grid_size=5)
        assert t.shape == (5, 5)

    def test_grid_size_one(self):
        t = generate_mock_temperatures(grid_size=1)
        assert t.shape == (1, 1)

    def test_large_grid(self):
        t = generate_mock_temperatures(grid_size=50)
        assert t.shape == (50, 50)


# ---------------------------------------------------------------------------
# Physical plausibility
# ---------------------------------------------------------------------------

class TestPhysicalPlausibility:
    def test_temperatures_at_least_base(self):
        base = 80.0
        t = generate_mock_temperatures(temperature=base)
        assert (t >= base).all(), "All temperatures should be >= base temperature"

    def test_center_hotter_than_corners(self):
        """Gaussian hotspot — centre should exceed all four corner pixels."""
        grid_size = 10
        t = generate_mock_temperatures(grid_size=grid_size, current=20.0, temperature=80.0)
        center = t[grid_size // 2, grid_size // 2]
        corners = [t[0, 0], t[0, -1], t[-1, 0], t[-1, -1]]
        assert all(center >= c for c in corners), \
            f"Centre {center:.2f} should be hottest, corners: {corners}"

    def test_higher_current_raises_temperature(self):
        """More current (higher I^2*R) should produce a hotter field."""
        t_low = generate_mock_temperatures(current=5.0, temperature=80.0)
        t_high = generate_mock_temperatures(current=40.0, temperature=80.0)
        assert t_high.max() > t_low.max()

    def test_higher_base_temperature_raises_all(self):
        t_cool = generate_mock_temperatures(temperature=60.0)
        t_warm = generate_mock_temperatures(temperature=90.0)
        assert (t_warm >= t_cool).all()

    def test_zero_current_no_ohmic_heating(self):
        """At I=0 ohmic_heating=0 so spatial_factor adds nothing."""
        t = generate_mock_temperatures(grid_size=10, current=0.0, temperature=75.0)
        np.testing.assert_allclose(t, 75.0, atol=1e-10)

    def test_returns_numpy_array(self):
        t = generate_mock_temperatures()
        assert isinstance(t, np.ndarray)

    def test_values_finite(self):
        t = generate_mock_temperatures(grid_size=10, current=20.0, temperature=80.0)
        assert np.isfinite(t).all()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_very_high_current(self):
        """Should not raise even for large currents."""
        t = generate_mock_temperatures(grid_size=5, current=1000.0, temperature=80.0)
        assert t.shape == (5, 5)
        assert np.isfinite(t).all()

    def test_negative_base_temperature_allowed(self):
        """Function does not guard against unphysical temperatures."""
        t = generate_mock_temperatures(grid_size=3, current=0.0, temperature=-10.0)
        assert t.shape == (3, 3)
