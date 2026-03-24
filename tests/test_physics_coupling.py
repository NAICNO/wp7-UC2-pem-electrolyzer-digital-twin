"""
Unit tests for digital_twin/backend/physics_coupling.py.

Tests PhysicsCoupling: interpolation, velocity averaging, and full
update_coupling round-trip, including shape and value correctness.
"""
import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'digital_twin'))

from backend.physics_coupling import PhysicsCoupling


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def coupling():
    return PhysicsCoupling(cfd_grid_size=50, pinn_grid_size=10)


@pytest.fixture
def coupling_small():
    """Smaller grids for arithmetic convenience."""
    return PhysicsCoupling(cfd_grid_size=20, pinn_grid_size=4)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestPhysicsCouplingInit:
    def test_stores_cfd_grid_size(self):
        pc = PhysicsCoupling(cfd_grid_size=30, pinn_grid_size=6)
        assert pc.cfd_grid_size == 30

    def test_stores_pinn_grid_size(self):
        pc = PhysicsCoupling(cfd_grid_size=30, pinn_grid_size=6)
        assert pc.pinn_grid_size == 6


# ---------------------------------------------------------------------------
# interpolate_temperature_to_cfd
# ---------------------------------------------------------------------------

class TestInterpolateTemperatureToCFD:
    def test_output_shape(self, coupling):
        temps_pinn = np.random.uniform(70, 90, (10, 10))
        result = coupling.interpolate_temperature_to_cfd(temps_pinn)
        assert result.shape == (50, 50)

    def test_uniform_temperature_preserved(self, coupling):
        """Uniform PINN field must produce same uniform CFD field."""
        temps_pinn = np.full((10, 10), 82.5)
        result = coupling.interpolate_temperature_to_cfd(temps_pinn)
        np.testing.assert_allclose(result, 82.5)

    def test_output_within_input_range(self, coupling):
        temps_pinn = np.random.uniform(60.0, 100.0, (10, 10))
        result = coupling.interpolate_temperature_to_cfd(temps_pinn)
        assert result.min() >= 60.0
        assert result.max() <= 100.0

    def test_single_cell_coarse_grid(self):
        """1×1 pinn grid should broadcast to full CFD grid."""
        pc = PhysicsCoupling(cfd_grid_size=4, pinn_grid_size=1)
        temps_pinn = np.array([[77.0]])
        result = pc.interpolate_temperature_to_cfd(temps_pinn)
        assert result.shape == (4, 4)
        np.testing.assert_allclose(result, 77.0)

    def test_values_come_from_source(self, coupling_small):
        """Interpolated values must all be drawn from the source array."""
        temps_pinn = np.arange(16, dtype=float).reshape(4, 4)
        result = coupling_small.interpolate_temperature_to_cfd(temps_pinn)
        source_vals = set(temps_pinn.ravel())
        for val in result.ravel():
            assert val in source_vals


# ---------------------------------------------------------------------------
# compute_velocity_average_for_pinn
# ---------------------------------------------------------------------------

class TestComputeVelocityAverage:
    def test_output_shape(self, coupling):
        vels_cfd = np.random.randn(50, 50, 2)
        result = coupling.compute_velocity_average_for_pinn(vels_cfd)
        assert result.shape == (10, 10, 2)

    def test_uniform_velocity_preserved(self, coupling):
        """Averaging a constant field must return that constant."""
        vels_cfd = np.ones((50, 50, 2)) * 0.05
        result = coupling.compute_velocity_average_for_pinn(vels_cfd)
        np.testing.assert_allclose(result, 0.05, atol=1e-10)

    def test_zero_velocity_field(self, coupling):
        vels_cfd = np.zeros((50, 50, 2))
        result = coupling.compute_velocity_average_for_pinn(vels_cfd)
        np.testing.assert_allclose(result, 0.0)

    def test_two_component_output(self, coupling):
        """Third axis (velocity components) must stay size 2."""
        vels_cfd = np.random.randn(50, 50, 2)
        result = coupling.compute_velocity_average_for_pinn(vels_cfd)
        assert result.shape[2] == 2

    def test_averaging_reduces_magnitude(self, coupling_small):
        """Averaged field magnitude must be <= max of input."""
        vels_cfd = np.random.randn(20, 20, 2)
        result = coupling_small.compute_velocity_average_for_pinn(vels_cfd)
        assert np.abs(result).max() <= np.abs(vels_cfd).max() + 1e-10


# ---------------------------------------------------------------------------
# update_coupling
# ---------------------------------------------------------------------------

class TestUpdateCoupling:
    def test_returns_two_arrays(self, coupling):
        temps_pinn = np.full((10, 10), 80.0)
        vels_cfd = np.zeros((50, 50, 2))
        result = coupling.update_coupling(temps_pinn, vels_cfd)
        assert len(result) == 2

    def test_temps_cfd_shape(self, coupling):
        temps_pinn = np.full((10, 10), 80.0)
        vels_cfd = np.zeros((50, 50, 2))
        temps_cfd, _ = coupling.update_coupling(temps_pinn, vels_cfd)
        assert temps_cfd.shape == (50, 50)

    def test_vels_pinn_shape(self, coupling):
        temps_pinn = np.full((10, 10), 80.0)
        vels_cfd = np.zeros((50, 50, 2))
        _, vels_pinn = coupling.update_coupling(temps_pinn, vels_cfd)
        assert vels_pinn.shape == (10, 10, 2)

    def test_consistent_with_individual_methods(self, coupling):
        temps_pinn = np.random.uniform(70, 90, (10, 10))
        vels_cfd = np.random.randn(50, 50, 2) * 0.01
        temps_cfd_combined, vels_pinn_combined = coupling.update_coupling(temps_pinn, vels_cfd)
        temps_cfd_direct = coupling.interpolate_temperature_to_cfd(temps_pinn)
        vels_pinn_direct = coupling.compute_velocity_average_for_pinn(vels_cfd)
        np.testing.assert_array_equal(temps_cfd_combined, temps_cfd_direct)
        np.testing.assert_array_equal(vels_pinn_combined, vels_pinn_direct)
