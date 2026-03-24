"""
Unit tests for digital_twin/backend/lbm_solver.py and lbm_gpu.py.

Tests LBMSolver (CPU) and LBMSolverGPU (CPU mode, no CuPy required):
initialization, D2Q9 weights, collision, streaming, boundaries, source
terms, velocity field extraction, and numerical properties.
"""
import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'digital_twin'))

from backend.lbm_solver import LBMSolver
from backend.lbm_gpu import LBMSolverGPU


# ---------------------------------------------------------------------------
# LBMSolver — initialization
# ---------------------------------------------------------------------------

class TestLBMSolverInit:
    def test_grid_size_stored(self):
        s = LBMSolver(grid_size=20)
        assert s.grid_size == 20

    def test_default_grid_size(self):
        s = LBMSolver()
        assert s.grid_size == 50

    def test_distribution_function_shape(self):
        s = LBMSolver(grid_size=10)
        assert s.f.shape == (10, 10, 9)

    def test_density_shape(self):
        s = LBMSolver(grid_size=10)
        assert s.rho.shape == (10, 10)

    def test_velocity_shape(self):
        s = LBMSolver(grid_size=10)
        assert s.u.shape == (10, 10, 2)

    def test_d2q9_weights_sum_to_one(self):
        s = LBMSolver()
        assert s.w.sum() == pytest.approx(1.0, abs=1e-10)

    def test_d2q9_lattice_directions(self):
        s = LBMSolver()
        assert s.c.shape == (9, 2)

    def test_initial_velocity_is_zero(self):
        s = LBMSolver(grid_size=10)
        np.testing.assert_allclose(s.u, 0.0)

    def test_initial_density_is_one(self):
        s = LBMSolver(grid_size=10)
        np.testing.assert_allclose(s.rho, 1.0)

    def test_initial_equilibrium_equals_f(self):
        s = LBMSolver(grid_size=10)
        np.testing.assert_allclose(s.f, s.f_eq, atol=1e-12)


# ---------------------------------------------------------------------------
# LBMSolver — collision step
# ---------------------------------------------------------------------------

class TestCollisionStep:
    def test_f_shape_preserved(self):
        s = LBMSolver(grid_size=10)
        s.collision_step(tau=1.0)
        assert s.f.shape == (10, 10, 9)

    def test_tau_one_relaxes_to_equilibrium(self):
        """With tau=1, f should equal f_eq after collision."""
        s = LBMSolver(grid_size=10)
        s.collision_step(tau=1.0)
        np.testing.assert_allclose(s.f, s.f_eq, atol=1e-12)

    def test_density_conserved_after_collision(self):
        """Total mass (sum of f over velocity directions) is conserved."""
        s = LBMSolver(grid_size=10)
        mass_before = s.f.sum()
        s.collision_step(tau=1.0)
        mass_after = s.f.sum()
        assert mass_before == pytest.approx(mass_after, rel=1e-6)

    def test_rho_updated(self):
        s = LBMSolver(grid_size=10)
        s.f[:, :, 0] += 0.1
        s.collision_step(tau=1.0)
        assert s.rho.mean() > 1.0


# ---------------------------------------------------------------------------
# LBMSolver — streaming step
# ---------------------------------------------------------------------------

class TestStreamingStep:
    def test_f_shape_preserved(self):
        s = LBMSolver(grid_size=10)
        s.streaming_step()
        assert s.f.shape == (10, 10, 9)

    def test_total_population_conserved(self):
        """Streaming is a permutation — total f must be unchanged."""
        s = LBMSolver(grid_size=10)
        total_before = s.f.sum()
        s.streaming_step()
        total_after = s.f.sum()
        assert total_before == pytest.approx(total_after, rel=1e-9)


# ---------------------------------------------------------------------------
# LBMSolver — apply_boundaries
# ---------------------------------------------------------------------------

class TestApplyBoundaries:
    def test_shapes_preserved(self):
        s = LBMSolver(grid_size=10)
        s.apply_boundaries()
        assert s.f.shape == (10, 10, 9)
        assert s.u.shape == (10, 10, 2)

    def test_inlet_velocity_set_at_top(self):
        s = LBMSolver(grid_size=10)
        s.apply_boundaries()
        # Inlet at top (y = grid_size - 1) sets downward velocity
        assert s.u[:, -1, 1] == pytest.approx(-0.01)

    def test_inlet_density_set_to_one(self):
        s = LBMSolver(grid_size=10)
        s.apply_boundaries()
        np.testing.assert_allclose(s.rho[:, -1], 1.0)


# ---------------------------------------------------------------------------
# LBMSolver — add_source_terms
# ---------------------------------------------------------------------------

class TestAddSourceTerms:
    def test_no_args_no_crash(self):
        s = LBMSolver(grid_size=10)
        s.add_source_terms()  # must not raise

    def test_temperature_modifies_vertical_velocity(self):
        s = LBMSolver(grid_size=10)
        u_before = s.u[:, :, 1].copy()
        temps = np.full((10, 10), 90.0)  # above T_ref (80°C) -> upward buoyancy
        s.add_source_terms(temperatures=temps)
        assert not np.allclose(s.u[:, :, 1], u_before)

    def test_cold_temperature_reduces_vertical_velocity(self):
        s = LBMSolver(grid_size=10)
        u_before = s.u[:, :, 1].copy()
        temps = np.full((10, 10), 70.0)  # below T_ref -> downward buoyancy
        s.add_source_terms(temperatures=temps)
        assert (s.u[:, :, 1] <= u_before).all()

    def test_temperature_resize_when_shape_differs(self):
        """Temperatures on a 5x5 grid must be resized to match the solver grid."""
        s = LBMSolver(grid_size=10)
        temps_small = np.full((5, 5), 85.0)
        s.add_source_terms(temperatures=temps_small)  # must not raise
        assert s.u.shape == (10, 10, 2)

    def test_void_fraction_reduces_velocity(self):
        """Non-zero void fraction must scale velocities down."""
        s = LBMSolver(grid_size=10)
        s.u[:] = 1.0
        vf = np.full((10, 10), 0.5)
        s.add_source_terms(void_fraction=vf)
        # drag_factor = 1 - 0.5*0.5 = 0.75
        np.testing.assert_allclose(s.u, 0.75, atol=1e-10)

    def test_zero_void_fraction_no_change(self):
        s = LBMSolver(grid_size=10)
        s.u[:] = 0.5
        vf = np.zeros((10, 10))
        s.add_source_terms(void_fraction=vf)
        np.testing.assert_allclose(s.u, 0.5, atol=1e-10)


# ---------------------------------------------------------------------------
# LBMSolver — step (full cycle)
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_does_not_raise(self):
        s = LBMSolver(grid_size=10)
        s.step(tau=1.0)

    def test_multiple_steps_stable(self):
        s = LBMSolver(grid_size=20)
        for _ in range(10):
            s.step(tau=1.0)
        assert np.isfinite(s.f).all()
        assert np.isfinite(s.u).all()

    def test_step_with_temperatures(self):
        s = LBMSolver(grid_size=10)
        temps = np.full((10, 10), 80.0)
        s.step(tau=1.0, temperatures=temps)
        assert np.isfinite(s.u).all()

    def test_step_with_void_fraction(self):
        s = LBMSolver(grid_size=10)
        vf = np.full((10, 10), 0.1)
        s.step(tau=1.0, void_fraction=vf)
        assert np.isfinite(s.u).all()


# ---------------------------------------------------------------------------
# LBMSolver — velocity field extraction
# ---------------------------------------------------------------------------

class TestVelocityField:
    def test_compute_vertical_velocity_shape(self):
        s = LBMSolver(grid_size=10)
        vz = s.compute_vertical_velocity()
        assert vz.shape == (10, 10)

    def test_get_velocity_field_shape(self):
        s = LBMSolver(grid_size=10)
        v = s.get_velocity_field()
        assert v.shape == (10, 10, 3)

    def test_get_velocity_field_channels(self):
        """First two channels come from u; third from continuity."""
        s = LBMSolver(grid_size=10)
        s.collision_step()
        v = s.get_velocity_field()
        np.testing.assert_array_equal(v[:, :, 0], s.u[:, :, 0])
        np.testing.assert_array_equal(v[:, :, 1], s.u[:, :, 1])

    def test_vertical_velocity_finite(self):
        s = LBMSolver(grid_size=10)
        s.step(tau=1.0)
        vz = s.compute_vertical_velocity()
        assert np.isfinite(vz).all()


# ---------------------------------------------------------------------------
# LBMSolverGPU (CPU fallback mode)
# ---------------------------------------------------------------------------

class TestLBMSolverGPUCPUMode:
    @pytest.fixture
    def gpu_cpu(self):
        return LBMSolverGPU(grid_size=10, use_gpu=False)

    def test_not_using_gpu(self, gpu_cpu):
        assert gpu_cpu.use_gpu is False

    def test_grid_size_stored(self, gpu_cpu):
        assert gpu_cpu.grid_size == 10

    def test_step_executes(self, gpu_cpu):
        gpu_cpu.step(tau=1.0)
        assert np.isfinite(gpu_cpu.u).all()

    def test_get_velocity_field_returns_numpy(self, gpu_cpu):
        v = gpu_cpu.get_velocity_field()
        assert isinstance(v, np.ndarray)
        assert v.shape == (10, 10, 3)

    def test_compute_vertical_velocity_returns_numpy(self, gpu_cpu):
        vz = gpu_cpu.compute_vertical_velocity()
        assert isinstance(vz, np.ndarray)
        assert vz.shape == (10, 10)

    def test_multiple_steps_stable(self, gpu_cpu):
        for _ in range(5):
            gpu_cpu.step(tau=1.0)
        assert np.isfinite(gpu_cpu.f).all()

    def test_add_source_terms_temperature(self, gpu_cpu):
        temps = np.full((10, 10), 85.0)
        gpu_cpu.add_source_terms(temperatures=temps)
        assert np.isfinite(gpu_cpu.u).all()

    def test_add_source_terms_void_fraction(self, gpu_cpu):
        vf = np.full((10, 10), 0.2)
        gpu_cpu.add_source_terms(void_fraction=vf)
        assert np.isfinite(gpu_cpu.u).all()

    def test_add_source_terms_mismatched_temperature(self, gpu_cpu):
        """Temperature array smaller than grid must be resized without error."""
        temps_small = np.full((5, 5), 82.0)
        gpu_cpu.add_source_terms(temperatures=temps_small)  # must not raise

    def test_streaming_step_conserves_total(self, gpu_cpu):
        total_before = np.array(gpu_cpu.f).sum()
        gpu_cpu.streaming_step()
        total_after = np.array(gpu_cpu.f).sum()
        assert total_before == pytest.approx(total_after, rel=1e-9)
