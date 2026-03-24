"""
GPU-accelerated LBM solver using CuPy.

Falls back to NumPy CPU implementation if GPU is unavailable.

Environment note: In the EESSI environment, CUDA shared libraries are not on
LD_LIBRARY_PATH by default. This module pre-loads them via ctypes before importing
CuPy, so that the import succeeds transparently.
"""

import numpy as np

# Pre-load CUDA shared libraries so CuPy can find them in the EESSI environment.
# These ctypes calls are no-ops if the libraries are already loaded or unavailable;
# they never raise on failure.
_CUDA_LIBS = [
    "/lib/x86_64-linux-gnu/libcuda.so.1",
    "/lib/x86_64-linux-gnu/libcudart.so.12",
    "/lib/x86_64-linux-gnu/libcublasLt.so.12",
    "/lib/x86_64-linux-gnu/libcublas.so.12",
    "/lib/x86_64-linux-gnu/libcufft.so.11",
    "/lib/x86_64-linux-gnu/libcurand.so.10",
    "/lib/x86_64-linux-gnu/libcusolver.so.11",
    "/lib/x86_64-linux-gnu/libcusparse.so.12",
    "/lib/x86_64-linux-gnu/libnvrtc-builtins.so.12.0",
    "/lib/x86_64-linux-gnu/libnvrtc.so.12",
]

import ctypes as _ctypes

for _lib_path in _CUDA_LIBS:
    try:
        _ctypes.CDLL(_lib_path)
    except OSError:
        pass  # Library not present; CuPy import will handle the error

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

from backend.lbm_solver import LBMSolver


class LBMSolverGPU(LBMSolver):
    """
    GPU-accelerated LBM solver using CuPy.

    Uses CuPy array operations (identical API to NumPy) for all compute-heavy
    steps. Falls back transparently to NumPy when use_gpu=False or GPU is
    unavailable.

    All methods returning arrays always return plain NumPy arrays so that the
    rest of the server code (which expects NumPy) works without modification.
    """

    def __init__(self, grid_size: int = 50, use_gpu: bool = True):
        """
        Initialize GPU-accelerated LBM solver.

        Args:
            grid_size: Grid resolution (default 50)
            use_gpu: Use GPU if available (default True)
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE

        if self.use_gpu:
            self.xp = cp
            print("LBM using GPU acceleration (CuPy)")
        else:
            self.xp = np
            if use_gpu and not GPU_AVAILABLE:
                print("GPU requested but CuPy not available, using CPU")

        # Parent __init__ allocates NumPy arrays and calls _compute_equilibrium().
        # Because _compute_equilibrium() is overridden and uses self.xp, we must
        # set self.xp BEFORE calling super().__init__().
        super().__init__(grid_size)

        # Move arrays to GPU if using CuPy
        if self.use_gpu:
            self._move_to_gpu()

    # ------------------------------------------------------------------
    # GPU memory management
    # ------------------------------------------------------------------

    def _move_to_gpu(self):
        """Move all LBM arrays from CPU (NumPy) to GPU (CuPy)."""
        self.c = cp.asarray(self.c)
        self.w = cp.asarray(self.w)
        self.f = cp.asarray(self.f)
        self.f_eq = cp.asarray(self.f_eq)
        self.rho = cp.asarray(self.rho)
        self.u = cp.asarray(self.u)

    # ------------------------------------------------------------------
    # Core LBM steps — override to use self.xp instead of hard-coded np
    # ------------------------------------------------------------------

    def _compute_equilibrium(self):
        """Compute equilibrium distribution using self.xp (GPU or CPU).

        This method is called both from the parent __init__ (when arrays are
        still NumPy) and after _move_to_gpu (when they are CuPy). We detect
        the current state by checking the type of self.f and use the matching
        array library rather than always self.xp, so the code is safe in both
        phases.
        """
        # Determine which library owns the current arrays.
        # During parent __init__, self.f is always NumPy.
        # After _move_to_gpu, self.f is CuPy.
        if GPU_AVAILABLE and isinstance(self.f, cp.ndarray):
            xp = cp
        else:
            xp = np

        self.f_eq = xp.zeros_like(self.f)

        # Ensure all operands are on the same device as f.
        # This guards against partial transfer state (e.g. mid-_move_to_gpu)
        # and the init phase where f is NumPy but self.xp is already cp.
        _same_device = lambda arr: isinstance(arr, type(self.f))
        c = self.c if _same_device(self.c) else xp.asarray(self.c)
        w = self.w if _same_device(self.w) else xp.asarray(self.w)
        rho = self.rho if _same_device(self.rho) else xp.asarray(self.rho)
        u = self.u if _same_device(self.u) else xp.asarray(self.u)

        for i in range(9):
            cu = xp.sum(c[i] * u, axis=2)   # c[i] · u  (grid, grid)
            usq = xp.sum(u ** 2, axis=2)     # |u|²

            self.f_eq[:, :, i] = w[i] * rho * (
                1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * usq
            )

    def collision_step(self, tau: float = 1.0):
        """BGK collision using self.xp."""
        xp = self.xp

        # self.c may still be NumPy if called before _move_to_gpu (shouldn't
        # normally happen, but guard defensively).
        c = xp.asarray(self.c) if not isinstance(self.c, type(self.f)) else self.c

        self.rho = xp.sum(self.f, axis=2)
        self.u[:, :, 0] = xp.sum(self.f * c[:, 0], axis=2) / self.rho
        self.u[:, :, 1] = xp.sum(self.f * c[:, 1], axis=2) / self.rho

        self._compute_equilibrium()

        self.f += -(self.f - self.f_eq) / tau

    def streaming_step(self):
        """Streaming step using self.xp."""
        xp = self.xp
        f_new = xp.zeros_like(self.f)

        for i in range(9):
            # .item() converts scalar CuPy/NumPy element to Python int safely
            cx = int(self.c[i, 0].item() if hasattr(self.c[i, 0], 'item') else self.c[i, 0])
            cy = int(self.c[i, 1].item() if hasattr(self.c[i, 1], 'item') else self.c[i, 1])
            f_new[:, :, i] = xp.roll(
                xp.roll(self.f[:, :, i], cx, axis=0), cy, axis=1
            )

        self.f = f_new

    def apply_boundaries(self):
        """
        Boundary conditions.

        The parent implementation uses `.copy()` on array slices, which works
        identically for both NumPy and CuPy, so we call super() directly.
        The only issue is _compute_equilibrium() being called inside — that is
        already overridden to use self.xp, so no further work needed here.
        """
        super().apply_boundaries()

    def add_source_terms(
        self,
        temperatures: np.ndarray = None,
        void_fraction: np.ndarray = None,
    ):
        """
        Add buoyancy and drag source terms.

        The parent implementation contains a Python resize loop (for
        temperatures) and uses np.newaxis. We accept NumPy inputs (as always
        provided by the server), convert to the appropriate array type, run the
        logic on GPU/CPU, and keep results on device.
        """
        xp = self.xp

        if temperatures is not None:
            beta = 3e-4
            g = 9.81
            T_ref = 80.0

            # Resize to grid if needed (using simple nearest-neighbour)
            if temperatures.shape != (self.grid_size, self.grid_size):
                temps_resized = np.zeros((self.grid_size, self.grid_size))
                scale_factor = self.grid_size / temperatures.shape[0]

                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        i_src = min(
                            int(i / scale_factor), temperatures.shape[0] - 1
                        )
                        j_src = min(
                            int(j / scale_factor), temperatures.shape[1] - 1
                        )
                        temps_resized[i, j] = temperatures[i_src, j_src]

                temperatures = temps_resized

            # Move to device
            temperatures_xp = xp.asarray(temperatures)
            delta_T = temperatures_xp - T_ref
            buoyancy_force = beta * g * delta_T
            self.u[:, :, 1] += buoyancy_force * 0.001

        if void_fraction is not None:
            void_fraction_xp = xp.asarray(void_fraction)
            drag_factor = 1.0 - 0.5 * void_fraction_xp
            self.u *= drag_factor[:, :, xp.newaxis]

    # ------------------------------------------------------------------
    # Output helpers — always return NumPy arrays for server compatibility
    # ------------------------------------------------------------------

    def compute_vertical_velocity(self) -> np.ndarray:
        """Compute vz from continuity — GPU-accelerated, returns NumPy."""
        xp = self.xp

        dux_dx = xp.zeros_like(self.u[:, :, 0])
        duy_dy = xp.zeros_like(self.u[:, :, 1])

        # Central differences (interior)
        dux_dx[1:-1, :] = (self.u[2:, :, 0] - self.u[:-2, :, 0]) / 2.0
        duy_dy[:, 1:-1] = (self.u[:, 2:, 1] - self.u[:, :-2, 1]) / 2.0

        # Forward/backward differences at boundaries
        dux_dx[0, :] = self.u[1, :, 0] - self.u[0, :, 0]
        dux_dx[-1, :] = self.u[-1, :, 0] - self.u[-2, :, 0]
        duy_dy[:, 0] = self.u[:, 1, 1] - self.u[:, 0, 1]
        duy_dy[:, -1] = self.u[:, -1, 1] - self.u[:, -2, 1]

        H_layer = 0.01
        vz = -(dux_dx + duy_dy) * H_layer

        if self.use_gpu:
            return cp.asnumpy(vz)
        return vz

    def get_velocity_field(self) -> np.ndarray:
        """
        Get 3D velocity field [vx, vy, vz].

        Returns:
            velocities: (grid_size, grid_size, 3) NumPy array
        """
        velocities = np.zeros((self.grid_size, self.grid_size, 3))

        u_host = cp.asnumpy(self.u) if self.use_gpu else self.u
        velocities[:, :, 0] = u_host[:, :, 0]
        velocities[:, :, 1] = u_host[:, :, 1]
        velocities[:, :, 2] = self.compute_vertical_velocity()

        return velocities
