import numpy as np

class LBMSolver:
    """
    Lattice Boltzmann Method solver using D2Q9 discretization.

    Velocity directions:
        6   2   5
          ↖ ↑ ↗
        3 ← 0 → 1
          ↙ ↓ ↘
        7   4   8
    """

    def __init__(self, grid_size: int = 50):
        self.grid_size = grid_size

        # D2Q9 lattice velocities
        self.c = np.array([
            [0, 0],    # 0: center
            [1, 0],    # 1: right
            [0, 1],    # 2: up
            [-1, 0],   # 3: left
            [0, -1],   # 4: down
            [1, 1],    # 5: right-up
            [-1, 1],   # 6: left-up
            [-1, -1],  # 7: left-down
            [1, -1]    # 8: right-down
        ])

        # D2Q9 weights
        self.w = np.array([
            4/9,       # 0: center
            1/9, 1/9, 1/9, 1/9,  # 1-4: axis
            1/36, 1/36, 1/36, 1/36  # 5-8: diagonals
        ])

        # Distribution function: f[x, y, direction]
        self.f = np.zeros((grid_size, grid_size, 9))

        # Macroscopic fields
        self.rho = np.ones((grid_size, grid_size))  # Density
        self.u = np.zeros((grid_size, grid_size, 2))  # Velocity [ux, uy]

        # Initialize to equilibrium
        self._compute_equilibrium()
        self.f = self.f_eq.copy()

    def _compute_equilibrium(self):
        """Compute equilibrium distribution function"""
        self.f_eq = np.zeros_like(self.f)

        for i in range(9):
            cu = np.sum(self.c[i] * self.u, axis=2)  # c · u
            usq = np.sum(self.u ** 2, axis=2)  # |u|²

            self.f_eq[:, :, i] = self.w[i] * self.rho * (
                1 + 3 * cu + 4.5 * cu**2 - 1.5 * usq
            )

    def collision_step(self, tau: float = 1.0):
        """
        BGK collision operator: relax toward equilibrium.

        Args:
            tau: Relaxation time (controls viscosity: ν = (tau - 0.5) / 3)
        """
        # Compute macroscopic density and velocity
        self.rho = np.sum(self.f, axis=2)
        self.u[:, :, 0] = np.sum(self.f * self.c[:, 0], axis=2) / self.rho
        self.u[:, :, 1] = np.sum(self.f * self.c[:, 1], axis=2) / self.rho

        # Compute equilibrium
        self._compute_equilibrium()

        # BGK collision: f_new = f - (f - f_eq) / tau
        self.f += -(self.f - self.f_eq) / tau

    def streaming_step(self):
        """
        Stream particles along lattice directions.
        Each f[:,:,i] propagates in direction c[i].
        """
        f_new = np.zeros_like(self.f)

        for i in range(9):
            cx, cy = self.c[i]

            # Roll distribution in direction (cx, cy)
            # axis=0 is x, axis=1 is y
            f_new[:, :, i] = np.roll(np.roll(self.f[:, :, i], cx, axis=0), cy, axis=1)

        self.f = f_new

    def apply_boundaries(self):
        """Apply boundary conditions: bounce-back walls, velocity inlet, pressure outlet"""
        # Bounce-back on left wall (x=0) - reverse x-direction velocities
        self.f[0, :, 1], self.f[0, :, 3] = self.f[0, :, 3].copy(), self.f[0, :, 1].copy()
        self.f[0, :, 5], self.f[0, :, 7] = self.f[0, :, 7].copy(), self.f[0, :, 5].copy()
        self.f[0, :, 8], self.f[0, :, 6] = self.f[0, :, 6].copy(), self.f[0, :, 8].copy()

        # Bounce-back on right wall (x=grid_size-1)
        self.f[-1, :, 1], self.f[-1, :, 3] = self.f[-1, :, 3].copy(), self.f[-1, :, 1].copy()
        self.f[-1, :, 5], self.f[-1, :, 7] = self.f[-1, :, 7].copy(), self.f[-1, :, 5].copy()
        self.f[-1, :, 8], self.f[-1, :, 6] = self.f[-1, :, 6].copy(), self.f[-1, :, 8].copy()

        # Velocity inlet at top (y=grid_size-1) - set downward velocity
        inlet_velocity = -0.01  # m/s downward
        self.u[:, -1, 1] = inlet_velocity
        self.rho[:, -1] = 1.0
        self._compute_equilibrium()
        self.f[:, -1, :] = self.f_eq[:, -1, :]

        # Pressure outlet at bottom (y=0) - zero-gradient
        self.f[:, 0, :] = self.f[:, 1, :]

    def add_source_terms(self, temperatures: np.ndarray = None, void_fraction: np.ndarray = None):
        """
        Add source terms to LBM: buoyancy, drag, and electroosmotic effects.

        Args:
            temperatures: Temperature field (Celsius) for buoyancy
            void_fraction: Gas void fraction for drag modification

        Note: Current implementation is simplified. Will be refined with proper
        physics coupling (Boussinesq buoyancy, bubble drag, electroosmotic flow).
        """
        if temperatures is not None:
            # Buoyancy force (Boussinesq approximation): F = β * g * ΔT
            # Simplified: apply upward force proportional to temperature
            beta = 3e-4  # Thermal expansion coefficient (1/K)
            g = 9.81  # Gravity (m/s²)
            T_ref = 80.0  # Reference temperature (°C)

            # Resize temperatures to match grid if needed
            if temperatures.shape != (self.grid_size, self.grid_size):
                # Simple nearest-neighbor interpolation without scipy
                temps_resized = np.zeros((self.grid_size, self.grid_size))
                scale_factor = self.grid_size / temperatures.shape[0]

                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        i_src = min(int(i / scale_factor), temperatures.shape[0] - 1)
                        j_src = min(int(j / scale_factor), temperatures.shape[1] - 1)
                        temps_resized[i, j] = temperatures[i_src, j_src]

                temperatures = temps_resized

            delta_T = temperatures - T_ref
            buoyancy_force = beta * g * delta_T

            # Apply force to velocity field (y-direction is vertical)
            self.u[:, :, 1] += buoyancy_force * 0.001  # Small timestep dt

        if void_fraction is not None:
            # Drag reduction due to gas bubbles (simplified)
            # Higher void fraction → lower effective viscosity
            drag_factor = 1.0 - 0.5 * void_fraction
            self.u *= drag_factor[:, :, np.newaxis]

    def step(self, tau: float = 1.0, temperatures: np.ndarray = None, void_fraction: np.ndarray = None):
        """
        Full LBM step: collision + streaming + boundaries + source terms

        Args:
            tau: Relaxation time parameter
            temperatures: Optional temperature field for buoyancy
            void_fraction: Optional void fraction for drag
        """
        self.collision_step(tau)
        self.streaming_step()
        self.apply_boundaries()
        self.add_source_terms(temperatures, void_fraction)

    def compute_vertical_velocity(self) -> np.ndarray:
        """
        Compute vertical velocity vz from continuity equation.

        For incompressible flow: ∂ux/∂x + ∂uy/∂y + ∂uz/∂z = 0

        Assuming constant layer thickness dz, we integrate:
        vz = -∫(∂ux/∂x + ∂uy/∂y) dz

        Returns:
            vz: (grid_size, grid_size) vertical velocity component
        """
        # Compute horizontal divergence using central differences
        dux_dx = np.zeros_like(self.u[:, :, 0])
        duy_dy = np.zeros_like(self.u[:, :, 1])

        # Central differences (2nd order accurate)
        dux_dx[1:-1, :] = (self.u[2:, :, 0] - self.u[:-2, :, 0]) / 2.0
        duy_dy[:, 1:-1] = (self.u[:, 2:, 1] - self.u[:, :-2, 1]) / 2.0

        # Boundaries: forward/backward differences
        dux_dx[0, :] = self.u[1, :, 0] - self.u[0, :, 0]
        dux_dx[-1, :] = self.u[-1, :, 0] - self.u[-2, :, 0]
        duy_dy[:, 0] = self.u[:, 1, 1] - self.u[:, 0, 1]
        duy_dy[:, -1] = self.u[:, -1, 1] - self.u[:, -2, 1]

        # Vertical velocity from continuity (assuming layer height H = 0.01 m)
        H_layer = 0.01  # meters
        vz = -(dux_dx + duy_dy) * H_layer

        return vz

    def get_velocity_field(self) -> np.ndarray:
        """
        Get 3D velocity field [vx, vy, vz] from 2D LBM + continuity.

        Returns:
            velocities: (grid_size, grid_size, 3) array
        """
        velocities = np.zeros((self.grid_size, self.grid_size, 3))
        velocities[:, :, 0] = self.u[:, :, 0]  # vx from LBM
        velocities[:, :, 1] = self.u[:, :, 1]  # vy from LBM
        velocities[:, :, 2] = self.compute_vertical_velocity()  # vz from continuity

        return velocities

