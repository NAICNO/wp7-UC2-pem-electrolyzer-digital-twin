# backend/physics_coupling.py
import numpy as np

class PhysicsCoupling:
    """
    Handles bidirectional coupling between LBM (CFD) and PINN (thermal).

    Current implementation is a placeholder - will be refined with proper
    interpolation, conservative remapping, and iterative convergence later.
    """

    def __init__(self, cfd_grid_size: int = 50, pinn_grid_size: int = 10):
        self.cfd_grid_size = cfd_grid_size
        self.pinn_grid_size = pinn_grid_size

    def interpolate_temperature_to_cfd(self, temperatures_pinn: np.ndarray) -> np.ndarray:
        """
        Interpolate coarse PINN temperature field to fine CFD grid.

        Args:
            temperatures_pinn: (pinn_grid_size, pinn_grid_size) temperature field

        Returns:
            temperatures_cfd: (cfd_grid_size, cfd_grid_size) interpolated field
        """
        # Simple nearest-neighbor interpolation (placeholder)
        scale_factor = self.cfd_grid_size / self.pinn_grid_size
        temperatures_cfd = np.zeros((self.cfd_grid_size, self.cfd_grid_size))

        for i in range(self.cfd_grid_size):
            for j in range(self.cfd_grid_size):
                # Map to coarse grid indices
                i_coarse = int(i / scale_factor)
                j_coarse = int(j / scale_factor)

                # Clamp to valid range
                i_coarse = min(i_coarse, self.pinn_grid_size - 1)
                j_coarse = min(j_coarse, self.pinn_grid_size - 1)

                temperatures_cfd[i, j] = temperatures_pinn[i_coarse, j_coarse]

        return temperatures_cfd

    def compute_velocity_average_for_pinn(self, velocities_cfd: np.ndarray) -> np.ndarray:
        """
        Average fine CFD velocity field to coarse PINN grid.

        Args:
            velocities_cfd: (cfd_grid_size, cfd_grid_size, 2) velocity field [u, v]

        Returns:
            velocities_pinn: (pinn_grid_size, pinn_grid_size, 2) averaged field
        """
        # Simple block averaging (placeholder)
        scale_factor = int(self.cfd_grid_size / self.pinn_grid_size)
        velocities_pinn = np.zeros((self.pinn_grid_size, self.pinn_grid_size, 2))

        for i in range(self.pinn_grid_size):
            for j in range(self.pinn_grid_size):
                # Extract block from fine grid
                i_start = i * scale_factor
                i_end = (i + 1) * scale_factor
                j_start = j * scale_factor
                j_end = (j + 1) * scale_factor

                # Average velocities in block
                velocities_pinn[i, j, :] = np.mean(
                    velocities_cfd[i_start:i_end, j_start:j_end, :],
                    axis=(0, 1)
                )

        return velocities_pinn

    def update_coupling(self, temperatures_pinn: np.ndarray, velocities_cfd: np.ndarray):
        """
        Perform one coupling iteration.

        Args:
            temperatures_pinn: Temperature field from PINN
            velocities_cfd: Velocity field from LBM

        Returns:
            temperatures_cfd: Interpolated temperatures for LBM
            velocities_pinn: Averaged velocities for PINN
        """
        temperatures_cfd = self.interpolate_temperature_to_cfd(temperatures_pinn)
        velocities_pinn = self.compute_velocity_average_for_pinn(velocities_cfd)

        return temperatures_cfd, velocities_pinn
