# backend/mock_pinn.py
"""
⚠️ DEPRECATED: This module is deprecated and will be removed in a future version.

Use backend.pinn_loader.PINNLoader instead for real PINN-based temperature predictions.

This mock implementation generates synthetic temperature fields and should only be
used for testing when the trained model is unavailable.
"""
import warnings
import numpy as np

warnings.warn(
    "backend.mock_pinn is deprecated. Use backend.pinn_loader.PINNLoader instead.",
    DeprecationWarning,
    stacklevel=2
)

def generate_mock_temperatures(grid_size: int = 10, current: float = 20.0, temperature: float = 80.0) -> np.ndarray:
    """
    Generate mock temperature field for testing physics coupling.

    Args:
        grid_size: Resolution of temperature field (default 10x10, lower than CFD grid)
        current: Operating current in Amps
        temperature: Operating temperature in Celsius

    Returns:
        temperatures: (grid_size, grid_size) array in Celsius
    """
    temperatures = np.zeros((grid_size, grid_size))

    # Base temperature from operating point
    base_temp = temperature

    # Add ohmic heating (proportional to I²R)
    ohmic_heating = 0.1 * (current / 20.0) ** 2

    # Add spatial variation (hotter in center due to current concentration)
    for i in range(grid_size):
        for j in range(grid_size):
            x = i / grid_size
            y = j / grid_size

            # Gaussian hotspot in center
            spatial_factor = np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)
            temperatures[i, j] = base_temp + ohmic_heating * spatial_factor * 5.0

    return temperatures
