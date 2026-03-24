"""
PINN Model Loader for Electrochemistry Prediction.

Loads trained PhysicsOriginal12Param model and provides inference
for electrochemical voltage prediction and spatial temperature field generation.

The model is 0D (predicts cell voltage from global operating conditions),
but generates spatial temperature fields by distributing ohmic heating
using a physics-based Gaussian pattern.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path for PINN model imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from models.physics_original_12param import PhysicsOriginal12Param


class PINNLoader:
    """
    Load and run inference with trained PINN model for electrochemistry prediction.

    The PINN predicts cell voltage from operating conditions (I, T, P), then
    generates spatial temperature fields by distributing ohmic heating using
    a Gaussian hotspot pattern (hotter in center where current density is higher).

    This bridges the 0D electrochemistry model with the spatial visualization
    needs of the digital twin.
    """

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Load trained PINN model.

        Args:
            model_path: Path to .pt checkpoint file (relative to project root)
            device: 'cpu' or 'cuda'

        Raises:
            FileNotFoundError: If model checkpoint doesn't exist
        """
        self.device = device

        # Handle both absolute and relative paths
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path

        self.model_path = model_path

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load model architecture
        self.model = PhysicsOriginal12Param(device=device)

        # Load trained weights
        checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print(f"✓ Loaded PINN model from {self.model_path.name}")

    def predict_temperatures(
        self,
        current: float,
        temperature: float,
        pressure: float,
        grid_size: int = 10
    ) -> np.ndarray:
        """
        Predict spatial temperature field for given operating conditions.

        Uses 0D PINN to predict voltage, calculates ohmic heating (P = I*V),
        then distributes heating spatially using Gaussian hotspot pattern.

        Args:
            current: Operating current (A)
            temperature: Operating temperature (°C)
            pressure: Total operating pressure (bar)
            grid_size: Resolution of output grid (default 10x10)

        Returns:
            temperatures: (grid_size, grid_size) array of temperatures in °C
                         Center is hotter than edges (Gaussian distribution)
        """
        with torch.no_grad():
            # PINN expects: (current, H2_pressure, O2_pressure, temperature)
            # Assume total pressure splits 50/50 for H2/O2
            H2_pressure = pressure / 2.0
            O2_pressure = pressure / 2.0

            # Create input tensor (batch size 1)
            current_tensor = torch.tensor([current], dtype=torch.float32, device=self.device)
            H2_pressure_tensor = torch.tensor([H2_pressure], dtype=torch.float32, device=self.device)
            O2_pressure_tensor = torch.tensor([O2_pressure], dtype=torch.float32, device=self.device)
            temp_tensor = torch.tensor([temperature], dtype=torch.float32, device=self.device)

            # Run PINN inference
            voltage, physics_params = self.model(
                current=current_tensor,
                H2_pressure=H2_pressure_tensor,
                O2_pressure=O2_pressure_tensor,
                temperature=temp_tensor
            )

            voltage = voltage.item()

        # Calculate ohmic heating: P = I * V
        power = current * voltage  # Watts

        # Generate spatial temperature field with Gaussian hotspot
        temps = np.zeros((grid_size, grid_size))

        for i in range(grid_size):
            for j in range(grid_size):
                # Normalized coordinates [0, 1]
                x = i / (grid_size - 1) if grid_size > 1 else 0.5
                y = j / (grid_size - 1) if grid_size > 1 else 0.5

                # Gaussian hotspot in center (current concentration)
                # Higher current density in center -> more heating
                spatial_factor = np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)

                # Temperature rise from ohmic heating
                # k_thermal converts W to °C (empirical coefficient)
                k_thermal = 0.5  # °C/W
                temp_rise = k_thermal * power * spatial_factor

                temps[i, j] = temperature + temp_rise

        return temps

    def predict_voltage(
        self,
        current: float,
        temperature: float,
        pressure: float
    ) -> float:
        """
        Predict cell voltage for given operating conditions (0D prediction).

        Args:
            current: Operating current (A)
            temperature: Operating temperature (°C)
            pressure: Total operating pressure (bar)

        Returns:
            voltage: Predicted cell voltage (V)
        """
        with torch.no_grad():
            # Split pressure 50/50 for H2/O2
            H2_pressure = pressure / 2.0
            O2_pressure = pressure / 2.0

            # Create input tensors
            current_tensor = torch.tensor([current], dtype=torch.float32, device=self.device)
            H2_pressure_tensor = torch.tensor([H2_pressure], dtype=torch.float32, device=self.device)
            O2_pressure_tensor = torch.tensor([O2_pressure], dtype=torch.float32, device=self.device)
            temp_tensor = torch.tensor([temperature], dtype=torch.float32, device=self.device)

            # Run inference
            voltage, physics_params = self.model(
                current=current_tensor,
                H2_pressure=H2_pressure_tensor,
                O2_pressure=O2_pressure_tensor,
                temperature=temp_tensor
            )

            return voltage.item()

    def compute_cell_voltages(
        self,
        current: float,
        temperature: float,
        pressure: float,
        cell_modifiers: list,
        cell_area: float = 50.0,
        R_ohm_base: float = 0.15,
        V_thermoneutral: float = 1.48,
    ) -> list:
        """
        Compute per-cell voltage, power, and efficiency for a multi-cell stack.

        Uses the 0D PINN to get the baseline healthy-cell voltage, then adds
        extra ohmic loss for each degraded cell based on its R_ohm_modifier.

        Args:
            current: Stack current in A (same for all cells — series connection)
            temperature: Inlet temperature in °C
            pressure: Operating pressure in bar
            cell_modifiers: List of dicts with 'R_ohm_modifier' and 'membraneHealth'
            cell_area: Active cell area in cm² (default 50 — project spec)
            R_ohm_base: Baseline ohmic resistance in Ω·cm² (default 0.15 — from EIS)
            V_thermoneutral: Thermoneutral voltage for efficiency calc (default 1.48 V)

        Returns:
            List of dicts: [{voltage, current, power, efficiency, membraneHealth}, ...]
        """
        if cell_area <= 0:
            raise ValueError(f"cell_area must be positive, got {cell_area}")

        V_base = self.predict_voltage(current, temperature, pressure)
        j = current / cell_area  # A/cm²

        results = []
        for modifier in cell_modifiers:
            R_mod = modifier.get('R_ohm_modifier', 1.0)
            health = modifier.get('membraneHealth', 1.0)

            extra_ohmic = j * R_ohm_base * (R_mod - 1.0)
            V_cell = V_base + extra_ohmic

            if V_cell <= 0:
                raise ValueError(
                    f"Cell voltage is non-physical ({V_cell:.4f} V). "
                    f"Check R_ohm_modifier={R_mod}."
                )

            power = current * V_cell
            efficiency = V_thermoneutral / V_cell if V_cell > 0 else 0.0

            results.append({
                'voltage': round(V_cell, 4),
                'current': round(current, 4),
                'power': round(power, 4),
                'efficiency': round(efficiency, 4),
                'membraneHealth': round(health, 4),
            })

        return results
