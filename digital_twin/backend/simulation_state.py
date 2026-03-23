# backend/simulation_state.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np

NUM_CELLS = 4

REQUIRED_KEYS = {'R_ohm_modifier', 'membraneHealth'}


def _default_cell_modifiers() -> List[Dict]:
    return [
        {'R_ohm_modifier': 1.0, 'membraneHealth': 1.0}
        for _ in range(NUM_CELLS)
    ]


@dataclass
class SimulationState:
    """Holds current simulation parameters and computed fields"""

    # Operating parameters
    current: float = 20.0  # Amperes
    temperature: float = 75.0  # Celsius
    pressure: float = 35.0  # bar

    # Simulation time
    time: float = 0.0  # seconds

    # Computed fields (initialized as None)
    velocities: Optional[np.ndarray] = None  # 50×50×3
    void_fractions: Optional[np.ndarray] = None  # 50×50
    temperatures: Optional[np.ndarray] = None  # 10×10 (from PINN)

    # Per-cell modifier tracking (4-cell SWVF stack)
    cell_modifiers: List[Dict] = field(default_factory=_default_cell_modifiers)

    def update_params(self, current: Optional[float] = None,
                     temperature: Optional[float] = None,
                     pressure: Optional[float] = None):
        """Update operating parameters"""
        if current is not None:
            self.current = current
        if temperature is not None:
            self.temperature = temperature
        if pressure is not None:
            self.pressure = pressure

    def update_cell_modifiers(self, modifiers: list) -> None:
        """Update per-cell modifiers. Must match number of cells."""
        if len(modifiers) != len(self.cell_modifiers):
            raise ValueError(
                f"Expected {len(self.cell_modifiers)} cell modifiers, got {len(modifiers)}"
            )
        for i, m in enumerate(modifiers):
            missing = REQUIRED_KEYS - m.keys()
            if missing:
                raise ValueError(f"Cell {i} modifier missing keys: {missing}")
        self.cell_modifiers = modifiers

    def to_dict(self):
        """Convert to JSON-serializable dict"""
        return {
            "current": self.current,
            "temperature": self.temperature,
            "pressure": self.pressure,
            "time": self.time,
            "velocities": self.velocities.tolist() if self.velocities is not None else None,
            "voidFractions": self.void_fractions.tolist() if self.void_fractions is not None else None,
            "temperatures": self.temperatures.tolist() if self.temperatures is not None else None,
        }
