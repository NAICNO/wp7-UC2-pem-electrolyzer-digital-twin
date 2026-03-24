"""
Unit tests for digital_twin/backend/simulation_state.py.

Tests SimulationState dataclass: default values, update_params,
update_cell_modifiers, to_dict, and all edge/error paths.
"""
import sys
import pytest
import numpy as np
from pathlib import Path

# Add digital_twin directory so 'backend.*' imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent / 'digital_twin'))

from backend.simulation_state import SimulationState, NUM_CELLS, REQUIRED_KEYS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _healthy_modifiers(n=NUM_CELLS):
    return [{'R_ohm_modifier': 1.0, 'membraneHealth': 1.0} for _ in range(n)]


# ---------------------------------------------------------------------------
# Default state
# ---------------------------------------------------------------------------

class TestSimulationStateDefaults:
    def test_default_current(self):
        s = SimulationState()
        assert s.current == 20.0

    def test_default_temperature(self):
        s = SimulationState()
        assert s.temperature == 75.0

    def test_default_pressure(self):
        s = SimulationState()
        assert s.pressure == 35.0

    def test_default_time(self):
        s = SimulationState()
        assert s.time == 0.0

    def test_default_velocities_none(self):
        s = SimulationState()
        assert s.velocities is None

    def test_default_void_fractions_none(self):
        s = SimulationState()
        assert s.void_fractions is None

    def test_default_temperatures_none(self):
        s = SimulationState()
        assert s.temperatures is None

    def test_default_cell_modifiers_length(self):
        s = SimulationState()
        assert len(s.cell_modifiers) == NUM_CELLS

    def test_default_cell_modifiers_values(self):
        s = SimulationState()
        for modifier in s.cell_modifiers:
            assert modifier['R_ohm_modifier'] == 1.0
            assert modifier['membraneHealth'] == 1.0

    def test_cell_modifiers_are_independent_across_instances(self):
        """Each instance gets its own list — no shared mutable default."""
        s1 = SimulationState()
        s2 = SimulationState()
        s1.cell_modifiers[0]['R_ohm_modifier'] = 999.0
        assert s2.cell_modifiers[0]['R_ohm_modifier'] == 1.0


# ---------------------------------------------------------------------------
# update_params
# ---------------------------------------------------------------------------

class TestUpdateParams:
    def test_update_current(self):
        s = SimulationState()
        s.update_params(current=50.0)
        assert s.current == 50.0

    def test_update_temperature(self):
        s = SimulationState()
        s.update_params(temperature=90.0)
        assert s.temperature == 90.0

    def test_update_pressure(self):
        s = SimulationState()
        s.update_params(pressure=10.0)
        assert s.pressure == 10.0

    def test_update_all_params(self):
        s = SimulationState()
        s.update_params(current=30.0, temperature=60.0, pressure=20.0)
        assert s.current == 30.0
        assert s.temperature == 60.0
        assert s.pressure == 20.0

    def test_update_partial_leaves_others_unchanged(self):
        s = SimulationState()
        s.update_params(current=5.0)
        assert s.temperature == 75.0
        assert s.pressure == 35.0

    def test_update_with_none_leaves_unchanged(self):
        s = SimulationState()
        s.update_params(current=None, temperature=None, pressure=None)
        assert s.current == 20.0
        assert s.temperature == 75.0
        assert s.pressure == 35.0

    def test_update_with_zero(self):
        s = SimulationState()
        s.update_params(current=0.0)
        assert s.current == 0.0

    def test_update_with_negative(self):
        s = SimulationState()
        s.update_params(current=-1.0)
        assert s.current == -1.0


# ---------------------------------------------------------------------------
# update_cell_modifiers
# ---------------------------------------------------------------------------

class TestUpdateCellModifiers:
    def test_valid_update(self):
        s = SimulationState()
        mods = [{'R_ohm_modifier': 1.5, 'membraneHealth': 0.8} for _ in range(NUM_CELLS)]
        s.update_cell_modifiers(mods)
        assert s.cell_modifiers[0]['R_ohm_modifier'] == 1.5
        assert s.cell_modifiers[0]['membraneHealth'] == 0.8

    def test_wrong_count_raises(self):
        s = SimulationState()
        with pytest.raises(ValueError, match="Expected"):
            s.update_cell_modifiers(_healthy_modifiers(NUM_CELLS - 1))

    def test_too_many_raises(self):
        s = SimulationState()
        with pytest.raises(ValueError, match="Expected"):
            s.update_cell_modifiers(_healthy_modifiers(NUM_CELLS + 1))

    def test_missing_required_key_r_ohm(self):
        s = SimulationState()
        mods = [{'membraneHealth': 1.0} for _ in range(NUM_CELLS)]
        with pytest.raises(ValueError, match="missing keys"):
            s.update_cell_modifiers(mods)

    def test_missing_required_key_membrane_health(self):
        s = SimulationState()
        mods = [{'R_ohm_modifier': 1.0} for _ in range(NUM_CELLS)]
        with pytest.raises(ValueError, match="missing keys"):
            s.update_cell_modifiers(mods)

    def test_extra_keys_are_allowed(self):
        s = SimulationState()
        mods = [{'R_ohm_modifier': 1.0, 'membraneHealth': 1.0, 'extra': 42}
                for _ in range(NUM_CELLS)]
        s.update_cell_modifiers(mods)
        assert s.cell_modifiers[0]['extra'] == 42

    def test_required_keys_set_content(self):
        assert REQUIRED_KEYS == {'R_ohm_modifier', 'membraneHealth'}


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------

class TestToDict:
    def test_scalar_fields_present(self):
        s = SimulationState()
        d = s.to_dict()
        assert d['current'] == 20.0
        assert d['temperature'] == 75.0
        assert d['pressure'] == 35.0
        assert d['time'] == 0.0

    def test_none_fields_serialized_as_none(self):
        s = SimulationState()
        d = s.to_dict()
        assert d['velocities'] is None
        assert d['voidFractions'] is None
        assert d['temperatures'] is None

    def test_numpy_velocities_serialized_as_list(self):
        s = SimulationState()
        s.velocities = np.zeros((50, 50, 3))
        d = s.to_dict()
        assert isinstance(d['velocities'], list)
        assert len(d['velocities']) == 50

    def test_numpy_void_fractions_serialized_as_list(self):
        s = SimulationState()
        s.void_fractions = np.ones((50, 50)) * 0.2
        d = s.to_dict()
        assert isinstance(d['voidFractions'], list)

    def test_numpy_temperatures_serialized_as_list(self):
        s = SimulationState()
        s.temperatures = np.full((10, 10), 80.0)
        d = s.to_dict()
        assert isinstance(d['temperatures'], list)
        assert d['temperatures'][0][0] == pytest.approx(80.0)

    def test_to_dict_keys(self):
        s = SimulationState()
        d = s.to_dict()
        expected_keys = {'current', 'temperature', 'pressure', 'time',
                         'velocities', 'voidFractions', 'temperatures'}
        assert set(d.keys()) == expected_keys
