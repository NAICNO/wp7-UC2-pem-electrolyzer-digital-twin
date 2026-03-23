"""
Model Definitions for PEM Electrolyzer PINN.

Contains all model architectures:
- HybridPhysicsMLP: Teacher model (8 params + MLP)
- PhysicsHybrid12Param: Student model (12 params, pure physics)
- PureMLP: Pure ML baseline
- BigMLP: Larger ML baseline
- SteadyStateTransformer: Transformer baseline
- get_model(): Factory function to instantiate models
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Tuple


# ============================================================================
# TEACHER MODEL: HybridPhysicsMLP
# ============================================================================

class HybridPhysicsMLP(nn.Module):
    """
    Hybrid Physics + MLP model for PEM electrolyzer voltage prediction.

    The model combines:
    1. Physics-based voltage from electrochemical equations
    2. MLP-based residual correction for non-modeled effects

    Input: [current (A), H2_pressure (bar), O2_pressure (bar), temperature (°C)]
    Output: Cell voltage (V)
    """

    def __init__(
        self,
        device: str = 'cuda',
        hidden_sizes: list = [128, 64],
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize the Hybrid Physics + MLP model.

        Args:
            device: Torch device ('cuda' or 'cpu')
            hidden_sizes: List of hidden layer sizes for MLP
            dropout: Dropout rate for MLP
            activation: Activation function ('gelu', 'relu', 'tanh', 'silu')
        """
        super().__init__()
        self.device = device

        # ====================================================================
        # Physical Constants (FIXED - not learnable)
        # ====================================================================
        self.R = 8.314e-3      # Gas constant [kJ/(mol·K)]
        self.F = 96485         # Faraday constant [C/mol]
        self.n = 2             # Electrons transferred
        self.A_cell = 50.0     # Active area [cm²]

        # Reference conditions
        self.T_ref = 353.15    # Reference temperature [K] (80°C)
        self.P_ref = 20.0      # Reference pressure [bar]

        # Fixed activation energies [kJ/mol]
        self.E_a_anode = 50.0
        self.E_a_cathode = 60.0
        self.E_R = 15.0

        # ====================================================================
        # Parameter Bounds (from NORCE expert knowledge)
        # ====================================================================
        self.i0_min, self.i0_max = 1e-6, 1.5
        self.R_ohm_min, self.R_ohm_max = 0.01, 2.0
        self.alpha_min, self.alpha_max = 0.1, 2.0
        self.offset_min, self.offset_max = -0.2, 0.2
        self.scale_min, self.scale_max = 0.5, 2.0
        self.linear_pressure_min, self.linear_pressure_max = 0.0, 10.0

        # ====================================================================
        # Learnable Physics Parameters (8 total)
        # ====================================================================
        # Exchange current densities (log-space for positivity)
        self.log_i0_a_ref = nn.Parameter(torch.tensor(-5.0, dtype=torch.float32))
        self.log_i0_c_ref = nn.Parameter(torch.tensor(-5.0, dtype=torch.float32))

        # Ohmic resistance (log-space)
        self.log_R_ohm_ref = nn.Parameter(torch.tensor(-1.6, dtype=torch.float32))

        # Transfer coefficients (sigmoid mapping to bounds)
        self.alpha_a_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.alpha_c_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Global corrections
        self.global_voltage_offset = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.global_resistance_scale_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.linear_pressure_coef_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # ====================================================================
        # MLP for Residual Correction
        # ====================================================================
        # Input: [i_norm, P_norm, T_norm, V_physics_norm]
        def get_activation(name):
            activations = {
                'gelu': nn.GELU,
                'tanh': nn.Tanh,
                'silu': nn.SiLU,
                'relu': nn.ReLU,
            }
            return activations.get(name.lower(), nn.GELU)()

        layers = []
        input_size = 4
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size),
                get_activation(activation),
                nn.Dropout(dropout),
            ])
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))

        self.mlp = nn.Sequential(*layers)

        # Residual scaling (learned, starts small)
        self.residual_scale = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

        # ====================================================================
        # Normalization Buffers (set during training)
        # NOTE: P_mean/P_std are for AVERAGE pressure (matching original model)
        # ====================================================================
        self.register_buffer('i_mean', torch.tensor(0.15))
        self.register_buffer('i_std', torch.tensor(0.05))
        self.register_buffer('P_mean', torch.tensor(25.0))  # Average pressure
        self.register_buffer('P_std', torch.tensor(10.0))   # Average pressure std
        self.register_buffer('T_mean', torch.tensor(350.0))
        self.register_buffer('T_std', torch.tensor(5.0))
        self.register_buffer('V_mean', torch.tensor(1.75))
        self.register_buffer('V_std', torch.tensor(0.05))

        # Move to device
        self.to(device)

    def set_normalization_stats(
        self,
        i_mean: float, i_std: float,
        P_mean: float, P_std: float,
        T_mean: float, T_std: float,
        V_mean: float, V_std: float
    ):
        """Set normalization statistics from training data.

        Note: P_mean/P_std are for AVERAGE pressure (H2+O2)/2, matching original model.
        """
        self.i_mean.fill_(i_mean)
        self.i_std.fill_(max(i_std, 1e-6))
        self.P_mean.fill_(P_mean)
        self.P_std.fill_(max(P_std, 1e-6))
        self.T_mean.fill_(T_mean)
        self.T_std.fill_(max(T_std, 1e-6))
        self.V_mean.fill_(V_mean)
        self.V_std.fill_(max(V_std, 1e-6))

    def _get_physics_params(self) -> Dict[str, torch.Tensor]:
        """Get constrained physics parameters."""
        # Exchange currents (exp of log, then clamp)
        i0_a_ref = torch.clamp(torch.exp(self.log_i0_a_ref), self.i0_min, self.i0_max)
        i0_c_ref = torch.clamp(torch.exp(self.log_i0_c_ref), self.i0_min, self.i0_max)

        # Ohmic resistance
        R_ohm_ref = torch.clamp(torch.exp(self.log_R_ohm_ref), self.R_ohm_min, self.R_ohm_max)

        # Transfer coefficients (sigmoid mapping)
        alpha_a = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.alpha_a_raw)
        alpha_c = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.alpha_c_raw)

        # Global corrections
        voltage_offset = self.offset_min + (self.offset_max - self.offset_min) * torch.sigmoid(self.global_voltage_offset)
        resistance_scale = self.scale_min + (self.scale_max - self.scale_min) * torch.sigmoid(self.global_resistance_scale_raw)
        linear_pressure_coef = self.linear_pressure_min + (self.linear_pressure_max - self.linear_pressure_min) * torch.sigmoid(self.linear_pressure_coef_raw)

        return {
            'i0_a_ref': i0_a_ref,
            'i0_c_ref': i0_c_ref,
            'R_ohm_ref': R_ohm_ref,
            'alpha_a': alpha_a,
            'alpha_c': alpha_c,
            'voltage_offset': voltage_offset,
            'resistance_scale': resistance_scale,
            'linear_pressure_coef': linear_pressure_coef,
        }

    def _compute_physics_voltage(
        self,
        current_density: torch.Tensor,
        H2_pressure: torch.Tensor,
        O2_pressure: torch.Tensor,
        temperature: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute voltage from electrochemical physics equations.

        Args:
            current_density: Current density [A/cm²]
            H2_pressure: H2 pressure [bar]
            O2_pressure: O2 pressure [bar]
            temperature: Temperature [K]
            params: Dictionary of physics parameters

        Returns:
            V_physics: Physics-based voltage [V]
        """
        # Clamp pressures to avoid log(0)
        P_MIN = 0.1
        H2_pressure = torch.clamp(H2_pressure, min=P_MIN)
        O2_pressure = torch.clamp(O2_pressure, min=P_MIN)

        # ================================================================
        # 1. Nernst (Reversible) Voltage
        # ================================================================
        # Temperature dependence
        V_rev = 1.23 - 0.9e-3 * (temperature - 298.15)

        # Pressure correction (Nernst equation)
        # CRITICAL: Use SEPARATE H2 and O2 pressures!
        # ln_arg = (P_H2 * sqrt(P_O2)) / P_H2O
        P_H2O = 0.05  # Reference water vapor pressure [bar]
        RT_nF = (self.R * temperature * 1000) / (self.n * self.F)
        ln_arg = (H2_pressure * torch.sqrt(O2_pressure)) / P_H2O
        V_nernst_pressure = RT_nF * torch.log(ln_arg + 1e-6)
        V_rev = V_rev + V_nernst_pressure

        # ================================================================
        # 2. Temperature-Dependent Parameters (Arrhenius)
        # ================================================================
        arrhenius_a = torch.exp(self.E_a_anode / (self.R * self.T_ref) * (1 - self.T_ref / temperature))
        arrhenius_c = torch.exp(self.E_a_cathode / (self.R * self.T_ref) * (1 - self.T_ref / temperature))
        arrhenius_R = torch.exp(self.E_R / (self.R * temperature) - self.E_R / (self.R * self.T_ref))

        i0_a = params['i0_a_ref'] * arrhenius_a
        i0_c = params['i0_c_ref'] * arrhenius_c
        R_ohm = params['R_ohm_ref'] * arrhenius_R * params['resistance_scale']

        # ================================================================
        # 3. Split Butler-Volmer (Activation Overpotentials)
        # ================================================================
        V_act_a = (RT_nF / params['alpha_a']) * torch.asinh(current_density / (2 * i0_a + 1e-10))
        V_act_c = (RT_nF / params['alpha_c']) * torch.asinh(current_density / (2 * i0_c + 1e-10))
        V_act = V_act_a + V_act_c

        # ================================================================
        # 4. Ohmic Loss
        # ================================================================
        V_ohm = current_density * R_ohm

        # ================================================================
        # 5. Linear Pressure Correction (use average pressure)
        # ================================================================
        pressure_avg = (H2_pressure + O2_pressure) / 2.0
        V_linear_pressure = params['linear_pressure_coef'] * (pressure_avg - self.P_ref) / 1000  # mV to V

        # ================================================================
        # Total Physics Voltage
        # ================================================================
        V_physics = V_rev + V_act + V_ohm + params['voltage_offset'] + V_linear_pressure

        return V_physics

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass: Physics + MLP residual.

        Args:
            x: Input tensor [batch, 4] with columns:
               - current [A] (NOT A/cm² - converted internally)
               - H2_pressure [bar]
               - O2_pressure [bar]
               - temperature [°C] (NOT K - converted internally)

        Returns:
            V_cell: Predicted cell voltage [V]
            params: Dictionary of physics parameters for logging
        """
        # Move to device
        x = x.to(self.device)

        # Extract inputs in ORIGINAL UNITS (A, bar, bar, °C)
        current = x[:, 0]         # [A]
        H2_pressure = x[:, 1]     # [bar]
        O2_pressure = x[:, 2]     # [bar]
        temperature = x[:, 3]     # [°C]

        # Convert units for physics calculations
        current_density = current / self.A_cell  # A -> A/cm²
        temperature_K = temperature + 273.15      # °C -> K

        # Get physics parameters
        params = self._get_physics_params()

        # Compute physics-based voltage (uses converted units)
        V_physics = self._compute_physics_voltage(current_density, H2_pressure, O2_pressure, temperature_K, params)

        # ================================================================
        # MLP Residual Correction
        # ================================================================
        # Normalize using ORIGINAL units (matching training stats)
        # IMPORTANT: Average pressures FIRST, then normalize (matching original model!)
        P_avg = (H2_pressure + O2_pressure) / 2.0
        i_norm = (current - self.i_mean) / (self.i_std + 1e-6)
        P_norm = (P_avg - self.P_mean) / (self.P_std + 1e-6)
        T_norm = (temperature - self.T_mean) / (self.T_std + 1e-6)
        V_norm = (V_physics - self.V_mean) / (self.V_std + 1e-6)

        mlp_input = torch.stack([i_norm, P_norm, T_norm, V_norm], dim=-1)

        # MLP prediction
        V_correction = self.mlp(mlp_input).squeeze(-1)
        V_correction = V_correction * torch.abs(self.residual_scale)

        # Clamp correction to ±100mV to prevent overfitting
        V_correction = torch.clamp(V_correction, -0.1, 0.1)

        # ================================================================
        # Total Voltage
        # ================================================================
        V_cell = V_physics + V_correction
        V_cell = torch.clamp(V_cell, 1.3, 2.5)  # Physical bounds

        # Prepare output parameters for logging
        output_params = {
            'i0_a_ref': params['i0_a_ref'].item(),
            'i0_c_ref': params['i0_c_ref'].item(),
            'R_ohm_ref': params['R_ohm_ref'].item(),
            'alpha_a': params['alpha_a'].item(),
            'alpha_c': params['alpha_c'].item(),
            'voltage_offset': params['voltage_offset'].item(),
            'resistance_scale': params['resistance_scale'].item(),
            'linear_pressure_coef': params['linear_pressure_coef'].item(),
            'residual_scale': self.residual_scale.item(),
            'V_physics_mean': V_physics.mean().item(),
            'V_correction_mean': V_correction.mean().item(),
            'V_correction_std': V_correction.std().item() if len(V_correction) > 1 else 0.0,
        }

        return V_cell, output_params

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# STUDENT MODEL: PhysicsHybrid12Param
# ============================================================================

class PhysicsHybrid12Param(nn.Module):
    """
    12-parameter hybrid physics model for PEM electrolyzer.

    This model uses more physics constraints than the teacher, making it
    more interpretable and better for OOD generalization when trained
    via knowledge distillation.

    Input: [current (A), H2_pressure (bar), O2_pressure (bar), temperature (°C)]
    Output: Cell voltage (V)
    """

    def __init__(
        self,
        # Reference conditions
        T_ref: float = 353.15,      # 80°C in Kelvin
        P_ref: float = 20.0,        # 20 bar reference pressure
        # Fixed physics constants
        beta_lim: float = 0.7,      # Fick's law exponent
        E_a_anode: float = 50.0,    # kJ/mol
        E_a_cathode: float = 60.0,  # kJ/mol
        E_R: float = 15.0,          # kJ/mol
        gamma_a: float = 0.2,       # Pressure exponent for i0_a
        gamma_c: float = 0.2,       # Pressure exponent for i0_c
        # Parameter bounds
        alpha_min: float = 0.3,
        alpha_max: float = 1.0,
        # Correction type
        correction_type: str = 'logistic',
    ):
        """
        Initialize the 12-parameter physics model.

        Args:
            T_ref: Reference temperature [K]
            P_ref: Reference pressure [bar]
            beta_lim: Fick's law exponent (FIXED)
            E_a_anode: Anode activation energy [kJ/mol] (FIXED)
            E_a_cathode: Cathode activation energy [kJ/mol] (FIXED)
            E_R: Resistance activation energy [kJ/mol] (FIXED)
            gamma_a: Anode i0 pressure exponent (FIXED)
            gamma_c: Cathode i0 pressure exponent (FIXED)
            alpha_min: Minimum transfer coefficient
            alpha_max: Maximum transfer coefficient
            correction_type: Type of hybrid correction ('logistic')
        """
        super().__init__()

        self.T_ref = T_ref
        self.P_ref = P_ref
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.correction_type = correction_type

        # Physical constants (truly fixed)
        self.F = 96485.0    # Faraday constant [C/mol]
        self.R = 8.314      # Gas constant [J/(mol·K)]
        self.n = 2          # Electrons transferred

        # ====================================================================
        # 6 LEARNABLE PHYSICS PARAMETERS
        # ====================================================================
        # Use log-space for strictly positive parameters (guaranteed > 0)
        self.log_i_lim_ref = nn.Parameter(torch.tensor(math.log(1.0)))   # ~1 A/cm²
        self.log_i0_a_ref = nn.Parameter(torch.tensor(math.log(1e-5)))   # ~1e-5 A/cm²
        self.log_i0_c_ref = nn.Parameter(torch.tensor(math.log(1e-3)))   # ~1e-3 A/cm²
        self.log_R_ohm_ref = nn.Parameter(torch.tensor(math.log(0.1)))   # ~0.1 Ω·cm²

        # Transfer coefficients (sigmoid mapping to [alpha_min, alpha_max])
        self.alpha_a_raw = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5
        self.alpha_c_raw = nn.Parameter(torch.tensor(0.0))

        # ====================================================================
        # 6 FIXED PHYSICS CONSTANTS (registered as buffers, NOT learnable)
        # These must stay at physically reasonable values for OOD generalization
        # ====================================================================
        self.register_buffer('beta_lim', torch.tensor(beta_lim))
        self.register_buffer('E_a_anode', torch.tensor(E_a_anode))
        self.register_buffer('E_a_cathode', torch.tensor(E_a_cathode))
        self.register_buffer('E_R', torch.tensor(E_R))
        self.register_buffer('gamma_a', torch.tensor(gamma_a))
        self.register_buffer('gamma_c', torch.tensor(gamma_c))

        # ====================================================================
        # 6 LEARNABLE HYBRID CORRECTION PARAMETERS
        # ====================================================================
        # Logistic correction: a * sigmoid(b * (i - c)) + d
        # Plus pressure/temperature modulation
        self.corr_a = nn.Parameter(torch.tensor(0.01))    # Amplitude
        self.corr_b = nn.Parameter(torch.tensor(1.0))     # Steepness
        self.corr_c = nn.Parameter(torch.tensor(1.0))     # Midpoint (current)
        self.corr_d = nn.Parameter(torch.tensor(0.0))     # Offset
        self.corr_p = nn.Parameter(torch.tensor(0.0))     # Pressure scaling
        self.corr_t = nn.Parameter(torch.tensor(0.0))     # Temperature scaling

    # ========================================================================
    # Property accessors for constrained parameters
    # ========================================================================

    @property
    def i_lim_ref(self) -> torch.Tensor:
        """Limiting current density reference [A/cm²]."""
        return torch.exp(self.log_i_lim_ref).clamp(0.1, 10.0)

    @property
    def i0_a_ref(self) -> torch.Tensor:
        """Anode exchange current density reference [A/cm²]."""
        return torch.exp(self.log_i0_a_ref).clamp(1e-8, 1e-2)

    @property
    def i0_c_ref(self) -> torch.Tensor:
        """Cathode exchange current density reference [A/cm²]."""
        return torch.exp(self.log_i0_c_ref).clamp(1e-6, 1.0)

    @property
    def R_ohm_ref(self) -> torch.Tensor:
        """Ohmic resistance reference [Ω·cm²]."""
        return torch.exp(self.log_R_ohm_ref).clamp(0.01, 1.0)

    @property
    def alpha_a(self) -> torch.Tensor:
        """Anode charge transfer coefficient."""
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.alpha_a_raw)

    @property
    def alpha_c(self) -> torch.Tensor:
        """Cathode charge transfer coefficient."""
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.alpha_c_raw)

    # ========================================================================
    # Physics Computations
    # ========================================================================

    def compute_nernst(self, T: torch.Tensor, H2_P: torch.Tensor, O2_P: torch.Tensor) -> torch.Tensor:
        """
        Compute Nernst (reversible) potential with pressure correction.

        V_rev = 1.23 - 0.9e-3*(T-298) + RT/(nF) * ln(P_H2 * sqrt(P_O2) / P_H2O)
        """
        # Standard potential at 25°C
        E0 = 1.229
        # Temperature correction
        E_rev = E0 - 0.9e-3 * (T - 298.15)

        # Pressure correction (Nernst equation)
        # CRITICAL: Use SEPARATE H2 and O2 pressures!
        P_H2O = 0.05  # bar (water vapor pressure reference)
        RT_nF = (self.R * T) / (self.n * self.F)
        # ln_arg = (P_H2 * sqrt(P_O2)) / P_H2O
        ln_arg = (H2_P * torch.sqrt(O2_P + 1e-6)) / P_H2O
        E_rev = E_rev + RT_nF * torch.log(ln_arg + 1e-6)

        return E_rev

    def compute_i_lim(self, P: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Compute limiting current density with pressure AND temperature scaling.

        Physics: i_lim depends on mass transport (diffusion), which varies with P and T.
        i_lim = i_lim_ref * (P/P_ref)^beta_lim * temperature_factor
        """
        # Pressure effect via Fick's law
        pressure_factor = (P / self.P_ref) ** self.beta_lim

        # Temperature effect on diffusivity
        T_norm = (T - self.T_ref) / self.T_ref
        temp_factor = 1.0 + 0.3 * T_norm  # ~30% increase per T_ref change

        i_lim = self.i_lim_ref * pressure_factor * temp_factor
        return i_lim.clamp(0.1, 10.0)

    def compute_exchange_currents(self, T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute exchange current densities with Arrhenius temperature scaling.

        i0 = i0_ref * exp(-E_a/R * (1/T - 1/T_ref)) * (T/T_ref)^gamma
        """
        inv_T_diff = (1.0 / T - 1.0 / self.T_ref)

        # Anode
        arr_factor_a = torch.exp(-self.E_a_anode * 1000 / self.R * inv_T_diff)
        temp_factor_a = (T / self.T_ref) ** self.gamma_a
        i0_a = self.i0_a_ref * arr_factor_a * temp_factor_a

        # Cathode
        arr_factor_c = torch.exp(-self.E_a_cathode * 1000 / self.R * inv_T_diff)
        temp_factor_c = (T / self.T_ref) ** self.gamma_c
        i0_c = self.i0_c_ref * arr_factor_c * temp_factor_c

        return i0_a, i0_c

    def compute_R_ohm(self, T: torch.Tensor) -> torch.Tensor:
        """
        Compute ohmic resistance with temperature scaling.

        R_ohm = R_ohm_ref * exp(-E_R/R * (1/T - 1/T_ref))
        """
        inv_T_diff = (1.0 / T - 1.0 / self.T_ref)
        arr_factor = torch.exp(-self.E_R * 1000 / self.R * inv_T_diff)
        return self.R_ohm_ref * arr_factor

    def compute_activation_overpotential(
        self,
        i: torch.Tensor,
        i0: torch.Tensor,
        alpha: torch.Tensor,
        T: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute activation overpotential using simplified Tafel equation.

        eta_act = (R*T)/(alpha*n*F) * arcsinh(i/(2*i0))
        """
        prefactor = (self.R * T) / (alpha * self.n * self.F)
        eta = prefactor * torch.asinh(i / (2.0 * i0 + 1e-10))
        return eta

    def compute_concentration_overpotential(
        self,
        i: torch.Tensor,
        i_lim: torch.Tensor,
        T: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute concentration overpotential.

        eta_conc = (R*T)/(n*F) * ln(1 - i/i_lim)
        """
        ratio = (i / (i_lim + 1e-10)).clamp(max=0.99)
        prefactor = (self.R * T) / (self.n * self.F)
        eta = -prefactor * torch.log(1.0 - ratio)
        return eta

    def compute_hybrid_correction(
        self,
        i: torch.Tensor,
        T: torch.Tensor,
        P: torch.Tensor,
        E_nernst: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hybrid correction term using logistic function.

        correction = (a * sigmoid(b*(i-c)) + d) * (1 + p_mod + t_mod)
        """
        # Logistic correction on current
        logistic = torch.sigmoid(self.corr_b * (i - self.corr_c))
        base_correction = self.corr_a * logistic + self.corr_d

        # Pressure/temperature modulation
        p_norm = (P - self.P_ref) / self.P_ref
        t_norm = (T - self.T_ref) / self.T_ref
        modulation = 1.0 + self.corr_p * p_norm + self.corr_t * t_norm

        correction = base_correction * modulation

        return correction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, 4] with columns:
               - current [A] (converted to A/cm² internally)
               - H2_pressure [bar]
               - O2_pressure [bar]
               - temperature [°C] (converted to K internally)

        Returns:
            V: Cell voltage [V]
        """
        # Extract inputs in ORIGINAL UNITS (A, bar, bar, °C)
        current = x[:, 0]      # [A]
        H2_P = x[:, 1]         # [bar]
        O2_P = x[:, 2]         # [bar]
        temperature = x[:, 3]  # [°C]

        # Convert to physics units
        i = current / 50.0        # A -> A/cm² (50 cm² active area)
        T = temperature + 273.15  # °C -> K

        # Average pressure for terms that use average
        P_avg = (H2_P + O2_P) / 2.0

        # Nernst potential (uses SEPARATE H2/O2 pressures!)
        E_nernst = self.compute_nernst(T, H2_P, O2_P)

        # Exchange current densities
        i0_a, i0_c = self.compute_exchange_currents(T)

        # Limiting current density (uses average pressure)
        i_lim = self.compute_i_lim(P_avg, T)

        # Ohmic resistance
        R_ohm = self.compute_R_ohm(T)

        # Activation overpotentials (split Butler-Volmer)
        eta_act_a = self.compute_activation_overpotential(i, i0_a, self.alpha_a, T)
        eta_act_c = self.compute_activation_overpotential(i, i0_c, self.alpha_c, T)

        # Ohmic overpotential
        eta_ohm = i * R_ohm

        # Concentration overpotential
        eta_conc = self.compute_concentration_overpotential(i, i_lim, T)

        # Hybrid correction (clamped to ±100mV) - uses average pressure
        correction = self.compute_hybrid_correction(i, T, P_avg, E_nernst)
        correction = torch.clamp(correction, -0.1, 0.1)

        # Total voltage
        V = E_nernst + eta_act_a + eta_act_c + eta_ohm + eta_conc + correction

        return V

    def get_physics_params(self) -> Dict[str, float]:
        """Get current physics parameter values."""
        return {
            'i_lim_ref': self.i_lim_ref.item(),
            'i0_a_ref': self.i0_a_ref.item(),
            'i0_c_ref': self.i0_c_ref.item(),
            'R_ohm_ref': self.R_ohm_ref.item(),
            'alpha_a': self.alpha_a.item(),
            'alpha_c': self.alpha_c.item(),
        }

    def get_hybrid_params(self) -> Dict[str, float]:
        """Get current hybrid correction parameter values."""
        return {
            'corr_a': self.corr_a.item(),
            'corr_b': self.corr_b.item(),
            'corr_c': self.corr_c.item(),
            'corr_d': self.corr_d.item(),
            'corr_p': self.corr_p.item(),
            'corr_t': self.corr_t.item(),
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# BASELINE MODELS
# ============================================================================

class PureMLP(nn.Module):
    """
    Pure MLP model without physics constraints.

    Input: [current (A), H2_pressure (bar), O2_pressure (bar), temperature (°C)]
    Output: Cell voltage (V)
    """

    def __init__(
        self,
        hidden_sizes: list = [128, 64],
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize Pure MLP model.

        Args:
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout rate
            activation: Activation function ('gelu', 'relu', 'tanh', 'silu')
        """
        super().__init__()

        self.hidden_sizes = hidden_sizes

        # Activation function
        activations = {
            'gelu': nn.GELU,
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'silu': nn.SiLU,
        }
        act_fn = activations.get(activation.lower(), nn.GELU)

        # Build MLP layers
        layers = []
        input_size = 4  # [current, H2_P, O2_P, temperature]

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                act_fn(),
                nn.Dropout(dropout),
            ])
            input_size = hidden_size

        # Output layer with bias initialized to typical voltage (~1.7V)
        output_layer = nn.Linear(input_size, 1)
        output_layer.bias.data.fill_(1.7)  # Initialize to typical voltage
        output_layer.weight.data.mul_(0.01)  # Small weights for stability
        layers.append(output_layer)

        self.mlp = nn.Sequential(*layers)

        # Normalization statistics (set during training)
        self.register_buffer('i_mean', torch.tensor(5.0))
        self.register_buffer('i_std', torch.tensor(1.0))
        self.register_buffer('P_mean', torch.tensor(25.0))
        self.register_buffer('P_std', torch.tensor(10.0))
        self.register_buffer('T_mean', torch.tensor(75.0))
        self.register_buffer('T_std', torch.tensor(5.0))

    def set_normalization_stats(
        self,
        i_mean: float, i_std: float,
        P_mean: float, P_std: float,
        T_mean: float, T_std: float,
    ):
        """Set normalization statistics from training data."""
        self.i_mean.fill_(i_mean)
        self.i_std.fill_(max(i_std, 1e-6))
        self.P_mean.fill_(P_mean)
        self.P_std.fill_(max(P_std, 1e-6))
        self.T_mean.fill_(T_mean)
        self.T_std.fill_(max(T_std, 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, 4] with columns:
               - current [A]
               - H2_pressure [bar]
               - O2_pressure [bar]
               - temperature [°C]

        Returns:
            V_cell: Predicted cell voltage [V]
        """
        # Extract inputs
        current = x[:, 0]
        H2_P = x[:, 1]
        O2_P = x[:, 2]
        temperature = x[:, 3]

        # Normalize inputs
        P_avg = (H2_P + O2_P) / 2.0
        i_norm = (current - self.i_mean) / (self.i_std + 1e-6)
        P_H2_norm = (H2_P - self.P_mean) / (self.P_std + 1e-6)
        P_O2_norm = (O2_P - self.P_mean) / (self.P_std + 1e-6)
        T_norm = (temperature - self.T_mean) / (self.T_std + 1e-6)

        # Stack all 4 normalized inputs
        mlp_input = torch.stack([i_norm, P_H2_norm, P_O2_norm, T_norm], dim=-1)

        # MLP prediction (no clamp - let it learn freely)
        V_pred = self.mlp(mlp_input).squeeze(-1)

        return V_pred

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BigMLP(nn.Module):
    """
    Big MLP model without physics constraints (~50K params).

    Input: [current (A), H2_pressure (bar), O2_pressure (bar), temperature (°C)]
    Output: Cell voltage (V)
    """

    def __init__(
        self,
        hidden_sizes: list = [256, 128, 64],
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize Big MLP model.

        Args:
            hidden_sizes: List of hidden layer sizes [256, 128, 64] for ~50K params
            dropout: Dropout rate
            activation: Activation function ('gelu', 'relu', 'tanh', 'silu')
        """
        super().__init__()

        self.hidden_sizes = hidden_sizes

        # Activation function
        activations = {
            'gelu': nn.GELU,
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'silu': nn.SiLU,
        }
        act_fn = activations.get(activation.lower(), nn.GELU)

        # Build MLP layers
        layers = []
        input_size = 4  # [current, H2_P, O2_P, temperature]

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size),  # Add layer norm for stability
                act_fn(),
                nn.Dropout(dropout),
            ])
            input_size = hidden_size

        # Output layer with bias initialized to typical voltage (~1.7V)
        output_layer = nn.Linear(input_size, 1)
        output_layer.bias.data.fill_(1.7)  # Initialize to typical voltage
        output_layer.weight.data.mul_(0.01)  # Small weights for stability
        layers.append(output_layer)

        self.mlp = nn.Sequential(*layers)

        # Normalization statistics (set during training)
        self.register_buffer('i_mean', torch.tensor(5.0))
        self.register_buffer('i_std', torch.tensor(1.0))
        self.register_buffer('P_mean', torch.tensor(25.0))
        self.register_buffer('P_std', torch.tensor(10.0))
        self.register_buffer('T_mean', torch.tensor(75.0))
        self.register_buffer('T_std', torch.tensor(5.0))

    def set_normalization_stats(
        self,
        i_mean: float, i_std: float,
        P_mean: float, P_std: float,
        T_mean: float, T_std: float,
    ):
        """Set normalization statistics from training data."""
        self.i_mean.fill_(i_mean)
        self.i_std.fill_(max(i_std, 1e-6))
        self.P_mean.fill_(P_mean)
        self.P_std.fill_(max(P_std, 1e-6))
        self.T_mean.fill_(T_mean)
        self.T_std.fill_(max(T_std, 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, 4] with columns:
               - current [A]
               - H2_pressure [bar]
               - O2_pressure [bar]
               - temperature [°C]

        Returns:
            V_cell: Predicted cell voltage [V]
        """
        # Extract inputs
        current = x[:, 0]
        H2_P = x[:, 1]
        O2_P = x[:, 2]
        temperature = x[:, 3]

        # Normalize inputs
        P_avg = (H2_P + O2_P) / 2.0
        i_norm = (current - self.i_mean) / (self.i_std + 1e-6)
        P_H2_norm = (H2_P - self.P_mean) / (self.P_std + 1e-6)
        P_O2_norm = (O2_P - self.P_mean) / (self.P_std + 1e-6)
        T_norm = (temperature - self.T_mean) / (self.T_std + 1e-6)

        # Stack all 4 normalized inputs
        mlp_input = torch.stack([i_norm, P_H2_norm, P_O2_norm, T_norm], dim=-1)

        # MLP prediction (no clamp - let it learn freely)
        V_pred = self.mlp(mlp_input).squeeze(-1)

        return V_pred

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for input features."""

    def __init__(self, d_model: int, max_len: int = 4):
        super().__init__()
        # Learnable position embeddings for each input feature
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pos_embedding[:x.size(1)]


class SteadyStateTransformer(nn.Module):
    """
    Transformer model for steady-state voltage prediction.

    Treats input features as tokens and uses self-attention
    to capture feature interactions.

    Input: [current (A), H2_pressure (bar), O2_pressure (bar), temperature (°C)]
    Output: Cell voltage (V)
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        """
        Initialize Transformer model.

        Args:
            d_model: Embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.n_features = 4

        # Input embedding: project each scalar feature to d_model dimensions
        self.input_proj = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.n_features)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection with initialization for typical voltage
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * self.n_features, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
        )
        # Initialize last layer bias to typical voltage
        self.output_proj[-1].bias.data.fill_(1.7)
        self.output_proj[-1].weight.data.mul_(0.01)

        # Normalization statistics
        self.register_buffer('i_mean', torch.tensor(5.0))
        self.register_buffer('i_std', torch.tensor(1.0))
        self.register_buffer('P_mean', torch.tensor(25.0))
        self.register_buffer('P_std', torch.tensor(10.0))
        self.register_buffer('T_mean', torch.tensor(75.0))
        self.register_buffer('T_std', torch.tensor(5.0))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)

    def set_normalization_stats(
        self,
        i_mean: float, i_std: float,
        P_mean: float, P_std: float,
        T_mean: float, T_std: float,
    ):
        """Set normalization statistics from training data."""
        self.i_mean.fill_(i_mean)
        self.i_std.fill_(max(i_std, 1e-6))
        self.P_mean.fill_(P_mean)
        self.P_std.fill_(max(P_std, 1e-6))
        self.T_mean.fill_(T_mean)
        self.T_std.fill_(max(T_std, 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, 4] with columns:
               - current [A]
               - H2_pressure [bar]
               - O2_pressure [bar]
               - temperature [°C]

        Returns:
            V_cell: Predicted cell voltage [V]
        """
        batch_size = x.size(0)

        # Extract and normalize inputs
        current = x[:, 0]
        H2_P = x[:, 1]
        O2_P = x[:, 2]
        temperature = x[:, 3]

        P_avg = (H2_P + O2_P) / 2.0
        i_norm = (current - self.i_mean) / (self.i_std + 1e-6)
        P_norm = (P_avg - self.P_mean) / (self.P_std + 1e-6)
        T_norm = (temperature - self.T_mean) / (self.T_std + 1e-6)

        # Normalize H2 and O2 pressures separately
        P_H2_norm = (H2_P - self.P_mean) / (self.P_std + 1e-6)
        P_O2_norm = (O2_P - self.P_mean) / (self.P_std + 1e-6)

        # Stack as features: [batch, n_features, 1]
        features = torch.stack([i_norm, P_H2_norm, P_O2_norm, T_norm], dim=1).unsqueeze(-1)

        # Project to d_model: [batch, n_features, d_model]
        embeddings = self.input_proj(features)

        # Add positional encoding
        embeddings = self.pos_encoder(embeddings)

        # Transformer encoder: [batch, n_features, d_model]
        transformer_out = self.transformer(embeddings)

        # Flatten and project to output: [batch, n_features * d_model] -> [batch, 1]
        flat = transformer_out.reshape(batch_size, -1)
        V_pred = self.output_proj(flat).squeeze(-1)

        # No clamp - let it learn freely (will fail on OOD)
        return V_pred

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# MODEL FACTORY FUNCTION
# ============================================================================

def get_model(name: str, device: str = 'cpu') -> nn.Module:
    """
    Factory function to instantiate models by name.

    Args:
        name: Model name ('teacher', 'student', 'pure_mlp', 'big_mlp', 'transformer')
        device: Device to create model on ('cpu' or 'cuda')

    Returns:
        model: Instantiated model

    Raises:
        ValueError: If model name is not recognized
    """
    models = {
        'teacher': lambda: HybridPhysicsMLP(device=device),
        'student': lambda: PhysicsHybrid12Param(),
        'pure_mlp': lambda: PureMLP(),
        'big_mlp': lambda: BigMLP(),
        'transformer': lambda: SteadyStateTransformer(),
    }

    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")

    return models[name]()


if __name__ == "__main__":
    """Quick test of models."""
    print("Testing model factory...")

    for name in ['teacher', 'student', 'pure_mlp', 'big_mlp', 'transformer']:
        model = get_model(name, device='cpu')
        print(f"{name}: {model.count_parameters():,} parameters")

    print("\nAll models loaded successfully!")
