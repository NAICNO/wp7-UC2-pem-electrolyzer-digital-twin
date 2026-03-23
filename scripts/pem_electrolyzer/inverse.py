"""
Inverse Pressure Optimizer for PEM Electrolyzer.

Simplified deployment version of src/inverse/pressure_optimizer.py.
Supports 12-param physics model only (CPU, ~5ms per query).

Given: V_max (safety limit), I (current), T (temperature)
Find:  P_max (maximum safe pressure before voltage exceeds limit)

Algorithm: Newton-Raphson with bisection fallback, exploiting V(P) monotonicity.
"""

import torch
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import time

from .models import PhysicsHybrid12Param

# Physical bounds
P_MIN = 0.1     # Minimum pressure [bar]
P_MAX = 40.0    # Maximum pressure [bar]
P_REF = 20.0    # Reference pressure [bar]
I_MIN = 5.0     # Minimum stable current [A]
I_MAX = 18.0    # Maximum safe current [A]

DEFAULT_TOL = 1e-4       # Convergence tolerance [V]
DEFAULT_MAX_ITER = 20    # Max iterations


@dataclass
class OptimizationResult:
    """Result of pressure optimization."""
    P_max: float              # Maximum pressure before safety limit [bar]
    P_safe: float             # Recommended safe pressure with margin [bar]
    V_achieved: float         # Voltage at P_max [V]
    converged: bool           # Whether optimization converged
    iterations: int           # Number of iterations used
    gradient_dVdP: float      # Sensitivity dV/dP at solution [V/bar]
    uncertainty_P: float      # Pressure uncertainty from model error [bar]
    latency_ms: float         # Computation time [ms]
    method: str               # 'newton', 'bisection', or 'hybrid'


@dataclass
class CurrentOptimizationResult:
    """Result of current optimization (find I_max)."""
    I_max: float              # Maximum current before voltage limit [A]
    I_safe: float             # Recommended safe current with margin [A]
    V_achieved: float         # Voltage at I_max [V]
    converged: bool           # Whether optimization converged
    feasible: bool            # Whether a solution exists
    iterations: int           # Number of iterations used
    gradient_dVdI: float      # Sensitivity dV/dI at solution [V/A]
    uncertainty_I: float      # Current uncertainty from model error [A]
    latency_ms: float         # Computation time [ms]
    method: str               # Algorithm used


def load_model(checkpoint_path: str, device: str = 'cpu') -> PhysicsHybrid12Param:
    """Load a trained 12-param model from checkpoint."""
    model = PhysicsHybrid12Param()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def predict_voltage(model: PhysicsHybrid12Param, I: float, T: float, P: float,
                    device: str = 'cpu') -> float:
    """Predict cell voltage for given operating conditions.

    Args:
        model: Trained PhysicsHybrid12Param model.
        I: Current [A].
        T: Temperature [°C].
        P: Pressure [bar] (symmetric: H2_P = O2_P = P).
        device: Computation device.

    Returns:
        Predicted voltage [V].
    """
    with torch.no_grad():
        # Model input: [current_A, H2_P_bar, O2_P_bar, temperature_C]
        x = torch.tensor([[I, P, P, T]], device=device, dtype=torch.float32)
        V = model(x)
    return V.item()


class PressureOptimizer:
    """Find maximum safe operating pressure for PEM electrolyzer.

    Given a trained forward model V = f(I, T, P), inverts to find P_max
    such that V(I, T, P_max) = V_target.

    Args:
        model: Trained PhysicsHybrid12Param model.
        device: Computation device ('cpu' recommended for 12-param model).
        model_uncertainty_mV: Model prediction uncertainty [mV] (default: 20).
    """

    def __init__(self, model: PhysicsHybrid12Param, device: str = 'cpu',
                 model_uncertainty_mV: float = 20.0):
        self.model = model
        self.device = device
        self.model_uncertainty_mV = model_uncertainty_mV

    def _evaluate_model(self, I: float, T: float, P: float) -> float:
        """Evaluate forward model at given operating point."""
        with torch.no_grad():
            # Model input: [current_A, H2_P_bar, O2_P_bar, temperature_C]
            x = torch.tensor([[I, P, P, T]],
                             device=self.device, dtype=torch.float32)
            V = self.model(x)
        return V.item()

    def _newton_step_P(self, P_current: float, V_target: float,
                       I: float, T: float):
        """Single Newton-Raphson step for pressure using autograd."""
        P = torch.tensor([P_current], requires_grad=True,
                         device=self.device, dtype=torch.float32)

        with torch.enable_grad():
            # Model input: [current_A, H2_P_bar, O2_P_bar, temperature_C]
            # Symmetric pressure: H2_P = O2_P = P
            x = torch.stack([
                torch.tensor([I], device=self.device, dtype=torch.float32),
                P,
                P,
                torch.tensor([T], device=self.device, dtype=torch.float32),
            ], dim=1)
            V_pred = self.model(x)

        dV_dP = torch.autograd.grad(
            V_pred, P, grad_outputs=torch.ones_like(V_pred), create_graph=False
        )[0]

        V = V_pred.item()
        residual = V - V_target
        gradient = dV_dP.item()

        if abs(gradient) < 1e-8:
            P_new = P_current - np.sign(residual) * 1.0
        else:
            P_new = P_current - residual / gradient

        return P_new, residual, gradient, V

    def _bisection(self, V_target: float, I: float, T: float,
                   P_low: float, P_high: float,
                   max_iter: int, tolerance: float) -> dict:
        """Bisection search (guaranteed convergence)."""
        V_low = self._evaluate_model(I, T, P_low)
        V_high = self._evaluate_model(I, T, P_high)

        if V_low > V_high:
            P_low, P_high = P_high, P_low

        converged = False
        P_mid = (P_low + P_high) / 2
        V_mid = self._evaluate_model(I, T, P_mid)
        residual = V_mid - V_target

        for iteration in range(max_iter):
            P_mid = (P_low + P_high) / 2
            V_mid = self._evaluate_model(I, T, P_mid)
            residual = V_mid - V_target

            if abs(residual) < tolerance:
                converged = True
                break

            if V_mid < V_target:
                P_low = P_mid
            else:
                P_high = P_mid

        # Compute gradient at final point
        _, _, gradient, _ = self._newton_step_P(P_mid, V_target, I, T)

        return {
            'P_max': P_mid,
            'V_achieved': V_mid,
            'converged': converged,
            'iterations': iteration + 1,
            'gradient': gradient,
            'method': 'bisection'
        }

    def _hybrid_optimization(self, V_target: float, I: float, T: float,
                             P_init: float, max_iter: int,
                             tolerance: float) -> dict:
        """Newton-Raphson with bisection fallback."""
        P = P_init
        P_low, P_high = P_MIN, P_MAX
        converged = False
        method_used = 'newton'
        gradient = 0.01
        V = self._evaluate_model(I, T, P)

        for iteration in range(max_iter):
            P_new, residual, gradient, V = self._newton_step_P(P, V_target, I, T)

            if abs(residual) < tolerance:
                converged = True
                break

            step_valid = (P_MIN <= P_new <= P_MAX and abs(P_new - P) < 15.0)

            if step_valid:
                P = P_new
            else:
                method_used = 'hybrid'
                P_mid = (P_low + P_high) / 2
                V_mid = self._evaluate_model(I, T, P_mid)

                if V_mid < V_target:
                    P_low = P_mid
                else:
                    P_high = P_mid

                P = P_mid
                V = V_mid

            if residual < 0:
                P_low = max(P_low, P)
            else:
                P_high = min(P_high, P)

        return {
            'P_max': P,
            'V_achieved': V,
            'converged': converged,
            'iterations': iteration + 1 if max_iter > 0 else 0,
            'gradient': gradient,
            'method': method_used
        }

    def find_P_max(
        self,
        V_target: float,
        I: float,
        T: float,
        safety_margin_mV: float = 40.0,
        pressure_derating: float = 0.95,
        method: str = 'hybrid',
        max_iter: int = None,
        tolerance: float = None,
    ) -> OptimizationResult:
        """Find maximum pressure for given voltage limit.

        Args:
            V_target: Maximum allowable voltage [V].
            I: Current [A].
            T: Temperature [°C].
            safety_margin_mV: Voltage safety margin [mV] (subtracted from V_target).
            pressure_derating: Additional pressure safety factor (0-1).
            method: 'newton', 'bisection', or 'hybrid'.
            max_iter: Maximum iterations (default: 20).
            tolerance: Convergence tolerance [V] (default: 1e-4).

        Returns:
            OptimizationResult with P_max, P_safe, and convergence info.
        """
        start_time = time.time()
        max_iter = max_iter or DEFAULT_MAX_ITER
        tolerance = tolerance or DEFAULT_TOL

        V_safe = V_target - safety_margin_mV / 1000.0

        if method == 'bisection':
            result = self._bisection(V_safe, I, T, P_MIN, P_MAX, max_iter, tolerance)
        else:
            result = self._hybrid_optimization(V_safe, I, T, P_REF, max_iter, tolerance)

        P_max = np.clip(result['P_max'], P_MIN, P_MAX)
        P_safe = P_max * pressure_derating

        gradient = result.get('gradient', 0.01)
        if abs(gradient) > 1e-10:
            uncertainty_P = (self.model_uncertainty_mV / 1000.0) / abs(gradient)
        else:
            uncertainty_P = 5.0

        latency_ms = (time.time() - start_time) * 1000

        return OptimizationResult(
            P_max=float(P_max),
            P_safe=float(P_safe),
            V_achieved=result['V_achieved'],
            converged=result['converged'],
            iterations=result['iterations'],
            gradient_dVdP=gradient,
            uncertainty_P=uncertainty_P,
            latency_ms=latency_ms,
            method=result['method']
        )

    # --- Current optimization (find I_max) ---

    def _newton_step_I(self, I_current: float, V_target: float,
                       P: float, T: float):
        """Single Newton-Raphson step for current using autograd."""
        I_t = torch.tensor([I_current], requires_grad=True,
                           device=self.device, dtype=torch.float32)

        with torch.enable_grad():
            # Model input: [current_A, H2_P_bar, O2_P_bar, temperature_C]
            x = torch.stack([
                I_t,
                torch.tensor([P], device=self.device, dtype=torch.float32),
                torch.tensor([P], device=self.device, dtype=torch.float32),
                torch.tensor([T], device=self.device, dtype=torch.float32),
            ], dim=1)
            V_pred = self.model(x)

        dV_dI = torch.autograd.grad(
            V_pred, I_t, grad_outputs=torch.ones_like(V_pred), create_graph=False
        )[0]

        V = V_pred.item()
        residual = V - V_target
        gradient = dV_dI.item()

        if abs(gradient) < 1e-8:
            I_new = I_current - np.sign(residual) * 0.5
        else:
            I_new = I_current - residual / gradient

        return I_new, residual, gradient, V

    def find_I_max(
        self,
        V_target: float,
        P: float,
        T: float,
        safety_margin_mV: float = 40.0,
        current_derating: float = 0.95,
        max_iter: int = None,
        tolerance: float = None,
    ) -> CurrentOptimizationResult:
        """Find maximum current for given voltage limit.

        Args:
            V_target: Maximum allowable voltage [V].
            P: Operating pressure [bar].
            T: Temperature [°C].
            safety_margin_mV: Voltage safety margin [mV].
            current_derating: Current safety factor (0-1).
            max_iter: Maximum iterations (default: 20).
            tolerance: Convergence tolerance [V] (default: 1e-4).

        Returns:
            CurrentOptimizationResult with I_max, I_safe, and convergence info.
        """
        start_time = time.time()
        max_iter = max_iter or DEFAULT_MAX_ITER
        tolerance = tolerance or DEFAULT_TOL

        V_safe = V_target - safety_margin_mV / 1000.0

        # Check feasibility: V at I_MIN must be below target
        V_at_I_min = self._evaluate_model(I_MIN, T, P)
        if V_at_I_min >= V_safe:
            latency_ms = (time.time() - start_time) * 1000
            return CurrentOptimizationResult(
                I_max=I_MIN, I_safe=I_MIN, V_achieved=V_at_I_min,
                converged=False, feasible=False, iterations=0,
                gradient_dVdI=0.0, uncertainty_I=float('inf'),
                latency_ms=latency_ms, method='none'
            )

        # Bisection for I (more robust than Newton for current)
        I_low, I_high = I_MIN, I_MAX
        converged = False
        gradient = 0.01
        V = V_at_I_min

        for iteration in range(max_iter):
            I_mid = (I_low + I_high) / 2
            V = self._evaluate_model(I_mid, T, P)
            residual = V - V_safe

            if abs(residual) < tolerance:
                converged = True
                break

            if V < V_safe:
                I_low = I_mid
            else:
                I_high = I_mid

        # Get gradient at solution
        _, _, gradient, _ = self._newton_step_I(I_mid, V_safe, P, T)

        I_max = np.clip(I_mid, I_MIN, I_MAX)
        I_safe = I_max * current_derating

        if abs(gradient) > 1e-10:
            uncertainty_I = (self.model_uncertainty_mV / 1000.0) / abs(gradient)
        else:
            uncertainty_I = 1.0

        latency_ms = (time.time() - start_time) * 1000

        return CurrentOptimizationResult(
            I_max=float(I_max),
            I_safe=float(I_safe),
            V_achieved=V,
            converged=converged,
            feasible=True,
            iterations=iteration + 1,
            gradient_dVdI=gradient,
            uncertainty_I=uncertainty_I,
            latency_ms=latency_ms,
            method='bisection'
        )
