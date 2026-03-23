from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import asyncio
import json
import uvicorn
import time
import numpy as np
from backend.simulation_state import SimulationState
from backend.lbm_gpu import LBMSolverGPU, GPU_AVAILABLE
from backend.physics_coupling import PhysicsCoupling
from backend.pinn_loader import PINNLoader
from pathlib import Path


def _to_numpy(arr) -> np.ndarray:
    """Convert a CuPy or NumPy array to a plain NumPy array.

    Uses .get() for CuPy arrays (np.asarray does not work with CuPy 12+),
    and is a no-op for arrays that are already NumPy.
    """
    if hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)

app = FastAPI(title="CFD Digital Twin Backend", version="0.1.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# Global simulation state
simulation_state = SimulationState()

# Initialize LBM solver (GPU if available, CPU fallback otherwise) and physics coupling
USE_GPU = GPU_AVAILABLE
lbm_solver = LBMSolverGPU(grid_size=50, use_gpu=USE_GPU)
if USE_GPU:
    print("CFD simulation using GPU acceleration (CuPy)")
    # Warm-up: run one step so JIT kernels are compiled at import time.
    # This prevents the first simulation_loop iteration from being unexpectedly
    # slow due to CuPy kernel compilation, and ensures the test that checks
    # for field population within 50ms works correctly.
    lbm_solver.step(tau=1.0)
    print("GPU warmup step complete")
else:
    print("CFD simulation using CPU (NumPy)")
physics_coupling = PhysicsCoupling(cfd_grid_size=50, pinn_grid_size=10)

# Initialize PINN loader (global)
import os as _os
_DEFAULT_MODEL = str(Path(__file__).parent.parent.parent / 'demonstrators' / 'student_model.pt')
PINN_MODEL_PATH = _os.environ.get('MODEL_PATH', _DEFAULT_MODEL)
try:
    pinn_loader = PINNLoader(model_path=str(PINN_MODEL_PATH), device='cpu')
    print("✓ Using real PINN model for temperature prediction")
except Exception as e:
    print(f"⚠ Failed to load PINN model: {e}")
    print("  Falling back to mock temperatures")
    pinn_loader = None
    from backend.mock_pinn import generate_mock_temperatures

# FPS tracking
last_frame_time = time.time()
frame_count = 0
fps = 0.0

async def simulation_loop():
    """Main simulation loop running at ~30 Hz with LBM solver"""
    global last_frame_time, frame_count, fps

    target_dt = 1.0 / 30.0  # 30 Hz

    while True:
        loop_start = time.time()

        # Update simulation time
        simulation_state.time += target_dt

        # Generate temperature field from PINN or mock
        if pinn_loader is not None:
            temperatures_pinn = pinn_loader.predict_temperatures(
                current=simulation_state.current,
                temperature=simulation_state.temperature,
                pressure=simulation_state.pressure,
                grid_size=10
            )
        else:
            # Fallback to mock
            temperatures_pinn = generate_mock_temperatures(
                grid_size=10,
                current=simulation_state.current,
                temperature=simulation_state.temperature
            )

        # Convert GPU arrays to NumPy for CPU-side code
        u_np = _to_numpy(lbm_solver.u)
        rho_np = _to_numpy(lbm_solver.rho)

        # Couple physics: interpolate temperatures to CFD grid
        temperatures_cfd, _ = physics_coupling.update_coupling(
            temperatures_pinn,
            u_np
        )

        # Compute void fractions from LBM density (before update)
        void_fractions = 1.0 - (rho_np / np.max(rho_np))
        void_fractions = np.clip(void_fractions, 0, 0.5)

        # Compute per-cell voltages using PINN + health modifiers
        if pinn_loader is not None:
            cell_telemetry = pinn_loader.compute_cell_voltages(
                current=simulation_state.current,
                temperature=simulation_state.temperature,
                pressure=simulation_state.pressure,
                cell_modifiers=simulation_state.cell_modifiers,
            )
        else:
            cell_telemetry = [
                {
                    'voltage': 1.75,
                    'current': simulation_state.current,
                    'power': simulation_state.current * 1.75,
                    'efficiency': round(1.48 / 1.75, 4),
                    'membraneHealth': m.get('membraneHealth', 1.0),
                }
                for m in simulation_state.cell_modifiers
            ]

        # Run LBM solver with physics coupling (multiple substeps for stability)
        for _ in range(5):
            lbm_solver.step(tau=1.0, temperatures=temperatures_cfd, void_fraction=void_fractions)

        # Extract velocity field from LBM (convert to 3D for compatibility)
        u_np = _to_numpy(lbm_solver.u)  # refresh after steps
        simulation_state.velocities = np.zeros((50, 50, 3))
        simulation_state.velocities[:, :, :2] = u_np  # vx, vy
        # vz will be added later when we have 3D flow

        # Store computed void fractions
        simulation_state.void_fractions = void_fractions

        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            current_time = time.time()
            fps = 30.0 / (current_time - last_frame_time)
            last_frame_time = current_time

        # Broadcast state to all connected clients
        message = {
            "type": "simulation_state",
            "data": {
                "time": simulation_state.time,
                "fps": fps,
                "velocities": simulation_state.velocities.tolist(),
                "voidFractions": simulation_state.void_fractions.tolist(),
                "temperatures": temperatures_cfd.tolist(),  # Now includes PINN temperatures
                "cells": cell_telemetry
            }
        }
        await manager.broadcast(message)

        # Sleep to maintain target frame rate
        elapsed = time.time() - loop_start
        sleep_time = max(0, target_dt - elapsed)
        await asyncio.sleep(sleep_time)

# Start simulation loop on startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(simulation_loop())

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def serve_html():
    html_path = Path(__file__).parent.parent / 'digital_twin_3d.html'
    return FileResponse(str(html_path), media_type='text/html')

@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()

            # Handle parameter updates
            if data.get("type") == "params":
                params = data.get("data", {})
                simulation_state.update_params(
                    current=params.get("current"),
                    temperature=params.get("temperature"),
                    pressure=params.get("pressure")
                )
                print(f"Parameters updated: I={simulation_state.current}A, "
                      f"T={simulation_state.temperature}°C, P={simulation_state.pressure}bar")

                # Accept per-cell modifiers from frontend (set by failure scenarios)
                if "cells" in params:
                    try:
                        simulation_state.update_cell_modifiers(params["cells"])
                    except (ValueError, KeyError, TypeError) as e:
                        print(f"Warning: invalid cell modifiers received: {e}")

                # Send acknowledgment
                await websocket.send_json({
                    "type": "params_ack",
                    "data": {
                        "current": simulation_state.current,
                        "temperature": simulation_state.temperature,
                        "pressure": simulation_state.pressure
                    }
                })
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
