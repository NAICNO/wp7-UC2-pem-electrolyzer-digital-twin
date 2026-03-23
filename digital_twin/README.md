# 3D Digital Twin — PEM Electrolyzer

Visualises a 4-cell PEM electrolyzer stack in real time using your trained PINN model.

## Quick Start (from the demonstrator notebook)

Run **Part 7** cells in `demonstrator-v1.orchestrator.ipynb` — they start the backend and open the twin automatically.

## Manual Start

```bash
MODEL_PATH=results/best_12param.pt python digital_twin/backend/server.py
```
Then open http://localhost:8000

## SSH Tunnel (remote VM)

```bash
ssh -L 8888:localhost:8888 -L 8000:localhost:8000 user@vm.sigma2.no
```

## Architecture

```
best_12param.pt  →  FastAPI (port 8000)  →  WebSocket  →  Three.js 3D viewer
                                          └─ GET /      →  digital_twin_3d.html
```
