# AI Agent Setup Instructions

## Quick Start

### Step 1: SSH into your NAIC VM

Replace the IP address with the one shown in the NAIC Orchestrator portal.
Do NOT type the angle brackets -- use the actual IP and key path.

```bash
# Example with a .pem key (common on NAIC):
ssh -i ~/.ssh/naic-vm.pem ubuntu@10.212.136.52

# Example with a standard key:
ssh -i ~/.ssh/id_rsa ubuntu@10.212.136.52

# If you get "Permission denied", check:
#   1. The key file has correct permissions:  chmod 600 ~/.ssh/naic-vm.pem
#   2. You are using the right username (ubuntu, not root)
#   3. The IP matches your VM in orchestrator.naic.no
```

### Step 2: Initialize VM (first time only)

```bash
curl -O https://raw.githubusercontent.com/NAICNO/wp7-UC2-pem-electrolyzer-digital-twin/main/vm-init.sh
chmod +x vm-init.sh
./vm-init.sh
```

### Step 3: Clone and setup

```bash
git clone https://github.com/NAICNO/wp7-UC2-pem-electrolyzer-digital-twin.git
cd uc2-pem-electrolyzer-pinn-optimizer
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### Step 4: Run quick test

```bash
python scripts/pem_electrolyzer/main.py --mode quick-test
```

### Step 5: Full training

```bash
python scripts/pem_electrolyzer/main.py --mode full --device cuda --epochs 100
```

## Jupyter Notebook Access

Start Jupyter on the VM, then create an SSH tunnel from your laptop.

**On the VM:**
```bash
cd uc2-pem-electrolyzer-pinn-optimizer
source venv/bin/activate
jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```

**On your laptop** (new terminal -- replace IP and key path with yours):
```bash
# Example:
ssh -v -N -L 8888:localhost:8888 -L 8000:localhost:8000 -i ~/.ssh/naic-vm.pem ubuntu@10.212.136.52
#                                  ↑ Also forwards Digital Twin backend (Part 7)

# Then open in your browser:
#   http://localhost:8888
```

Common mistakes:
- Do NOT keep the angle brackets. `ubuntu@<VM_IP>` means type `ubuntu@10.212.136.52` (your actual IP).
- `-N` means "no remote command" -- the terminal will appear to hang. That is normal.
- `-v` enables verbose output so you can see connection progress.
- If port 8888 is already in use locally, pick another: `-L 9999:localhost:8888` then open `http://localhost:9999`.

## Verification Steps

1. Check Python: `python3 --version` (need 3.8+)
2. Check GPU: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check data: `ls dataset/*.csv` (should show 3 files)
4. Quick test: `python scripts/pem_electrolyzer/main.py --mode quick-test`
5. Check results: `ls results/` (should show results.json after training)

## Project Structure

```
uc2-pem-electrolyzer-pinn-optimizer/
├── AGENT.md                    # This file
├── AGENT.yaml                  # Machine-readable version
├── README.md                   # Project overview
├── setup.sh                    # Environment setup
├── vm-init.sh                  # VM initialization
├── requirements.txt            # ML dependencies
├── requirements-docs.txt       # Documentation dependencies
├── demonstrator-v1.orchestrator.ipynb  # Interactive notebook
├── utils.py                    # Cluster utilities
├── widgets.py                  # Jupyter widgets
├── dataset/                    # NORCE experimental data
│   ├── test2_subset.csv        # OOD evaluation (current sweep)
│   ├── test3_subset.csv        # OOD evaluation (pressure swap)
│   └── test4_subset.csv        # Training data (long-term stability)
├── scripts/pem_electrolyzer/   # ML training scripts
│   ├── main.py                 # CLI entry point
│   ├── models.py               # Model definitions
│   ├── inverse.py              # Inverse pressure optimizer
│   ├── dataloader.py           # Data loading
│   ├── trainer.py              # Teacher training
│   ├── distillation.py         # Knowledge distillation
│   ├── evaluation.py           # OOD evaluation
│   └── ablation.py             # Ablation study
├── results/                    # Output directory
└── content/                    # Sphinx documentation
```

## CLI Reference

```bash
python scripts/pem_electrolyzer/main.py [OPTIONS]

Modes:
  --mode full          Train teacher + student, evaluate OOD (default)
  --mode quick-test    5 epochs, fast verification
  --mode teacher-only  Train only teacher model
  --mode ablation      7 experiments × 3 seeds = 21 runs
  --mode inverse       Find max safe pressure (requires trained checkpoint)

Training Options:
  --data-dir PATH      Dataset directory (default: dataset/)
  --output-dir PATH    Output directory (default: results/)
  --epochs N           Training epochs (default: 100)
  --seed N             Random seed (default: 42)
  --alpha FLOAT        Distillation weight (default: 0.1)
  --batch-size N       Batch size (default: 4096)
  --lr FLOAT           Learning rate (default: 0.01)
  --device DEVICE      cuda/cpu/auto (default: auto)

Inverse Solver Options (--mode inverse):
  --voltage FLOAT      Target voltage [V] (required for P_max search)
  --current FLOAT      Current [A]
  --temperature FLOAT  Temperature [°C]
  --pressure FLOAT     Pressure [bar] (for voltage prediction)
  --checkpoint PATH    Model checkpoint (default: results/best_12param.pt)
  --safety-margin FLOAT Safety margin [mV] (default: 40)
  --json               Output as JSON
```

### Inverse Solver Examples

Find maximum safe pressure:
```bash
python scripts/pem_electrolyzer/main.py --mode inverse \
    --voltage 1.85 --current 10 --temperature 75
```

Predict voltage at given conditions:
```bash
python scripts/pem_electrolyzer/main.py --mode inverse \
    --current 10 --temperature 75 --pressure 20
```

JSON output for integration:
```bash
python scripts/pem_electrolyzer/main.py --mode inverse \
    --voltage 1.85 --current 10 --temperature 75 --json
```

### Parameter Sweeps

Sweep current at fixed temperature and pressure to find safe operating envelope:
```bash
for I in 5 8 10 12 15; do
  python scripts/pem_electrolyzer/main.py --mode inverse \
      --voltage 1.85 --current $I --temperature 75 --json
done
```

Sweep temperature at fixed current:
```bash
for T in 60 65 70 75 80; do
  python scripts/pem_electrolyzer/main.py --mode inverse \
      --voltage 1.85 --current 10 --temperature $T --json
done
```

Voltage map across operating conditions:
```bash
for I in 5 8 10 12 15; do
  for P in 5 10 15 20 25 30; do
    python scripts/pem_electrolyzer/main.py --mode inverse \
        --current $I --temperature 75 --pressure $P --json
  done
done
```

Ablation study (multiple seeds):
```bash
for SEED in 42 123 456; do
  python scripts/pem_electrolyzer/main.py --mode full \
      --seed $SEED --epochs 100 --device cuda
done
```

## Models

| Model | Type | Params | Description |
|-------|------|--------|-------------|
| Teacher (HybridPhysicsMLP) | Physics+MLP | ~9,354 | 8 physics params + MLP residual |
| Student (PhysicsHybrid12Param) | Physics | 12 | 6 physics + 6 hybrid correction |
| PureMLP | ML | ~9,300 | No-physics baseline |
| BigMLP | ML | ~50,000 | Large no-physics baseline |
| Transformer | ML | ~50,000 | Self-attention baseline |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| SSH "Permission denied (publickey)" | Check key permissions: `chmod 600 ~/.ssh/your-key.pem` |
| SSH hangs or times out | Verify VM IP in orchestrator.naic.no; check VPN if required |
| Typed `<VM_IP>` literally | Replace with your actual IP, e.g. `ubuntu@10.212.136.52` |
| Jupyter tunnel not working | Make sure `-N` flag is present; check port isn't already used |
| CUDA out of memory | Reduce batch size: `--batch-size 1024` |
| ModuleNotFoundError | Activate venv: `source venv/bin/activate` |
| Permission denied (scripts) | `chmod +x setup.sh vm-init.sh` |
| No GPU detected | Check `nvidia-smi`; install CUDA drivers |
| Data not found | Check `ls dataset/*.csv`; run `dataset/extract_data.py` if needed |
| Checkpoint not found (inverse) | Train first with `--mode full`, or provide `--checkpoint` path |
