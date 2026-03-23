# Setting Up the Environment

```{objectives}
- Connect to your VM via SSH
- Initialize a fresh VM with required packages
- Clone the repository and set up the Python environment
- Start Jupyter Lab with SSH tunneling
- Verify PyTorch and GPU access
```

## 1. Connect to Your VM

Connect to your VM using SSH (see Episode 02 for Windows-specific instructions):

````{tabs}
```{tab} macOS / Linux / Git Bash
chmod 600 /path/to/your-key.pem
ssh -i /path/to/your-key.pem ubuntu@<VM_IP>
```

```{tab} Windows (PowerShell)
ssh -i "C:\Users\YourName\Downloads\your-key.pem" ubuntu@<VM_IP>
```
````

```{note}
**Windows users**: If you see "Permissions for key are too open", fix the key permissions first. See Episode 02, Step 7 for detailed instructions. Git Bash is recommended — it supports `chmod` natively.
```

## 2. System Setup (Fresh VM)

On a fresh NAIC VM, install required system packages:

```bash
sudo apt update -y
sudo apt install -y build-essential git python3-dev python3-venv python3-pip libssl-dev zlib1g-dev
```

This installs:
- `git` -- For cloning the repository
- `build-essential` -- Compiler toolchain (gcc, make)
- `python3-dev`, `python3-venv`, `python3-pip` -- Python development tools
- `libssl-dev`, `zlib1g-dev` -- Required for building Python packages

Alternatively, the repository includes a `vm-init.sh` script that automates system setup:

```bash
curl -O https://raw.githubusercontent.com/NAICNO/wp7-UC2-pem-electrolyzer-digital-twin/main/vm-init.sh
chmod +x vm-init.sh
./vm-init.sh
```

This will detect if module system (EasyBuild/Lmod) is available, install system packages if needed, and check GPU availability.

## 3. Clone and Setup

```bash
git clone https://github.com/NAICNO/wp7-UC2-pem-electrolyzer-digital-twin.git
cd uc2-pem-electrolyzer-pinn-optimizer
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

The `setup.sh` script automatically:
1. Loads the Python module (if available via Lmod)
2. Validates Python version (3.8+ required)
3. Checks GPU and sets up CUDA symlinks
4. Creates a Python virtual environment
5. Installs dependencies from `requirements.txt`
6. Verifies PyTorch installation

## 4. Quick Verification

```bash
# Quick test (5 epochs, should complete in under a minute)
python scripts/pem_electrolyzer/main.py --mode quick-test

# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 5. Start Jupyter Lab (Optional)

For interactive exploration, start Jupyter Lab inside a tmux session for persistence:

```bash
# Use tmux for persistence
tmux new -s jupyter
cd ~/uc2-pem-electrolyzer-pinn-optimizer
source venv/bin/activate
jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --NotebookApp.token='' --NotebookApp.password=''
# Detach with Ctrl+B, then D
```

## 6. Create SSH Tunnel (on your local machine)

To access Jupyter Lab from your local browser, create an SSH tunnel. Open a **new terminal** on your local machine (not the VM):

````{tabs}
```{tab} macOS / Linux / Git Bash
# Verbose mode (recommended - shows connection status)
ssh -v -N -L 8888:localhost:8888 -i /path/to/your-key.pem ubuntu@<VM_IP>
```

```{tab} Windows (PowerShell)
ssh -v -N -L 8888:localhost:8888 -i "C:\Users\YourName\Downloads\your-key.pem" ubuntu@<VM_IP>
```
````

> **Note:** The tunnel will appear to "hang" after connecting -- this is normal! It means the tunnel is active. Keep the terminal open while using Jupyter.

**If port 8888 is already in use**, use an alternative port:

```bash
ssh -v -N -L 9999:localhost:8888 -i /path/to/your-key.pem ubuntu@<VM_IP>
# Then access via http://localhost:9999
```

Then navigate to: **http://localhost:8888/lab/tree/demonstrator-v1.orchestrator.ipynb**

To close the tunnel, press `Ctrl+C` in the terminal.

## Project Structure

After cloning, you will have:

```
uc2-pem-electrolyzer-pinn-optimizer/
├── dataset/
│   ├── test2_subset.csv
│   ├── test3_subset.csv
│   └── test4_subset.csv
├── scripts/
│   └── pem_electrolyzer/
│       ├── main.py
│       ├── models.py
│       ├── dataloader.py
│       ├── trainer.py
│       ├── distillation.py
│       ├── evaluation.py
│       └── plotting.py
├── content/           (Sphinx docs)
├── setup.sh
├── vm-init.sh
├── utils.py
├── widgets.py
├── demonstrator-v1.orchestrator.ipynb
├── requirements.txt
└── requirements-docs.txt
```

## Key Dependencies

The following packages are installed via `setup.sh`:

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework (PINN training) |
| `numpy` | Numerical computation |
| `pandas` | Data loading and manipulation |
| `scipy` | Scientific computing (optimization) |
| `matplotlib` | Plotting and visualization |
| `tqdm` | Progress bars for training loops |
| `jupyterlab` | Interactive notebook environment |
| `ipywidgets` | Interactive widgets for demonstrator |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Check PYTHONPATH: `export PYTHONPATH=$(pwd)` |
| CUDA not available | Install GPU drivers: `nvidia-smi` to verify |
| SSH connection refused | Check key permissions: `chmod 600 /path/to/key.pem` |
| SSH "Permissions too open" (Windows) | Use Git Bash (`chmod 600`) or fix via icacls — see Episode 02 |
| SSH connection timed out | Your IP may not be whitelisted — add it at orchestrator.naic.no |
| Port 8888 already in use | Use alternative port: `-L 9999:localhost:8888` |
| `pip install` fails | Update pip: `pip install --upgrade pip` |
| `ImportError` after install | Activate venv first: `source venv/bin/activate` |
| Jupyter notebook won't start | Check installation: `pip install jupyterlab` |
| Data files not found | Verify dataset directory: `ls dataset/` |
| SSH tunnel appears to hang | This is normal -- tunnel is active, keep terminal open |
| Host key verification failed | Remove old key: `ssh-keygen -R <VM_IP>` |

```{keypoints}
- Set SSH key permissions with `chmod 600` before connecting (use Git Bash on Windows)
- Initialize fresh VMs with `sudo apt install -y build-essential git python3-dev python3-venv`
- Clone this repository directly -- all code and data are included
- Run `./setup.sh` to automatically set up the Python environment
- Use tmux for persistent Jupyter Lab sessions
- Create an SSH tunnel to access Jupyter from your local browser
- Use `--mode quick-test` for fast verification after setup
- Windows users: Git Bash is recommended for the best experience with SSH and Unix commands
```
