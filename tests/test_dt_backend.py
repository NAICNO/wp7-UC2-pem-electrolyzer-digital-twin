"""Content assertions for digital_twin/backend/."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, 'digital_twin', 'backend')

def _read(filename):
    with open(os.path.join(BACKEND, filename)) as f:
        return f.read()

def test_server_imports_fileresponse():
    src = _read('server.py')
    assert 'FileResponse' in src, \
        "server.py must import FileResponse to serve the HTML"

def test_server_has_root_route():
    src = _read('server.py')
    assert '@app.get("/")' in src, \
        "server.py must have a GET / route"

def test_server_model_path_uses_env():
    src = _read('server.py')
    assert 'MODEL_PATH' in src and 'environ' in src, \
        "server.py must read MODEL_PATH from os.environ"

def test_server_health_endpoint():
    src = _read('server.py')
    assert '/health' in src, \
        "server.py must have /health endpoint"

def test_pinn_loader_exists():
    assert os.path.isfile(os.path.join(BACKEND, 'pinn_loader.py')), \
        "digital_twin/backend/pinn_loader.py must exist"

def test_lbm_gpu_exists():
    assert os.path.isfile(os.path.join(BACKEND, 'lbm_gpu.py')), \
        "digital_twin/backend/lbm_gpu.py must exist"

def test_simulation_state_exists():
    assert os.path.isfile(os.path.join(BACKEND, 'simulation_state.py')), \
        "digital_twin/backend/simulation_state.py must exist"

def test_requirements_has_fastapi():
    req = _read('requirements.txt')
    assert 'fastapi' in req.lower(), \
        "requirements.txt must list fastapi"

def test_requirements_has_uvicorn():
    req = _read('requirements.txt')
    assert 'uvicorn' in req.lower(), \
        "requirements.txt must list uvicorn"
