"""
Smoke test: start the backend with the student model, hit /health and GET /.
Skips gracefully if model is missing (CI environment) or port 8000 is in use.
"""
import os
import sys
import socket
import subprocess
import time
import pytest
try:
    import requests
except ImportError:
    requests = None
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / 'results' / 'best_12param.pt'

# pinn_loader.py adds PROJECT_ROOT/src to sys.path, where PROJECT_ROOT is
# digital_twin/.  That directory may not exist; fall back to the sibling
# pem-electrolyzer-inverse project's src/ which holds the real model code.
_DT_SRC = ROOT / 'digital_twin' / 'src'
_PEM_SRC = Path('/home/ubuntu/pem-electrolyzer-inverse/src')
_EXTRA_PYTHONPATH = str(_DT_SRC if _DT_SRC.exists() else _PEM_SRC)


def _port_in_use(port: int) -> bool:
    """Return True if the given TCP port is already bound on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(('127.0.0.1', port)) == 0


@pytest.fixture(scope='module')
def backend():
    """Start backend, yield, then terminate."""
    if requests is None:
        pytest.skip('requests not installed — skipping smoke tests')

    if not MODEL_PATH.exists():
        pytest.skip(f'Model not found at {MODEL_PATH} — run notebook Parts 1-6 first')

    if _port_in_use(8000):
        pytest.skip('Port 8000 is already in use — skipping live server smoke tests')

    # Compose PYTHONPATH so that both the backend package and the PINN model
    # code (models.physics_original_12param) are importable.
    existing_pp = os.environ.get('PYTHONPATH', '')
    pythonpath = os.pathsep.join(filter(None, [
        str(ROOT / 'digital_twin'),  # makes 'import backend.xxx' work
        _EXTRA_PYTHONPATH,           # makes 'import models.xxx' work
        existing_pp,
    ]))

    env = {**os.environ, 'MODEL_PATH': str(MODEL_PATH), 'PYTHONPATH': pythonpath}
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / 'digital_twin' / 'backend' / 'server.py')],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
    )
    # Wait up to 20s for /health to respond
    ready = False
    for _ in range(40):
        try:
            r = requests.get('http://localhost:8000/health', timeout=1)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            time.sleep(0.5)

    if not ready:
        # Print backend output to help debug
        proc.terminate()
        out = proc.stdout.read().decode(errors='replace')
        pytest.fail(f'Backend did not start within 20s. Output:\n{out}')

    yield proc
    proc.terminate()
    proc.wait(timeout=5)


def test_health_endpoint(backend):
    r = requests.get('http://localhost:8000/health', timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data.get('status') == 'healthy', f"Unexpected health response: {data}"


def test_html_served_at_root(backend):
    r = requests.get('http://localhost:8000/', timeout=5)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    assert 'text/html' in r.headers.get('content-type', ''), \
        f"Expected text/html, got {r.headers.get('content-type')}"
    # The HTML should contain some marker from digital_twin_3d.html
    assert 'THREE' in r.text or 'electrolyzer' in r.text.lower() or 'digital_twin' in r.text.lower(), \
        "HTML response should contain Three.js or electrolyzer content"
