"""Content assertions for digital_twin/ directory structure."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT_DIR = os.path.join(ROOT, 'digital_twin')
BACKEND_DIR = os.path.join(DT_DIR, 'backend')

def test_digital_twin_dir_exists():
    assert os.path.isdir(DT_DIR), "digital_twin/ must exist"

def test_backend_dir_exists():
    assert os.path.isdir(BACKEND_DIR), "digital_twin/backend/ must exist"

def test_readme_exists():
    assert os.path.isfile(os.path.join(DT_DIR, 'README.md')), \
        "digital_twin/README.md must exist"

def test_server_py_exists():
    assert os.path.isfile(os.path.join(BACKEND_DIR, 'server.py')), \
        "digital_twin/backend/server.py must exist"

def test_pinn_loader_exists():
    assert os.path.isfile(os.path.join(BACKEND_DIR, 'pinn_loader.py')), \
        "digital_twin/backend/pinn_loader.py must exist"

def test_html_exists():
    assert os.path.isfile(os.path.join(DT_DIR, 'digital_twin_3d.html')), \
        "digital_twin/digital_twin_3d.html must exist"

def test_requirements_exists():
    assert os.path.isfile(os.path.join(BACKEND_DIR, 'requirements.txt')), \
        "digital_twin/backend/requirements.txt must exist"
