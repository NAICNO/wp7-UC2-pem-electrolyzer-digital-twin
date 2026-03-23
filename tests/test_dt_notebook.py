"""Content assertions for Part 7 digital twin cells in the demonstrator notebook."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(ROOT, 'demonstrator-v1.orchestrator.ipynb')

def _nb():
    with open(NB_PATH) as f:
        return json.load(f)

def _all_source(nb):
    """Concatenate all cell sources into one string."""
    return '\n'.join(
        ''.join(cell['source'])
        for cell in nb['cells']
    )

def test_part7_heading_present():
    nb = _nb()
    src = _all_source(nb)
    assert 'Part 7' in src, "Notebook must have Part 7 section"

def test_model_path_assertion():
    nb = _nb()
    src = _all_source(nb)
    assert 'best_12param.pt' in src, \
        "Part 7 must reference best_12param.pt"

def test_subprocess_popen_present():
    nb = _nb()
    src = _all_source(nb)
    assert 'subprocess.Popen' in src, \
        "Part 7 must start backend via subprocess.Popen"

def test_model_path_env_var():
    nb = _nb()
    src = _all_source(nb)
    assert 'MODEL_PATH' in src, \
        "Part 7 must pass MODEL_PATH env var to backend"

def test_health_check_cell():
    nb = _nb()
    src = _all_source(nb)
    assert '/health' in src, \
        "Part 7 must poll /health endpoint to confirm backend is up"

def test_launch_link_cell():
    nb = _nb()
    src = _all_source(nb)
    assert 'localhost:8000' in src, \
        "Part 7 must show launch link to http://localhost:8000"

def test_cleanup_cell():
    nb = _nb()
    src = _all_source(nb)
    assert 'proc.terminate' in src, \
        "Part 7 must have cleanup cell with proc.terminate()"
