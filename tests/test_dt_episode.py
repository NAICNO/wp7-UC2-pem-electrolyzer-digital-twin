"""Content assertions for Sphinx Episode 09."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EP_PATH = os.path.join(ROOT, 'content', 'episodes', '09-digital-twin.md')
INDEX_PATH = os.path.join(ROOT, 'content', 'index.rst')

def _ep():
    with open(EP_PATH) as f:
        return f.read()

def test_episode_file_exists():
    assert os.path.isfile(EP_PATH), "content/episodes/09-digital-twin.md must exist"

def test_episode_has_title():
    src = _ep()
    assert src.startswith('# '), "Episode must start with a markdown h1 heading"

def test_episode_mentions_architecture():
    src = _ep()
    assert 'FastAPI' in src or 'WebSocket' in src or 'backend' in src.lower(), \
        "Episode must explain the architecture"

def test_episode_explains_panels():
    src = _ep()
    for panel in ['TWIN', 'SIMULATE', 'PREDICT', 'FIX']:
        assert panel in src, f"Episode must explain the {panel} panel"

def test_episode_has_exercises():
    src = _ep()
    assert 'Exercise' in src or 'exercise' in src, \
        "Episode must have exercises section"

def test_episode_mentions_model_path():
    src = _ep()
    assert 'MODEL_PATH' in src or 'best_12param' in src, \
        "Episode must mention MODEL_PATH or best_12param.pt"

def test_index_includes_episode09():
    with open(INDEX_PATH) as f:
        idx = f.read()
    assert '09-digital-twin' in idx, \
        "content/index.rst must include episodes/09-digital-twin in toctree"
