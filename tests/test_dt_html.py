"""Content assertions for digital_twin/digital_twin_3d.html."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML_PATH = os.path.join(ROOT, 'digital_twin', 'digital_twin_3d.html')

def _html():
    with open(HTML_PATH) as f:
        return f.read()

def test_html_file_exists():
    assert os.path.isfile(HTML_PATH), "digital_twin/digital_twin_3d.html must exist"

def test_ws_url_uses_location_host():
    html = _html()
    assert 'window.location.host' in html, \
        "WS URL must use window.location.host (includes port) not hardcoded hostname:port"

def test_no_hardcoded_localhost_ws():
    html = _html()
    assert 'ws://localhost:8000' not in html, \
        "Must not hardcode ws://localhost:8000"

def test_three_js_present():
    html = _html()
    assert 'THREE' in html or 'three' in html.lower(), \
        "Three.js must be referenced in the HTML"

def test_scenario_buttons_present():
    html = _html()
    assert 'runScenario' in html, "runScenario function must exist"

def test_predict_panel_present():
    html = _html()
    assert 'predict-body' in html or 'PREDICT' in html, \
        "Predict panel must be present"
