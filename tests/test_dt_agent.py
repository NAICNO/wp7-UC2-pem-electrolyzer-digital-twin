"""Content assertions for AGENT.md port 8000 tunnel."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_PATH = os.path.join(ROOT, 'AGENT.md')

def _agent():
    with open(AGENT_PATH) as f:
        return f.read()

def test_port_8000_in_agent():
    src = _agent()
    assert '8000' in src, \
        "AGENT.md must mention port 8000 for digital twin SSH tunnel"

def test_both_ports_in_same_ssh_command():
    src = _agent()
    lines = src.splitlines()
    found = any('8888' in line and '8000' in line for line in lines)
    assert found, \
        "AGENT.md must have a single SSH command forwarding both port 8888 and 8000"
