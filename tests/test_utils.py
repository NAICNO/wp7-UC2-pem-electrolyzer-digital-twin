"""
Unit tests for utils.py (root-level cluster utilities).

Focuses on testable logic — get_available_nodes filtering/shuffling and
submit_slurm_job command construction — using mock SSH clients so no
real cluster is required.
"""
import sys
import random
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import utils


# ---------------------------------------------------------------------------
# get_available_nodes
# ---------------------------------------------------------------------------

class TestGetAvailableNodes:
    def _make_client(self, stdout_lines):
        """Return a mock paramiko-style SSH client."""
        client = MagicMock()
        stdout = MagicMock()
        stdout.read.return_value = '\n'.join(stdout_lines).encode()
        client.exec_command.return_value = (MagicMock(), stdout, MagicMock())
        return client

    def test_returns_list(self):
        client = self._make_client(['node01 4/0/0/4', 'node02 4/0/0/4'])
        result = utils.get_available_nodes(client)
        assert isinstance(result, list)

    def test_returns_at_most_four(self):
        lines = [f'node{i:02d} 4/0/0/4' for i in range(10)]
        client = self._make_client(lines)
        result = utils.get_available_nodes(client)
        assert len(result) <= 4

    def test_ignore_list_excludes_nodes(self):
        client = self._make_client(['node01 4/0/0/4', 'node02 4/0/0/4'])
        result = utils.get_available_nodes(client, ignore_list=['node01'])
        assert 'node01' not in result

    def test_empty_stdout_returns_empty(self):
        client = self._make_client([])
        result = utils.get_available_nodes(client)
        assert result == []

    def test_malformed_lines_skipped(self):
        """Lines with fewer than 2 fields must be ignored silently."""
        client = self._make_client(['', 'onlyone', 'node01 info'])
        result = utils.get_available_nodes(client)
        assert 'node01' in result

    def test_default_ignore_list_is_none(self):
        client = self._make_client(['node01 4/0/0/4'])
        # Should not raise when ignore_list is omitted
        result = utils.get_available_nodes(client)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# submit_slurm_job
# ---------------------------------------------------------------------------

class TestSubmitSlurmJob:
    def _make_client(self, output='Submitted batch job 12345'):
        client = MagicMock()
        stdout = MagicMock()
        stdout.read.return_value = output.encode()
        stderr = MagicMock()
        stderr.read.return_value = b''
        client.exec_command.return_value = (MagicMock(), stdout, stderr)
        return client

    def test_returns_string(self):
        client = self._make_client()
        result = utils.submit_slurm_job(client, 'node01', 'myaccount', '--mode full')
        assert isinstance(result, str)

    def test_exec_command_called(self):
        client = self._make_client()
        utils.submit_slurm_job(client, 'node01', 'myaccount', '--mode full')
        client.exec_command.assert_called_once()

    def test_command_contains_sbatch(self):
        client = self._make_client()
        utils.submit_slurm_job(client, 'node01', 'myaccount', '--mode full')
        cmd = client.exec_command.call_args[0][0]
        assert 'sbatch' in cmd

    def test_command_contains_node(self):
        client = self._make_client()
        utils.submit_slurm_job(client, 'node42', 'myaccount', '--mode full')
        cmd = client.exec_command.call_args[0][0]
        assert 'node42' in cmd

    def test_command_contains_account(self):
        client = self._make_client()
        utils.submit_slurm_job(client, 'node01', 'proj_account', '--mode full')
        cmd = client.exec_command.call_args[0][0]
        assert 'proj_account' in cmd

    def test_gpu_flag_included_when_true(self):
        client = self._make_client()
        utils.submit_slurm_job(client, 'node01', 'acc', '', gpu=True)
        cmd = client.exec_command.call_args[0][0]
        assert '--gres=gpu:1' in cmd

    def test_no_gpu_flag_when_false(self):
        client = self._make_client()
        utils.submit_slurm_job(client, 'node01', 'acc', '', gpu=False)
        cmd = client.exec_command.call_args[0][0]
        assert '--gres=gpu:1' not in cmd

    def test_script_args_included(self):
        client = self._make_client()
        utils.submit_slurm_job(client, 'node01', 'acc', '--epochs 100 --lr 0.01')
        cmd = client.exec_command.call_args[0][0]
        assert '--epochs 100 --lr 0.01' in cmd

    def test_error_output_does_not_raise(self):
        """stderr content should be printed but not cause an exception."""
        client = MagicMock()
        stdout = MagicMock()
        stdout.read.return_value = b'Submitted batch job 1'
        stderr = MagicMock()
        stderr.read.return_value = b'some warning'
        client.exec_command.return_value = (MagicMock(), stdout, stderr)
        result = utils.submit_slurm_job(client, 'node01', 'acc', '')
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# connect_ssh — guards against missing USERNAME
# ---------------------------------------------------------------------------

class TestConnectSSH:
    def test_missing_username_raises_value_error(self, monkeypatch):
        monkeypatch.setattr(utils, 'USERNAME', '')
        with pytest.raises(Exception):
            # connect_ssh will try to import paramiko; mock it
            with patch.dict('sys.modules', {'paramiko': MagicMock()}):
                utils.connect_ssh()
