import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(autouse=True)
def _mock_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("BEDROCK_MODE", "mock")
    monkeypatch.setenv("AGENT_WORKDIR", str(tmp_path / "runs"))
    monkeypatch.delenv("GATEWAY_API_KEY", raising=False)
