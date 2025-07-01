import os
import importlib
import pytest
from pydantic import ValidationError

from enhanced_csp.main import RuntimeSettings, SchedulingPolicy


def test_default_policy():
    settings = RuntimeSettings()
    assert settings.scheduling_policy == SchedulingPolicy.round_robin


def test_env_override(monkeypatch):
    monkeypatch.setenv("RUNTIME_SCHEDULING_POLICY", "priority")
    # Recreate settings after env var
    settings = RuntimeSettings()
    assert settings.scheduling_policy == SchedulingPolicy.priority


def test_invalid_policy(monkeypatch):
    monkeypatch.setenv("RUNTIME_SCHEDULING_POLICY", "foobar")
    with pytest.raises(ValidationError):
        RuntimeSettings()
