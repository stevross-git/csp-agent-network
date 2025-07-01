import importlib
import os
import pytest


@pytest.mark.asyncio
async def test_initialize_sets_active():
    from monitoring.csp_monitoring import get_default

    monitor = get_default()
    await monitor.initialize()
    assert monitor.is_active
    await monitor.shutdown()
    assert not monitor.is_active


def test_import_works_with_and_without_flag(monkeypatch):
    monkeypatch.delenv("MONITORING_ENABLED", raising=False)
    mod = importlib.reload(importlib.import_module("enhanced_csp.backend.config.settings"))
    assert mod.settings.MONITORING_ENABLED is False
    from monitoring import csp_monitoring  # should import without error

    monkeypatch.setenv("MONITORING_ENABLED", "true")
    mod = importlib.reload(importlib.import_module("enhanced_csp.backend.config.settings"))
    assert mod.settings.MONITORING_ENABLED is True
    from monitoring import csp_monitoring as _  # reimport to ensure still works

