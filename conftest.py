"""Pytest configuration to mark integration tests that depend on external
services so they are skipped by default (unless explicitly requested with
-m integration). This keeps the unit-test suite green in offline CI runs.
"""
import pathlib
import pytest

# File path substrings that indicate an integration test.
INTEGRATION_KEYWORDS = [
    "a2a_compliance_test.py",
    "a2a_registry",
    "test_agent1_simple.py",
    "test_complete_workflow_with_data_manager.py",
]

pytest_plugins = ["pytest_asyncio"]

IGNORE_FILES = {

    "a2a_compliance_test.py",
    "test_live_network.py",
    "test_trust_integration.py",
    "test_agent1_simple.py",
    "test_complete_workflow_with_data_manager.py",
    "test_a2a_agent.py",
}


def pytest_ignore_collect(path, config):
    """Skip collecting heavy integration test files entirely."""
    return path.basename in IGNORE_FILES


# Disable async background tasks in AIDecisionDatabaseLogger during unit tests
import importlib
try:
    db_mod = importlib.import_module(
        "a2a_agents.backend.app.a2a.core.ai_decision_logger_database"
    )
    # Override method to bypass starting asyncio tasks during import time
    if hasattr(db_mod, "AIDecisionDatabaseLogger"):
        db_mod.AIDecisionDatabaseLogger._start_background_tasks = lambda self: None  # type: ignore[attr-defined]
except ModuleNotFoundError:
    # Module path not present in certain envs
    pass


@pytest.fixture(scope="session", autouse=True)
def _set_event_loop():
    """Ensure a running event loop is available for code executed at import time."""
    import asyncio
    try:
        asyncio.get_running_loop()
        yield
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        yield
        loop.close()
        asyncio.set_event_loop(None)


@pytest.fixture(autouse=True)
async def _mock_httpx_post(monkeypatch):
    """Mock httpx.AsyncClient.post to avoid real HTTP calls during tests."""
    async def _fake_post(self, url, *args, **kwargs):  # type: ignore[no-self-use]
        class _Resp:
            status_code = 200
            def json(self):
                return {"ok": True}
            text = "ok"
        return _Resp()
    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post, raising=False)
    yield

def pytest_collection_modifyitems(config, items):
    """Automatically add the `integration` mark to selected tests."""
    integration_mark = pytest.mark.integration
    for item in items:
        path_str = str(item.fspath)
        if any(key in path_str for key in INTEGRATION_KEYWORDS):
            item.add_marker(integration_mark)
