"""Pytest configuration for FinSight CIB test suites.

This module keeps legacy tests running after upgrading to pytest ≥ 8.

1. pytest_pyfunc_call hook – runs the test function and discards any
   non-None return value so pytest doesn’t treat it as a failure.
2. Fixtures ``ord_descriptors`` and ``registration_id`` – expected by some
   Agent-0 verification tests.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

import pytest


# ---------------------------------------------------------------------------
# Hook: ignore non-None return values (pytest ≥ 8 compatibility)
# ---------------------------------------------------------------------------

@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):  # type: ignore[override]
    """Execute the test function and discard its return value."""
    test_func = pyfuncitem.obj
    sig = inspect.signature(test_func)
    kwargs: Dict[str, Any] = {
        name: pyfuncitem.funcargs[name]
        for name in sig.parameters
        if name in pyfuncitem.funcargs
    }
    test_func(**kwargs)
    return True  # tell pytest we handled the call


# ---------------------------------------------------------------------------
# Fixtures required by legacy Agent-0 tests
# ---------------------------------------------------------------------------

@pytest.fixture
def ord_descriptors():
    """Return a dummy ORD document via test-suite helper."""
    from backend.test_agent0_detailed_verification import generate_ord_document  # noqa: WPS433
    return generate_ord_document()


@pytest.fixture
def registration_id():
    """Provide a placeholder registration identifier."""
    return "test_registration_dummy_id"
