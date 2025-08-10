"""Top-level package for the A2A Network.

This allows Python to treat the Solidity project directory as a package so that
`import a2a_network.python_sdk ...` works from other repos (e.g. a2a_agents).

Nothing here depends on having the Solidity toolchain installed; it merely
exposes the embedded `python_sdk` sub-package.
"""

from importlib import import_module as _import_module

# Re-export the python_sdk sub-package under a shorter alias if desired
try:
    _import_module("a2a_network.python_sdk")
except ModuleNotFoundError:
    # The sub-package may not be available in some minimal builds; ignore.
    pass
