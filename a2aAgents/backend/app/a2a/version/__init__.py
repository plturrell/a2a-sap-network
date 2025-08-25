"""
A2A Version Management System
Manages compatibility between a2aAgents and a2aNetwork
"""

from .versionManager import VersionManager, get_version_manager
from .compatibilityChecker import CompatibilityChecker
from .dependencyResolver import DependencyResolver

__version__ = "1.0.0"
__a2a_protocol_version__ = "0.2.9"
__network_compatibility__ = ["1.0.0", "1.0.1", "1.1.0"]

__all__ = [
    "VersionManager",
    "get_version_manager",
    "CompatibilityChecker",
    "DependencyResolver",
    "__version__",
    "__a2a_protocol_version__",
    "__network_compatibility__"
]
