"""
SDK Mixins for A2A Agents
"""

# Import all mixins
from .blockchainQueueMixin import BlockchainQueueMixin

# Define what gets imported with "from mixins import *"
__all__ = ['BlockchainQueueMixin']

# Create stub classes for optional mixins that may not be available
try:
    from .performanceMonitor import PerformanceMonitorMixin
    __all__.append('PerformanceMonitorMixin')
except ImportError:
    class PerformanceMonitorMixin:
        """Stub for PerformanceMonitorMixin when not available"""
        pass

try:
    from .securityHardened import SecurityHardenedMixin
    __all__.append('SecurityHardenedMixin')
except ImportError:
    class SecurityHardenedMixin:
        """Stub for SecurityHardenedMixin when not available"""
        pass

try:
    from .caching import CachingMixin
    __all__.append('CachingMixin')
except ImportError:
    class CachingMixin:
        """Stub for CachingMixin when not available"""
        pass

try:
    from .telemetry import TelemetryMixin


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
    __all__.append('TelemetryMixin')
except ImportError:
    class TelemetryMixin:
        """Stub for TelemetryMixin when not available"""
        pass
