"""
SDK Mixins for A2A Agents
Provides common functionality for agents
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class PerformanceMonitorMixin:
    """
    Mixin for performance monitoring capabilities
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._performance_metrics = {
            "requests_processed": 0,
            "average_response_time": 0.0,
            "error_count": 0,
            "success_rate": 100.0
        }

    def update_performance_metrics(self, response_time: float, success: bool = True):
        """Update performance metrics"""
        self._performance_metrics["requests_processed"] += 1

        # Update average response time
        current_avg = self._performance_metrics["average_response_time"]
        count = self._performance_metrics["requests_processed"]
        self._performance_metrics["average_response_time"] = (
            (current_avg * (count - 1) + response_time) / count
        )

        # Update success rate
        if not success:
            self._performance_metrics["error_count"] += 1

        self._performance_metrics["success_rate"] = (
            (count - self._performance_metrics["error_count"]) / count * 100
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self._performance_metrics.copy()


class SecurityHardenedMixin:
    """
    Mixin for security hardening capabilities
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._security_config = {
            "rate_limiting_enabled": True,
            "input_validation_enabled": True,
            "audit_logging_enabled": True,
            "max_requests_per_minute": 100
        }
        self._request_counts = {}

    def validate_request(self, request_data: Dict[str, Any]) -> bool:
        """Validate incoming request for security"""
        if not self._security_config["input_validation_enabled"]:
            return True

        # Basic validation
        if not isinstance(request_data, dict):
            logger.warning("Request validation failed: not a dictionary")
            return False

        # Check for required fields
        if "method" not in request_data:
            logger.warning("Request validation failed: missing method")
            return False

        return True

    def check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits"""
        if not self._security_config["rate_limiting_enabled"]:
            return True

        import time
        current_time = time.time()

        # Clean old entries (older than 1 minute)
        cutoff_time = current_time - 60
        for cid in list(self._request_counts.keys()):
            self._request_counts[cid] = [
                t for t in self._request_counts[cid] if t > cutoff_time
            ]
            if not self._request_counts[cid]:
                del self._request_counts[cid]

        # Check current client rate
        if client_id not in self._request_counts:
            self._request_counts[client_id] = []

        client_requests = self._request_counts[client_id]
        if len(client_requests) >= self._security_config["max_requests_per_minute"]:
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            return False

        # Add current request
        client_requests.append(current_time)
        return True

    def audit_log(self, event: str, details: Dict[str, Any]):
        """Log security-relevant events"""
        if self._security_config["audit_logging_enabled"]:
            logger.info(f"AUDIT: {event} - {details}")


class CachingMixin:
    """
    Mixin for caching capabilities
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "hit_rate": 0.0
        }

    def cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self._cache:
            self._cache_stats["hits"] += 1
            self._update_cache_stats()
            return self._cache[key]
        else:
            self._cache_stats["misses"] += 1
            self._update_cache_stats()
            return None

    def cache_set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        import time
        cache_entry = {
            "value": value,
            "timestamp": time.time(),
            "ttl": ttl
        }
        self._cache[key] = cache_entry

    def cache_clear(self):
        """Clear the cache"""
        self._cache.clear()
        self._cache_stats = {"hits": 0, "misses": 0, "hit_rate": 0.0}

    def _update_cache_stats(self):
        """Update cache hit rate statistics"""
        total = self._cache_stats["hits"] + self._cache_stats["misses"]
        if total > 0:
            self._cache_stats["hit_rate"] = self._cache_stats["hits"] / total * 100

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self._cache_stats.copy()


class TelemetryMixin:
    """
    Mixin for telemetry and monitoring
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._telemetry_data = {
            "events": [],
            "metrics": {},
            "traces": []
        }

    def emit_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Emit a metric"""
        import time
        metric_entry = {
            "name": name,
            "value": value,
            "timestamp": time.time(),
            "tags": tags or {}
        }

        if name not in self._telemetry_data["metrics"]:
            self._telemetry_data["metrics"][name] = []

        self._telemetry_data["metrics"][name].append(metric_entry)

    def emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event"""
        import time
        event_entry = {
            "type": event_type,
            "data": data,
            "timestamp": time.time()
        }
        self._telemetry_data["events"].append(event_entry)

    def start_trace(self, trace_name: str) -> str:
        """Start a trace"""
        import uuid
        import time
        trace_id = str(uuid.uuid4())

        trace_entry = {
            "id": trace_id,
            "name": trace_name,
            "start_time": time.time(),
            "end_time": None,
            "spans": []
        }
        self._telemetry_data["traces"].append(trace_entry)
        return trace_id

    def end_trace(self, trace_id: str):
        """End a trace"""
        import time
        for trace in self._telemetry_data["traces"]:
            if trace["id"] == trace_id:
                trace["end_time"] = time.time()
                break

    def get_telemetry_data(self) -> Dict[str, Any]:
        """Get all telemetry data"""
        return self._telemetry_data.copy()