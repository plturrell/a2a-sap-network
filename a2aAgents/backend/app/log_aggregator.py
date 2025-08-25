"""
Log Aggregator for A2A Platform
Collects and aggregates logs from all agents and services
"""
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import deque, defaultdict
import re
import aiofiles
import os
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/logs", tags=["Logs"])

# Log storage (in-memory for now, can be extended to use Redis/DB)
class LogEntry(BaseModel):
    timestamp: datetime
    level: str
    service: str
    message: str
    metadata: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None
    agent_id: Optional[str] = None

class LogAggregator:
    def __init__(self, max_logs: int = 10000):
        self.logs: deque = deque(maxlen=max_logs)
        self.log_levels = defaultdict(int)
        self.service_counts = defaultdict(int)
        self.error_patterns = defaultdict(int)
        self.log_file_positions = {}
        
    async def add_log(self, entry: LogEntry):
        """Add a log entry to the aggregator"""
        self.logs.append(entry)
        self.log_levels[entry.level] += 1
        self.service_counts[entry.service] += 1
        
        # Track error patterns
        if entry.level in ["ERROR", "CRITICAL"]:
            pattern = self._extract_error_pattern(entry.message)
            if pattern:
                self.error_patterns[pattern] += 1
    
    def _extract_error_pattern(self, message: str) -> Optional[str]:
        """Extract error pattern from message"""
        # Common error patterns
        patterns = [
            (r"ConnectionError.*", "Connection Error"),
            (r"TimeoutError.*", "Timeout Error"),
            (r"ValueError.*", "Value Error"),
            (r"KeyError.*", "Key Error"),
            (r".*authentication.*", "Authentication Error"),
            (r".*permission.*", "Permission Error"),
            (r".*memory.*", "Memory Error"),
            (r".*disk.*", "Disk Error"),
        ]
        
        for pattern, name in patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return name
        
        return None
    
    def get_logs(
        self,
        service: Optional[str] = None,
        level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        search: Optional[str] = None
    ) -> List[LogEntry]:
        """Get logs with filtering"""
        filtered_logs = list(self.logs)
        
        # Apply filters
        if service:
            filtered_logs = [log for log in filtered_logs if log.service == service]
        
        if level:
            filtered_logs = [log for log in filtered_logs if log.level == level]
        
        if start_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp >= start_time]
        
        if end_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp <= end_time]
        
        if search:
            filtered_logs = [
                log for log in filtered_logs 
                if search.lower() in log.message.lower()
            ]
        
        # Sort by timestamp descending and limit
        filtered_logs.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_logs[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get log statistics"""
        if not self.logs:
            return {
                "total_logs": 0,
                "log_levels": {},
                "services": {},
                "error_patterns": {},
                "time_range": None
            }
        
        oldest = min(self.logs, key=lambda x: x.timestamp).timestamp
        newest = max(self.logs, key=lambda x: x.timestamp).timestamp
        
        return {
            "total_logs": len(self.logs),
            "log_levels": dict(self.log_levels),
            "services": dict(self.service_counts),
            "error_patterns": dict(self.error_patterns),
            "time_range": {
                "start": oldest.isoformat(),
                "end": newest.isoformat(),
                "duration_minutes": (newest - oldest).total_seconds() / 60
            }
        }

# Global aggregator instance
aggregator = LogAggregator()

# Background task to read log files
async def tail_log_file(file_path: str, service_name: str):
    """Tail a log file and add entries to aggregator"""
    if not os.path.exists(file_path):
        return
    
    # Get last position
    last_position = aggregator.log_file_positions.get(file_path, 0)
    
    try:
        async with aiofiles.open(file_path, 'r') as f:
            await f.seek(last_position)
            
            while True:
                line = await f.readline()
                if not line:
                    aggregator.log_file_positions[file_path] = await f.tell()
                    await asyncio.sleep(1)
                    continue
                
                # Parse log line (adjust based on your log format)
                try:
                    # Try JSON format first
                    log_data = json.loads(line)
                    entry = LogEntry(
                        timestamp=datetime.fromisoformat(log_data.get("timestamp", datetime.utcnow().isoformat())),
                        level=log_data.get("level", "INFO"),
                        service=service_name,
                        message=log_data.get("message", line),
                        metadata=log_data.get("metadata"),
                        trace_id=log_data.get("trace_id"),
                        agent_id=log_data.get("agent_id")
                    )
                except json.JSONDecodeError:
                    # Fallback to text parsing
                    parts = line.strip().split(" ", 3)
                    if len(parts) >= 4:
                        entry = LogEntry(
                            timestamp=datetime.utcnow(),
                            level=parts[1] if parts[1] in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] else "INFO",
                            service=service_name,
                            message=line.strip()
                        )
                    else:
                        continue
                
                await aggregator.add_log(entry)
                
    except Exception as e:
        print(f"Error tailing log file {file_path}: {e}")

# Start log collection tasks
async def start_log_collection():
    """Start background tasks to collect logs from various sources"""
    log_sources = [
        ("/app/logs/startup.log", "startup"),
        ("/app/logs/agent0.log", "agent0"),
        ("/app/logs/agent1.log", "agent1"),
        ("/app/logs/agent2.log", "agent2"),
        ("/app/logs/agent3.log", "agent3"),
        ("/app/logs/agent4.log", "agent4"),
        ("/app/logs/agent5.log", "agent5"),
        ("/app/logs/network.log", "network"),
        ("/app/logs/frontend.log", "frontend"),
        ("/app/logs/error.log", "errors"),
    ]
    
    tasks = []
    for file_path, service_name in log_sources:
        task = asyncio.create_task(tail_log_file(file_path, service_name))
        tasks.append(task)
    
    return tasks

# API Endpoints
@router.get("/stream")
async def get_logs(
    service: Optional[str] = Query(None, description="Filter by service"),
    level: Optional[str] = Query(None, description="Filter by log level"),
    start_time: Optional[datetime] = Query(None, description="Start time filter"),
    end_time: Optional[datetime] = Query(None, description="End time filter"),
    limit: int = Query(100, description="Maximum number of logs to return"),
    search: Optional[str] = Query(None, description="Search in log messages")
):
    """Get aggregated logs with filtering"""
    logs = aggregator.get_logs(
        service=service,
        level=level,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        search=search
    )
    
    return {
        "count": len(logs),
        "logs": [log.dict() for log in logs]
    }

@router.get("/statistics")
async def get_log_statistics():
    """Get log statistics and metrics"""
    return aggregator.get_statistics()

@router.get("/tail/{service}")
async def tail_service_logs(
    service: str,
    lines: int = Query(50, description="Number of recent lines")
):
    """Get recent logs for a specific service"""
    logs = aggregator.get_logs(service=service, limit=lines)
    
    if not logs:
        raise HTTPException(status_code=404, detail=f"No logs found for service: {service}")
    
    return {
        "service": service,
        "count": len(logs),
        "logs": [log.dict() for log in logs]
    }

@router.get("/errors")
async def get_error_logs(
    minutes: int = Query(60, description="Time window in minutes"),
    limit: int = Query(100, description="Maximum number of errors")
):
    """Get recent error logs"""
    start_time = datetime.utcnow() - timedelta(minutes=minutes)
    
    error_logs = []
    for log in aggregator.logs:
        if log.timestamp >= start_time and log.level in ["ERROR", "CRITICAL"]:
            error_logs.append(log)
    
    # Sort by timestamp descending
    error_logs.sort(key=lambda x: x.timestamp, reverse=True)
    error_logs = error_logs[:limit]
    
    # Group errors by pattern
    error_groups = defaultdict(list)
    for log in error_logs:
        pattern = aggregator._extract_error_pattern(log.message) or "Other"
        error_groups[pattern].append(log)
    
    return {
        "time_window_minutes": minutes,
        "total_errors": len(error_logs),
        "error_groups": {
            pattern: {
                "count": len(logs),
                "recent_examples": [log.dict() for log in logs[:3]]
            }
            for pattern, logs in error_groups.items()
        },
        "all_errors": [log.dict() for log in error_logs]
    }

@router.get("/search")
async def search_logs(
    query: str = Query(..., description="Search query"),
    regex: bool = Query(False, description="Use regex search"),
    case_sensitive: bool = Query(False, description="Case sensitive search"),
    limit: int = Query(100, description="Maximum results")
):
    """Search logs with text or regex"""
    results = []
    
    for log in aggregator.logs:
        if regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            if re.search(query, log.message, flags):
                results.append(log)
        else:
            if case_sensitive:
                if query in log.message:
                    results.append(log)
            else:
                if query.lower() in log.message.lower():
                    results.append(log)
    
    # Sort by timestamp descending
    results.sort(key=lambda x: x.timestamp, reverse=True)
    results = results[:limit]
    
    return {
        "query": query,
        "regex": regex,
        "case_sensitive": case_sensitive,
        "count": len(results),
        "results": [log.dict() for log in results]
    }

@router.post("/ingest")
async def ingest_log(entry: LogEntry):
    """Manually ingest a log entry"""
    await aggregator.add_log(entry)
    return {"status": "ok", "message": "Log entry added"}

# Initialize log collection on module import
# This will be started when the FastAPI app starts
log_collection_tasks = []

async def init_log_aggregator():
    """Initialize log aggregator - call this from FastAPI startup"""
    global log_collection_tasks
    log_collection_tasks = await start_log_collection()
    return log_collection_tasks

async def shutdown_log_aggregator():
    """Shutdown log aggregator - call this from FastAPI shutdown"""
    global log_collection_tasks
    for task in log_collection_tasks:
        task.cancel()
    await asyncio.gather(*log_collection_tasks, return_exceptions=True)