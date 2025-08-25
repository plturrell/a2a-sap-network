"""
Common imports for A2A agents
Reduces redundant imports across the platform
"""

# Standard library imports
import asyncio
import logging
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from uuid import uuid4
from pathlib import Path

# A2A core imports (commonly used across agents)
try:
    from app.a2a.sdk.types import A2AMessage, MessagePart, AgentCard
    from app.a2a.core.secure_agent_base import SecureA2AAgent
    from app.core.loggingConfig import get_logger, LogCategory
except ImportError:
    # Fallback imports for standalone operation
    pass

__all__ = [
    'asyncio', 'logging', 'time', 'json', 'os',
    'Dict', 'List', 'Any', 'Optional', 'Tuple', 'Union',
    'datetime', 'timedelta', 'uuid4', 'Path',
    'A2AMessage', 'MessagePart', 'AgentCard', 'SecureA2AAgent',
    'get_logger', 'LogCategory'
]
