"""
Data Manager Agent - A2A Microservice
Central data persistence service for the A2A network
"""

from .agent import DataManagerAgent, DataRecord, QueryResult, StorageBackend
from .router import create_a2a_router

__all__ = [
    "DataManagerAgent",
    "DataRecord", 
    "QueryResult",
    "StorageBackend",
    "create_a2a_router"
]