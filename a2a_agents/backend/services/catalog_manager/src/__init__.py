"""
Catalog Manager Agent - A2A Microservice
ORD Registry and Data Product Catalog
"""

from .agent import (
    CatalogManagerAgent, 
    ORDResource, 
    DataProductEntry,
    ResourceType,
    ResourceStatus
)
from .router import create_a2a_router

__all__ = [
    "CatalogManagerAgent",
    "ORDResource",
    "DataProductEntry", 
    "ResourceType",
    "ResourceStatus",
    "create_a2a_router"
]